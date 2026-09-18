#include "lidar_cone_detector/gpu_postprocess.hpp"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lidar_cone_detector
{
namespace
{

constexpr int32_t kNmsThreads = 64;

// GPU 后处理主路径处理 anchor-free CenterHead 原始输出：
//   hm       [1, 2, 160, 80] raw logits 或 score
//   center   [1, 2, 160, 80]
//   center_z [1, 1, 160, 80]
//   dim      [1, 3, 160, 80]
//   rot      [1, 2, 160, 80]
// pipeline:
//   1. heatmap 阈值 + 3x3 local max，得到稀疏候选。
//   2. 在 GPU 上 decode center/z/dim/yaw 到真实 3D 框。
//   3. Thrust 在 GPU 上按 score 降序排序，相当于 TopK。
//   4. CUDA 计算 rotated BEV IoU 的 NMS bitmask。
//   5. GPU 单线程做最终贪心选择，只把最终 detections 拷回 CPU。
//
// 这样避免每帧把完整 head tensor 全量 D2H，post 延迟主要受候选数量和
// gpu_nms_pre_max_size 影响，而不是 raw tensor 大小。

struct DevicePoint2D
{
  float x;
  float y;
};

struct DevicePostprocessParams
{
  float raw_score_threshold;
  float nms_threshold;
  float x_min;
  float y_min;
  float z_min;
  float x_max;
  float y_max;
  float z_max;
  float min_box_size;
  float max_box_size;
  float max_z_margin;
  float voxel_x;
  float voxel_y;
  int32_t num_classes;
  int32_t num_preds;
  int32_t heatmap_h;
  int32_t heatmap_w;
  int32_t feature_map_stride;
  int32_t max_candidates;
  int32_t filter_invalid_boxes;
  int32_t cls_output_is_logits;
};

void checkCuda(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

__host__ __device__ bool scoreGreater(
  const GpuPostprocessDetection & a, const GpuPostprocessDetection & b)
{
  return a.score > b.score;
}

struct ScoreGreater
{
  __host__ __device__ bool operator()(
    const GpuPostprocessDetection & a, const GpuPostprocessDetection & b) const
  {
    return scoreGreater(a, b);
  }
};

__device__ float sigmoidDevice(float x)
{
  if (x >= 0.0F) {
    const float z = expf(-x);
    return 1.0F / (1.0F + z);
  }
  const float z = expf(x);
  return z / (1.0F + z);
}

__device__ bool validDetectionDevice(
  const GpuPostprocessDetection & det, const DevicePostprocessParams & params)
{
  if (params.filter_invalid_boxes == 0) {
    return true;
  }
  if (
    !isfinite(det.x) || !isfinite(det.y) || !isfinite(det.z) ||
    !isfinite(det.length) || !isfinite(det.width) || !isfinite(det.height) ||
    !isfinite(det.yaw) || !isfinite(det.score))
  {
    return false;
  }

  const bool center_inside_xy =
    det.x >= params.x_min && det.x <= params.x_max &&
    det.y >= params.y_min && det.y <= params.y_max;
  const bool center_inside_z =
    det.z >= params.z_min - params.max_z_margin &&
    det.z <= params.z_max + params.max_z_margin;
  const bool size_ok =
    det.length >= params.min_box_size && det.length <= params.max_box_size &&
    det.width >= params.min_box_size && det.width <= params.max_box_size &&
    det.height >= params.min_box_size && det.height <= params.max_box_size;

  return center_inside_xy && center_inside_z && size_ok;
}

__device__ bool isLocalHeatmapMax(
  const float * __restrict__ hm,
  int32_t cls,
  int32_t yy,
  int32_t xx,
  float value,
  int32_t height,
  int32_t width)
{
  // CenterPoint 的 decode 通常先对 heatmap 做 3x3 max-pool NMS。
  // sigmoid 单调递增，因此 raw logit 或 score 都可以直接比较。
  const int32_t y0 = max(0, yy - 1);
  const int32_t y1 = min(height - 1, yy + 1);
  const int32_t x0 = max(0, xx - 1);
  const int32_t x1 = min(width - 1, xx + 1);
  const size_t class_base = static_cast<size_t>(cls) * height * width;
  for (int32_t y = y0; y <= y1; ++y) {
    for (int32_t x = x0; x <= x1; ++x) {
      if (y == yy && x == xx) {
        continue;
      }
      if (hm[class_base + static_cast<size_t>(y) * width + x] > value) {
        return false;
      }
    }
  }
  return true;
}

__global__ void filterCenterHeadKernel(
  const float * __restrict__ hm,
  const float * __restrict__ center,
  const float * __restrict__ center_z,
  const float * __restrict__ dim,
  const float * __restrict__ rot,
  GpuPostprocessDetection * __restrict__ candidates,
  int32_t * __restrict__ candidate_count,
  DevicePostprocessParams params)
{
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= params.num_preds) {
    return;
  }

  const int32_t cells_per_class = params.heatmap_h * params.heatmap_w;
  const int32_t class_id = i / cells_per_class;
  const int32_t cell = i - class_id * cells_per_class;
  const int32_t yy = cell / params.heatmap_w;
  const int32_t xx = cell - yy * params.heatmap_w;

  const size_t heatmap_idx =
    static_cast<size_t>(class_id) * cells_per_class + static_cast<size_t>(cell);
  const float heatmap_value = hm[heatmap_idx];
  if (heatmap_value < params.raw_score_threshold) {
    return;
  }
  if (!isLocalHeatmapMax(
      hm, class_id, yy, xx, heatmap_value, params.heatmap_h, params.heatmap_w)) {
    return;
  }

  const size_t plane = static_cast<size_t>(cells_per_class);
  const float offset_x = center[0U * plane + cell];
  const float offset_y = center[1U * plane + cell];
  const float z = center_z[cell];
  const float length = expf(dim[0U * plane + cell]);
  const float width = expf(dim[1U * plane + cell]);
  const float height = expf(dim[2U * plane + cell]);
  const float sin_yaw = rot[0U * plane + cell];
  const float cos_yaw = rot[1U * plane + cell];

  const float real_stride_x = params.voxel_x * static_cast<float>(params.feature_map_stride);
  const float real_stride_y = params.voxel_y * static_cast<float>(params.feature_map_stride);

  GpuPostprocessDetection det;
  det.x = (static_cast<float>(xx) + offset_x) * real_stride_x + params.x_min;
  det.y = (static_cast<float>(yy) + offset_y) * real_stride_y + params.y_min;
  det.z = z;
  det.length = length;
  det.width = width;
  det.height = height;
  det.yaw = atan2f(sin_yaw, cos_yaw);
  det.score = params.cls_output_is_logits != 0 ? sigmoidDevice(heatmap_value) : heatmap_value;
  det.label = class_id + 1;

  if (!validDetectionDevice(det, params)) {
    return;
  }

  const int32_t dst = atomicAdd(candidate_count, 1);
  if (dst < params.max_candidates) {
    candidates[dst] = det;
  }
}

__global__ void initializeCandidatesKernel(
  GpuPostprocessDetection * __restrict__ candidates,
  int32_t max_candidates)
{
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_candidates) {
    return;
  }

  GpuPostprocessDetection det;
  det.x = 0.0F;
  det.y = 0.0F;
  det.z = 0.0F;
  det.length = 0.0F;
  det.width = 0.0F;
  det.height = 0.0F;
  det.yaw = 0.0F;
  det.score = -1.0e30F;
  det.label = 0;
  candidates[idx] = det;
}

__global__ void clampCountKernel(
  const int32_t * __restrict__ input_count,
  int32_t * __restrict__ output_count,
  int32_t max_count)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int32_t count = max(0, *input_count);
    *output_count = min(count, max_count);
  }
}

__device__ float crossDevice(
  const DevicePoint2D & a, const DevicePoint2D & b, const DevicePoint2D & c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ DevicePoint2D lineIntersectionDevice(
  const DevicePoint2D & p1, const DevicePoint2D & p2,
  const DevicePoint2D & q1, const DevicePoint2D & q2)
{
  const float a1 = p2.y - p1.y;
  const float b1 = p1.x - p2.x;
  const float c1 = a1 * p1.x + b1 * p1.y;
  const float a2 = q2.y - q1.y;
  const float b2 = q1.x - q2.x;
  const float c2 = a2 * q1.x + b2 * q1.y;
  const float det = a1 * b2 - a2 * b1;
  if (fabsf(det) < 1.0e-6F) {
    return p2;
  }
  DevicePoint2D out;
  out.x = (b2 * c1 - b1 * c2) / det;
  out.y = (a1 * c2 - a2 * c1) / det;
  return out;
}

__device__ void boxCornersDevice(const GpuPostprocessDetection & box, DevicePoint2D corners[4])
{
  const float c = cosf(box.yaw);
  const float s = sinf(box.yaw);
  const float hx = box.length * 0.5F;
  const float hy = box.width * 0.5F;

  const float local_x[4] = {hx, -hx, -hx, hx};
  const float local_y[4] = {hy, hy, -hy, -hy};
  for (int32_t i = 0; i < 4; ++i) {
    corners[i].x = box.x + local_x[i] * c - local_y[i] * s;
    corners[i].y = box.y + local_x[i] * s + local_y[i] * c;
  }
}

__device__ float polygonAreaDevice(const DevicePoint2D * poly, int32_t count)
{
  if (count < 3) {
    return 0.0F;
  }
  float area = 0.0F;
  for (int32_t i = 0; i < count; ++i) {
    const DevicePoint2D & p = poly[i];
    const DevicePoint2D & q = poly[(i + 1) % count];
    area += p.x * q.y - p.y * q.x;
  }
  return fabsf(area) * 0.5F;
}

__device__ int32_t clipPolygonWithEdgeDevice(
  const DevicePoint2D * subject,
  int32_t subject_count,
  const DevicePoint2D & edge_start,
  const DevicePoint2D & edge_end,
  DevicePoint2D * output)
{
  if (subject_count <= 0) {
    return 0;
  }

  int32_t output_count = 0;
  DevicePoint2D previous = subject[subject_count - 1];
  bool previous_inside = crossDevice(edge_start, edge_end, previous) >= -1.0e-6F;
  for (int32_t i = 0; i < subject_count; ++i) {
    const DevicePoint2D current = subject[i];
    const bool current_inside = crossDevice(edge_start, edge_end, current) >= -1.0e-6F;
    if (current_inside != previous_inside && output_count < 8) {
      output[output_count++] = lineIntersectionDevice(previous, current, edge_start, edge_end);
    }
    if (current_inside && output_count < 8) {
      output[output_count++] = current;
    }
    previous = current;
    previous_inside = current_inside;
  }
  return output_count;
}

__device__ float orientedIouBevDevice(
  const GpuPostprocessDetection & a, const GpuPostprocessDetection & b)
{
  if (a.length <= 0.0F || a.width <= 0.0F || b.length <= 0.0F || b.width <= 0.0F) {
    return 0.0F;
  }

  DevicePoint2D corners_a[4];
  DevicePoint2D corners_b[4];
  boxCornersDevice(a, corners_a);
  boxCornersDevice(b, corners_b);

  DevicePoint2D poly_a[8];
  DevicePoint2D poly_b[8];
  int32_t poly_count = 4;
  for (int32_t i = 0; i < 4; ++i) {
    poly_a[i] = corners_a[i];
  }

  for (int32_t edge = 0; edge < 4; ++edge) {
    poly_count = clipPolygonWithEdgeDevice(
      poly_a, poly_count, corners_b[edge], corners_b[(edge + 1) & 3], poly_b);
    if (poly_count <= 0) {
      return 0.0F;
    }
    for (int32_t i = 0; i < poly_count; ++i) {
      poly_a[i] = poly_b[i];
    }
  }

  const float inter_area = polygonAreaDevice(poly_a, poly_count);
  const float area_a = a.length * a.width;
  const float area_b = b.length * b.width;
  const float denom = area_a + area_b - inter_area;
  return denom > 1.0e-6F ? inter_area / denom : 0.0F;
}

__global__ void nmsMaskKernel(
  const GpuPostprocessDetection * __restrict__ boxes,
  int32_t num_boxes,
  float nms_threshold,
  uint64_t * __restrict__ mask,
  int32_t col_blocks,
  const int32_t * __restrict__ active_count)
{
  const int32_t active_boxes = min(num_boxes, max(0, *active_count));
  const int32_t row_start = blockIdx.y;
  const int32_t col_start = blockIdx.x;
  if (row_start > col_start) {
    return;
  }

  const int32_t row_size = min(active_boxes - row_start * kNmsThreads, kNmsThreads);
  const int32_t col_size = min(active_boxes - col_start * kNmsThreads, kNmsThreads);
  if (row_size <= 0 || col_size <= 0) {
    return;
  }

  __shared__ GpuPostprocessDetection block_boxes[kNmsThreads];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x] = boxes[col_start * kNmsThreads + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x >= row_size) {
    return;
  }

  const int32_t cur_box_idx = row_start * kNmsThreads + threadIdx.x;
  const GpuPostprocessDetection cur_box = boxes[cur_box_idx];
  const int32_t start = row_start == col_start ? threadIdx.x + 1 : 0;
  uint64_t bits = 0ULL;
  for (int32_t i = start; i < col_size; ++i) {
    if (orientedIouBevDevice(cur_box, block_boxes[i]) > nms_threshold) {
      bits |= 1ULL << static_cast<uint32_t>(i);
    }
  }
  mask[static_cast<size_t>(cur_box_idx) * col_blocks + col_start] = bits;
}

__global__ void nmsSelectKernel(
  const GpuPostprocessDetection * __restrict__ boxes,
  const uint64_t * __restrict__ mask,
  uint64_t * __restrict__ removed,
  GpuPostprocessDetection * __restrict__ kept,
  int32_t * __restrict__ kept_count,
  int32_t num_boxes,
  int32_t col_blocks,
  int32_t max_kept,
  const int32_t * __restrict__ active_count)
{
  // NMS 的贪心选择本身是串行依赖：第 i 个框是否保留取决于所有更高分框的 suppress 结果。
  // 这里用单线程在 GPU 上做最后的 bitset 合并，避免把 K x ceil(K/64) 的 mask 拷回 CPU。
  // 对默认 K=1024，循环只有约 16K 个 64-bit OR，远小于 rotated IoU mask 计算本身。
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int32_t out_count = 0;
  const int32_t active_boxes = min(num_boxes, max(0, *active_count));
  for (int32_t i = 0; i < active_boxes; ++i) {
    const int32_t block = i / kNmsThreads;
    const int32_t in_block = i % kNmsThreads;
    const uint64_t bit = 1ULL << static_cast<uint32_t>(in_block);
    if ((removed[block] & bit) != 0ULL) {
      continue;
    }

    if (out_count < max_kept) {
      kept[out_count] = boxes[i];
    }
    ++out_count;
    if (out_count >= max_kept) {
      break;
    }

    const uint64_t * row = mask + static_cast<size_t>(i) * col_blocks;
    for (int32_t j = block; j < col_blocks; ++j) {
      removed[j] |= row[j];
    }
  }
  *kept_count = min(out_count, max_kept);
}

}  // namespace

GpuPostprocessor::GpuPostprocessor(const PostprocessConfig & config)
: config_(config)
{
  max_candidates_ = std::max(1, config_.gpu_candidate_buffer_size);
  const int32_t requested_gpu_pre_nms =
    config_.gpu_nms_pre_max_size > 0 ? config_.gpu_nms_pre_max_size : config_.nms_pre_max_size;
  // CPU reference keeps the historical nms_pre_max_size. GPU NMS has O(K^2)
  // rotated-IoU work and an O(K * ceil(K/64)) bitmask copy, so it gets its own
  // conservative TopK. For cones, K=1024 is usually far above the number of
  // meaningful boxes while avoiding the p95 spikes from 4096-way NMS.
  nms_pre_max_size_ = std::max(
    1, std::min({config_.nms_pre_max_size, requested_gpu_pre_nms, max_candidates_}));
  nms_post_max_size_ = std::max(1, config_.nms_post_max_size);
  nms_col_blocks_ = ceilDiv(nms_pre_max_size_, kNmsThreads);
  allocateBuffers();
}

GpuPostprocessor::~GpuPostprocessor()
{
  freeBuffers();
}

int32_t GpuPostprocessor::ceilDiv(int32_t a, int32_t b)
{
  return (a + b - 1) / b;
}

float GpuPostprocessor::rawScoreThreshold(float score_threshold, bool logits)
{
  if (!logits) {
    return score_threshold;
  }
  const float eps = 1.0e-6F;
  const float clamped = std::max(eps, std::min(1.0F - eps, score_threshold));
  return std::log(clamped / (1.0F - clamped));
}

void GpuPostprocessor::allocateBuffers()
{
  const size_t candidate_bytes =
    static_cast<size_t>(max_candidates_) * sizeof(GpuPostprocessDetection);
  const size_t kept_bytes =
    static_cast<size_t>(nms_post_max_size_) * sizeof(GpuPostprocessDetection);
  const size_t mask_bytes =
    static_cast<size_t>(nms_pre_max_size_) * nms_col_blocks_ * sizeof(uint64_t);

  checkCuda(cudaMalloc(&d_candidates_, candidate_bytes), "cudaMalloc(gpu post candidates)");
  checkCuda(cudaMalloc(&d_kept_, kept_bytes), "cudaMalloc(gpu post kept)");
  checkCuda(cudaMalloc(&d_candidate_count_, sizeof(int32_t)), "cudaMalloc(gpu post candidate count)");
  checkCuda(cudaMalloc(&d_active_candidate_count_, sizeof(int32_t)), "cudaMalloc(gpu post active count)");
  checkCuda(cudaMalloc(&d_kept_count_, sizeof(int32_t)), "cudaMalloc(gpu post kept count)");
  checkCuda(cudaMalloc(&d_nms_mask_, mask_bytes), "cudaMalloc(gpu post nms mask)");
  checkCuda(cudaMalloc(&d_removed_, static_cast<size_t>(nms_col_blocks_) * sizeof(uint64_t)), "cudaMalloc(gpu post removed mask)");

  checkCuda(
    cudaMallocHost(reinterpret_cast<void **>(&h_candidate_count_), sizeof(int32_t)),
    "cudaMallocHost(gpu post count)");
  checkCuda(
    cudaMallocHost(reinterpret_cast<void **>(&h_active_candidate_count_), sizeof(int32_t)),
    "cudaMallocHost(gpu post active count)");
  checkCuda(
    cudaMallocHost(reinterpret_cast<void **>(&h_kept_count_), sizeof(int32_t)),
    "cudaMallocHost(gpu post kept count)");
  checkCuda(
    cudaMallocHost(reinterpret_cast<void **>(&h_kept_), kept_bytes),
    "cudaMallocHost(gpu post kept)");
}

void GpuPostprocessor::freeBuffers()
{
  cudaFree(d_candidates_);
  cudaFree(d_kept_);
  cudaFree(d_candidate_count_);
  cudaFree(d_active_candidate_count_);
  cudaFree(d_kept_count_);
  cudaFree(d_nms_mask_);
  cudaFree(d_removed_);
  cudaFreeHost(h_candidate_count_);
  cudaFreeHost(h_active_candidate_count_);
  cudaFreeHost(h_kept_count_);
  cudaFreeHost(h_kept_);
  d_candidates_ = nullptr;
  d_kept_ = nullptr;
  d_candidate_count_ = nullptr;
  d_active_candidate_count_ = nullptr;
  d_kept_count_ = nullptr;
  d_nms_mask_ = nullptr;
  d_removed_ = nullptr;
  h_candidate_count_ = nullptr;
  h_active_candidate_count_ = nullptr;
  h_kept_count_ = nullptr;
  h_kept_ = nullptr;
}

std::vector<Detection> GpuPostprocessor::decode(
  const RawNetworkOutputs & outputs, cudaStream_t stream)
{
  if (
    outputs.output_type != NetworkOutputType::kCenterHeadRaw ||
    outputs.hm_device == nullptr || outputs.center_device == nullptr ||
    outputs.center_z_device == nullptr || outputs.dim_device == nullptr ||
    outputs.rot_device == nullptr)
  {
    return {};
  }

  const bool centerhead_shape_ok =
    outputs.hm_c == config_.num_classes &&
    outputs.center_c == 2 && outputs.center_z_c == 1 &&
    outputs.dim_c == 3 && outputs.rot_c == 2 &&
    outputs.hm_h == outputs.center_h && outputs.hm_h == outputs.center_z_h &&
    outputs.hm_h == outputs.dim_h && outputs.hm_h == outputs.rot_h &&
    outputs.hm_w == outputs.center_w && outputs.hm_w == outputs.center_z_w &&
    outputs.hm_w == outputs.dim_w && outputs.hm_w == outputs.rot_w;

  if (!centerhead_shape_ok) {
    throw std::runtime_error("Unexpected network output shape for GPU postprocess");
  }

  DevicePostprocessParams params;
  params.raw_score_threshold = rawScoreThreshold(config_.score_threshold, config_.cls_output_is_logits);
  params.nms_threshold = config_.nms_threshold;
  params.x_min = config_.point_cloud_range[0];
  params.y_min = config_.point_cloud_range[1];
  params.z_min = config_.point_cloud_range[2];
  params.x_max = config_.point_cloud_range[3];
  params.y_max = config_.point_cloud_range[4];
  params.z_max = config_.point_cloud_range[5];
  params.min_box_size = config_.min_box_size;
  params.max_box_size = config_.max_box_size;
  params.max_z_margin = config_.max_z_margin;
  params.voxel_x = config_.voxel_size[0];
  params.voxel_y = config_.voxel_size[1];
  params.num_classes = config_.num_classes;
  params.num_preds = outputs.hm_c * outputs.hm_h * outputs.hm_w;
  params.heatmap_h = outputs.hm_h;
  params.heatmap_w = outputs.hm_w;
  params.feature_map_stride = config_.feature_map_stride;
  params.max_candidates = max_candidates_;
  params.filter_invalid_boxes = config_.filter_invalid_boxes ? 1 : 0;
  params.cls_output_is_logits = config_.cls_output_is_logits ? 1 : 0;

  constexpr int32_t threads = 256;
  checkCuda(cudaMemsetAsync(d_candidate_count_, 0, sizeof(int32_t), stream), "cudaMemsetAsync(gpu post count)");
  checkCuda(cudaMemsetAsync(d_active_candidate_count_, 0, sizeof(int32_t), stream), "cudaMemsetAsync(gpu post active count)");
  const int32_t init_blocks = ceilDiv(max_candidates_, threads);
  initializeCandidatesKernel<<<init_blocks, threads, 0, stream>>>(d_candidates_, max_candidates_);
  checkCuda(cudaGetLastError(), "initializeCandidatesKernel launch");

  const int32_t blocks = ceilDiv(params.num_preds, threads);
  filterCenterHeadKernel<<<blocks, threads, 0, stream>>>(
    outputs.hm_device, outputs.center_device, outputs.center_z_device,
    outputs.dim_device, outputs.rot_device, d_candidates_, d_candidate_count_, params);
  checkCuda(cudaGetLastError(), "filterCenterHeadKernel launch");

  clampCountKernel<<<1, 1, 0, stream>>>(
    d_candidate_count_, d_active_candidate_count_, max_candidates_);
  checkCuda(cudaGetLastError(), "clampCountKernel launch");

  thrust::device_ptr<GpuPostprocessDetection> begin(d_candidates_);
  thrust::device_ptr<GpuPostprocessDetection> end(d_candidates_ + max_candidates_);
  thrust::sort(thrust::cuda::par.on(stream), begin, end, ScoreGreater{});

  const int32_t nms_input_count = nms_pre_max_size_;
  const int32_t col_blocks = nms_col_blocks_;
  const size_t active_mask_bytes =
    static_cast<size_t>(nms_input_count) * col_blocks * sizeof(uint64_t);
  checkCuda(cudaMemsetAsync(d_nms_mask_, 0, active_mask_bytes, stream), "cudaMemsetAsync(gpu nms mask)");

  const dim3 nms_grid(static_cast<unsigned int>(col_blocks), static_cast<unsigned int>(col_blocks));
  nmsMaskKernel<<<nms_grid, kNmsThreads, 0, stream>>>(
    d_candidates_, nms_input_count, config_.nms_threshold, d_nms_mask_, col_blocks,
    d_active_candidate_count_);
  checkCuda(cudaGetLastError(), "nmsMaskKernel launch");

  checkCuda(
    cudaMemsetAsync(d_removed_, 0, static_cast<size_t>(col_blocks) * sizeof(uint64_t), stream),
    "cudaMemsetAsync(gpu nms removed)");
  checkCuda(cudaMemsetAsync(d_kept_count_, 0, sizeof(int32_t), stream), "cudaMemsetAsync(gpu kept count)");
  nmsSelectKernel<<<1, 1, 0, stream>>>(
    d_candidates_, d_nms_mask_, d_removed_, d_kept_, d_kept_count_,
    nms_input_count, col_blocks, nms_post_max_size_, d_active_candidate_count_);
  checkCuda(cudaGetLastError(), "nmsSelectKernel launch");

  checkCuda(
    cudaMemcpyAsync(
      h_candidate_count_, d_candidate_count_, sizeof(int32_t),
      cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync(gpu post raw count D2H)");
  checkCuda(
    cudaMemcpyAsync(
      h_active_candidate_count_, d_active_candidate_count_, sizeof(int32_t),
      cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync(gpu post active count D2H)");
  checkCuda(
    cudaMemcpyAsync(
      h_kept_count_, d_kept_count_, sizeof(int32_t),
      cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync(gpu post kept count D2H)");
  checkCuda(
    cudaMemcpyAsync(
      h_kept_, d_kept_, static_cast<size_t>(nms_post_max_size_) * sizeof(GpuPostprocessDetection),
      cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync(gpu post kept D2H)");
  checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(gpu post)");

  last_raw_candidate_count_ = std::max(0, *h_candidate_count_);
  last_candidate_overflow_ = last_raw_candidate_count_ > max_candidates_;
  last_nms_input_count_ = std::min(
    std::max(0, *h_active_candidate_count_), nms_pre_max_size_);
  const int32_t kept_count = std::max(0, std::min(*h_kept_count_, nms_post_max_size_));
  std::vector<Detection> kept;
  kept.reserve(static_cast<size_t>(kept_count));
  for (int32_t i = 0; i < kept_count; ++i) {
    const auto & src = h_kept_[i];
    Detection det;
    det.x = src.x;
    det.y = src.y;
    det.z = src.z;
    det.length = src.length;
    det.width = src.width;
    det.height = src.height;
    det.yaw = src.yaw;
    det.score = src.score;
    det.label = src.label;
    kept.push_back(det);
  }

  return kept;
}

}  // namespace lidar_cone_detector
