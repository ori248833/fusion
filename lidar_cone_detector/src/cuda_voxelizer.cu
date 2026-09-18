#include "lidar_cone_detector/cuda_voxelizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace lidar_cone_detector
{
namespace
{

void checkCuda(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

class ScopedCudaEventTimer
{
public:
  // 这个小工具用 CUDA event 测 GPU stream 上的耗时。
  // out_ms 为空时完全不计时，避免关闭延时测试后产生额外同步。
  // event 采用 thread_local 复用：latency_test=true 时每帧会测 H2D/reset/voxel/pad
  // 多个小段，如果每段都 create/destroy event，host p95 会被计时工具本身污染。
  ScopedCudaEventTimer(cudaStream_t stream, double * out_ms)
  : stream_(stream), out_ms_(out_ms)
  {
    if (out_ms_ != nullptr) {
      start_ = eventStart();
      stop_ = eventStop();
      if (start_ == nullptr) {
        checkCuda(cudaEventCreate(&eventStart()), "cudaEventCreate(start)");
        start_ = eventStart();
      }
      if (stop_ == nullptr) {
        checkCuda(cudaEventCreate(&eventStop()), "cudaEventCreate(stop)");
        stop_ = eventStop();
      }
      checkCuda(cudaEventRecord(start_, stream_), "cudaEventRecord(start)");
    }
  }

  ~ScopedCudaEventTimer()
  {
    if (out_ms_ != nullptr) {
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float elapsed = 0.0F;
      cudaEventElapsedTime(&elapsed, start_, stop_);
      *out_ms_ = static_cast<double>(elapsed);
    }
  }

private:
  static cudaEvent_t & eventStart()
  {
    thread_local cudaEvent_t event{nullptr};
    return event;
  }

  static cudaEvent_t & eventStop()
  {
    thread_local cudaEvent_t event{nullptr};
    return event;
  }

  cudaStream_t stream_{nullptr};
  double * out_ms_{nullptr};
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
};

__device__ __forceinline__ int32_t divFloorToInt(float value)
{
  return static_cast<int32_t>(floorf(value));
}

template<typename T>
__device__ __forceinline__ T loadUnaligned(const uint8_t * ptr)
{
  T value{};
  uint8_t * out = reinterpret_cast<uint8_t *>(&value);
#pragma unroll
  for (int32_t i = 0; i < static_cast<int32_t>(sizeof(T)); ++i) {
    out[i] = ptr[i];
  }
  return value;
}

__device__ __forceinline__ float readPointFieldAsFloatDevice(const uint8_t * ptr, uint8_t datatype)
{
  // sensor_msgs::msg::PointField 的枚举值：
  // INT8=1, UINT8=2, INT16=3, UINT16=4, INT32=5, UINT32=6, FLOAT32=7, FLOAT64=8。
  switch (datatype) {
    case 1:
      return static_cast<float>(loadUnaligned<int8_t>(ptr));
    case 2:
      return static_cast<float>(loadUnaligned<uint8_t>(ptr));
    case 3:
      return static_cast<float>(loadUnaligned<int16_t>(ptr));
    case 4:
      return static_cast<float>(loadUnaligned<uint16_t>(ptr));
    case 5:
      return static_cast<float>(loadUnaligned<int32_t>(ptr));
    case 6:
      return static_cast<float>(loadUnaligned<uint32_t>(ptr));
    case 7:
      return loadUnaligned<float>(ptr);
    case 8:
      return static_cast<float>(loadUnaligned<double>(ptr));
    default:
      return 0.0F;
  }
}

bool canPackCompactXyzi(const PointCloudLayout & layout)
{
  // 当前 Hesai topic 是 point_step=26，但 x/y/z/intensity 正好是前 16 字节 float32。
  // 推理不使用 ring/timestamp，所以可以只搬运 XYZI，减少 pinned copy 和 H2D 字节数。
  constexpr uint8_t kFloat32Datatype = 7U;
  return layout.point_step > 16 &&
    layout.x_offset == 0 && layout.y_offset == 4 && layout.z_offset == 8 && layout.intensity_offset == 12 &&
    layout.x_datatype == kFloat32Datatype &&
    layout.y_datatype == kFloat32Datatype &&
    layout.z_datatype == kFloat32Datatype &&
    layout.intensity_valid &&
    layout.intensity_datatype == kFloat32Datatype;
}

PointCloudLayout compactXyziLayout()
{
  constexpr uint8_t kFloat32Datatype = 7U;
  PointCloudLayout layout;
  layout.point_step = 16;
  layout.x_offset = 0;
  layout.y_offset = 4;
  layout.z_offset = 8;
  layout.intensity_offset = 12;
  layout.x_datatype = kFloat32Datatype;
  layout.y_datatype = kFloat32Datatype;
  layout.z_datatype = kFloat32Datatype;
  layout.intensity_datatype = kFloat32Datatype;
  layout.intensity_valid = true;
  return layout;
}

__device__ __forceinline__ void insertPointIntoVoxel(
  float x, float y, float z, float intensity,
  float * voxels, int32_t * coords, int32_t * num_points_per_voxel,
  int32_t * voxel_count, int32_t * voxel_map,
  float min_x, float min_y, float min_z,
  float max_x, float max_y, float max_z,
  float voxel_x, float voxel_y, float voxel_z,
  int32_t grid_x, int32_t grid_y, int32_t grid_z,
  int32_t max_points_per_voxel, int32_t max_voxels)
{
  // 和 OpenPCDet 的 range 过滤保持一致：右边界不包含。
  if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
      x < min_x || x >= max_x ||
      y < min_y || y >= max_y ||
      z < min_z || z >= max_z) {
    return;
  }

  const int32_t x_idx = divFloorToInt((x - min_x) / voxel_x);
  const int32_t y_idx = divFloorToInt((y - min_y) / voxel_y);
  const int32_t z_idx = divFloorToInt((z - min_z) / voxel_z);
  if (x_idx < 0 || x_idx >= grid_x || y_idx < 0 || y_idx >= grid_y || z_idx < 0 || z_idx >= grid_z) {
    return;
  }

  const int32_t map_idx = (z_idx * grid_y + y_idx) * grid_x + x_idx;

  // -1 表示空格子，-2 表示其他线程正在给这个格子分配 voxel id。
  // atomicCAS 保证同一个 grid cell 只会创建一个 voxel，其他线程会复用该 voxel id。
  int32_t voxel_idx = atomicCAS(&voxel_map[map_idx], -1, -2);
  if (voxel_idx == -1) {
    voxel_idx = atomicAdd(voxel_count, 1);
    if (voxel_idx >= max_voxels) {
      // -3 表示 voxel 数超过上限，这个 cell 后续点都直接丢弃。
      atomicExch(&voxel_map[map_idx], -3);
      return;
    }

    coords[voxel_idx * 4 + 0] = 0;      // batch index
    coords[voxel_idx * 4 + 1] = z_idx;  // OpenPCDet 坐标顺序：[batch, z, y, x]
    coords[voxel_idx * 4 + 2] = y_idx;
    coords[voxel_idx * 4 + 3] = x_idx;
    __threadfence();
    atomicExch(&voxel_map[map_idx], voxel_idx);
  } else {
    while (voxel_idx == -2) {
      voxel_idx = atomicAdd(&voxel_map[map_idx], 0);
      __nanosleep(32);
    }
    if (voxel_idx < 0 || voxel_idx >= max_voxels) {
      return;
    }
  }

  const int32_t point_offset = atomicAdd(&num_points_per_voxel[voxel_idx], 1);
  if (point_offset >= max_points_per_voxel) {
    return;
  }

  const int64_t base = (static_cast<int64_t>(voxel_idx) * max_points_per_voxel + point_offset) * 4;
  voxels[base + 0] = x;
  voxels[base + 1] = y;
  voxels[base + 2] = z;
  voxels[base + 3] = isfinite(intensity) ? intensity : 0.0F;
}

__global__ void voxelizeKernel(
  const float * points, int32_t num_points,
  float * voxels, int32_t * coords, int32_t * num_points_per_voxel,
  int32_t * voxel_count, int32_t * voxel_map,
  float min_x, float min_y, float min_z,
  float max_x, float max_y, float max_z,
  float voxel_x, float voxel_y, float voxel_z,
  int32_t grid_x, int32_t grid_y, int32_t grid_z,
  int32_t max_points_per_voxel, int32_t max_voxels)
{
  const int32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) {
    return;
  }

  const float x = points[point_idx * 4 + 0];
  const float y = points[point_idx * 4 + 1];
  const float z = points[point_idx * 4 + 2];
  const float intensity = points[point_idx * 4 + 3];

  insertPointIntoVoxel(
    x, y, z, intensity,
    voxels, coords, num_points_per_voxel, voxel_count, voxel_map,
    min_x, min_y, min_z, max_x, max_y, max_z, voxel_x, voxel_y, voxel_z,
    grid_x, grid_y, grid_z, max_points_per_voxel, max_voxels);
}

__global__ void voxelizeRawPointCloud2Kernel(
  const uint8_t * raw_points, int32_t num_points, PointCloudLayout layout,
  float * voxels, int32_t * coords, int32_t * num_points_per_voxel,
  int32_t * voxel_count, int32_t * voxel_map,
  float min_x, float min_y, float min_z,
  float max_x, float max_y, float max_z,
  float voxel_x, float voxel_y, float voxel_z,
  int32_t grid_x, int32_t grid_y, int32_t grid_z,
  int32_t max_points_per_voxel, int32_t max_voxels)
{
  const int32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) {
    return;
  }

  const uint8_t * ptr = raw_points + static_cast<int64_t>(point_idx) * layout.point_step;
  const float x = readPointFieldAsFloatDevice(ptr + layout.x_offset, layout.x_datatype);
  const float y = readPointFieldAsFloatDevice(ptr + layout.y_offset, layout.y_datatype);
  const float z = readPointFieldAsFloatDevice(ptr + layout.z_offset, layout.z_datatype);
  const float intensity = layout.intensity_valid ?
    readPointFieldAsFloatDevice(ptr + layout.intensity_offset, layout.intensity_datatype) : 0.0F;

  insertPointIntoVoxel(
    x, y, z, intensity,
    voxels, coords, num_points_per_voxel, voxel_count, voxel_map,
    min_x, min_y, min_z, max_x, max_y, max_z, voxel_x, voxel_y, voxel_z,
    grid_x, grid_y, grid_z, max_points_per_voxel, max_voxels);
}

__global__ void clampVoxelPointCountsKernel(int32_t * num_points_per_voxel, int32_t max_voxels, int32_t max_points)
{
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_voxels) {
    return;
  }
  if (num_points_per_voxel[idx] > max_points) {
    num_points_per_voxel[idx] = max_points;
  }
}

__global__ void padVoxelBuffersKernel(
  float * voxels, int32_t * coords, int32_t * num_points_per_voxel,
  int32_t real_voxels, int32_t target_voxels, int32_t max_points_per_voxel)
{
  const int32_t dst = real_voxels + blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= target_voxels) {
    return;
  }

  for (int32_t j = 0; j < max_points_per_voxel * 4; ++j) {
    voxels[dst * max_points_per_voxel * 4 + j] = 0.0F;
  }
  coords[dst * 4 + 0] = 0;
  coords[dst * 4 + 1] = 0;
  coords[dst * 4 + 2] = 0;
  coords[dst * 4 + 3] = 0;
  // 这里绝对不能写 0。PillarVFE 会用 num_points 做除法，0 会导致 NaN。
  num_points_per_voxel[dst] = 1;
}

}  // namespace

CudaVoxelizer::CudaVoxelizer(const VoxelizerConfig & config)
: config_(config)
{
  // 根据 my_cone_anchorfree.yaml 的 point_cloud_range 和 voxel_size 计算网格大小。
  // 当前配置是 x:19.2m / 0.12 = 160, y:38.4m / 0.12 = 320, z:4m / 4 = 1。
  grid_x_ = static_cast<int32_t>(
    std::round((config_.point_cloud_range[3] - config_.point_cloud_range[0]) / config_.voxel_size[0]));
  grid_y_ = static_cast<int32_t>(
    std::round((config_.point_cloud_range[4] - config_.point_cloud_range[1]) / config_.voxel_size[1]));
  grid_z_ = static_cast<int32_t>(
    std::round((config_.point_cloud_range[5] - config_.point_cloud_range[2]) / config_.voxel_size[2]));
  grid_cells_ = grid_x_ * grid_y_ * grid_z_;

  if (grid_x_ <= 0 || grid_y_ <= 0 || grid_z_ <= 0) {
    throw std::runtime_error("Invalid voxel grid size from point_cloud_range and voxel_size");
  }
  if (config_.max_points_per_voxel <= 0 || config_.max_voxels <= 0 || config_.max_input_points <= 0) {
    throw std::runtime_error("Invalid voxelizer max limits");
  }

  checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(voxelizer)");
  allocateDeviceMemory();
  // Reserve the raw PointCloud2 staging buffers up front so normal frames do not
  // hit cudaMallocHost/cudaMalloc growth paths. Direct raw mode needs room for
  // Hesai's 26 byte point_step, so reserve a conservative 32 bytes per point.
  const size_t reserve_point_step = (config_.direct_raw_h2d || !config_.compact_raw_h2d) ? 32U : 16U;
  ensureRawBuffers(static_cast<size_t>(config_.max_input_points) * reserve_point_step);
}

CudaVoxelizer::~CudaVoxelizer()
{
  freeDeviceMemory();
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

void CudaVoxelizer::allocateDeviceMemory()
{
  checkCuda(
    cudaMalloc(&d_points_, static_cast<size_t>(config_.max_input_points) * 4 * sizeof(float)),
    "cudaMalloc(points)");
  checkCuda(
    cudaMalloc(
      &d_voxels_,
      static_cast<size_t>(config_.max_voxels) * config_.max_points_per_voxel * 4 * sizeof(float)),
    "cudaMalloc(voxels)");
  checkCuda(cudaMalloc(&d_coords_, static_cast<size_t>(config_.max_voxels) * 4 * sizeof(int32_t)), "cudaMalloc(coords)");
  checkCuda(cudaMalloc(&d_num_points_, static_cast<size_t>(config_.max_voxels) * sizeof(int32_t)), "cudaMalloc(num_points)");
  checkCuda(cudaMalloc(&d_voxel_count_, sizeof(int32_t)), "cudaMalloc(voxel_count)");
  checkCuda(cudaMalloc(&d_voxel_map_, static_cast<size_t>(grid_cells_) * sizeof(int32_t)), "cudaMalloc(voxel_map)");
}

void CudaVoxelizer::freeDeviceMemory()
{
  if (h_pinned_raw_ != nullptr) {
    cudaFreeHost(h_pinned_raw_);
  }
  cudaFree(d_raw_points_);
  cudaFree(d_points_);
  cudaFree(d_voxels_);
  cudaFree(d_coords_);
  cudaFree(d_num_points_);
  cudaFree(d_voxel_count_);
  cudaFree(d_voxel_map_);
  h_pinned_raw_ = nullptr;
  h_pinned_raw_bytes_ = 0U;
  d_raw_points_ = nullptr;
  d_raw_points_bytes_ = 0U;
  d_points_ = nullptr;
  d_voxels_ = nullptr;
  d_coords_ = nullptr;
  d_num_points_ = nullptr;
  d_voxel_count_ = nullptr;
  d_voxel_map_ = nullptr;
}

void CudaVoxelizer::ensureRawBuffers(size_t bytes)
{
  if (bytes == 0U) {
    return;
  }
  if (h_pinned_raw_bytes_ < bytes) {
    if (h_pinned_raw_ != nullptr) {
      checkCuda(cudaFreeHost(h_pinned_raw_), "cudaFreeHost(raw)");
      h_pinned_raw_ = nullptr;
      h_pinned_raw_bytes_ = 0U;
    }
    // pinned host memory 让 raw PointCloud2 -> device raw buffer 的 H2D 可以真正异步。
    checkCuda(cudaMallocHost(reinterpret_cast<void **>(&h_pinned_raw_), bytes), "cudaMallocHost(raw)");
    h_pinned_raw_bytes_ = bytes;
  }
  if (d_raw_points_bytes_ < bytes) {
    cudaFree(d_raw_points_);
    d_raw_points_ = nullptr;
    d_raw_points_bytes_ = 0U;
    checkCuda(cudaMalloc(&d_raw_points_, bytes), "cudaMalloc(raw PointCloud2)");
    d_raw_points_bytes_ = bytes;
  }
}

int32_t CudaVoxelizer::voxelize(const float * host_points, int32_t num_points, VoxelizerTimings * timings)
{
  if (timings != nullptr) {
    *timings = VoxelizerTimings{};
  }

  const int32_t clamped_points = std::max(0, std::min(num_points, config_.max_input_points));

  {
    // 点云先从 host 拷到 device。后续 voxelize、TensorRT 都在 GPU 上继续处理。
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->h2d_points_ms : nullptr);
    if (clamped_points > 0 && host_points != nullptr) {
      checkCuda(
        cudaMemcpyAsync(
          d_points_, host_points, static_cast<size_t>(clamped_points) * 4 * sizeof(float),
          cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(points H2D)");
    }
  }

  {
    // 每帧都要清空 voxel 输出、voxel 计数器和 grid->voxel 映射表。
    // voxel_map 用 0xFF 初始化，相当于 int32_t 的 -1。
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->reset_ms : nullptr);
    // 正常 split 路径只使用 [0, real_voxels) 的 coords，以及每个 voxel 的
    // [0, voxel_num_points) 真实点；padding 点不会被 CUDA VFE 读取。因此默认跳过
    // d_voxels_/d_coords_ 的全量清零，少搬运约 10MB/帧。排查输入残留时可用
    // clear_voxel_buffers:=true 恢复完整清零。
    if (config_.clear_voxel_buffers) {
      checkCuda(
        cudaMemsetAsync(
          d_voxels_, 0,
          static_cast<size_t>(config_.max_voxels) * config_.max_points_per_voxel * 4 * sizeof(float),
          stream_),
        "cudaMemsetAsync(voxels)");
      checkCuda(
        cudaMemsetAsync(
          d_coords_, 0, static_cast<size_t>(config_.max_voxels) * 4 * sizeof(int32_t), stream_),
        "cudaMemsetAsync(coords)");
    }
    checkCuda(cudaMemsetAsync(d_num_points_, 0, static_cast<size_t>(config_.max_voxels) * sizeof(int32_t), stream_), "cudaMemsetAsync(num_points)");
    checkCuda(cudaMemsetAsync(d_voxel_count_, 0, sizeof(int32_t), stream_), "cudaMemsetAsync(voxel_count)");
    checkCuda(cudaMemsetAsync(d_voxel_map_, 0xFF, static_cast<size_t>(grid_cells_) * sizeof(int32_t), stream_), "cudaMemsetAsync(voxel_map)");
  }

  {
    // 一个 CUDA 线程处理一个点，符合范围的点会被散到对应 pillar。
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->voxelize_ms : nullptr);
    if (clamped_points > 0) {
      constexpr int32_t threads = 256;
      const int32_t blocks = (clamped_points + threads - 1) / threads;
      voxelizeKernel<<<blocks, threads, 0, stream_>>>(
        d_points_, clamped_points,
        d_voxels_, d_coords_, d_num_points_, d_voxel_count_, d_voxel_map_,
        config_.point_cloud_range[0], config_.point_cloud_range[1], config_.point_cloud_range[2],
        config_.point_cloud_range[3], config_.point_cloud_range[4], config_.point_cloud_range[5],
        config_.voxel_size[0], config_.voxel_size[1], config_.voxel_size[2],
        grid_x_, grid_y_, grid_z_,
        config_.max_points_per_voxel, config_.max_voxels);
      checkCuda(cudaGetLastError(), "voxelizeKernel launch");
    }

    constexpr int32_t threads = 256;
    const int32_t blocks = (config_.max_voxels + threads - 1) / threads;
    clampVoxelPointCountsKernel<<<blocks, threads, 0, stream_>>>(
      d_num_points_, config_.max_voxels, config_.max_points_per_voxel);
    checkCuda(cudaGetLastError(), "clampVoxelPointCountsKernel launch");
  }

  int32_t host_count = 0;
  // 只同步一个 voxel_count 回 CPU，因为 TensorRT dynamic shape 需要知道本帧 N。
  const auto sync_begin = std::chrono::steady_clock::now();
  checkCuda(
    cudaMemcpyAsync(&host_count, d_voxel_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync(voxel_count D2H)");
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(voxelizer)");
  const auto sync_end = std::chrono::steady_clock::now();
  if (timings != nullptr) {
    timings->count_sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_begin).count();
  }

  return std::max(0, std::min(host_count, config_.max_voxels));
}

int32_t CudaVoxelizer::voxelizePointCloud2Raw(
  const uint8_t * host_raw, size_t host_raw_bytes, int32_t num_points,
  const PointCloudLayout & layout, VoxelizerTimings * timings)
{
  if (timings != nullptr) {
    *timings = VoxelizerTimings{};
  }

  if (host_raw == nullptr || host_raw_bytes == 0U || layout.point_step <= 0 || num_points <= 0) {
    return 0;
  }

  const int32_t available_points = static_cast<int32_t>(
    std::min(static_cast<size_t>(std::max(0, num_points)), host_raw_bytes / static_cast<size_t>(layout.point_step)));
  const int32_t clamped_points = std::max(0, std::min(available_points, config_.max_input_points));
  const bool pack_compact_xyzi =
    config_.compact_raw_h2d && !config_.direct_raw_h2d && canPackCompactXyzi(layout);
  const size_t device_point_step = pack_compact_xyzi ? 16U : static_cast<size_t>(layout.point_step);
  const size_t raw_bytes = static_cast<size_t>(clamped_points) * device_point_step;
  const PointCloudLayout device_layout = pack_compact_xyzi ? compactXyziLayout() : layout;
  ensureRawBuffers(raw_bytes);

  {
    // 只复制 TensorRT 前处理需要的 bytes，不再在 CPU 上展开成 std::vector<float>。
    // 若 x/y/z/intensity 已经是前 16 字节 float32，则丢掉 ring/timestamp，打包成紧凑 XYZI。
    const auto copy_begin = std::chrono::steady_clock::now();
    if (pack_compact_xyzi) {
      for (int32_t i = 0; i < clamped_points; ++i) {
        std::memcpy(
          h_pinned_raw_ + static_cast<size_t>(i) * 16U,
          host_raw + static_cast<size_t>(i) * static_cast<size_t>(layout.point_step),
          16U);
      }
    } else {
      std::memcpy(h_pinned_raw_, host_raw, raw_bytes);
    }
    const auto copy_end = std::chrono::steady_clock::now();
    if (timings != nullptr) {
      timings->pinned_copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
    }
  }

  {
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->h2d_points_ms : nullptr);
    checkCuda(
      cudaMemcpyAsync(d_raw_points_, h_pinned_raw_, raw_bytes, cudaMemcpyHostToDevice, stream_),
      "cudaMemcpyAsync(raw PointCloud2 H2D)");
  }

  {
    // 每帧都清空 voxel 输出、voxel 计数器和 grid->voxel 映射表。
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->reset_ms : nullptr);
    if (config_.clear_voxel_buffers) {
      checkCuda(
        cudaMemsetAsync(
          d_voxels_, 0,
          static_cast<size_t>(config_.max_voxels) * config_.max_points_per_voxel * 4 * sizeof(float),
          stream_),
        "cudaMemsetAsync(voxels)");
      checkCuda(
        cudaMemsetAsync(
          d_coords_, 0, static_cast<size_t>(config_.max_voxels) * 4 * sizeof(int32_t), stream_),
        "cudaMemsetAsync(coords)");
    }
    checkCuda(cudaMemsetAsync(d_num_points_, 0, static_cast<size_t>(config_.max_voxels) * sizeof(int32_t), stream_), "cudaMemsetAsync(num_points)");
    checkCuda(cudaMemsetAsync(d_voxel_count_, 0, sizeof(int32_t), stream_), "cudaMemsetAsync(voxel_count)");
    checkCuda(cudaMemsetAsync(d_voxel_map_, 0xFF, static_cast<size_t>(grid_cells_) * sizeof(int32_t), stream_), "cudaMemsetAsync(voxel_map)");
  }

  {
    // 一个 CUDA kernel 同时完成 PointCloud2 raw 解析、range filter 和 voxelization。
    ScopedCudaEventTimer timer(stream_, timings != nullptr ? &timings->voxelize_ms : nullptr);
    constexpr int32_t threads = 256;
    const int32_t blocks = (clamped_points + threads - 1) / threads;
    voxelizeRawPointCloud2Kernel<<<blocks, threads, 0, stream_>>>(
      d_raw_points_, clamped_points, device_layout,
      d_voxels_, d_coords_, d_num_points_, d_voxel_count_, d_voxel_map_,
      config_.point_cloud_range[0], config_.point_cloud_range[1], config_.point_cloud_range[2],
      config_.point_cloud_range[3], config_.point_cloud_range[4], config_.point_cloud_range[5],
      config_.voxel_size[0], config_.voxel_size[1], config_.voxel_size[2],
      grid_x_, grid_y_, grid_z_,
      config_.max_points_per_voxel, config_.max_voxels);
    checkCuda(cudaGetLastError(), "voxelizeRawPointCloud2Kernel launch");

    const int32_t clamp_blocks = (config_.max_voxels + threads - 1) / threads;
    clampVoxelPointCountsKernel<<<clamp_blocks, threads, 0, stream_>>>(
      d_num_points_, config_.max_voxels, config_.max_points_per_voxel);
    checkCuda(cudaGetLastError(), "clampVoxelPointCountsKernel launch");
  }

  int32_t host_count = 0;
  const auto sync_begin = std::chrono::steady_clock::now();
  checkCuda(
    cudaMemcpyAsync(&host_count, d_voxel_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync(voxel_count D2H)");
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(voxelizer raw)");
  const auto sync_end = std::chrono::steady_clock::now();
  if (timings != nullptr) {
    timings->count_sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_begin).count();
  }

  return std::max(0, std::min(host_count, config_.max_voxels));
}

void CudaVoxelizer::padToCount(int32_t real_voxels, int32_t target_voxels, double * pad_ms)
{
  if (pad_ms != nullptr) {
    *pad_ms = 0.0;
  }
  const int32_t clamped_real = std::max(0, std::min(real_voxels, config_.max_voxels));
  const int32_t clamped_target = std::max(0, std::min(target_voxels, config_.max_voxels));
  if (clamped_target <= clamped_real) {
    return;
  }

  ScopedCudaEventTimer timer(stream_, pad_ms);
  constexpr int32_t threads = 256;
  const int32_t pad_count = clamped_target - clamped_real;
  const int32_t blocks = (pad_count + threads - 1) / threads;
  padVoxelBuffersKernel<<<blocks, threads, 0, stream_>>>(
    d_voxels_, d_coords_, d_num_points_, clamped_real, clamped_target, config_.max_points_per_voxel);
  checkCuda(cudaGetLastError(), "padVoxelBuffersKernel launch");
  // TRT 使用另一个 stream；这里同步一次，确保 padding 完成后再交给 TRT。
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(padToCount)");
}

}  // namespace lidar_cone_detector
