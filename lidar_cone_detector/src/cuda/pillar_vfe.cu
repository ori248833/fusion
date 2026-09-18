#include "lidar_cone_detector/cuda_vfe.hpp"
#include "lidar_cone_detector/cuda_scatter.hpp"
#include "lidar_cone_detector/vfe_pfn_fused_weights.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace lidar_cone_detector
{
namespace
{

static_assert(pp_vfe_weights::VFE_IN_DIM == kPillarVfeInDim, "Unexpected VFE input dimension");
static_assert(pp_vfe_weights::VFE_OUT_DIM == kPillarVfeOutDim, "Unexpected VFE output dimension");
static_assert(
  pp_vfe_weights::MAX_POINTS_PER_VOXEL == kPillarVfeMaxPoints,
  "Unexpected max points per voxel");

// constant memory 广播读很适合这种所有 voxel 共享的小权重。
__constant__ float c_vfe_w[kPillarVfeOutDim * kPillarVfeInDim];
__constant__ float c_vfe_b[kPillarVfeOutDim];

void checkCuda(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

__global__ void pillarVfeKernel(
  const float * __restrict__ voxels,
  const int32_t * __restrict__ voxel_num_points,
  const int32_t * __restrict__ voxel_coords,
  float * __restrict__ pillar_features,
  int32_t num_voxels)
{
  const int32_t voxel_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t channel = static_cast<int32_t>(threadIdx.x);
  if (voxel_idx >= num_voxels || channel >= kPillarVfeOutDim) {
    return;
  }

  const float * voxel_ptr = voxels + voxel_idx * kPillarVfeMaxPoints * 4;
  const int32_t * coord_ptr = voxel_coords + voxel_idx * 4;

  __shared__ int32_t s_num_points;
  __shared__ float s_mean_x;
  __shared__ float s_mean_y;
  __shared__ float s_mean_z;
  __shared__ float s_center_x;
  __shared__ float s_center_y;
  __shared__ float s_center_z;
  __shared__ float s_point_x[kPillarVfeMaxPoints];
  __shared__ float s_point_y[kPillarVfeMaxPoints];
  __shared__ float s_point_z[kPillarVfeMaxPoints];
  __shared__ float s_point_i[kPillarVfeMaxPoints];

  if (channel == 0) {
    int32_t num_points = voxel_num_points[voxel_idx];
    if (num_points < 0) {
      num_points = 0;
    } else if (num_points > kPillarVfeMaxPoints) {
      num_points = kPillarVfeMaxPoints;
    }
    s_num_points = num_points;

    // OpenPCDet 坐标顺序是 [batch_idx, z_idx, y_idx, x_idx]。
    const int32_t z_idx = coord_ptr[1];
    const int32_t y_idx = coord_ptr[2];
    const int32_t x_idx = coord_ptr[3];
    s_center_x = static_cast<float>(x_idx) * pp_vfe_weights::VOXEL_X + pp_vfe_weights::X_OFFSET;
    s_center_y = static_cast<float>(y_idx) * pp_vfe_weights::VOXEL_Y + pp_vfe_weights::Y_OFFSET;
    s_center_z = static_cast<float>(z_idx) * pp_vfe_weights::VOXEL_Z + pp_vfe_weights::Z_OFFSET;
  }
  __syncthreads();

  const int32_t num_points = s_num_points;
  if (channel < kPillarVfeMaxPoints) {
    if (channel < num_points) {
      const float * pt = voxel_ptr + channel * 4;
      s_point_x[channel] = pt[0];
      s_point_y[channel] = pt[1];
      s_point_z[channel] = pt[2];
      s_point_i[channel] = pt[3];
    } else {
      s_point_x[channel] = 0.0F;
      s_point_y[channel] = 0.0F;
      s_point_z[channel] = 0.0F;
      s_point_i[channel] = 0.0F;
    }
  }
  __syncthreads();

  if (channel == 0) {
    float mean_x = 0.0F;
    float mean_y = 0.0F;
    float mean_z = 0.0F;
    for (int32_t p = 0; p < num_points; ++p) {
      mean_x += s_point_x[p];
      mean_y += s_point_y[p];
      mean_z += s_point_z[p];
    }
    if (num_points > 0) {
      const float inv_n = 1.0F / static_cast<float>(num_points);
      mean_x *= inv_n;
      mean_y *= inv_n;
      mean_z *= inv_n;
    }
    s_mean_x = mean_x;
    s_mean_y = mean_y;
    s_mean_z = mean_z;
  }
  __syncthreads();

  // OpenPCDet 会让 padding 点的 10 维输入为 0。padding 点经过融合层后就是 ReLU(B)，
  // 所以这里直接把它作为初值，后面只遍历真实点，避免每个 pillar 固定多算 32 次。
  float max_val = -FLT_MAX;
  if (num_points < kPillarVfeMaxPoints) {
    max_val = c_vfe_b[channel];
    if (max_val < 0.0F) {
      max_val = 0.0F;
    }
  }
  const float * w = c_vfe_w + channel * kPillarVfeInDim;
  for (int32_t p = 0; p < num_points; ++p) {
    const float x = s_point_x[p];
    const float y = s_point_y[p];
    const float z = s_point_z[p];
    const float intensity = s_point_i[p];

    // 直接展开 10 维 fused Linear+BN，避免每个线程构造 feat[10] 临时数组。
    // 特征顺序必须和 export_vfe_fused_weights.py / OpenPCDet PillarVFE 保持一致。
    float acc = c_vfe_b[channel] +
      w[0] * x +
      w[1] * y +
      w[2] * z +
      w[3] * intensity +
      w[4] * (x - s_mean_x) +
      w[5] * (y - s_mean_y) +
      w[6] * (z - s_mean_z) +
      w[7] * (x - s_center_x) +
      w[8] * (y - s_center_y) +
      w[9] * (z - s_center_z);
    if (acc < 0.0F) {
      acc = 0.0F;
    }
    if (acc > max_val) {
      max_val = acc;
    }
  }

  pillar_features[voxel_idx * kPillarVfeOutDim + channel] = max_val;
}

__global__ void pillarVfeScatterKernel(
  const float * __restrict__ voxels,
  const int32_t * __restrict__ voxel_num_points,
  const int32_t * __restrict__ voxel_coords,
  float * __restrict__ spatial_features,
  int32_t num_voxels)
{
  const int32_t voxel_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t channel = static_cast<int32_t>(threadIdx.x);
  if (voxel_idx >= num_voxels || channel >= kPillarVfeOutDim) {
    return;
  }

  const float * voxel_ptr = voxels + voxel_idx * kPillarVfeMaxPoints * 4;
  const int32_t * coord_ptr = voxel_coords + voxel_idx * 4;

  __shared__ int32_t s_num_points;
  __shared__ int32_t s_x_idx;
  __shared__ int32_t s_y_idx;
  __shared__ float s_mean_x;
  __shared__ float s_mean_y;
  __shared__ float s_mean_z;
  __shared__ float s_center_x;
  __shared__ float s_center_y;
  __shared__ float s_center_z;
  __shared__ float s_point_x[kPillarVfeMaxPoints];
  __shared__ float s_point_y[kPillarVfeMaxPoints];
  __shared__ float s_point_z[kPillarVfeMaxPoints];
  __shared__ float s_point_i[kPillarVfeMaxPoints];

  if (channel == 0) {
    int32_t num_points = voxel_num_points[voxel_idx];
    if (num_points < 0) {
      num_points = 0;
    } else if (num_points > kPillarVfeMaxPoints) {
      num_points = kPillarVfeMaxPoints;
    }
    s_num_points = num_points;

    const int32_t z_idx = coord_ptr[1];
    const int32_t y_idx = coord_ptr[2];
    const int32_t x_idx = coord_ptr[3];
    s_x_idx = x_idx;
    s_y_idx = y_idx;
    s_center_x = static_cast<float>(x_idx) * pp_vfe_weights::VOXEL_X + pp_vfe_weights::X_OFFSET;
    s_center_y = static_cast<float>(y_idx) * pp_vfe_weights::VOXEL_Y + pp_vfe_weights::Y_OFFSET;
    s_center_z = static_cast<float>(z_idx) * pp_vfe_weights::VOXEL_Z + pp_vfe_weights::Z_OFFSET;
  }
  __syncthreads();

  const int32_t num_points = s_num_points;
  if (channel < kPillarVfeMaxPoints) {
    if (channel < num_points) {
      const float * pt = voxel_ptr + channel * 4;
      s_point_x[channel] = pt[0];
      s_point_y[channel] = pt[1];
      s_point_z[channel] = pt[2];
      s_point_i[channel] = pt[3];
    } else {
      s_point_x[channel] = 0.0F;
      s_point_y[channel] = 0.0F;
      s_point_z[channel] = 0.0F;
      s_point_i[channel] = 0.0F;
    }
  }
  __syncthreads();

  if (channel == 0) {
    float mean_x = 0.0F;
    float mean_y = 0.0F;
    float mean_z = 0.0F;
    for (int32_t p = 0; p < num_points; ++p) {
      mean_x += s_point_x[p];
      mean_y += s_point_y[p];
      mean_z += s_point_z[p];
    }
    if (num_points > 0) {
      const float inv_n = 1.0F / static_cast<float>(num_points);
      mean_x *= inv_n;
      mean_y *= inv_n;
      mean_z *= inv_n;
    }
    s_mean_x = mean_x;
    s_mean_y = mean_y;
    s_mean_z = mean_z;
  }
  __syncthreads();

  float max_val = -FLT_MAX;
  if (num_points < kPillarVfeMaxPoints) {
    max_val = c_vfe_b[channel];
    if (max_val < 0.0F) {
      max_val = 0.0F;
    }
  }
  const float * w = c_vfe_w + channel * kPillarVfeInDim;
  for (int32_t p = 0; p < num_points; ++p) {
    const float x = s_point_x[p];
    const float y = s_point_y[p];
    const float z = s_point_z[p];
    const float intensity = s_point_i[p];

    float acc = c_vfe_b[channel] +
      w[0] * x +
      w[1] * y +
      w[2] * z +
      w[3] * intensity +
      w[4] * (x - s_mean_x) +
      w[5] * (y - s_mean_y) +
      w[6] * (z - s_mean_z) +
      w[7] * (x - s_center_x) +
      w[8] * (y - s_center_y) +
      w[9] * (z - s_center_z);
    if (acc < 0.0F) {
      acc = 0.0F;
    }
    if (acc > max_val) {
      max_val = acc;
    }
  }

  if (s_x_idx < 0 || s_x_idx >= kBevWidth || s_y_idx < 0 || s_y_idx >= kBevHeight) {
    return;
  }
  const int32_t out_idx = channel * kBevHeight * kBevWidth + s_y_idx * kBevWidth + s_x_idx;
  spatial_features[out_idx] = max_val;
}

}  // namespace

void initPillarVfeWeights()
{
  checkCuda(
    cudaMemcpyToSymbol(
      c_vfe_w, pp_vfe_weights::W_FUSED,
      sizeof(float) * kPillarVfeOutDim * kPillarVfeInDim),
    "cudaMemcpyToSymbol(VFE W)");
  checkCuda(
    cudaMemcpyToSymbol(c_vfe_b, pp_vfe_weights::B_FUSED, sizeof(float) * kPillarVfeOutDim),
    "cudaMemcpyToSymbol(VFE B)");
}

void launchPillarVfe(
  const float * voxels,
  const int32_t * voxel_num_points,
  const int32_t * voxel_coords,
  float * pillar_features,
  int32_t num_voxels,
  cudaStream_t stream)
{
  if (num_voxels <= 0) {
    return;
  }

  const dim3 block(kPillarVfeOutDim);
  const dim3 grid(static_cast<unsigned int>(num_voxels));
  pillarVfeKernel<<<grid, block, 0, stream>>>(
    voxels, voxel_num_points, voxel_coords, pillar_features, num_voxels);
  checkCuda(cudaGetLastError(), "pillarVfeKernel launch");
}

void launchPillarVfeScatter(
  const float * voxels,
  const int32_t * voxel_num_points,
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream)
{
  if (num_voxels <= 0) {
    return;
  }

  const dim3 block(kPillarVfeOutDim);
  const dim3 grid(static_cast<unsigned int>(num_voxels));
  pillarVfeScatterKernel<<<grid, block, 0, stream>>>(
    voxels, voxel_num_points, voxel_coords, spatial_features, num_voxels);
  checkCuda(cudaGetLastError(), "pillarVfeScatterKernel launch");
}

}  // namespace lidar_cone_detector
