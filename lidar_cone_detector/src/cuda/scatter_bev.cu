#include "lidar_cone_detector/cuda_scatter.hpp"

#include <cuda_runtime.h>

#include <cstdint>
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

__global__ void scatterPillarToBevKernel(
  const float * __restrict__ pillar_features,
  const int32_t * __restrict__ voxel_coords,
  float * __restrict__ spatial_features,
  int32_t num_voxels)
{
  const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t total = num_voxels * kBevChannels;
  if (idx >= total) {
    return;
  }

  const int32_t voxel_idx = idx / kBevChannels;
  const int32_t channel = idx - voxel_idx * kBevChannels;

  // OpenPCDet voxel_coords: [batch_idx, z_idx, y_idx, x_idx]。
  const int32_t y = voxel_coords[voxel_idx * 4 + 2];
  const int32_t x = voxel_coords[voxel_idx * 4 + 3];
  if (x < 0 || x >= kBevWidth || y < 0 || y >= kBevHeight) {
    return;
  }

  const int32_t out_idx = channel * kBevHeight * kBevWidth + y * kBevWidth + x;
  spatial_features[out_idx] = pillar_features[voxel_idx * kBevChannels + channel];
}

__global__ void clearBevCellsKernel(
  const int32_t * __restrict__ voxel_coords,
  float * __restrict__ spatial_features,
  int32_t num_voxels)
{
  const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t total = num_voxels * kBevChannels;
  if (idx >= total) {
    return;
  }

  const int32_t voxel_idx = idx / kBevChannels;
  const int32_t channel = idx - voxel_idx * kBevChannels;
  const int32_t y = voxel_coords[voxel_idx * 4 + 2];
  const int32_t x = voxel_coords[voxel_idx * 4 + 3];
  if (x < 0 || x >= kBevWidth || y < 0 || y >= kBevHeight) {
    return;
  }

  const int32_t out_idx = channel * kBevHeight * kBevWidth + y * kBevWidth + x;
  spatial_features[out_idx] = 0.0F;
}

}  // namespace

void launchScatterPillarToBev(
  const float * pillar_features,
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream)
{
  const size_t bytes =
    static_cast<size_t>(kBevChannels) * kBevHeight * kBevWidth * sizeof(float);

  // 每帧必须清空 BEV，否则没有 pillar 的格子会残留上一帧特征。
  checkCuda(cudaMemsetAsync(spatial_features, 0, bytes, stream), "cudaMemsetAsync(spatial_features)");
  if (num_voxels <= 0) {
    return;
  }

  constexpr int32_t threads = 256;
  const int32_t total = num_voxels * kBevChannels;
  const int32_t blocks = (total + threads - 1) / threads;
  scatterPillarToBevKernel<<<blocks, threads, 0, stream>>>(
    pillar_features, voxel_coords, spatial_features, num_voxels);
  checkCuda(cudaGetLastError(), "scatterPillarToBevKernel launch");
}

void launchClearBevCells(
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream)
{
  if (num_voxels <= 0) {
    return;
  }

  constexpr int32_t threads = 256;
  const int32_t total = num_voxels * kBevChannels;
  const int32_t blocks = (total + threads - 1) / threads;
  clearBevCellsKernel<<<blocks, threads, 0, stream>>>(voxel_coords, spatial_features, num_voxels);
  checkCuda(cudaGetLastError(), "clearBevCellsKernel launch");
}

}  // namespace lidar_cone_detector
