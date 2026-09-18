#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lidar_cone_detector
{

constexpr int32_t kBevChannels = 64;
constexpr int32_t kBevHeight = 320;
constexpr int32_t kBevWidth = 160;

// 将稀疏 pillar feature 写回 OpenPCDet 的 BEV dense feature map：
//   pillar_features  [N, 64]
//   voxel_coords     [N, 4]，顺序为 [batch, z, y, x]
//   spatial_features [1, 64, 320, 160]，NCHW 连续布局。
void launchScatterPillarToBev(
  const float * pillar_features,
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream);

// 只清理上一帧实际写过的 BEV cell，比每帧 memset 整张 [64,320,160] 更省带宽。
void launchClearBevCells(
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream);

}  // namespace lidar_cone_detector
