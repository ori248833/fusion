#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lidar_cone_detector
{

// OpenPCDet PillarVFE 的固定形状：每个 pillar 最多 32 点，10 维手工特征，输出 64 维。
constexpr int32_t kPillarVfeMaxPoints = 32;
constexpr int32_t kPillarVfeInDim = 10;
constexpr int32_t kPillarVfeOutDim = 64;

// 将导出的 Linear+BN 融合权重拷到 CUDA constant memory。节点启动时调用一次即可。
void initPillarVfeWeights();

// 手写 CUDA VFE：
//   voxels           [N, 32, 4]，每个点为 x/y/z/intensity
//   voxel_num_points [N]
//   voxel_coords     [N, 4]，OpenPCDet 顺序为 [batch, z, y, x]
//   pillar_features  [N, 64]
void launchPillarVfe(
  const float * voxels,
  const int32_t * voxel_num_points,
  const int32_t * voxel_coords,
  float * pillar_features,
  int32_t num_voxels,
  cudaStream_t stream);

// 融合版 CUDA VFE + scatter：直接把每个 pillar 的 64 维特征写入 BEV dense map。
// 调用前需要先清理 spatial_features 中上一帧写过的位置。
void launchPillarVfeScatter(
  const float * voxels,
  const int32_t * voxel_num_points,
  const int32_t * voxel_coords,
  float * spatial_features,
  int32_t num_voxels,
  cudaStream_t stream);

}  // namespace lidar_cone_detector
