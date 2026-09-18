#pragma once

#include "lidar_cone_detector/pointpillars_common.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace lidar_cone_detector
{

class CudaVoxelizer
{
public:
  explicit CudaVoxelizer(const VoxelizerConfig & config);
  ~CudaVoxelizer();

  CudaVoxelizer(const CudaVoxelizer &) = delete;
  CudaVoxelizer & operator=(const CudaVoxelizer &) = delete;

  // 输入 host 端 XYZI 点云，输出已经放在 device 端的 TensorRT 三个输入。
  // 返回真实 voxel 数；如果为 0，调用者仍应给 TensorRT 传 1 个空 voxel。
  int32_t voxelize(const float * host_points, int32_t num_points, VoxelizerTimings * timings);

  // 生产路径：直接从 PointCloud2 原始 data buffer 做 GPU 解析、range filter 和 voxelization。
  // host_raw 会先复制到 pinned host buffer，再异步 H2D 到 device raw buffer；
  // kernel 直接按 PointCloudLayout 读取 x/y/z/intensity，不再生成 CPU XYZI 中间 vector。
  int32_t voxelizePointCloud2Raw(
    const uint8_t * host_raw, size_t host_raw_bytes, int32_t num_points,
    const PointCloudLayout & layout, VoxelizerTimings * timings);

  // 将 [real_voxels, target_voxels) 的输入槽位填成安全 padding。
  // padded voxels/coords 为 0，voxel_num_points 必须为 1，避免 PillarVFE 除以 0。
  void padToCount(int32_t real_voxels, int32_t target_voxels, double * pad_ms);

  float * device_voxels() const { return d_voxels_; }
  int32_t * device_coords() const { return d_coords_; }
  int32_t * device_num_points() const { return d_num_points_; }
  cudaStream_t stream() const { return stream_; }
  int32_t max_voxels() const { return config_.max_voxels; }
  int32_t max_points_per_voxel() const { return config_.max_points_per_voxel; }

private:
  void allocateDeviceMemory();
  void freeDeviceMemory();
  void ensureRawBuffers(size_t bytes);

  VoxelizerConfig config_;
  int32_t grid_x_{0};
  int32_t grid_y_{0};
  int32_t grid_z_{0};
  int32_t grid_cells_{0};

  cudaStream_t stream_{nullptr};
  uint8_t * h_pinned_raw_{nullptr};
  size_t h_pinned_raw_bytes_{0};
  uint8_t * d_raw_points_{nullptr};
  size_t d_raw_points_bytes_{0};
  float * d_points_{nullptr};
  float * d_voxels_{nullptr};
  int32_t * d_coords_{nullptr};
  int32_t * d_num_points_{nullptr};
  int32_t * d_voxel_count_{nullptr};
  int32_t * d_voxel_map_{nullptr};
};

}  // namespace lidar_cone_detector
