#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lidar_cone_detector
{

// 一帧中最终要发布的检测框。坐标定义保持 OpenPCDet 的
// [x, y, z, length, width, height, yaw]，label 从 1 开始。
struct Detection
{
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  float length{0.0F};
  float width{0.0F};
  float height{0.0F};
  float yaw{0.0F};
  float score{0.0F};
  int32_t label{0};
};

// Anchor-free CenterPoint/CenterHead 版本的模型常量来自
// pointpillars_int8_tools/configs/my_cone_anchorfree.yaml。
// 只要 YAML 里的 POINT_CLOUD_RANGE、VOXEL_SIZE、FEATURE_MAP_STRIDE 或类别数变了，
// 这里、VFE 融合权重、TensorRT engine 和后处理 decode 必须同步更新。
constexpr int32_t kDefaultNumClasses = 2;
constexpr int32_t kDefaultFeatureMapStride = 2;
constexpr int32_t kDefaultCenterHeadHeight = 160;
constexpr int32_t kDefaultCenterHeadWidth = 80;

// CUDA voxelizer 的配置，数值默认来自 my_cone_anchorfree.yaml。
struct VoxelizerConfig
{
  std::array<float, 3> voxel_size{0.12F, 0.12F, 4.0F};
  std::array<float, 6> point_cloud_range{0.0F, -19.2F, -3.0F, 19.2F, 19.2F, 1.0F};
  int32_t max_points_per_voxel{32};
  int32_t max_voxels{30000};
  int32_t max_input_points{300000};
  bool compact_raw_h2d{true};
  bool direct_raw_h2d{true};
  // 默认不清 d_voxels_/d_coords_ 全量缓冲。split CUDA VFE 和 OpenPCDet VFE 都只按
  // voxel_num_points 读取真实点，padding 区域不会参与计算；跳过这两个大 memset
  // 可以减少 voxel_reset 的固定带宽开销。如果要排查输入残留问题，可临时打开。
  bool clear_voxel_buffers{false};
};

// PointCloud2 原始 buffer 的字段布局。datatype 使用 sensor_msgs::msg::PointField 的枚举值：
// INT8=1, UINT8=2, INT16=3, UINT16=4, INT32=5, UINT32=6, FLOAT32=7, FLOAT64=8。
// 这里不直接依赖 ROS header，方便 CUDA voxelizer 在 .cu 文件里使用。
struct PointCloudLayout
{
  int32_t point_step{0};
  int32_t x_offset{-1};
  int32_t y_offset{-1};
  int32_t z_offset{-1};
  int32_t intensity_offset{-1};
  uint8_t x_datatype{0U};
  uint8_t y_datatype{0U};
  uint8_t z_datatype{0U};
  uint8_t intensity_datatype{0U};
  bool intensity_valid{false};
};

// 后处理配置。CenterHead engine 输出 hm/center/center_z/dim/rot 原始 head tensor；
// 部署端在 GPU/CPU 后处理里完成 heatmap topK、box decode、range filter 和 rotated NMS。
struct PostprocessConfig
{
  std::array<float, 6> point_cloud_range{0.0F, -19.2F, -3.0F, 19.2F, 19.2F, 1.0F};
  std::array<float, 3> voxel_size{0.12F, 0.12F, 4.0F};
  float score_threshold{0.30F};
  float nms_threshold{0.1F};
  int32_t feature_map_stride{kDefaultFeatureMapStride};
  bool cls_output_is_logits{true};
  bool filter_invalid_boxes{true};
  float min_box_size{0.03F};
  float max_box_size{2.0F};
  float max_z_margin{0.0F};
  int32_t num_classes{kDefaultNumClasses};
  int32_t nms_pre_max_size{4096};
  int32_t nms_post_max_size{500};
  int32_t gpu_nms_pre_max_size{512};
  int32_t gpu_candidate_buffer_size{2048};
};

struct VoxelizerTimings
{
  double pinned_copy_ms{0.0};
  double h2d_points_ms{0.0};
  double reset_ms{0.0};
  double voxelize_ms{0.0};
  double count_sync_ms{0.0};
  double pad_ms{0.0};
};

struct TrtTimings
{
  double enqueue_host_ms{0.0};
  double trt_gpu_ms{0.0};
  double d2h_outputs_ms{0.0};
};

struct FrameTimings
{
  double parse_ms{0.0};
  double pinned_copy_ms{0.0};
  double h2d_points_ms{0.0};
  double voxel_reset_ms{0.0};
  double voxelize_ms{0.0};
  double voxel_count_sync_ms{0.0};
  double voxel_pad_ms{0.0};
  double bev_clear_ms{0.0};
  double cuda_vfe_ms{0.0};
  double scatter_ms{0.0};
  double trt_enqueue_host_ms{0.0};
  double trt_gpu_ms{0.0};
  double d2h_outputs_ms{0.0};
  double postprocess_ms{0.0};
  double publish_ms{0.0};
  double other_ms{0.0};
  double total_ms{0.0};
};

struct TrackedCone
{
  std::string class_name;
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  float length{0.0F};
  float width{0.0F};
  float height{0.0F};
};

}  // namespace lidar_cone_detector
