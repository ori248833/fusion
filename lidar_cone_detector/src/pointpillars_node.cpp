#include "lidar_cone_detector/cuda_voxelizer.hpp"
#include "lidar_cone_detector/cuda_scatter.hpp"
#include "lidar_cone_detector/cuda_vfe.hpp"
#include "lidar_cone_detector/gpu_postprocess.hpp"
#include "lidar_cone_detector/pointpillars_common.hpp"
#include "lidar_cone_detector/postprocess.hpp"
#include "lidar_cone_detector/trt_engine.hpp"
#include "lidar_cone_detector/vfe_pfn_fused_weights.hpp"

#include "lidar_cone_detector/msg/three_d_cone.hpp"
#include "lidar_cone_detector/msg/three_d_cone_array.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lidar_cone_detector
{
namespace
{

using Clock = std::chrono::steady_clock;

enum class BevClearMode
{
  kPrevCells,
  kFull,
  kAuto
};

// launch 发送 SIGINT/SIGTERM 时，先让主循环主动退出，再按顺序释放 CUDA/TensorRT 资源。
// 这样比直接把信号交给默认处理更容易避免 Ctrl-C 后进程挂住。
volatile std::sig_atomic_t g_shutdown_requested = 0;

void handleSignal(int)
{
  g_shutdown_requested = 1;
}

double elapsedMs(const Clock::time_point & start, const Clock::time_point & end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void checkCuda(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

BevClearMode parseBevClearMode(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (
    value == "prev_cells" || value == "prev-cells" || value == "prev" ||
    value == "occupied" || value == "occupied_cells" || value == "occupied-cells") {
    return BevClearMode::kPrevCells;
  }
  if (value == "full" || value == "memset") {
    return BevClearMode::kFull;
  }
  if (value == "auto") {
    return BevClearMode::kAuto;
  }
  throw std::runtime_error(
    "unsupported bev_clear_mode '" + value + "', expected occupied/prev_cells, full, or auto");
}

const char * bevClearModeName(BevClearMode mode)
{
  switch (mode) {
    case BevClearMode::kPrevCells:
      return "prev_cells";
    case BevClearMode::kFull:
      return "full";
    case BevClearMode::kAuto:
      return "auto";
  }
  return "unknown";
}

float readFloat32(const uint8_t * ptr)
{
  float value = 0.0F;
  std::memcpy(&value, ptr, sizeof(float));
  return value;
}

double readFloat64(const uint8_t * ptr)
{
  double value = 0.0;
  std::memcpy(&value, ptr, sizeof(double));
  return value;
}

uint16_t readUInt16(const uint8_t * ptr)
{
  uint16_t value = 0U;
  std::memcpy(&value, ptr, sizeof(uint16_t));
  return value;
}

uint32_t readUInt32(const uint8_t * ptr)
{
  uint32_t value = 0U;
  std::memcpy(&value, ptr, sizeof(uint32_t));
  return value;
}

int16_t readInt16(const uint8_t * ptr)
{
  int16_t value = 0;
  std::memcpy(&value, ptr, sizeof(int16_t));
  return value;
}

int32_t readInt32(const uint8_t * ptr)
{
  int32_t value = 0;
  std::memcpy(&value, ptr, sizeof(int32_t));
  return value;
}

std::string pointFieldTypeName(uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return "int8";
    case sensor_msgs::msg::PointField::UINT8:
      return "uint8";
    case sensor_msgs::msg::PointField::INT16:
      return "int16";
    case sensor_msgs::msg::PointField::UINT16:
      return "uint16";
    case sensor_msgs::msg::PointField::INT32:
      return "int32";
    case sensor_msgs::msg::PointField::UINT32:
      return "uint32";
    case sensor_msgs::msg::PointField::FLOAT32:
      return "float32";
    case sensor_msgs::msg::PointField::FLOAT64:
      return "float64";
    default:
      return "unknown";
  }
}

size_t pointFieldTypeSize(uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
    case sensor_msgs::msg::PointField::UINT8:
      return 1U;
    case sensor_msgs::msg::PointField::INT16:
    case sensor_msgs::msg::PointField::UINT16:
      return 2U;
    case sensor_msgs::msg::PointField::INT32:
    case sensor_msgs::msg::PointField::UINT32:
    case sensor_msgs::msg::PointField::FLOAT32:
      return 4U;
    case sensor_msgs::msg::PointField::FLOAT64:
      return 8U;
    default:
      return 0U;
  }
}

float readPointFieldAsFloat(const uint8_t * ptr, uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return static_cast<float>(*reinterpret_cast<const int8_t *>(ptr));
    case sensor_msgs::msg::PointField::UINT8:
      return static_cast<float>(*ptr);
    case sensor_msgs::msg::PointField::INT16:
      return static_cast<float>(readInt16(ptr));
    case sensor_msgs::msg::PointField::UINT16:
      return static_cast<float>(readUInt16(ptr));
    case sensor_msgs::msg::PointField::INT32:
      return static_cast<float>(readInt32(ptr));
    case sensor_msgs::msg::PointField::UINT32:
      return static_cast<float>(readUInt32(ptr));
    case sensor_msgs::msg::PointField::FLOAT32:
      return readFloat32(ptr);
    case sensor_msgs::msg::PointField::FLOAT64:
      return static_cast<float>(readFloat64(ptr));
    default:
      return 0.0F;
  }
}

float clampFloat(float value, float min_value, float max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

std::array<float, 3> toFloatArray3(const std::vector<double> & values, const std::array<float, 3> & fallback)
{
  if (values.size() != 3) {
    return fallback;
  }
  return {static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2])};
}

std::array<float, 6> toFloatArray6(const std::vector<double> & values, const std::array<float, 6> & fallback)
{
  if (values.size() != 6) {
    return fallback;
  }
  return {
    static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]),
    static_cast<float>(values[3]), static_cast<float>(values[4]), static_cast<float>(values[5])};
}

std::string defaultBackboneHeadEnginePath()
{
  const auto share = ament_index_cpp::get_package_share_directory("lidar_cone_detector");
  return share + "/pointpillars_centerhead_full_int8_qat.engine";
}

std::string defaultTxtSavePath()
{
  const auto share = ament_index_cpp::get_package_share_directory("lidar_cone_detector");
  return share + "/runtime/unique_detected_cones.txt";
}

}  // namespace

// ROS2 主节点：
// 订阅 PointCloud2，完成点云解析、CUDA voxelization、TensorRT 推理、
// CenterHead 后处理，然后沿用旧包的话题发布 Marker 和 ThreeDConeArray。
class PointPillarsTrtNode final : public rclcpp::Node
{
public:
  PointPillarsTrtNode()
  : Node("lidar_cone_detector_trt_cuda")
  {
    declareAndReadParameters();
    validateAnchorfreeGeometry();
    // latency_test 打开时这三个 event 每帧都会用于拆分 bev_clear 和 fused_vfe_scatter。
    // 事件创建/销毁本身可能产生 host 端尖峰，所以节点生命周期内复用一组。
    checkCuda(cudaEventCreate(&clear_start_event_), "cudaEventCreate(clear_start)");
    checkCuda(cudaEventCreate(&clear_done_event_), "cudaEventCreate(clear_done)");
    checkCuda(cudaEventCreate(&fused_done_event_), "cudaEventCreate(fused_done)");

    // voxelizer 输出 device 指针，后面直接交给 TensorRT，避免 device->host->device 来回拷贝。
    voxelizer_ = std::make_unique<CudaVoxelizer>(voxelizer_config_);
    // 手写 CUDA VFE/scatter 使用和当前 anchor-free checkpoint 对齐的融合权重。
    initPillarVfeWeights();
    allocateSplitBuffers();
    backbone_head_engine_ = std::make_unique<TrtEngine>(backbone_head_engine_path_);
    if (!backbone_head_engine_->supports_spatial_features_input()) {
      throw std::runtime_error("CenterHead engine does not expose spatial_features input");
    }
    if (!backbone_head_engine_->supports_centerhead_outputs()) {
      throw std::runtime_error("CenterHead engine must expose hm, center, center_z, dim, and rot outputs");
    }
    // 后处理负责把 raw head tensor 还原成可发布的 3D 框。
    postprocessor_ = std::make_unique<PointPillarsPostprocessor>(postprocess_config_);
    if (use_gpu_postprocess_) {
      gpu_postprocessor_ = std::make_unique<GpuPostprocessor>(postprocess_config_);
    }

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PointPillarsTrtNode::pointCloudCallback, this, std::placeholders::_1));
    if (publish_markers_) {
      marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/detected_cones_markers", 10);
    }
    bbox_pub_ = create_publisher<lidar_cone_detector::msg::ThreeDConeArray>("/detected_cones_bbox", 10);

    RCLCPP_INFO(get_logger(), "TensorRT C++/CUDA node started.");
    RCLCPP_INFO(get_logger(), "centerhead_engine_path: %s", backbone_head_engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "input_topic: %s", input_topic_.c_str());
    RCLCPP_INFO(get_logger(), "marker_topic: %s", publish_markers_ ? "/detected_cones_markers" : "disabled");
    RCLCPP_INFO(get_logger(), "bbox_topic: /detected_cones_bbox");
    RCLCPP_INFO(
      get_logger(), "score_thresh: %.3f, nms_thresh: %.3f", score_threshold_, nms_threshold_);
    RCLCPP_INFO(
      get_logger(), "gpu_postprocess: %s | gpu_nms_pre_max_size: %d | gpu_post_max_candidates: %d",
      use_gpu_postprocess_ ? "enabled" : "disabled",
      postprocess_config_.gpu_nms_pre_max_size,
      postprocess_config_.gpu_candidate_buffer_size);
    RCLCPP_INFO(
      get_logger(), "gpu_preprocess: %s | compact_raw_h2d: %s | direct_raw_h2d: %s | clear_voxel_buffers: %s",
      gpu_preprocess_ ? "enabled" : "disabled",
      voxelizer_config_.compact_raw_h2d ? "enabled" : "disabled",
      voxelizer_config_.direct_raw_h2d ? "enabled" : "disabled",
      voxelizer_config_.clear_voxel_buffers ? "enabled" : "disabled");
    RCLCPP_INFO(get_logger(), "cuda_vfe_weights: initialized");
    RCLCPP_INFO(get_logger(), "pipeline: CUDA VFE/scatter -> CenterHead TensorRT -> GPU/CPU postprocess");
    RCLCPP_INFO(
      get_logger(), "bev_clear_mode: %s | auto_full_threshold: %d",
      bevClearModeName(bev_clear_mode_), bev_clear_auto_full_threshold_);
    RCLCPP_INFO(get_logger(), "fixed_hesai_layout: %s", fixed_hesai_layout_ ? "enabled" : "disabled");
    RCLCPP_INFO(
      get_logger(), "latency test: %s", latency_enabled_ ? "enabled" : "disabled");
    if (enable_tracking_) {
      RCLCPP_INFO(get_logger(), "tracking enabled, TXT will be saved to: %s", txt_save_path_.c_str());
    }
  }

  ~PointPillarsTrtNode() override
  {
    if (clear_start_event_ != nullptr) {
      cudaEventDestroy(clear_start_event_);
    }
    if (clear_done_event_ != nullptr) {
      cudaEventDestroy(clear_done_event_);
    }
    if (fused_done_event_ != nullptr) {
      cudaEventDestroy(fused_done_event_);
    }
    cudaFree(d_spatial_features_);
    cudaFree(d_prev_voxel_coords_);
    d_spatial_features_ = nullptr;
    d_prev_voxel_coords_ = nullptr;
  }

  void saveConesToTxt()
  {
    if (!enable_tracking_) {
      return;
    }
    if (global_cones_.empty()) {
      RCLCPP_WARN(get_logger(), "tracking list is empty, nothing was saved.");
      return;
    }

    const std::filesystem::path save_path(txt_save_path_);
    const std::filesystem::path parent = save_path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        RCLCPP_ERROR(
          get_logger(), "failed to create TXT output directory %s: %s",
          parent.c_str(), ec.message().c_str());
        return;
      }
    }

    std::ofstream out(txt_save_path_);
    if (!out) {
      RCLCPP_ERROR(get_logger(), "failed to open TXT save path: %s", txt_save_path_.c_str());
      return;
    }
    out << "Class,X,Y,Z,Length,Width,Height\n";
    for (const auto & cone : global_cones_) {
      out << cone.class_name << ','
          << cone.x << ',' << cone.y << ',' << cone.z << ','
          << cone.length << ',' << cone.width << ',' << cone.height << '\n';
    }
    RCLCPP_INFO(get_logger(), "saved %zu unique cones to %s", global_cones_.size(), txt_save_path_.c_str());
  }

private:
  void declareAndReadParameters()
  {
    // 基本运行参数。print_latency 是为了兼容旧 launch；新参数推荐用 enable_latency_test。
    backbone_head_engine_path_ = declare_parameter<std::string>(
      "backbone_head_engine_path", defaultBackboneHeadEnginePath());
    input_topic_ = declare_parameter<std::string>("input_topic", "/lidar_points");
    // my_cone_anchorfree.yaml 的 POST_PROCESSING.SCORE_THRESH 是 0.30。
    // 低阈值会让 CenterHead heatmap 的弱峰值大量进入 NMS，召回高但后处理更慢。
    score_threshold_ = clampFloat(
      static_cast<float>(declare_parameter<double>("score_thresh", 0.30)), 1.0e-4F, 0.9999F);
    nms_threshold_ = clampFloat(
      static_cast<float>(declare_parameter<double>("nms_thresh", 0.1)), 0.0F, 1.0F);
    enable_tracking_ = declare_parameter<bool>("enable_tracking", false);
    publish_markers_ = declare_parameter<bool>("publish_markers", true);
    const bool enable_latency_test = declare_parameter<bool>("enable_latency_test", false);
    const bool legacy_print_latency = declare_parameter<bool>("print_latency", false);
    latency_enabled_ = enable_latency_test || legacy_print_latency;
    const int32_t legacy_latency_log_every_n =
      static_cast<int32_t>(declare_parameter<int64_t>("latency_log_every_n", 10));
    const int32_t latency_print_interval =
      static_cast<int32_t>(declare_parameter<int64_t>("latency_print_interval", 0));
    latency_log_every_n_ = std::max<int32_t>(
      1, latency_print_interval > 0 ? latency_print_interval : legacy_latency_log_every_n);
    marker_lifetime_sec_ = declare_parameter<double>("marker_lifetime_sec", 0.5);
    bev_clear_mode_ = parseBevClearMode(declare_parameter<std::string>("bev_clear_mode", "occupied"));
    bev_clear_auto_full_threshold_ = std::max<int32_t>(
      0, static_cast<int32_t>(declare_parameter<int64_t>("bev_clear_auto_full_threshold", 12000)));
    txt_save_path_ = declare_parameter<std::string>("txt_save_path", defaultTxtSavePath());
    dedup_distance_thresh_ = std::max(
      0.01F, static_cast<float>(declare_parameter<double>("dedup_distance_thresh", 0.5)));
    // 生产运行默认打开几条和 pointpillars_int8_tools 部署路线匹配的轻量优化：
    // 1. gpu_preprocess：CUDA 直接从 PointCloud2 raw bytes 解析并 voxelize，不生成 CPU XYZI vector。
    // 2. direct_raw_h2d：Hesai point_step=26 时避免 32 万次 stride compact 小拷贝，整帧 raw 一次 H2D；
    //    虽然多搬一些字节，但通常能压低 pinned_copy p95。
    // 3. compact_raw_h2d：direct_raw_h2d 关闭时才生效，只搬运模型需要的前 16 字节 XYZI。
    // 4. clear_voxel_buffers：默认 false，跳过 d_voxels/d_coords 全量清零，降低 voxel_reset 带宽。
    // 5. fast_pointcloud_parser/cpu_range_filter：仅 CPU fallback 路径使用。
    gpu_preprocess_ = declare_parameter<bool>("gpu_preprocess", true);
    fixed_hesai_layout_ = declare_parameter<bool>("fixed_hesai_layout", true);
    voxelizer_config_.compact_raw_h2d = declare_parameter<bool>("compact_raw_h2d", true);
    voxelizer_config_.direct_raw_h2d = declare_parameter<bool>("direct_raw_h2d", true);
    fast_pointcloud_parser_ = declare_parameter<bool>("fast_pointcloud_parser", true);
    cpu_range_filter_ = declare_parameter<bool>("cpu_range_filter", true);

    // 这些默认值来自 my_cone_anchorfree.yaml：
    // range [0,-19.2,-3,19.2,19.2,1]，voxel [0.12,0.12,4.0]，
    // BEV [1,64,320,160]，CenterHead heatmap [1,2,160,80]。
    // 如果以后重新训练或重新导出模型，point_cloud_range / voxel_size / max_voxels 必须一起核对。
    voxelizer_config_.max_input_points =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("max_input_points", 500000)));
    voxelizer_config_.max_voxels =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("max_voxels", 30000)));
    voxelizer_config_.max_points_per_voxel =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("max_points_per_voxel", 32)));
    if (voxelizer_config_.max_points_per_voxel != kPillarVfeMaxPoints) {
      std::ostringstream oss;
      oss << "max_points_per_voxel must be " << kPillarVfeMaxPoints
          << " because the fused CUDA VFE kernel and exported weights are fixed to that value.";
      throw std::runtime_error(oss.str());
    }
    voxelizer_config_.clear_voxel_buffers = declare_parameter<bool>("clear_voxel_buffers", false);
    voxelizer_config_.point_cloud_range = toFloatArray6(
      declare_parameter<std::vector<double>>("point_cloud_range", {0.0, -19.2, -3.0, 19.2, 19.2, 1.0}),
      voxelizer_config_.point_cloud_range);
    voxelizer_config_.voxel_size = toFloatArray3(
      declare_parameter<std::vector<double>>("voxel_size", {0.12, 0.12, 4.0}),
      voxelizer_config_.voxel_size);

    // 后处理参数：CenterHead engine 输出 hm/center/center_z/dim/rot，节点做 heatmap decode + NMS。
    // score_thresh、nms_thresh 保持可调，方便现场按误检/漏检情况微调。
    postprocess_config_.point_cloud_range = voxelizer_config_.point_cloud_range;
    postprocess_config_.voxel_size = voxelizer_config_.voxel_size;
    postprocess_config_.score_threshold = score_threshold_;
    postprocess_config_.nms_threshold = nms_threshold_;
    postprocess_config_.feature_map_stride =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("feature_map_stride", kDefaultFeatureMapStride)));
    postprocess_config_.cls_output_is_logits =
      declare_parameter<bool>("cls_output_is_logits", true);
    postprocess_config_.filter_invalid_boxes =
      declare_parameter<bool>("filter_invalid_boxes", true);
    postprocess_config_.min_box_size =
      std::max(0.001F, static_cast<float>(declare_parameter<double>("min_box_size", 0.03)));
    postprocess_config_.max_box_size =
      std::max(
        postprocess_config_.min_box_size,
        static_cast<float>(declare_parameter<double>("max_box_size", 2.0)));
    postprocess_config_.max_z_margin =
      std::max(0.0F, static_cast<float>(declare_parameter<double>("max_z_margin", 0.0)));
    postprocess_config_.nms_pre_max_size =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("nms_pre_max_size", 4096)));
    postprocess_config_.nms_post_max_size =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("nms_post_max_size", 500)));
    use_gpu_postprocess_ = declare_parameter<bool>("use_gpu_postprocess", true);
    postprocess_config_.gpu_nms_pre_max_size =
      std::max<int32_t>(
        1, static_cast<int32_t>(declare_parameter<int64_t>("gpu_nms_pre_max_size", 512)));
    postprocess_config_.gpu_candidate_buffer_size =
      std::max<int32_t>(
        postprocess_config_.gpu_nms_pre_max_size,
        static_cast<int32_t>(declare_parameter<int64_t>("gpu_post_max_candidates", 2048)));
  }

  void allocateSplitBuffers()
  {
    const size_t spatial_bytes =
      static_cast<size_t>(kBevChannels) * kBevHeight * kBevWidth * sizeof(float);
    const size_t prev_coords_bytes =
      static_cast<size_t>(voxelizer_config_.max_voxels) * 4U * sizeof(int32_t);

    checkCuda(cudaMalloc(&d_spatial_features_, spatial_bytes), "cudaMalloc(spatial_features)");
    checkCuda(cudaMalloc(&d_prev_voxel_coords_, prev_coords_bytes), "cudaMalloc(prev_voxel_coords)");
  }

  void validateAnchorfreeGeometry() const
  {
    auto closeEnough = [](float a, float b) {
      return std::fabs(a - b) < 1.0e-4F;
    };

    const int32_t bev_w = static_cast<int32_t>(std::round(
      (voxelizer_config_.point_cloud_range[3] - voxelizer_config_.point_cloud_range[0]) /
      voxelizer_config_.voxel_size[0]));
    const int32_t bev_h = static_cast<int32_t>(std::round(
      (voxelizer_config_.point_cloud_range[4] - voxelizer_config_.point_cloud_range[1]) /
      voxelizer_config_.voxel_size[1]));
    const int32_t hm_h = bev_h / std::max(1, postprocess_config_.feature_map_stride);
    const int32_t hm_w = bev_w / std::max(1, postprocess_config_.feature_map_stride);
    if (bev_h != kBevHeight || bev_w != kBevWidth ||
        hm_h != kDefaultCenterHeadHeight || hm_w != kDefaultCenterHeadWidth) {
      std::ostringstream oss;
      oss << "Anchor-free geometry mismatch: voxel config gives BEV ["
          << bev_h << ',' << bev_w << "] and CenterHead ["
          << hm_h << ',' << hm_w << "], but code expects BEV ["
          << kBevHeight << ',' << kBevWidth << "] and CenterHead ["
          << kDefaultCenterHeadHeight << ',' << kDefaultCenterHeadWidth
          << "]. Update cuda_scatter.hpp, TensorRT engine, and CenterHead decode together.";
      throw std::runtime_error(oss.str());
    }

    const float expected_x_offset = voxelizer_config_.point_cloud_range[0] + voxelizer_config_.voxel_size[0] * 0.5F;
    const float expected_y_offset = voxelizer_config_.point_cloud_range[1] + voxelizer_config_.voxel_size[1] * 0.5F;
    const float expected_z_offset = voxelizer_config_.point_cloud_range[2] + voxelizer_config_.voxel_size[2] * 0.5F;
    if (
      !closeEnough(pp_vfe_weights::VOXEL_X, voxelizer_config_.voxel_size[0]) ||
      !closeEnough(pp_vfe_weights::VOXEL_Y, voxelizer_config_.voxel_size[1]) ||
      !closeEnough(pp_vfe_weights::VOXEL_Z, voxelizer_config_.voxel_size[2]) ||
      !closeEnough(pp_vfe_weights::X_OFFSET, expected_x_offset) ||
      !closeEnough(pp_vfe_weights::Y_OFFSET, expected_y_offset) ||
      !closeEnough(pp_vfe_weights::Z_OFFSET, expected_z_offset)) {
      throw std::runtime_error(
        "VFE fused weights do not match point_cloud_range/voxel_size. "
        "Regenerate vfe_pfn_fused_weights.hpp from the same my_cone_anchorfree.yaml and checkpoint.");
    }
  }

  RawNetworkOutputs runSplitPipeline(
    int32_t real_voxels, FrameTimings * timings, bool copy_outputs_to_host = true)
  {
    if (
      backbone_head_engine_ == nullptr || d_spatial_features_ == nullptr ||
      d_prev_voxel_coords_ == nullptr) {
      throw std::runtime_error("split pipeline requested but backbone/head engine or buffers are not initialized");
    }

    cudaStream_t stream = voxelizer_->stream();
    const bool measure = latency_enabled_ && timings != nullptr;
    if (measure) {
      checkCuda(cudaEventRecord(clear_start_event_, stream), "cudaEventRecord(clear_start)");
    }

    const bool use_full_clear =
      !spatial_features_initialized_ || bev_clear_mode_ == BevClearMode::kFull ||
      (bev_clear_mode_ == BevClearMode::kAuto &&
      prev_split_voxels_ >= bev_clear_auto_full_threshold_);
    if (use_full_clear) {
      const size_t spatial_bytes =
        static_cast<size_t>(kBevChannels) * kBevHeight * kBevWidth * sizeof(float);
      checkCuda(cudaMemsetAsync(d_spatial_features_, 0, spatial_bytes, stream), "cudaMemsetAsync(spatial_features)");
    } else {
      launchClearBevCells(d_prev_voxel_coords_, d_spatial_features_, prev_split_voxels_, stream);
    }
    spatial_features_initialized_ = true;
    if (measure) {
      checkCuda(cudaEventRecord(clear_done_event_, stream), "cudaEventRecord(clear_done)");
    }

    launchPillarVfeScatter(
      voxelizer_->device_voxels(), voxelizer_->device_num_points(), voxelizer_->device_coords(),
      d_spatial_features_, real_voxels, stream);
    if (measure) {
      checkCuda(cudaEventRecord(fused_done_event_, stream), "cudaEventRecord(fused_done)");
    }

    TrtTimings trt_timings;
    const RawNetworkOutputs raw = backbone_head_engine_->inferSpatialFeatures(
      d_spatial_features_, stream, measure ? &trt_timings : nullptr, copy_outputs_to_host);

    if (real_voxels > 0) {
      const size_t coords_bytes = static_cast<size_t>(real_voxels) * 4U * sizeof(int32_t);
      checkCuda(
        cudaMemcpyAsync(
          d_prev_voxel_coords_, voxelizer_->device_coords(), coords_bytes,
          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync(prev voxel coords)");
    }
    prev_split_voxels_ = real_voxels;

    if (measure) {
      float clear_ms = 0.0F;
      float fused_ms = 0.0F;
      checkCuda(
        cudaEventElapsedTime(&clear_ms, clear_start_event_, clear_done_event_),
        "cudaEventElapsedTime(bev_clear)");
      checkCuda(
        cudaEventElapsedTime(&fused_ms, clear_done_event_, fused_done_event_),
        "cudaEventElapsedTime(fused_vfe_scatter)");
      timings->bev_clear_ms = static_cast<double>(clear_ms);
      timings->cuda_vfe_ms = static_cast<double>(fused_ms);
      timings->scatter_ms = 0.0;
      timings->trt_enqueue_host_ms = trt_timings.enqueue_host_ms;
      timings->trt_gpu_ms = trt_timings.trt_gpu_ms;
      timings->d2h_outputs_ms = trt_timings.d2h_outputs_ms;
    }

    if (!first_split_trt_log_done_) {
      first_split_trt_log_done_ = true;
      RCLCPP_INFO(
        get_logger(), "first split backbone/head inference finished | %s",
        rawOutputSummary(raw).c_str());
    }
    return raw;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // ROS 回调里捕获异常，避免某一帧坏消息或 TensorRT 错误直接把进程打崩。
    try {
      processPointCloud(msg);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "PointCloud callback failed: %s", e.what());
    }
  }

  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg)
  {
    const auto total_start = Clock::now();
    FrameTimings timings;

    // 1+2. PointCloud2 前处理和 CUDA voxelization。
    // 默认走 GPU 融合路径：raw PointCloud2 bytes -> CUDA kernel 解析/range filter/voxelize。
    // 如果字段布局不支持，再退回旧 CPU XYZI vector 路径。
    VoxelizerTimings voxel_timings;
    int32_t input_points_for_log = 0;
    int32_t real_voxels = 0;
    bool used_gpu_preprocess = false;
    PointCloudLayout layout;
    const auto parse_start = Clock::now();
    if (gpu_preprocess_ && buildPointCloudLayout(*msg, layout)) {
      input_points_for_log = usableRawPointCount(*msg, layout);
      const auto parse_end = Clock::now();
      timings.parse_ms = elapsedMs(parse_start, parse_end);
      used_gpu_preprocess = true;
      real_voxels = voxelizer_->voxelizePointCloud2Raw(
        msg->data.empty() ? nullptr : msg->data.data(), msg->data.size(), input_points_for_log,
        layout, latency_enabled_ ? &voxel_timings : nullptr);
    } else {
      const std::vector<float> & points = parsePointCloud(*msg);
      const auto parse_end = Clock::now();
      timings.parse_ms = elapsedMs(parse_start, parse_end);
      input_points_for_log = static_cast<int32_t>(points.size() / 4);
      real_voxels = voxelizer_->voxelize(
        points.empty() ? nullptr : points.data(), input_points_for_log,
        latency_enabled_ ? &voxel_timings : nullptr);
    }
    timings.pinned_copy_ms = voxel_timings.pinned_copy_ms;
    timings.h2d_points_ms = voxel_timings.h2d_points_ms;
    timings.voxel_reset_ms = voxel_timings.reset_ms;
    timings.voxelize_ms = voxel_timings.voxelize_ms;
    timings.voxel_count_sync_ms = voxel_timings.count_sync_ms;
    timings.voxel_pad_ms = voxel_timings.pad_ms;
    if (!first_cloud_logged_) {
      first_cloud_logged_ = true;
      RCLCPP_INFO(
        get_logger(),
        "first PointCloud2 received | frame_id '%s' | width %u | height %u | point_step %u | input points %d | preprocess %s",
        msg->header.frame_id.c_str(), msg->width, msg->height, msg->point_step,
        input_points_for_log, used_gpu_preprocess ? "gpu_raw" : "cpu_xyzi");
    }
    if (!first_voxel_log_done_) {
      first_voxel_log_done_ = true;
      RCLCPP_INFO(
        get_logger(), "first frame voxelized | input points %zu | voxels %d",
        static_cast<size_t>(input_points_for_log), real_voxels);
    }

    // 3. 推理。anchor-free 主路径：
    // CUDA voxelization -> fused CUDA VFE/scatter -> CenterHead TensorRT engine。
    // TensorRT 输出可保持在 device 上，直接交给 CUDA 后处理。
    const bool use_gpu_post_this_frame =
      use_gpu_postprocess_ && gpu_postprocessor_ != nullptr;
    RawNetworkOutputs raw = runSplitPipeline(real_voxels, &timings, !use_gpu_post_this_frame);

    // 4. CenterHead 后处理。
    // CenterHead 输出 hm/center/center_z/dim/rot；部署端做：
    //   heatmap 阈值/local-max -> TopK/sort -> box decode -> class-agnostic rotated NMS。
    // GPU 路径只把最终检测框拷回 CPU；CPU 路径保留为 reference/fallback。
    const auto post_start = Clock::now();
    const std::vector<Detection> detections =
      use_gpu_post_this_frame ?
      gpu_postprocessor_->decode(raw, voxelizer_->stream()) :
      postprocessor_->decode(raw);
    const auto post_end = Clock::now();
    timings.postprocess_ms = elapsedMs(post_start, post_end);
    if (
      use_gpu_post_this_frame && gpu_postprocessor_->last_candidate_overflow() &&
      (frame_index_ % static_cast<uint64_t>(latency_log_every_n_) == 0U))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "GPU postprocess candidate buffer overflow: raw candidates %d > buffer %d. "
        "Increase gpu_post_max_candidates or raise score_thresh.",
        gpu_postprocessor_->last_raw_candidate_count(),
        postprocess_config_.gpu_candidate_buffer_size);
    }
    if (!first_postprocess_log_done_) {
      first_postprocess_log_done_ = true;
      RCLCPP_INFO(
        get_logger(), "first postprocess finished | detections %zu", detections.size());
      if (!detections.empty()) {
        const auto & best = detections.front();
        RCLCPP_INFO(
          get_logger(),
          "first best detection | score %.3f | label %d | xyz [%.2f, %.2f, %.2f] | size [%.2f, %.2f, %.2f]",
          best.score, best.label, best.x, best.y, best.z, best.length, best.width, best.height);
      }
    }
    if (static_cast<int32_t>(detections.size()) >= postprocess_config_.nms_post_max_size) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "detections reached nms_post_max_size=%d. Boxes are probably low-confidence clutter; "
        "raise score_thresh toward the training value 0.30.",
        postprocess_config_.nms_post_max_size);
    }

    // 5. 发布 RViz Marker 和自定义 ThreeDConeArray；tracking 打开时顺便做全局去重。
    const auto publish_start = Clock::now();
    publishAndTrack(detections, msg->header);
    const auto publish_end = Clock::now();
    timings.publish_ms = elapsedMs(publish_start, publish_end);
    timings.total_ms = elapsedMs(total_start, publish_end);

    ++frame_index_;
    if (latency_enabled_ && (frame_index_ % static_cast<uint64_t>(latency_log_every_n_) == 0U)) {
      logLatency(timings, input_points_for_log, real_voxels, detections.size());
    }
  }

  static std::string rawOutputSummary(const RawNetworkOutputs & out)
  {
    std::ostringstream oss;
    if (out.output_type == NetworkOutputType::kCenterHeadRaw) {
      oss << "CenterHead"
          << " | hm [" << out.hm_c << ',' << out.hm_h << ',' << out.hm_w << ']'
          << " | center [" << out.center_c << ',' << out.center_h << ',' << out.center_w << ']'
          << " | center_z [" << out.center_z_c << ',' << out.center_z_h << ',' << out.center_z_w << ']'
          << " | dim [" << out.dim_c << ',' << out.dim_h << ',' << out.dim_w << ']'
          << " | rot [" << out.rot_c << ',' << out.rot_h << ',' << out.rot_w << ']';
    } else {
      oss << "unknown";
    }
    return oss.str();
  }

  int32_t usableRawPointCount(const sensor_msgs::msg::PointCloud2 & msg, const PointCloudLayout & layout) const
  {
    if (layout.point_step <= 0) {
      return 0;
    }
    const size_t declared_count = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
    const size_t available_count = msg.data.size() / static_cast<size_t>(layout.point_step);
    const size_t usable_count = std::min(
      {declared_count, available_count, static_cast<size_t>(voxelizer_config_.max_input_points)});
    return static_cast<int32_t>(usable_count);
  }

  bool buildPointCloudLayout(const sensor_msgs::msg::PointCloud2 & msg, PointCloudLayout & layout)
  {
    if (tryBuildFixedHesaiLayout(msg, layout)) {
      return true;
    }

    struct FieldReader
    {
      int32_t offset{-1};
      uint8_t datatype{0U};
      bool valid{false};
    };

    if (!point_fields_logged_) {
      std::ostringstream oss;
      for (const auto & field : msg.fields) {
        oss << field.name << "(offset=" << field.offset
            << ", type=" << pointFieldTypeName(field.datatype)
            << ", count=" << field.count << ") ";
      }
      RCLCPP_INFO(get_logger(), "PointCloud2 fields | point_step %u | %s", msg.point_step, oss.str().c_str());
      point_fields_logged_ = true;
    }

    FieldReader x_field;
    FieldReader y_field;
    FieldReader z_field;
    FieldReader intensity_field;
    for (const auto & field : msg.fields) {
      const bool readable = pointFieldTypeSize(field.datatype) > 0U;
      if (field.name == "x") {
        x_field = {static_cast<int32_t>(field.offset), field.datatype, readable};
      } else if (field.name == "y") {
        y_field = {static_cast<int32_t>(field.offset), field.datatype, readable};
      } else if (field.name == "z") {
        z_field = {static_cast<int32_t>(field.offset), field.datatype, readable};
      } else if (field.name == "intensity") {
        intensity_field = {static_cast<int32_t>(field.offset), field.datatype, readable};
      } else if (field.name == "reflectivity" && !intensity_field.valid) {
        intensity_field = {static_cast<int32_t>(field.offset), field.datatype, readable};
      }
    }

    if (!x_field.valid || !y_field.valid || !z_field.valid || msg.point_step == 0 || msg.data.empty()) {
      return false;
    }
    if (msg.height > 1U && msg.row_step != msg.point_step * msg.width) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "GPU raw preprocess only supports contiguous PointCloud2 rows; falling back to CPU parser.");
      return false;
    }

    const auto fieldEnd = [](const FieldReader & field) {
      return field.valid ? static_cast<size_t>(field.offset) + pointFieldTypeSize(field.datatype) : 0U;
    };
    const size_t max_field_end = std::max(
      {fieldEnd(x_field), fieldEnd(y_field), fieldEnd(z_field), fieldEnd(intensity_field)});
    if (max_field_end > msg.point_step) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "PointCloud2 field offset exceeds point_step.");
      return false;
    }

    layout.point_step = static_cast<int32_t>(msg.point_step);
    layout.x_offset = x_field.offset;
    layout.y_offset = y_field.offset;
    layout.z_offset = z_field.offset;
    layout.intensity_offset = intensity_field.offset;
    layout.x_datatype = x_field.datatype;
    layout.y_datatype = y_field.datatype;
    layout.z_datatype = z_field.datatype;
    layout.intensity_datatype = intensity_field.datatype;
    layout.intensity_valid = intensity_field.valid;

    if (!point_parser_logged_) {
      point_parser_logged_ = true;
      const bool compact_xyzi =
        msg.point_step == 16U &&
        x_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        y_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        z_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        intensity_field.valid &&
        intensity_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        x_field.offset == 0 && y_field.offset == 4 && z_field.offset == 8 && intensity_field.offset == 12;
      RCLCPP_INFO(
        get_logger(),
        "PointCloud2 preprocess | gpu_raw enabled | point_step %u | compact_xyzi_step16 %s | compact_raw_h2d %s | direct_raw_h2d %s | model range [%.1f %.1f %.1f %.1f %.1f %.1f]",
        msg.point_step, compact_xyzi ? "yes" : "no",
        voxelizer_config_.compact_raw_h2d ? "enabled" : "disabled",
        voxelizer_config_.direct_raw_h2d ? "enabled" : "disabled",
        voxelizer_config_.point_cloud_range[0], voxelizer_config_.point_cloud_range[1],
        voxelizer_config_.point_cloud_range[2], voxelizer_config_.point_cloud_range[3],
        voxelizer_config_.point_cloud_range[4], voxelizer_config_.point_cloud_range[5]);
    }

    return true;
  }

  bool tryBuildFixedHesaiLayout(const sensor_msgs::msg::PointCloud2 & msg, PointCloudLayout & layout)
  {
    if (!fixed_hesai_layout_) {
      return false;
    }
    if (msg.data.empty() || (msg.point_step != 16U && msg.point_step != 26U)) {
      return false;
    }
    if (msg.height > 1U && msg.row_step != msg.point_step * msg.width) {
      return false;
    }

    if (!fixed_hesai_layout_checked_) {
      fixed_hesai_layout_checked_ = true;
      fixed_hesai_layout_supported_ = validateFixedHesaiFields(msg);
      if (!fixed_hesai_layout_supported_) {
        RCLCPP_WARN(
          get_logger(),
          "fixed_hesai_layout requested but PointCloud2 fields are not x/y/z/intensity float32 at offsets 0/4/8/12; using generic parser.");
      }
    }
    if (!fixed_hesai_layout_supported_) {
      return false;
    }

    // 比赛上车固定 Hesai 格式时走这个快速路径：每帧不再扫描 fields。
    layout.point_step = static_cast<int32_t>(msg.point_step);
    layout.x_offset = 0;
    layout.y_offset = 4;
    layout.z_offset = 8;
    layout.intensity_offset = 12;
    layout.x_datatype = sensor_msgs::msg::PointField::FLOAT32;
    layout.y_datatype = sensor_msgs::msg::PointField::FLOAT32;
    layout.z_datatype = sensor_msgs::msg::PointField::FLOAT32;
    layout.intensity_datatype = sensor_msgs::msg::PointField::FLOAT32;
    layout.intensity_valid = true;

    if (!point_parser_logged_) {
      point_parser_logged_ = true;
      RCLCPP_INFO(
        get_logger(),
        "PointCloud2 preprocess | fixed_hesai_layout enabled | point_step %u | compact_xyzi_step16 %s | compact_raw_h2d %s | direct_raw_h2d %s | model range [%.1f %.1f %.1f %.1f %.1f %.1f]",
        msg.point_step, msg.point_step == 16U ? "yes" : "no",
        voxelizer_config_.compact_raw_h2d ? "enabled" : "disabled",
        voxelizer_config_.direct_raw_h2d ? "enabled" : "disabled",
        voxelizer_config_.point_cloud_range[0], voxelizer_config_.point_cloud_range[1],
        voxelizer_config_.point_cloud_range[2], voxelizer_config_.point_cloud_range[3],
        voxelizer_config_.point_cloud_range[4], voxelizer_config_.point_cloud_range[5]);
    }
    return true;
  }

  bool validateFixedHesaiFields(const sensor_msgs::msg::PointCloud2 & msg)
  {
    bool has_x = false;
    bool has_y = false;
    bool has_z = false;
    bool has_intensity = false;
    std::ostringstream oss;
    for (const auto & field : msg.fields) {
      oss << field.name << "(offset=" << field.offset
          << ", type=" << pointFieldTypeName(field.datatype)
          << ", count=" << field.count << ") ";
      const bool is_f32 = field.datatype == sensor_msgs::msg::PointField::FLOAT32;
      has_x = has_x || (field.name == "x" && field.offset == 0U && is_f32);
      has_y = has_y || (field.name == "y" && field.offset == 4U && is_f32);
      has_z = has_z || (field.name == "z" && field.offset == 8U && is_f32);
      has_intensity = has_intensity || (field.name == "intensity" && field.offset == 12U && is_f32);
    }
    if (!point_fields_logged_) {
      RCLCPP_INFO(get_logger(), "PointCloud2 fields | point_step %u | %s", msg.point_step, oss.str().c_str());
      point_fields_logged_ = true;
    }
    return has_x && has_y && has_z && has_intensity;
  }

  const std::vector<float> & parsePointCloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    point_buffer_.clear();

    struct FieldReader
    {
      int32_t offset{-1};
      uint8_t datatype{0U};
      bool valid{false};
    };

    // 不假设 x/y/z/intensity 固定偏移，而是从 PointField 里查偏移。
    // Hesai 点云常见 point_step=26，intensity 可能是 uint8/uint16/float32；
    // 如果强度读错，模型输入分布会和训练时不同，检测框会明显乱飞。
    FieldReader x_field;
    FieldReader y_field;
    FieldReader z_field;
    FieldReader intensity_field;

    if (!point_fields_logged_) {
      std::ostringstream oss;
      for (const auto & field : msg.fields) {
        oss << field.name << "(offset=" << field.offset
            << ", type=" << pointFieldTypeName(field.datatype)
            << ", count=" << field.count << ") ";
      }
      RCLCPP_INFO(get_logger(), "PointCloud2 fields | point_step %u | %s", msg.point_step, oss.str().c_str());
      point_fields_logged_ = true;
    }

    for (const auto & field : msg.fields) {
      if (field.name == "x") {
        x_field = {static_cast<int32_t>(field.offset), field.datatype, pointFieldTypeSize(field.datatype) > 0U};
      } else if (field.name == "y") {
        y_field = {static_cast<int32_t>(field.offset), field.datatype, pointFieldTypeSize(field.datatype) > 0U};
      } else if (field.name == "z") {
        z_field = {static_cast<int32_t>(field.offset), field.datatype, pointFieldTypeSize(field.datatype) > 0U};
      } else if (field.name == "intensity") {
        intensity_field = {static_cast<int32_t>(field.offset), field.datatype, pointFieldTypeSize(field.datatype) > 0U};
      } else if (field.name == "reflectivity" && !intensity_field.valid) {
        // 某些 Hesai 驱动把反射强度叫 reflectivity；训练转换脚本最终也是把它作为 intensity 数值用。
        intensity_field = {static_cast<int32_t>(field.offset), field.datatype, pointFieldTypeSize(field.datatype) > 0U};
      }
    }

    if (!x_field.valid || !y_field.valid || !z_field.valid) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "PointCloud2 must contain readable x/y/z fields.");
      return point_buffer_;
    }
    if (msg.point_step == 0 || msg.data.empty()) {
      return point_buffer_;
    }

    const size_t point_count = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
    const size_t usable_count = std::min(point_count, static_cast<size_t>(voxelizer_config_.max_input_points));
    // point_buffer_ 是成员变量，会跨帧复用 capacity，避免每帧重新申请几 MB 内存。
    // 打开 cpu_range_filter 后，最终保留下来的点远少于原始 32 万点，所以初始 reserve 保守一点。
    const size_t reserve_points = cpu_range_filter_ ? std::min<size_t>(usable_count, 100000U) : usable_count;
    point_buffer_.reserve(reserve_points * 4U);
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float min_intensity = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();
    float max_intensity = std::numeric_limits<float>::lowest();

    // 防止损坏的 PointCloud2 消息让下面的 memcpy 越界。
    const auto fieldEnd = [](const FieldReader & field) {
      return field.valid ? static_cast<size_t>(field.offset) + pointFieldTypeSize(field.datatype) : 0U;
    };
    const size_t max_field_end = std::max(
      {fieldEnd(x_field), fieldEnd(y_field), fieldEnd(z_field), fieldEnd(intensity_field)});
    if (max_field_end > msg.point_step) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "PointCloud2 field offset exceeds point_step.");
      return point_buffer_;
    }

    const bool fast_float32_xyzi =
      fast_pointcloud_parser_ &&
      x_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
      y_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
      z_field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
      (!intensity_field.valid || intensity_field.datatype == sensor_msgs::msg::PointField::FLOAT32);
    if (!point_parser_logged_) {
      point_parser_logged_ = true;
      RCLCPP_INFO(
        get_logger(),
        "PointCloud2 parser | fast_float32_xyzi %s | cpu_range_filter %s | model range [%.1f %.1f %.1f %.1f %.1f %.1f]",
        fast_float32_xyzi ? "enabled" : "disabled",
        cpu_range_filter_ ? "enabled" : "disabled",
        voxelizer_config_.point_cloud_range[0], voxelizer_config_.point_cloud_range[1],
        voxelizer_config_.point_cloud_range[2], voxelizer_config_.point_cloud_range[3],
        voxelizer_config_.point_cloud_range[4], voxelizer_config_.point_cloud_range[5]);
    }

    const auto & range = voxelizer_config_.point_cloud_range;
    auto insideModelRange = [&range](float x, float y, float z) {
      // OpenPCDet 和 CUDA voxelizer 都是不包含右边界，这里保持同样规则。
      return x >= range[0] && x < range[3] &&
        y >= range[1] && y < range[4] &&
        z >= range[2] && z < range[5];
    };

    size_t finite_count = 0U;
    size_t kept_count = 0U;
    bool has_finite_intensity = false;
    for (size_t i = 0; i < usable_count; ++i) {
      const size_t base = i * msg.point_step;
      if (base + msg.point_step > msg.data.size()) {
        break;
      }
      const uint8_t * ptr = msg.data.data() + base;
      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      float intensity = 0.0F;
      if (fast_float32_xyzi) {
        x = readFloat32(ptr + x_field.offset);
        y = readFloat32(ptr + y_field.offset);
        z = readFloat32(ptr + z_field.offset);
        intensity = intensity_field.valid ? readFloat32(ptr + intensity_field.offset) : 0.0F;
      } else {
        x = readPointFieldAsFloat(ptr + x_field.offset, x_field.datatype);
        y = readPointFieldAsFloat(ptr + y_field.offset, y_field.datatype);
        z = readPointFieldAsFloat(ptr + z_field.offset, z_field.datatype);
        intensity = intensity_field.valid ?
          readPointFieldAsFloat(ptr + intensity_field.offset, intensity_field.datatype) : 0.0F;
      }
      // NaN/Inf 点会破坏 voxel 索引计算，直接丢掉。
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      ++finite_count;
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      min_z = std::min(min_z, z);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
      max_z = std::max(max_z, z);
      if (std::isfinite(intensity)) {
        has_finite_intensity = true;
        min_intensity = std::min(min_intensity, intensity);
        max_intensity = std::max(max_intensity, intensity);
      }

      if (cpu_range_filter_ && !insideModelRange(x, y, z)) {
        continue;
      }
      point_buffer_.push_back(x);
      point_buffer_.push_back(y);
      point_buffer_.push_back(z);
      point_buffer_.push_back(std::isfinite(intensity) ? intensity : 0.0F);
      ++kept_count;
    }
    if (!has_finite_intensity) {
      min_intensity = 0.0F;
      max_intensity = 0.0F;
    }
    if (!point_stats_logged_ && finite_count > 0U) {
      point_stats_logged_ = true;
      RCLCPP_INFO(
        get_logger(),
        "first parsed cloud stats | raw %zu | finite %zu | kept %zu | x [%.2f, %.2f] | y [%.2f, %.2f] | z [%.2f, %.2f] | intensity [%.2f, %.2f]",
        usable_count, finite_count, kept_count, min_x, max_x, min_y, max_y, min_z, max_z,
        min_intensity, max_intensity);
    }
    return point_buffer_;
  }

  void publishAndTrack(
    const std::vector<Detection> & detections, const std_msgs::msg::Header & header)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    if (publish_markers_) {
      // 先发 DELETEALL，避免当前帧检测数少于上一帧时，RViz 残留旧框。
      marker_array.markers.reserve(detections.size() + 1U);
      visualization_msgs::msg::Marker clear_marker;
      clear_marker.header = header;
      clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
      marker_array.markers.push_back(clear_marker);
    }

    lidar_cone_detector::msg::ThreeDConeArray cone_array_msg;
    cone_array_msg.header = header;
    cone_array_msg.cones.reserve(detections.size());

    // Marker 保留时间只在 RViz 发布打开时计算；比赛关闭 Marker 后不做这部分工作。
    int32_t lifetime_sec = 0;
    uint32_t lifetime_nanosec = 0U;
    if (publish_markers_) {
      const int64_t lifetime_ns =
        static_cast<int64_t>(std::max(0.0, marker_lifetime_sec_) * 1000000000.0);
      lifetime_sec = static_cast<int32_t>(lifetime_ns / 1000000000LL);
      lifetime_nanosec = static_cast<uint32_t>(lifetime_ns % 1000000000LL);
    }

    int32_t marker_id = 0;
    for (const auto & det : detections) {
      if (enable_tracking_) {
        addToGlobalCones(det);
      }

      if (publish_markers_) {
        // Marker.CUBE 的 pose 表示中心和旋转，scale 表示长宽高。
        // yaw 转四元数后，RViz 里的框会和模型预测朝向一致。
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "cone";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = det.x;
        marker.pose.position.y = det.y;
        marker.pose.position.z = det.z;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = std::sin(static_cast<double>(det.yaw) * 0.5);
        marker.pose.orientation.w = std::cos(static_cast<double>(det.yaw) * 0.5);
        marker.scale.x = det.length;
        marker.scale.y = det.width;
        marker.scale.z = det.height;
        if (det.label == 1) {
          marker.color.r = 1.0F;
          marker.color.g = 1.0F;
          marker.color.b = 0.0F;
          marker.color.a = 0.9F;
        } else {
          marker.color.r = 0.0F;
          marker.color.g = 0.5F;
          marker.color.b = 1.0F;
          marker.color.a = 0.9F;
        }
        // Marker 保留时间给 RViz 留一点缓冲；点云频率较低时不会一闪就消失。
        marker.lifetime.sec = lifetime_sec;
        marker.lifetime.nanosec = lifetime_nanosec;
        marker_array.markers.push_back(marker);
      }

      lidar_cone_detector::msg::ThreeDCone cone;
      cone.center.x = det.x;
      cone.center.y = det.y;
      cone.center.z = det.z;
      cone.size.x = det.length;
      cone.size.y = det.width;
      cone.size.z = det.height;
      cone.label = det.label;
      cone.yaw = det.yaw;
      cone_array_msg.cones.push_back(cone);
    }

    if (publish_markers_ && marker_pub_ != nullptr) {
      marker_pub_->publish(marker_array);
    }
    bbox_pub_->publish(cone_array_msg);
  }

  void addToGlobalCones(const Detection & det)
  {
    // 简单的空间距离去重：同一个锥桶被多帧重复检测时，只保存第一次出现的位置。
    const float dedup_distance_sq = dedup_distance_thresh_ * dedup_distance_thresh_;
    for (const auto & cone : global_cones_) {
      const float dx = det.x - cone.x;
      const float dy = det.y - cone.y;
      const float dz = det.z - cone.z;
      if (dx * dx + dy * dy + dz * dz < dedup_distance_sq) {
        return;
      }
    }

    TrackedCone tracked;
    tracked.class_name = det.label == 1 ? "Cone" : "BigCone";
    tracked.x = det.x;
    tracked.y = det.y;
    tracked.z = det.z;
    tracked.length = det.length;
    tracked.width = det.width;
    tracked.height = det.height;
    global_cones_.push_back(tracked);
  }

  void logLatency(const FrameTimings & t, int32_t point_count, int32_t voxel_count, size_t det_count)
  {
    // 延时拆成 CPU 解析、CUDA 预处理、TensorRT、后处理和发布，方便定位瓶颈。
    const double measured_ms =
      t.parse_ms + t.pinned_copy_ms + t.h2d_points_ms + t.voxel_reset_ms + t.voxelize_ms +
      t.voxel_count_sync_ms + t.voxel_pad_ms + t.bev_clear_ms + t.cuda_vfe_ms + t.scatter_ms +
      t.trt_enqueue_host_ms + t.trt_gpu_ms + t.d2h_outputs_ms + t.postprocess_ms + t.publish_ms;
    const double other_ms = std::max(0.0, t.total_ms - measured_ms);
    RCLCPP_INFO(
      get_logger(),
      "Latency ms | total %.2f | parse %.2f | pinned_copy %.2f | H2D %.2f | voxel_reset %.2f | voxel %.2f | "
      "count %.2f | pad %.2f | bev_clear %.2f | fused_vfe_scatter %.2f | scatter %.2f | "
      "TRT_enqueue_host %.2f | TRT_gpu %.2f | D2H %.2f | post %.2f | pub %.2f | other %.2f | "
      "points %d | voxels %d | det %zu",
      t.total_ms, t.parse_ms, t.pinned_copy_ms, t.h2d_points_ms, t.voxel_reset_ms, t.voxelize_ms,
      t.voxel_count_sync_ms, t.voxel_pad_ms, t.bev_clear_ms, t.cuda_vfe_ms, t.scatter_ms,
      t.trt_enqueue_host_ms, t.trt_gpu_ms, t.d2h_outputs_ms, t.postprocess_ms, t.publish_ms,
      other_ms, point_count, voxel_count, det_count);
  }

  std::string backbone_head_engine_path_;
  std::string input_topic_;
  std::string txt_save_path_;
  float score_threshold_{0.30F};
  float nms_threshold_{0.1F};
  float dedup_distance_thresh_{0.5F};
  double marker_lifetime_sec_{0.5};
  bool enable_tracking_{false};
  bool publish_markers_{true};
  bool latency_enabled_{false};
  bool gpu_preprocess_{true};
  bool fixed_hesai_layout_{true};
  bool fixed_hesai_layout_checked_{false};
  bool fixed_hesai_layout_supported_{false};
  bool fast_pointcloud_parser_{true};
  bool cpu_range_filter_{true};
  bool use_gpu_postprocess_{true};
  bool spatial_features_initialized_{false};
  BevClearMode bev_clear_mode_{BevClearMode::kPrevCells};
  bool point_fields_logged_{false};
  bool point_parser_logged_{false};
  bool point_stats_logged_{false};
  bool first_cloud_logged_{false};
  bool first_voxel_log_done_{false};
  bool first_split_trt_log_done_{false};
  bool first_postprocess_log_done_{false};
  int32_t latency_log_every_n_{10};
  int32_t bev_clear_auto_full_threshold_{12000};
  uint64_t frame_index_{0};
  std::vector<float> point_buffer_;

  VoxelizerConfig voxelizer_config_;
  PostprocessConfig postprocess_config_;

  std::unique_ptr<TrtEngine> backbone_head_engine_;
  std::unique_ptr<CudaVoxelizer> voxelizer_;
  std::unique_ptr<PointPillarsPostprocessor> postprocessor_;
  std::unique_ptr<GpuPostprocessor> gpu_postprocessor_;
  float * d_spatial_features_{nullptr};
  int32_t * d_prev_voxel_coords_{nullptr};
  cudaEvent_t clear_start_event_{nullptr};
  cudaEvent_t clear_done_event_{nullptr};
  cudaEvent_t fused_done_event_{nullptr};
  int32_t prev_split_voxels_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<lidar_cone_detector::msg::ThreeDConeArray>::SharedPtr bbox_pub_;
  std::vector<TrackedCone> global_cones_;
};

}  // namespace lidar_cone_detector

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // 覆盖默认信号处理后，主循环能在收到 Ctrl-C 时自己收尾。
  // CUDA/TensorRT 对象会随着 node.reset() 析构，退出会干净很多。
  std::signal(SIGINT, lidar_cone_detector::handleSignal);
  std::signal(SIGTERM, lidar_cone_detector::handleSignal);

  std::shared_ptr<lidar_cone_detector::PointPillarsTrtNode> node;
  try {
    node = std::make_shared<lidar_cone_detector::PointPillarsTrtNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && lidar_cone_detector::g_shutdown_requested == 0) {
      executor.spin_once(std::chrono::milliseconds(100));
    }
    executor.remove_node(node);
  } catch (const std::exception & e) {
    if (node) {
      RCLCPP_ERROR(node->get_logger(), "node stopped by exception: %s", e.what());
    } else {
      std::cerr << "node failed before logger was ready: " << e.what() << std::endl;
    }
  }

  if (node) {
    node->saveConesToTxt();
    node.reset();
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
