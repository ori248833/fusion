#pragma once

#include "lidar_cone_detector/pointpillars_common.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lidar_cone_detector
{

enum class NetworkOutputType
{
  kUnknown,
  kCenterHeadRaw
};

struct RawNetworkOutputs
{
  NetworkOutputType output_type{NetworkOutputType::kUnknown};

  // Anchor-free CenterHead raw outputs:
  //   hm       [1, 2, 160, 80]  raw logits unless export includes sigmoid
  //   center   [1, 2, 160, 80]
  //   center_z [1, 1, 160, 80]
  //   dim      [1, 3, 160, 80]
  //   rot      [1, 2, 160, 80]
  const float * hm{nullptr};
  const float * center{nullptr};
  const float * center_z{nullptr};
  const float * dim{nullptr};
  const float * rot{nullptr};
  const float * hm_device{nullptr};
  const float * center_device{nullptr};
  const float * center_z_device{nullptr};
  const float * dim_device{nullptr};
  const float * rot_device{nullptr};
  int32_t hm_c{0};
  int32_t hm_h{0};
  int32_t hm_w{0};
  int32_t center_c{0};
  int32_t center_h{0};
  int32_t center_w{0};
  int32_t center_z_c{0};
  int32_t center_z_h{0};
  int32_t center_z_w{0};
  int32_t dim_c{0};
  int32_t dim_h{0};
  int32_t dim_w{0};
  int32_t rot_c{0};
  int32_t rot_h{0};
  int32_t rot_w{0};
};

class TrtLogger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

class TrtEngine
{
public:
  explicit TrtEngine(const std::string & engine_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  RawNetworkOutputs inferSpatialFeatures(
    float * d_spatial_features, cudaStream_t stream, TrtTimings * timings,
    bool copy_outputs_to_host = true);

  bool supports_spatial_features_input() const { return supports_spatial_features_input_; }
  bool supports_centerhead_outputs() const { return output_type_ == NetworkOutputType::kCenterHeadRaw; }
  NetworkOutputType output_type() const { return output_type_; }

private:
  struct TensorBuffer
  {
    std::string name;
    nvinfer1::TensorIOMode mode{nvinfer1::TensorIOMode::kNONE};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    std::vector<int64_t> dims;
    size_t bytes{0};
    void * device{nullptr};
    float * host_float{nullptr};
    size_t host_elems{0};
  };

  static size_t dataTypeSize(nvinfer1::DataType dtype);
  static size_t volume(const nvinfer1::Dims & dims);
  static std::vector<int64_t> dimsToVector(const nvinfer1::Dims & dims);
  static void setDims4(nvinfer1::Dims & dims, int32_t d0, int32_t d1, int32_t d2, int32_t d3);

  void loadEngine(const std::string & engine_path);
  void discoverTensors();
  void resizeOutputBuffer(TensorBuffer & buffer, const nvinfer1::Dims & dims);
  const std::vector<std::string> & activeOutputNames() const;
  void resizeActiveOutputBuffers();
  void bindActiveOutputBuffers();
  void copyActiveOutputsToHost(cudaStream_t stream);
  RawNetworkOutputs outputsFromCurrentBuffers(bool host_valid);
  TensorBuffer & tensor(const std::string & name);
  bool hasTensor(const std::string & name) const;

  TrtLogger logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  cudaStream_t stream_{nullptr};
  cudaEvent_t infer_start_event_{nullptr};
  cudaEvent_t infer_done_event_{nullptr};
  cudaEvent_t copy_done_event_{nullptr};
  std::vector<TensorBuffer> tensors_;
  std::unordered_map<std::string, size_t> tensor_index_;
  std::string hm_output_name_{"hm"};
  std::string center_output_name_{"center"};
  std::string center_z_output_name_{"center_z"};
  std::string dim_output_name_{"dim"};
  std::string rot_output_name_{"rot"};
  std::string spatial_input_name_{"spatial_features"};
  std::vector<std::string> active_output_names_;
  NetworkOutputType output_type_{NetworkOutputType::kUnknown};
  bool supports_spatial_features_input_{false};
  bool spatial_shape_set_{false};
  bool spatial_io_bound_{false};
};

}  // namespace lidar_cone_detector
