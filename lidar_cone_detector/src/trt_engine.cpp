#include "lidar_cone_detector/trt_engine.hpp"
#include "lidar_cone_detector/cuda_scatter.hpp"

#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

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

void checkTrt(bool status, const char * what)
{
  if (!status) {
    throw std::runtime_error(std::string("TensorRT call failed: ") + what);
  }
}

}  // namespace

void TrtLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << msg << std::endl;
  }
}

TrtEngine::TrtEngine(const std::string & engine_path)
{
  // 单独的 stream 用于 TensorRT enqueue 和输出 D2H 拷贝。
  // voxelizer 有自己的 stream；节点层面按顺序调用，保证数据已经准备好。
  checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(trt)");
  // latency_test 打开时每帧都会记录 TRT/D2H 时间。event 创建销毁本身会带来
  // host 端抖动，所以这里创建一次并在整段生命周期复用。
  checkCuda(cudaEventCreate(&infer_start_event_), "cudaEventCreate(trt infer_start)");
  checkCuda(cudaEventCreate(&infer_done_event_), "cudaEventCreate(trt infer_done)");
  checkCuda(cudaEventCreate(&copy_done_event_), "cudaEventCreate(trt copy_done)");
  loadEngine(engine_path);
  discoverTensors();
}

TrtEngine::~TrtEngine()
{
  for (auto & tensor : tensors_) {
    cudaFree(tensor.device);
    cudaFreeHost(tensor.host_float);
    tensor.device = nullptr;
    tensor.host_float = nullptr;
    tensor.host_elems = 0U;
  }
  delete context_;
  delete engine_;
  delete runtime_;
  if (infer_start_event_ != nullptr) {
    cudaEventDestroy(infer_start_event_);
  }
  if (infer_done_event_ != nullptr) {
    cudaEventDestroy(infer_done_event_);
  }
  if (copy_done_event_ != nullptr) {
    cudaEventDestroy(copy_done_event_);
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

size_t TrtEngine::dataTypeSize(nvinfer1::DataType dtype)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      throw std::runtime_error("Unsupported TensorRT data type");
  }
}

size_t TrtEngine::volume(const nvinfer1::Dims & dims)
{
  size_t v = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("Dynamic output dimension was not resolved");
    }
    v *= static_cast<size_t>(dims.d[i]);
  }
  return v;
}

std::vector<int64_t> TrtEngine::dimsToVector(const nvinfer1::Dims & dims)
{
  std::vector<int64_t> out;
  out.reserve(static_cast<size_t>(dims.nbDims));
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    out.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return out;
}

void TrtEngine::setDims4(nvinfer1::Dims & dims, int32_t d0, int32_t d1, int32_t d2, int32_t d3)
{
  dims.nbDims = 4;
  dims.d[0] = d0;
  dims.d[1] = d1;
  dims.d[2] = d2;
  dims.d[3] = d3;
}

void TrtEngine::loadEngine(const std::string & engine_path)
{
  // 直接加载 .engine，不在 ROS 节点里 build engine。
  // 注意 engine 与 GPU 架构、TensorRT 版本强绑定，Orin 上建议重新生成。
  std::ifstream file(engine_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open TensorRT engine: " + engine_path);
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (blob.empty()) {
    throw std::runtime_error("TensorRT engine file is empty: " + engine_path);
  }

  initLibNvInferPlugins(&logger_, "");
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (runtime_ == nullptr) {
    throw std::runtime_error("createInferRuntime failed");
  }

  engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
  if (engine_ == nullptr) {
    throw std::runtime_error(
      "deserializeCudaEngine failed. If this happens on Orin, rebuild the engine with the "
      "same TensorRT version and GPU architecture as the target device.");
  }

  context_ = engine_->createExecutionContext();
  if (context_ == nullptr) {
    throw std::runtime_error("createExecutionContext failed");
  }
  if (engine_->getNbOptimizationProfiles() > 0) {
    checkTrt(context_->setOptimizationProfileAsync(0, stream_), "setOptimizationProfileAsync(0)");
  }

}

void TrtEngine::discoverTensors()
{
  // TensorRT 10 推荐按 tensor name 设置 shape/address，避免旧 binding index API 的歧义。
  const int32_t nb = engine_->getNbIOTensors();
  tensors_.reserve(static_cast<size_t>(nb));
  for (int32_t i = 0; i < nb; ++i) {
    const char * name = engine_->getIOTensorName(i);
    TensorBuffer tensor;
    tensor.name = name;
    tensor.mode = engine_->getTensorIOMode(name);
    tensor.dtype = engine_->getTensorDataType(name);
    tensor.dims = dimsToVector(engine_->getTensorShape(name));
    tensor_index_[tensor.name] = tensors_.size();
    tensors_.push_back(std::move(tensor));
  }

  supports_spatial_features_input_ = hasTensor(spatial_input_name_);

  if (!supports_spatial_features_input_) {
    throw std::runtime_error(
      "CenterHead TensorRT engine must expose split input tensor spatial_features.");
  }

  const bool has_centerhead =
    hasTensor(hm_output_name_) && hasTensor(center_output_name_) &&
    hasTensor(center_z_output_name_) && hasTensor(dim_output_name_) &&
    hasTensor(rot_output_name_);

  if (!has_centerhead) {
    throw std::runtime_error(
      "CenterHead TensorRT engine must expose output tensors hm, center, center_z, dim, and rot.");
  }

  // Anchor-free CenterPoint/CenterHead split engine:
  // spatial_features [1,64,320,160] -> hm/center/center_z/dim/rot。
  output_type_ = NetworkOutputType::kCenterHeadRaw;
  active_output_names_ = {
    hm_output_name_,
    center_output_name_,
    center_z_output_name_,
    dim_output_name_,
    rot_output_name_,
  };
}

TrtEngine::TensorBuffer & TrtEngine::tensor(const std::string & name)
{
  const auto it = tensor_index_.find(name);
  if (it == tensor_index_.end()) {
    throw std::runtime_error("TensorRT engine is missing tensor: " + name);
  }
  return tensors_[it->second];
}

bool TrtEngine::hasTensor(const std::string & name) const
{
  return tensor_index_.count(name) != 0U;
}

void TrtEngine::resizeOutputBuffer(TensorBuffer & buffer, const nvinfer1::Dims & dims)
{
  if (buffer.mode != nvinfer1::TensorIOMode::kOUTPUT) {
    return;
  }
  if (buffer.dtype != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("Only FP32 output tensors are supported by this postprocess path: " + buffer.name);
  }

  const size_t elems = volume(dims);
  const size_t bytes = elems * dataTypeSize(buffer.dtype);
  if (buffer.bytes != bytes) {
    // 输出 shape 固定时只会分配一次；如果未来换 dynamic 输出，这里也能按需重分配。
    cudaFree(buffer.device);
    cudaFreeHost(buffer.host_float);
    buffer.device = nullptr;
    buffer.host_float = nullptr;
    buffer.host_elems = 0U;
    checkCuda(cudaMalloc(&buffer.device, bytes), ("cudaMalloc(output " + buffer.name + ")").c_str());
    // 输出直接拷到 pinned host memory，避免 pageable host D2H 的隐性 staging 开销。
    checkCuda(
      cudaMallocHost(reinterpret_cast<void **>(&buffer.host_float), elems * sizeof(float)),
      ("cudaMallocHost(output " + buffer.name + ")").c_str());
    buffer.host_elems = elems;
    buffer.bytes = bytes;
  }
  buffer.dims = dimsToVector(dims);
}

const std::vector<std::string> & TrtEngine::activeOutputNames() const
{
  return active_output_names_;
}

void TrtEngine::resizeActiveOutputBuffers()
{
  for (const auto & name : activeOutputNames()) {
    auto & buffer = tensor(name);
    resizeOutputBuffer(buffer, context_->getTensorShape(name.c_str()));
  }
}

void TrtEngine::bindActiveOutputBuffers()
{
  for (const auto & name : activeOutputNames()) {
    auto & buffer = tensor(name);
    checkTrt(context_->setTensorAddress(name.c_str(), buffer.device), ("setTensorAddress(" + name + ")").c_str());
  }
}

void TrtEngine::copyActiveOutputsToHost(cudaStream_t stream)
{
  for (const auto & name : activeOutputNames()) {
    auto & buffer = tensor(name);
    checkCuda(
      cudaMemcpyAsync(buffer.host_float, buffer.device, buffer.bytes, cudaMemcpyDeviceToHost, stream),
      ("cudaMemcpyAsync(" + name + " D2H)").c_str());
  }
}

RawNetworkOutputs TrtEngine::outputsFromCurrentBuffers(bool host_valid)
{
  RawNetworkOutputs out;
  out.output_type = output_type_;

  auto fill_nchw = [host_valid](const TensorBuffer & src, const float *& host, const float *& device,
      int32_t & channels, int32_t & height, int32_t & width) {
      host = host_valid ? src.host_float : nullptr;
      device = static_cast<const float *>(src.device);
      if (src.dims.size() == 4) {
        channels = static_cast<int32_t>(src.dims[1]);
        height = static_cast<int32_t>(src.dims[2]);
        width = static_cast<int32_t>(src.dims[3]);
      } else if (src.dims.size() == 3) {
        channels = static_cast<int32_t>(src.dims[0]);
        height = static_cast<int32_t>(src.dims[1]);
        width = static_cast<int32_t>(src.dims[2]);
      }
    };

  if (output_type_ == NetworkOutputType::kCenterHeadRaw) {
    fill_nchw(tensor(hm_output_name_), out.hm, out.hm_device, out.hm_c, out.hm_h, out.hm_w);
    fill_nchw(
      tensor(center_output_name_), out.center, out.center_device,
      out.center_c, out.center_h, out.center_w);
    fill_nchw(
      tensor(center_z_output_name_), out.center_z, out.center_z_device,
      out.center_z_c, out.center_z_h, out.center_z_w);
    fill_nchw(tensor(dim_output_name_), out.dim, out.dim_device, out.dim_c, out.dim_h, out.dim_w);
    fill_nchw(tensor(rot_output_name_), out.rot, out.rot_device, out.rot_c, out.rot_h, out.rot_w);
    return out;
  }

  return out;
}

RawNetworkOutputs TrtEngine::inferSpatialFeatures(
  float * d_spatial_features, cudaStream_t stream, TrtTimings * timings, bool copy_outputs_to_host)
{
  if (timings != nullptr) {
    *timings = TrtTimings{};
  }
  if (!supports_spatial_features_input_) {
    throw std::runtime_error("inferSpatialFeatures called on a TensorRT engine without spatial_features input");
  }

  if (!spatial_shape_set_) {
    nvinfer1::Dims spatial_shape;
    setDims4(spatial_shape, 1, kBevChannels, kBevHeight, kBevWidth);
    const nvinfer1::Dims engine_shape = engine_->getTensorShape(spatial_input_name_.c_str());
    bool needs_shape = false;
    for (int32_t i = 0; i < engine_shape.nbDims; ++i) {
      needs_shape = needs_shape || engine_shape.d[i] < 0;
    }
    if (needs_shape) {
      checkTrt(
        context_->setInputShape(spatial_input_name_.c_str(), spatial_shape),
        "setInputShape(spatial_features)");
    }
    spatial_shape_set_ = true;
    spatial_io_bound_ = false;
  }

  if (!spatial_io_bound_) {
    // Split CenterHead engine 的 input/output shape 固定。首次推理完成 shape 解析、
    // 输出 buffer 分配和 tensor address 绑定，后续帧直接 enqueue，避免热路径反复查 shape
    // 和 setTensorAddress。d_spatial_features_ 在节点初始化时分配，生命周期覆盖整个节点。
    resizeActiveOutputBuffers();
    checkTrt(
      context_->setTensorAddress(spatial_input_name_.c_str(), d_spatial_features),
      "setTensorAddress(spatial_features)");
    bindActiveOutputBuffers();
    spatial_io_bound_ = true;
  }

  if (timings != nullptr) {
    checkCuda(cudaEventRecord(infer_start_event_, stream), "cudaEventRecord(split_infer_start)");
  }

  const auto enqueue_begin = std::chrono::steady_clock::now();
  checkTrt(context_->enqueueV3(stream), "enqueueV3(spatial_features)");
  const auto enqueue_end = std::chrono::steady_clock::now();
  if (timings != nullptr) {
    timings->enqueue_host_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_begin).count();
    checkCuda(cudaEventRecord(infer_done_event_, stream), "cudaEventRecord(split_infer_done)");
  }

  if (copy_outputs_to_host) {
    copyActiveOutputsToHost(stream);
    if (timings != nullptr) {
      checkCuda(cudaEventRecord(copy_done_event_, stream), "cudaEventRecord(split_copy_done)");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(split trt)");
  } else if (timings != nullptr) {
    // GPU 后处理会在同一 stream 上排队，因此不拷 raw output 时无需在这里同步；
    // 但 latency 日志需要 TRT_gpu 数字，所以 enable_latency_test=true 时会等待 infer_done。
    checkCuda(cudaEventSynchronize(infer_done_event_), "cudaEventSynchronize(split infer_done)");
  }

  if (timings != nullptr) {
    float trt_ms = 0.0F;
    checkCuda(
      cudaEventElapsedTime(&trt_ms, infer_start_event_, infer_done_event_),
      "cudaEventElapsedTime(split trt)");
    timings->trt_gpu_ms = static_cast<double>(trt_ms);
    if (copy_outputs_to_host) {
      float d2h_ms = 0.0F;
      checkCuda(
        cudaEventElapsedTime(&d2h_ms, infer_done_event_, copy_done_event_),
        "cudaEventElapsedTime(split d2h)");
      timings->d2h_outputs_ms = static_cast<double>(d2h_ms);
    }
  }

  return outputsFromCurrentBuffers(copy_outputs_to_host);
}

}  // namespace lidar_cone_detector
