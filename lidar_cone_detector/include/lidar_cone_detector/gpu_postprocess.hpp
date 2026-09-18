#pragma once

#include "lidar_cone_detector/pointpillars_common.hpp"
#include "lidar_cone_detector/trt_engine.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace lidar_cone_detector
{

struct GpuPostprocessDetection
{
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float score;
  int32_t label;
};

class GpuPostprocessor
{
public:
  explicit GpuPostprocessor(const PostprocessConfig & config);
  ~GpuPostprocessor();

  GpuPostprocessor(const GpuPostprocessor &) = delete;
  GpuPostprocessor & operator=(const GpuPostprocessor &) = delete;

  std::vector<Detection> decode(const RawNetworkOutputs & outputs, cudaStream_t stream);

  int32_t last_raw_candidate_count() const { return last_raw_candidate_count_; }
  int32_t last_nms_input_count() const { return last_nms_input_count_; }
  bool last_candidate_overflow() const { return last_candidate_overflow_; }

private:
  static int32_t ceilDiv(int32_t a, int32_t b);
  static float rawScoreThreshold(float score_threshold, bool logits);
  void allocateBuffers();
  void freeBuffers();

  PostprocessConfig config_;
  int32_t max_candidates_{0};
  int32_t nms_pre_max_size_{0};
  int32_t nms_post_max_size_{0};
  int32_t nms_col_blocks_{0};

  GpuPostprocessDetection * d_candidates_{nullptr};
  GpuPostprocessDetection * d_kept_{nullptr};
  int32_t * d_candidate_count_{nullptr};
  int32_t * d_active_candidate_count_{nullptr};
  int32_t * d_kept_count_{nullptr};
  uint64_t * d_nms_mask_{nullptr};
  uint64_t * d_removed_{nullptr};

  int32_t * h_candidate_count_{nullptr};
  int32_t * h_active_candidate_count_{nullptr};
  int32_t * h_kept_count_{nullptr};
  GpuPostprocessDetection * h_kept_{nullptr};

  int32_t last_raw_candidate_count_{0};
  int32_t last_nms_input_count_{0};
  bool last_candidate_overflow_{false};
};

}  // namespace lidar_cone_detector
