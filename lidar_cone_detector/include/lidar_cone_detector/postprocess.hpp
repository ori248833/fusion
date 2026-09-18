#pragma once

#include "lidar_cone_detector/pointpillars_common.hpp"
#include "lidar_cone_detector/trt_engine.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace lidar_cone_detector
{

class PointPillarsPostprocessor
{
public:
  explicit PointPillarsPostprocessor(const PostprocessConfig & config);

  std::vector<Detection> decode(const RawNetworkOutputs & outputs) const;

private:
  struct Candidate
  {
    Detection det;
    bool suppressed{false};
  };

  std::vector<Detection> decodeCenterHead(const RawNetworkOutputs & outputs) const;
  bool isValidDetection(const Detection & det) const;
  std::vector<Detection> nms(std::vector<Candidate> candidates) const;

  static float sigmoid(float x);
  static float orientedIouBev(const Detection & a, const Detection & b);

  PostprocessConfig config_;
};

}  // namespace lidar_cone_detector
