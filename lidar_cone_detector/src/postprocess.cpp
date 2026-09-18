#include "lidar_cone_detector/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lidar_cone_detector
{
namespace
{

struct Point2D
{
  float x{0.0F};
  float y{0.0F};
};

float cross(const Point2D & a, const Point2D & b, const Point2D & c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

std::array<Point2D, 4> boxCorners(const Detection & box)
{
  const float c = std::cos(box.yaw);
  const float s = std::sin(box.yaw);
  const float hx = box.length * 0.5F;
  const float hy = box.width * 0.5F;
  const std::array<Point2D, 4> local = {{
    {hx, hy}, {-hx, hy}, {-hx, -hy}, {hx, -hy}
  }};

  std::array<Point2D, 4> out;
  for (size_t i = 0; i < local.size(); ++i) {
    out[i].x = box.x + local[i].x * c - local[i].y * s;
    out[i].y = box.y + local[i].x * s + local[i].y * c;
  }
  return out;
}

float polygonArea(const std::vector<Point2D> & poly)
{
  if (poly.size() < 3) {
    return 0.0F;
  }
  float area = 0.0F;
  for (size_t i = 0; i < poly.size(); ++i) {
    const Point2D & p = poly[i];
    const Point2D & q = poly[(i + 1) % poly.size()];
    area += p.x * q.y - p.y * q.x;
  }
  return std::fabs(area) * 0.5F;
}

Point2D lineIntersection(const Point2D & p1, const Point2D & p2, const Point2D & q1, const Point2D & q2)
{
  const float a1 = p2.y - p1.y;
  const float b1 = p1.x - p2.x;
  const float c1 = a1 * p1.x + b1 * p1.y;

  const float a2 = q2.y - q1.y;
  const float b2 = q1.x - q2.x;
  const float c2 = a2 * q1.x + b2 * q1.y;

  const float det = a1 * b2 - a2 * b1;
  if (std::fabs(det) < 1e-6F) {
    return p2;
  }
  return {(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}

float rawScoreThreshold(float score_threshold, bool logits)
{
  if (!logits) {
    return score_threshold;
  }

  // sigmoid 是单调函数，所以 raw logit 可以直接和 logit(score_thresh) 比较。
  // 这样不会改变精度，却能避免对 80000xnum_classes 个输出都调用 exp。
  const float eps = 1.0e-6F;
  const float clamped = std::max(eps, std::min(1.0F - eps, score_threshold));
  return std::log(clamped / (1.0F - clamped));
}

std::vector<Point2D> clipPolygonWithEdge(
  const std::vector<Point2D> & subject, const Point2D & edge_start, const Point2D & edge_end)
{
  std::vector<Point2D> output;
  if (subject.empty()) {
    return output;
  }

  auto inside = [&](const Point2D & p) {
    // boxCorners 生成的是逆时针顶点，左侧为内部。
    return cross(edge_start, edge_end, p) >= -1e-6F;
  };

  Point2D previous = subject.back();
  bool previous_inside = inside(previous);
  for (const auto & current : subject) {
    const bool current_inside = inside(current);
    if (current_inside != previous_inside) {
      output.push_back(lineIntersection(previous, current, edge_start, edge_end));
    }
    if (current_inside) {
      output.push_back(current);
    }
    previous = current;
    previous_inside = current_inside;
  }
  return output;
}

}  // namespace

PointPillarsPostprocessor::PointPillarsPostprocessor(const PostprocessConfig & config)
: config_(config)
{
}

std::vector<Detection> PointPillarsPostprocessor::decode(const RawNetworkOutputs & outputs) const
{
  if (outputs.output_type != NetworkOutputType::kCenterHeadRaw) {
    throw std::runtime_error("Unexpected network output type: CenterHead raw outputs are required");
  }
  if (
    outputs.hm == nullptr || outputs.center == nullptr || outputs.center_z == nullptr ||
    outputs.dim == nullptr || outputs.rot == nullptr) {
    return {};
  }
  return decodeCenterHead(outputs);
}

std::vector<Detection> PointPillarsPostprocessor::decodeCenterHead(const RawNetworkOutputs & outputs) const
{
  const bool shape_ok =
    outputs.hm_c == config_.num_classes &&
    outputs.center_c == 2 && outputs.center_z_c == 1 &&
    outputs.dim_c == 3 && outputs.rot_c == 2 &&
    outputs.hm_h == outputs.center_h && outputs.hm_h == outputs.center_z_h &&
    outputs.hm_h == outputs.dim_h && outputs.hm_h == outputs.rot_h &&
    outputs.hm_w == outputs.center_w && outputs.hm_w == outputs.center_z_w &&
    outputs.hm_w == outputs.dim_w && outputs.hm_w == outputs.rot_w;
  if (!shape_ok) {
    throw std::runtime_error("Unexpected CenterHead output shape");
  }

  const int32_t h = outputs.hm_h;
  const int32_t w = outputs.hm_w;
  const int32_t cells_per_class = h * w;
  const bool logits = config_.cls_output_is_logits;
  const float raw_threshold = rawScoreThreshold(config_.score_threshold, logits);
  const float real_stride_x = config_.voxel_size[0] * static_cast<float>(config_.feature_map_stride);
  const float real_stride_y = config_.voxel_size[1] * static_cast<float>(config_.feature_map_stride);
  std::vector<Candidate> candidates;
  candidates.reserve(256);

  auto heatmapAt = [&](int32_t cls, int32_t yy, int32_t xx) {
    return outputs.hm[static_cast<size_t>(cls) * cells_per_class + static_cast<size_t>(yy) * w + xx];
  };
  auto localMax = [&](int32_t cls, int32_t yy, int32_t xx, float value) {
    const int32_t y0 = std::max(0, yy - 1);
    const int32_t y1 = std::min(h - 1, yy + 1);
    const int32_t x0 = std::max(0, xx - 1);
    const int32_t x1 = std::min(w - 1, xx + 1);
    for (int32_t y = y0; y <= y1; ++y) {
      for (int32_t x = x0; x <= x1; ++x) {
        if (y == yy && x == xx) {
          continue;
        }
        if (heatmapAt(cls, y, x) > value) {
          return false;
        }
      }
    }
    return true;
  };

  for (int32_t cls = 0; cls < config_.num_classes; ++cls) {
    for (int32_t yy = 0; yy < h; ++yy) {
      for (int32_t xx = 0; xx < w; ++xx) {
        const int32_t cell = yy * w + xx;
        const float heatmap_value = heatmapAt(cls, yy, xx);
        if (heatmap_value < raw_threshold || !localMax(cls, yy, xx, heatmap_value)) {
          continue;
        }

        const size_t plane = static_cast<size_t>(cells_per_class);
        const float offset_x = outputs.center[0U * plane + cell];
        const float offset_y = outputs.center[1U * plane + cell];
        Detection det;
        det.x = (static_cast<float>(xx) + offset_x) * real_stride_x + config_.point_cloud_range[0];
        det.y = (static_cast<float>(yy) + offset_y) * real_stride_y + config_.point_cloud_range[1];
        det.z = outputs.center_z[cell];
        det.length = std::exp(outputs.dim[0U * plane + cell]);
        det.width = std::exp(outputs.dim[1U * plane + cell]);
        det.height = std::exp(outputs.dim[2U * plane + cell]);
        det.yaw = std::atan2(outputs.rot[0U * plane + cell], outputs.rot[1U * plane + cell]);
        det.score = logits ? sigmoid(heatmap_value) : heatmap_value;
        det.label = cls + 1;

        if (isValidDetection(det)) {
          candidates.push_back({det, false});
        }
      }
    }
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const Candidate & a, const Candidate & b) { return a.det.score > b.det.score; });
  if (static_cast<int32_t>(candidates.size()) > config_.nms_pre_max_size) {
    candidates.resize(static_cast<size_t>(config_.nms_pre_max_size));
  }
  return nms(std::move(candidates));
}

bool PointPillarsPostprocessor::isValidDetection(const Detection & det) const
{
  if (!config_.filter_invalid_boxes) {
    return true;
  }

  // 低分误检有时会解出极端尺寸或跑出点云范围；发布前过滤掉这些物理上不合理的框，
  // 防止 RViz 被无意义的大框/飞框刷满。
  if (!std::isfinite(det.x) || !std::isfinite(det.y) || !std::isfinite(det.z) ||
      !std::isfinite(det.length) || !std::isfinite(det.width) || !std::isfinite(det.height) ||
      !std::isfinite(det.yaw) || !std::isfinite(det.score)) {
    return false;
  }

  const bool center_inside_xy =
    det.x >= config_.point_cloud_range[0] && det.x <= config_.point_cloud_range[3] &&
    det.y >= config_.point_cloud_range[1] && det.y <= config_.point_cloud_range[4];
  const bool center_inside_z =
    det.z >= config_.point_cloud_range[2] - config_.max_z_margin &&
    det.z <= config_.point_cloud_range[5] + config_.max_z_margin;
  const bool size_ok =
    det.length >= config_.min_box_size && det.length <= config_.max_box_size &&
    det.width >= config_.min_box_size && det.width <= config_.max_box_size &&
    det.height >= config_.min_box_size && det.height <= config_.max_box_size;

  return center_inside_xy && center_inside_z && size_ok;
}

std::vector<Detection> PointPillarsPostprocessor::nms(std::vector<Candidate> candidates) const
{
  // class_agnostic_nms：不按类别分别 NMS，而是所有类别候选框一起抑制。
  std::vector<Detection> kept;
  kept.reserve(static_cast<size_t>(config_.nms_post_max_size));

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].suppressed) {
      continue;
    }
    kept.push_back(candidates[i].det);
    if (static_cast<int32_t>(kept.size()) >= config_.nms_post_max_size) {
      break;
    }

    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!candidates[j].suppressed &&
          orientedIouBev(candidates[i].det, candidates[j].det) > config_.nms_threshold) {
        candidates[j].suppressed = true;
      }
    }
  }

  return kept;
}

float PointPillarsPostprocessor::sigmoid(float x)
{
  if (x >= 0.0F) {
    const float z = std::exp(-x);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(x);
  return z / (1.0F + z);
}

float PointPillarsPostprocessor::orientedIouBev(const Detection & a, const Detection & b)
{
  // BEV 旋转框 IoU：先把两个框转成四边形，再做多边形裁剪得到交集面积。
  // 这里是 CPU 实现，足够直观；如果后处理成为瓶颈，可再替换为 CUDA NMS。
  if (a.length <= 0.0F || a.width <= 0.0F || b.length <= 0.0F || b.width <= 0.0F) {
    return 0.0F;
  }

  const auto corners_a = boxCorners(a);
  const auto corners_b = boxCorners(b);
  std::vector<Point2D> inter(corners_a.begin(), corners_a.end());
  for (size_t i = 0; i < corners_b.size(); ++i) {
    inter = clipPolygonWithEdge(inter, corners_b[i], corners_b[(i + 1) % corners_b.size()]);
    if (inter.empty()) {
      return 0.0F;
    }
  }

  const float inter_area = polygonArea(inter);
  const float area_a = a.length * a.width;
  const float area_b = b.length * b.width;
  const float denom = area_a + area_b - inter_area;
  return denom > 1e-6F ? inter_area / denom : 0.0F;
}

}  // namespace lidar_cone_detector
