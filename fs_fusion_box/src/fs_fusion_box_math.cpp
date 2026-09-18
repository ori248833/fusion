#include "fs_fusion_box/fs_fusion_box_math.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/eigen.hpp>

namespace fs_fusion_box {

namespace {

struct MatchCandidate {
    size_t lidar_index{0};
    size_t camera_index{0};
    double iou{0.0};
};

int parse_class_id(
    const vision_msgs::msg::ObjectHypothesisWithPose& hypothesis,
    int fallback = -1) {
    if (hypothesis.hypothesis.class_id.empty()) {
        return fallback;
    }
    try {
        return std::stoi(hypothesis.hypothesis.class_id);
    } catch (...) {
        return fallback;
    }
}

bool is_supported_map_color(int color) {
    return color == drd25_msgs::msg::Cone::BLUE ||
           color == drd25_msgs::msg::Cone::RED ||
           color == drd25_msgs::msg::Cone::YELLOW_BIG ||
           color == drd25_msgs::msg::Cone::YELLOW_SMALL;
}

bool is_confirmed_map_color(uint8_t color) {
    return color == drd25_msgs::msg::Cone::BLUE ||
           color == drd25_msgs::msg::Cone::RED ||
           color == drd25_msgs::msg::Cone::YELLOW_BIG ||
           color == drd25_msgs::msg::Cone::YELLOW_SMALL;
}

uint8_t unknown_color_from_lidar_detection(
    const vision_msgs::msg::Detection3D& lidar_detection) {
    int lidar_label = -1;
    if (!lidar_detection.results.empty()) {
        lidar_label = parse_class_id(lidar_detection.results.front());
    }

    if (lidar_label == 1) {
        return drd25_msgs::msg::Cone::UNKNOWN_SMALL;
    }
    if (lidar_label == 2) {
        return drd25_msgs::msg::Cone::UNKNOWN_BIG;
    }
    return drd25_msgs::msg::Cone::UNKNOWN;
}

uint8_t resolve_color(
    int mapped_camera_color,
    const vision_msgs::msg::Detection3D& lidar_detection) {
    if (!is_supported_map_color(mapped_camera_color)) {
        return drd25_msgs::msg::Cone::UNKNOWN;
    }

    if (mapped_camera_color == drd25_msgs::msg::Cone::YELLOW_BIG ||
        mapped_camera_color == drd25_msgs::msg::Cone::YELLOW_SMALL) {
        // LiDAR label semantics are fixed by the detector contract:
        // 1 = small cone, 2 = big cone.
        int lidar_label = -1;
        if (!lidar_detection.results.empty()) {
            lidar_label = parse_class_id(lidar_detection.results.front());
        }
        if (lidar_label == 1) {
            return drd25_msgs::msg::Cone::YELLOW_SMALL;
        }
        if (lidar_label == 2) {
            return drd25_msgs::msg::Cone::YELLOW_BIG;
        }
    }

    return static_cast<uint8_t>(mapped_camera_color);
}

}  // namespace

std::vector<ProjectedBox> project_3d_boxes_to_2d(
    const vision_msgs::msg::Detection3DArray::ConstSharedPtr& lidar_msg,
    const CalibrationParams& params) {
    std::vector<ProjectedBox> results;
    results.reserve(lidar_msg->detections.size());

    for (size_t i = 0; i < lidar_msg->detections.size(); ++i) {
        const auto& detection = lidar_msg->detections[i];

        const double cx = detection.bbox.center.position.x;
        const double cy = detection.bbox.center.position.y;
        const double cz = detection.bbox.center.position.z;
        const double dx = detection.bbox.size.x / 2.0;
        const double dy = detection.bbox.size.y / 2.0;
        const double dz = detection.bbox.size.z / 2.0;

        Eigen::Quaterniond quaternion(1.0, 0.0, 0.0, 0.0);
        if (!detection.results.empty()) {
            const auto& orientation =
                detection.results.front().pose.pose.orientation;
            quaternion = Eigen::Quaterniond(
                orientation.w,
                orientation.x,
                orientation.y,
                orientation.z);
            if (quaternion.norm() < 1e-9) {
                quaternion = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            } else {
                quaternion.normalize();
            }
        }

        const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();
        const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
        const Eigen::Matrix3d rotation_z =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        const Eigen::Vector3d center(cx, cy, cz);

        constexpr int signs[8][3] = {
            {1, 1, 1},
            {1, 1, -1},
            {1, -1, 1},
            {1, -1, -1},
            {-1, 1, 1},
            {-1, 1, -1},
            {-1, -1, 1},
            {-1, -1, -1},
        };

        std::vector<cv::Point2f> image_points;
        image_points.reserve(8);
        for (const auto& sign : signs) {
            const Eigen::Vector3d raw_corner(
                cx + static_cast<double>(sign[0]) * dx,
                cy + static_cast<double>(sign[1]) * dy,
                cz + static_cast<double>(sign[2]) * dz);
            const Eigen::Vector3d rotated_corner =
                rotation_z * (raw_corner - center) + center;
            const Eigen::Vector4d lidar_point(
                rotated_corner.x(),
                rotated_corner.y(),
                rotated_corner.z(),
                1.0);
            const Eigen::Vector3d camera_point =
                (params.T_l2c * lidar_point).head<3>();

            if (camera_point.z() <= 0.1) {
                continue;
            }
            const double u =
                camera_point.x() * params.K.at<double>(0, 0) /
                    camera_point.z() +
                params.K.at<double>(0, 2);
            const double v =
                camera_point.y() * params.K.at<double>(1, 1) /
                    camera_point.z() +
                params.K.at<double>(1, 2);
            image_points.emplace_back(
                static_cast<float>(u), static_cast<float>(v));
        }

        const Eigen::Vector4d lidar_center(cx, cy, cz, 1.0);
        ProjectedBox projected_box;
        projected_box.original_index = i;
        projected_box.depth = (params.T_l2c * lidar_center).z();
        projected_box.valid = false;

        if (!image_points.empty()) {
            cv::Rect rectangle = cv::boundingRect(image_points);
            rectangle &= cv::Rect(0, 0, params.img_w, params.img_h);
            if (rectangle.area() > 0) {
                projected_box.rect = rectangle;
                projected_box.valid = true;
            }
        }
        results.push_back(projected_box);
    }
    return results;
}

double calculate_overlap(const cv::Rect& box1, const cv::Rect& box2) {
    const cv::Rect intersection = box1 & box2;
    if (intersection.area() <= 0) {
        return 0.0;
    }

    const double intersection_area = intersection.area();
    const double union_area =
        box1.area() + box2.area() - intersection_area;
    return union_area > 0.0 ? intersection_area / union_area : 0.0;
}

FusionResult fuse_measurements(
    const std::vector<ProjectedBox>& projected_boxes,
    const vision_msgs::msg::Detection3DArray::ConstSharedPtr& lidar_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& camera_msg,
    double overlap_threshold) {
    FusionResult result;
    result.fused_cones.reserve(lidar_msg->detections.size());

    // LiDAR is authoritative: create exactly one output cone for every LiDAR
    // detection before considering any camera observation.
    for (const auto& lidar_detection : lidar_msg->detections) {
        drd25_msgs::msg::Cone cone;
        cone.x = lidar_detection.bbox.center.position.x;
        cone.y = lidar_detection.bbox.center.position.y;
        cone.color = unknown_color_from_lidar_detection(lidar_detection);
        result.fused_cones.push_back(cone);
    }

    constexpr double kVerticalScale = 0.25;
    constexpr double kMinimumVerticalPixels = 5.0;
    constexpr double kHorizontalScale = 1.5;
    constexpr double kMinimumHorizontalPixels = 15.0;

    std::vector<MatchCandidate> candidates;
    for (const auto& projected_box : projected_boxes) {
        if (!projected_box.valid ||
            projected_box.original_index >= result.fused_cones.size()) {
            continue;
        }

        const double lidar_bottom =
            projected_box.rect.y + projected_box.rect.height;
        const double lidar_center_x =
            projected_box.rect.x + projected_box.rect.width / 2.0;
        const double vertical_tolerance = std::max(
            kVerticalScale * projected_box.rect.height,
            kMinimumVerticalPixels);
        const double horizontal_tolerance = std::max(
            kHorizontalScale * projected_box.rect.width,
            kMinimumHorizontalPixels);

        // Radar-centric search: this projected LiDAR box examines every YOLO
        // box that passes the deterministic geometry gates.
        for (size_t camera_index = 0;
             camera_index < camera_msg->detections.size();
             ++camera_index) {
            const auto& camera_box =
                camera_msg->detections[camera_index].bbox;
            const cv::Rect camera_rectangle(
                cvRound(camera_box.center.position.x - camera_box.size_x / 2.0),
                cvRound(camera_box.center.position.y - camera_box.size_y / 2.0),
                cvRound(camera_box.size_x),
                cvRound(camera_box.size_y));

            if (camera_rectangle.width <= 0 || camera_rectangle.height <= 0) {
                continue;
            }

            const double camera_bottom =
                camera_rectangle.y + camera_rectangle.height;
            const double camera_center_x =
                camera_rectangle.x + camera_rectangle.width / 2.0;
            if (std::abs(camera_bottom - lidar_bottom) > vertical_tolerance ||
                std::abs(camera_center_x - lidar_center_x) >
                    horizontal_tolerance) {
                continue;
            }

            const double iou =
                calculate_overlap(projected_box.rect, camera_rectangle);
            if (iou > overlap_threshold) {
                candidates.push_back(MatchCandidate{
                    projected_box.original_index, camera_index, iou});
            }
        }
    }

    // Global greedy assignment makes the relationship strictly one-to-one.
    // With the small number of cones normally present, this is deterministic,
    // inexpensive, and avoids introducing an uncalibrated weighted penalty.
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const MatchCandidate& lhs, const MatchCandidate& rhs) {
            return lhs.iou > rhs.iou;
        });

    std::vector<bool> lidar_matched(lidar_msg->detections.size(), false);
    std::vector<bool> camera_matched(camera_msg->detections.size(), false);

    for (const auto& candidate : candidates) {
        if (lidar_matched[candidate.lidar_index] ||
            camera_matched[candidate.camera_index]) {
            continue;
        }

        const auto& camera_detection =
            camera_msg->detections[candidate.camera_index];
        if (camera_detection.results.empty()) {
            continue;
        }

        const int mapped_camera_color =
            parse_class_id(camera_detection.results.front());
        const uint8_t final_color = resolve_color(
            mapped_camera_color,
            lidar_msg->detections[candidate.lidar_index]);
        if (!is_confirmed_map_color(final_color)) {
            continue;
        }

        result.fused_cones[candidate.lidar_index].color = final_color;
        lidar_matched[candidate.lidar_index] = true;
        camera_matched[candidate.camera_index] = true;
        result.matches.push_back(FusionMatch{
            candidate.lidar_index,
            candidate.camera_index,
            candidate.iou,
            final_color});
    }

    for (size_t i = 0; i < camera_matched.size(); ++i) {
        if (!camera_matched[i]) {
            result.unmatched_camera_indices.push_back(i);
        }
    }

    // Unmatched LiDAR cones remain in fused_cones with UNKNOWN. The node's
    // short-lived Track color memory may supply a previously confirmed color;
    // a missed YOLO detection never deletes a LiDAR cone.
    return result;
}

}  // namespace fs_fusion_box
