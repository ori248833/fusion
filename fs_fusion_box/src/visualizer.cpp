#include "fs_fusion_box/visualizer.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

constexpr double kObjectLabelFontScale = 0.35;
constexpr double kTimestampFontScale = 0.42;
constexpr int kLabelThickness = 1;
constexpr int kLabelOutlineThickness = 3;

cv::Point rect_center(const cv::Rect& rect) {
    return cv::Point(
        rect.x + rect.width / 2,
        rect.y + rect.height / 2);
}

cv::Rect camera_rect(
    const cone_interfaces::msg::Cone& cone,
    const cv::Size& canvas_size) {
    cv::Rect rect(
        cvRound(cone.center.x - cone.size.x / 2.0),
        cvRound(cone.center.y - cone.size.y / 2.0),
        cvRound(cone.size.x),
        cvRound(cone.size.y));
    return rect & cv::Rect(0, 0, canvas_size.width, canvas_size.height);
}

void draw_dashed_line(
    cv::Mat& canvas,
    const cv::Point& start,
    const cv::Point& end,
    const cv::Scalar& color,
    int thickness = 1,
    double dash_length = 6.0,
    double gap_length = 4.0) {
    const cv::Point2d delta(
        static_cast<double>(end.x - start.x),
        static_cast<double>(end.y - start.y));
    const double length = std::hypot(delta.x, delta.y);
    if (length < 1.0) {
        return;
    }

    const cv::Point2d direction(delta.x / length, delta.y / length);
    for (double offset = 0.0; offset < length;
         offset += dash_length + gap_length) {
        const double segment_end = std::min(offset + dash_length, length);
        const cv::Point p1(
            cvRound(start.x + direction.x * offset),
            cvRound(start.y + direction.y * offset));
        const cv::Point p2(
            cvRound(start.x + direction.x * segment_end),
            cvRound(start.y + direction.y * segment_end));
        cv::line(canvas, p1, p2, color, thickness, cv::LINE_AA);
    }
}

void draw_dashed_rect(
    cv::Mat& canvas,
    const cv::Rect& rect,
    const cv::Scalar& color,
    int thickness = 1) {
    const cv::Point top_left(rect.x, rect.y);
    const cv::Point top_right(rect.x + rect.width, rect.y);
    const cv::Point bottom_left(rect.x, rect.y + rect.height);
    const cv::Point bottom_right(
        rect.x + rect.width, rect.y + rect.height);
    draw_dashed_line(canvas, top_left, top_right, color, thickness);
    draw_dashed_line(canvas, top_right, bottom_right, color, thickness);
    draw_dashed_line(canvas, bottom_right, bottom_left, color, thickness);
    draw_dashed_line(canvas, bottom_left, top_left, color, thickness);
}

int overlap_area(
    const cv::Rect& candidate,
    const std::vector<cv::Rect>& occupied) {
    int area = 0;
    for (const auto& rect : occupied) {
        area += (candidate & rect).area();
    }
    return area;
}

cv::Rect draw_label(
    cv::Mat& canvas,
    const std::string& text,
    const cv::Rect& anchor,
    bool prefer_above,
    const cv::Scalar& foreground,
    double font_scale,
    std::vector<cv::Rect>& occupied) {
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        text,
        cv::FONT_HERSHEY_SIMPLEX,
        font_scale,
        kLabelThickness,
        &baseline);
    const int width = text_size.width + 8;
    const int height = text_size.height + baseline + 6;
    const int gap = 3;

    std::vector<cv::Point> origins;
    origins.reserve(6);
    if (prefer_above) {
        origins.emplace_back(anchor.x, anchor.y - height - gap);
        origins.emplace_back(
            anchor.x + anchor.width - width,
            anchor.y - height - gap);
        origins.emplace_back(anchor.x, anchor.y + anchor.height + gap);
        origins.emplace_back(
            anchor.x + anchor.width - width,
            anchor.y + anchor.height + gap);
    } else {
        origins.emplace_back(anchor.x, anchor.y + anchor.height + gap);
        origins.emplace_back(
            anchor.x + anchor.width - width,
            anchor.y + anchor.height + gap);
        origins.emplace_back(anchor.x, anchor.y - height - gap);
        origins.emplace_back(
            anchor.x + anchor.width - width,
            anchor.y - height - gap);
    }
    origins.emplace_back(anchor.x + anchor.width + gap, anchor.y);
    origins.emplace_back(anchor.x - width - gap, anchor.y);

    cv::Rect best;
    int best_score = std::numeric_limits<int>::max();
    for (size_t i = 0; i < origins.size(); ++i) {
        const int x = std::clamp(
            origins[i].x, 0, std::max(0, canvas.cols - width));
        const int y = std::clamp(
            origins[i].y, 0, std::max(0, canvas.rows - height));
        const cv::Rect candidate(x, y, width, height);
        const int score = overlap_area(candidate, occupied) * 100 +
                          static_cast<int>(i);
        if (score < best_score) {
            best = candidate;
            best_score = score;
        }
    }

    const cv::Point text_origin(
        best.x + 4,
        best.y + 3 + text_size.height);
    cv::putText(
        canvas,
        text,
        text_origin,
        cv::FONT_HERSHEY_SIMPLEX,
        font_scale,
        cv::Scalar(0, 0, 0),
        kLabelOutlineThickness,
        cv::LINE_AA);
    cv::putText(
        canvas,
        text,
        text_origin,
        cv::FONT_HERSHEY_SIMPLEX,
        font_scale,
        foreground,
        kLabelThickness,
        cv::LINE_AA);
    occupied.push_back(best);
    return best;
}

std::string format_track_id(uint64_t track_id) {
    return track_id == 0 ? "T--" : "T" + std::to_string(track_id);
}

std::string format_timestamp(const builtin_interfaces::msg::Time& stamp) {
    std::ostringstream stream;
    stream << "Time " << stamp.sec << '.' << std::setfill('0')
           << std::setw(9) << stamp.nanosec;
    return stream.str();
}

std::string color_name(uint8_t color) {
    switch (color) {
        case drd25_msgs::msg::Cone::BLUE:
            return "Blue";
        case drd25_msgs::msg::Cone::RED:
            return "Red";
        case drd25_msgs::msg::Cone::YELLOW_BIG:
            return "Yellow-L";
        case drd25_msgs::msg::Cone::YELLOW_SMALL:
            return "Yellow-S";
        case drd25_msgs::msg::Cone::UNKNOWN_BIG:
            return "Unknown-L";
        case drd25_msgs::msg::Cone::UNKNOWN_SMALL:
            return "Unknown-S";
        default:
            return "Unknown";
    }
}

std::string decision_tag(const std::string& decision, bool matched) {
    if (decision.find("沿用历史颜色") != std::string::npos) {
        return " H";
    }
    if (decision.find("颜色冲突") != std::string::npos) {
        return " !";
    }
    if (!matched) {
        return " U";
    }
    return "";
}

}  // namespace

namespace fs_fusion_box {

Visualizer::Visualizer(rclcpp::Node* node, const std::string& frame_id) 
    : frame_id_(frame_id) 
{
    marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("fusion/markers", 10);
    image_pub_ = node->create_publisher<sensor_msgs::msg::Image>("fusion/debug_image", 1);
}

// ============================================================
// 1. publishFusedCones - RViz 可视化（MarkerArray）
// ============================================================
void Visualizer::publishFusedCones(
    const std::vector<drd25_msgs::msg::Cone>& cones, 
    const lidar_cone_detector::msg::ThreeDConeArray::ConstSharedPtr& lidar_msg,
    const std_msgs::msg::Header& /*header*/) 
{
    // ---------- 防御：检查 lidar_msg 是否为空 ----------
    if (!lidar_msg) {
        RCLCPP_WARN(rclcpp::get_logger("visualizer"), 
                    "publishFusedCones: lidar_msg is nullptr, skip");
        return;
    }

    visualization_msgs::msg::MarkerArray markers;
    
    // 清空旧 Marker
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_all.header.frame_id = frame_id_;
    markers.markers.push_back(delete_all);

    int id = 0;
    for (size_t i = 0; i < cones.size(); ++i) {
        const auto& cone = cones[i];

        visualization_msgs::msg::Marker m;
        m.header = lidar_msg->header; 
        m.header.frame_id = frame_id_;
        m.ns = "fused_cones";
        m.id = ++id;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.type = visualization_msgs::msg::Marker::CUBE;

        // ---------- 默认尺寸 ----------
        double l = 0.2, w = 0.2, h = 0.32;
        double cone_z = h / 2.0;
        double yaw = 0.0;

        // ---------- 从雷达读取真实尺寸（仅当索引有效） ----------
        if (i < lidar_msg->cones.size()) {
            l = lidar_msg->cones[i].size.x;
            w = lidar_msg->cones[i].size.y;
            h = lidar_msg->cones[i].size.z;
            cone_z = lidar_msg->cones[i].center.z;
            yaw = lidar_msg->cones[i].yaw;

            if (h < 0.05) h = 0.05;

            Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());
            Eigen::Quaterniond q(yaw_rot);
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();
            m.pose.orientation.w = q.w();
        }

        m.scale.x = l;
        m.scale.y = w;
        m.scale.z = h;
        m.pose.position.x = cone.x;
        m.pose.position.y = cone.y;
        m.pose.position.z = cone_z;

        // ============================================================
        // 【修改点】颜色映射：完全不显示橙色（cone.color == 3）
        // 策略：将橙色（3）强制改为蓝色（0）或灰色（4）
        // ============================================================
        uint8_t color_to_display = cone.color;
        
        // 如果颜色是橙色（3），强制改为蓝色（0）或灰色（4）
        if (color_to_display == 3) {
            color_to_display = 0;  // 改为蓝色，你也可以改成灰色（4）
        }

        m.color.a = 0.8;
        switch (color_to_display) {
            case 0:  // 蓝色
                m.color.b = 1.0;
                break;
            case 1:  // 红色
                m.color.r = 1.0;
                break;
            case 2:  // 黄色
                m.color.r = 1.0;
                m.color.g = 1.0;
                break;
            case 3:  // 橙色（已强制转换，不会进入此分支）
                m.color.r = 1.0;
                m.color.g = 0.5;
                break;
            default: // 灰色（包括 UNKNOWN）
                m.color.r = 0.5;
                m.color.g = 0.5;
                m.color.b = 0.5;
                break;
        }

        m.lifetime = rclcpp::Duration::from_seconds(0.2);
        markers.markers.push_back(m);
    }
    
    marker_pub_->publish(markers);
}

// ============================================================
// 2. publishSyntheticView - 调试图像（仅保留一个定义！）
// ============================================================
void Visualizer::publishSyntheticView(
    const cone_interfaces::msg::ConeArray::ConstSharedPtr& camera_msg,
    const lidar_cone_detector::msg::ThreeDConeArray::ConstSharedPtr& lidar_msg,
    const CalibrationParams& params,
    const std::vector<ProjectedBox>& projected_boxes,
    const std::vector<FusionMatch>& matches,
    const std::vector<uint64_t>& track_ids,
    const std::vector<uint8_t>& final_colors,
    const std::vector<std::string>& decisions)
{
    // ---------- 防御1：lidar_msg 不能为空 ----------
    if (!lidar_msg) {
        RCLCPP_WARN(rclcpp::get_logger("visualizer"), 
                    "publishSyntheticView: lidar_msg is nullptr, skip");
        return;
    }

    // ---------- 防御2：图像尺寸必须有效 ----------
    if (params.img_w <= 0 || params.img_h <= 0) {
        RCLCPP_WARN(rclcpp::get_logger("visualizer"), 
                    "publishSyntheticView: invalid image size (%dx%d)", 
                    params.img_w, params.img_h);
        return;
    }

    cv::Mat canvas = cv::Mat::zeros(params.img_h, params.img_w, CV_8UC3);
    const cv::Size canvas_size(canvas.cols, canvas.rows);
    const cv::Scalar camera_color(0, 255, 0);
    const cv::Scalar lidar_color(255, 128, 0);
    const cv::Scalar unmatched_color(0, 165, 255);
    const cv::Scalar link_color(210, 210, 210);

    const size_t lidar_count = lidar_msg->cones.size();
    const size_t camera_count = camera_msg ? camera_msg->cones.size() : 0;
    std::vector<cv::Rect> lidar_rects(lidar_count);
    std::vector<bool> lidar_rect_valid(lidar_count, false);
    for (const auto& projected_box : projected_boxes) {
        if (!projected_box.valid ||
            projected_box.original_index < 0 ||
            static_cast<size_t>(projected_box.original_index) >= lidar_count) {
            continue;
        }
        const size_t lidar_index =
            static_cast<size_t>(projected_box.original_index);
        const cv::Rect clipped = projected_box.rect &
            cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (clipped.area() > 0) {
            lidar_rects[lidar_index] = clipped;
            lidar_rect_valid[lidar_index] = true;
        }
    }

    std::vector<cv::Rect> camera_rects(camera_count);
    std::vector<bool> camera_rect_valid(camera_count, false);
    if (camera_msg) {
        for (size_t i = 0; i < camera_count; ++i) {
            camera_rects[i] = camera_rect(camera_msg->cones[i], canvas_size);
            camera_rect_valid[i] = camera_rects[i].area() > 0;
        }
    }

    std::vector<int> lidar_to_camera(lidar_count, -1);
    std::vector<int> camera_to_lidar(camera_count, -1);
    std::vector<double> lidar_iou(lidar_count, 0.0);
    for (const auto& match : matches) {
        if (match.lidar_index >= lidar_count ||
            match.camera_index >= camera_count) {
            continue;
        }
        lidar_to_camera[match.lidar_index] =
            static_cast<int>(match.camera_index);
        camera_to_lidar[match.camera_index] =
            static_cast<int>(match.lidar_index);
        lidar_iou[match.lidar_index] = match.iou;
    }

    // Draw association lines first so boxes and text stay visually dominant.
    for (size_t lidar_index = 0; lidar_index < lidar_count; ++lidar_index) {
        const int camera_index = lidar_to_camera[lidar_index];
        if (camera_index < 0 || !lidar_rect_valid[lidar_index] ||
            !camera_rect_valid[static_cast<size_t>(camera_index)]) {
            continue;
        }
        cv::line(
            canvas,
            rect_center(lidar_rects[lidar_index]),
            rect_center(camera_rects[static_cast<size_t>(camera_index)]),
            link_color,
            1,
            cv::LINE_AA);
    }

    for (size_t i = 0; i < camera_count; ++i) {
        if (!camera_rect_valid[i]) {
            continue;
        }
        if (camera_to_lidar[i] >= 0) {
            cv::rectangle(canvas, camera_rects[i], camera_color, 2, cv::LINE_AA);
        } else {
            draw_dashed_rect(canvas, camera_rects[i], unmatched_color, 1);
        }
    }
    for (size_t i = 0; i < lidar_count; ++i) {
        if (!lidar_rect_valid[i]) {
            continue;
        }
        if (lidar_to_camera[i] >= 0) {
            cv::rectangle(canvas, lidar_rects[i], lidar_color, 2, cv::LINE_AA);
        } else {
            draw_dashed_rect(canvas, lidar_rects[i], lidar_color, 1);
        }
    }

    std::vector<cv::Rect> occupied;
    occupied.reserve(lidar_count + camera_count + matches.size() + 1);
    draw_label(
        canvas,
        format_timestamp(lidar_msg->header.stamp),
        cv::Rect(8, 0, 1, 1),
        false,
        cv::Scalar(235, 235, 235),
        kTimestampFontScale,
        occupied);
    occupied.insert(occupied.end(), lidar_rects.begin(), lidar_rects.end());
    occupied.insert(occupied.end(), camera_rects.begin(), camera_rects.end());

    for (size_t i = 0; i < camera_count; ++i) {
        if (!camera_rect_valid[i]) {
            continue;
        }
        uint64_t track_id = 0;
        const int lidar_index = camera_to_lidar[i];
        if (lidar_index >= 0 &&
            static_cast<size_t>(lidar_index) < track_ids.size()) {
            track_id = track_ids[static_cast<size_t>(lidar_index)];
        }
        std::ostringstream label;
        label << format_track_id(track_id) << " Y" << i + 1
              << " C" << std::fixed << std::setprecision(2)
              << camera_msg->cones[i].confidence;
        draw_label(
            canvas,
            label.str(),
            camera_rects[i],
            true,
            camera_to_lidar[i] >= 0 ? camera_color : unmatched_color,
            kObjectLabelFontScale,
            occupied);
    }

    for (size_t i = 0; i < lidar_count; ++i) {
        if (!lidar_rect_valid[i]) {
            continue;
        }
        const uint64_t track_id = i < track_ids.size() ? track_ids[i] : 0;
        const uint8_t final_color = i < final_colors.size()
            ? final_colors[i]
            : drd25_msgs::msg::Cone::UNKNOWN;
        const bool matched = lidar_to_camera[i] >= 0;
        const std::string decision = i < decisions.size()
            ? decisions[i]
            : std::string();
        std::ostringstream label;
        label << format_track_id(track_id) << " L" << i + 1 << ' '
              << color_name(final_color)
              << decision_tag(decision, matched);
        draw_label(
            canvas,
            label.str(),
            lidar_rects[i],
            false,
            lidar_color,
            kObjectLabelFontScale,
            occupied);
    }

    for (size_t lidar_index = 0; lidar_index < lidar_count; ++lidar_index) {
        const int camera_index = lidar_to_camera[lidar_index];
        if (camera_index < 0 || !lidar_rect_valid[lidar_index] ||
            !camera_rect_valid[static_cast<size_t>(camera_index)]) {
            continue;
        }
        const cv::Point lidar_center = rect_center(lidar_rects[lidar_index]);
        const cv::Point yolo_center =
            rect_center(camera_rects[static_cast<size_t>(camera_index)]);
        const cv::Point midpoint(
            (lidar_center.x + yolo_center.x) / 2,
            (lidar_center.y + yolo_center.y) / 2);
        std::ostringstream label;
        label << "L" << lidar_index + 1 << "<->Y" << camera_index + 1
              << " IoU " << std::fixed << std::setprecision(2)
              << lidar_iou[lidar_index];
        draw_label(
            canvas,
            label.str(),
            cv::Rect(midpoint.x, midpoint.y, 1, 1),
            true,
            link_color,
            kObjectLabelFontScale,
            occupied);
    }

    // ---------- 发布图像 ----------
    std_msgs::msg::Header header = lidar_msg->header; 
    cv_bridge::CvImage out_msg;
    out_msg.header = header; 
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = canvas;

    image_pub_->publish(*out_msg.toImageMsg());
}

} // namespace fs_fusion_box
