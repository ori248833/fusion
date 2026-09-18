#include "fs_fusion_box/visualizer.hpp"
#include <cmath>  // 需要 std::isfinite

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
    const CalibrationParams& params)
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

    // ============================================================
    // 【核心修改】YOLO 框：只有 camera_msg 非空时才绘制
    // ============================================================
    if (camera_msg) {
        for (const auto& cone : camera_msg->cones) {
            cv::Point top_left(
                cone.center.x - cone.size.x / 2.0,
                cone.center.y - cone.size.y / 2.0
            );
            cv::Point bottom_right(
                cone.center.x + cone.size.x / 2.0,
                cone.center.y + cone.size.y / 2.0
            );

            cv::rectangle(canvas, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
            
            std::string label = "YOLO " + std::to_string(int(cone.confidence * 100)) + "%";
            cv::putText(canvas, label, cv::Point(top_left.x, top_left.y - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }
    } else {
        // 可选：在图像上显示"无相机数据"提示
        cv::putText(canvas, "No Camera Data", cv::Point(10, 30), 
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    // ============================================================
    // 雷达投影框（lidar_msg 已确保非空）
    // ============================================================
    for (const auto& detection : lidar_msg->cones) {
        double cx = detection.center.x; 
        double cy = detection.center.y; 
        double cz = detection.center.z;
        double dx = detection.size.x / 2.0; 
        double dy = detection.size.y / 2.0; 
        double dz = detection.size.z / 2.0;

        double yaw = detection.yaw;
        
        Eigen::Matrix3d rot_z;
        rot_z << cos(yaw), -sin(yaw), 0,
                sin(yaw),  cos(yaw), 0,
                0,        0,        1;

        std::vector<cv::Point3f> object_points;
        int signs[8][3] = {
            {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1},
            {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1}
        };

        for(int k=0; k<8; ++k) {
            Eigen::Vector3d pt_raw(
                signs[k][0]*dx, 
                signs[k][1]*dy, 
                signs[k][2]*dz
            );
            Eigen::Vector3d pt_rot = rot_z * pt_raw;
            object_points.push_back(cv::Point3f(
                cx + pt_rot.x(), 
                cy + pt_rot.y(), 
                cz + pt_rot.z()
            ));
        }

        std::vector<cv::Point2f> image_points;
        bool all_points_valid = true;
        for(const auto& pt3 : object_points) {
            Eigen::Vector4d pt_l(pt3.x, pt3.y, pt3.z, 1.0);
            Eigen::Vector4d pt_c_eigen = params.T_l2c * pt_l;
            cv::Point3f pt_c(pt_c_eigen.x(), pt_c_eigen.y(), pt_c_eigen.z());

            if (pt_c.z <= 0.1) { 
                all_points_valid = false; 
                break; 
            }
            
            double u = (pt_c.x * params.K.at<double>(0,0) / pt_c.z) + params.K.at<double>(0,2);
            double v = (pt_c.y * params.K.at<double>(1,1) / pt_c.z) + params.K.at<double>(1,2);
            
            // ---------- 防御：检查 u/v 是否有效 ----------
            if (!std::isfinite(u) || !std::isfinite(v)) {
                all_points_valid = false;
                break;
            }
            
            image_points.push_back(cv::Point2f(u, v));
        }

        if (!all_points_valid || image_points.empty()) continue;

        cv::Rect rect = cv::boundingRect(image_points);
        rect = rect & cv::Rect(0, 0, canvas.cols, canvas.rows); 
        
        if (rect.area() > 0) {
            cv::rectangle(canvas, rect, cv::Scalar(255, 0, 0), 2);
            cv::putText(canvas, "Lidar", cv::Point(rect.x, rect.y + rect.height + 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
        }
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