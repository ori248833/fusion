#ifndef FS_FUSION_BOX_VISUALIZER_HPP
#define FS_FUSION_BOX_VISUALIZER_HPP

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <drd25_msgs/msg/cone.hpp>
// [修改] 引用新的头文件
#include <lidar_cone_detector/msg/three_d_cone_array.hpp>
#include <cone_interfaces/msg/cone_array.hpp> 
#include <sensor_msgs/msg/image.hpp> 
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>     
#include <vector>
#include <string>
#include <set>

#include "fs_fusion_box/fs_fusion_box_math.hpp" 

namespace fs_fusion_box {

class Visualizer {
public:
    Visualizer(rclcpp::Node* node, const std::string& frame_id);

    // 1. Rviz Marker 发布
    // [修改] 命名空间变更
    // void publishFusedCones(
    //     const std::vector<drd25_msgs::msg::Cone>& cones, 
    //     const lidar_cone_detector::msg::ThreeDConeArray::ConstSharedPtr& lidar_msg,
    //     const std::vector<size_t>& recovered_indices, 
    //     const std_msgs::msg::Header& header
    // );

    void publishFusedCones(
        const std::vector<drd25_msgs::msg::Cone>& cones, 
        const lidar_cone_detector::msg::ThreeDConeArray::ConstSharedPtr& lidar_msg,
        const std_msgs::msg::Header& header
    );

    // 2. 纯数据驱动的虚拟视图
    // [修改] 命名空间变更
    void publishSyntheticView(
        const cone_interfaces::msg::ConeArray::ConstSharedPtr& camera_msg,
        const lidar_cone_detector::msg::ThreeDConeArray::ConstSharedPtr& lidar_msg,
        const CalibrationParams& params
    );

private:
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_; 
    std::string frame_id_;
};

} // namespace fs_fusion_box

#endif // FS_FUSION_BOX_VISUALIZER_HPP