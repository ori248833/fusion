#ifndef FS_FUSION_BOX_MATH_HPP  // 加上 _BOX_
#define FS_FUSION_BOX_MATH_HPP

#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <drd25_msgs/msg/cone.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fs_fusion_box {

    // 标定参数结构体
    struct CalibrationParams {
        cv::Mat K;              // 内参矩阵 3x3
        cv::Mat D;              // 畸变系数
        Eigen::Matrix4d T_l2c;  // 雷达(Lidar) -> 相机(Camera) 的变换矩阵
        int img_w;              // 图像宽度
        int img_h;              // 图像高度
    };

    // 辅助结构：用于存储投影后的2D框
    struct ProjectedBox {
        cv::Rect rect;      // 投影在图像上的矩形
        int original_index; // 对应原始3D检测框的索引
        double depth;       // 锥筒中心的深度
        bool valid;         // 投影是否成功（是否在图像内）
    };

    /**
     * @brief 核心功能1：将3D检测框投影到2D平面
     * 逻辑：取3D框的8个角点 -> 变换坐标 -> 投影 -> 取2D外包络矩形
     */
    std::vector<ProjectedBox> project_3d_boxes_to_2d(
        const vision_msgs::msg::Detection3DArray::ConstSharedPtr& lidar_msg,
        const CalibrationParams& params
    );

    /**
     * @brief 计算两个矩形的交并比 (IoU) 或 重叠率
     * 用于判断投影框和YOLO框是不是同一个物体
     */
    double calculate_overlap(const cv::Rect& box1, const cv::Rect& box2);

    /**
     * @brief 核心功能2：执行融合逻辑
     * 逻辑：
     * 1. 遍历投影框和YOLO框，计算重叠度
     * 2. 匹配成功的 -> 赋予YOLO的颜色
     * 3. 雷达有但YOLO没看到的 -> 标记为 Unknown
     * 4. YOLO有但雷达没看到的 -> 标记为“待补漏” (返回索引)
    */
    struct FusionMatch {
        std::size_t lidar_index{0};
        std::size_t camera_index{0};
        double iou{0.0};
        uint8_t final_color{drd25_msgs::msg::Cone::UNKNOWN};
    };

    struct FusionResult {
        std::vector<drd25_msgs::msg::Cone> fused_cones;
        std::vector<FusionMatch> matches;
        std::vector<std::size_t> unmatched_camera_indices;
    };

    FusionResult fuse_measurements(
        const std::vector<ProjectedBox>& proj_boxes,
        const vision_msgs::msg::Detection3DArray::ConstSharedPtr& lidar_msg,
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr& camera_msg,
        double overlap_threshold = 0.3 // 置信度阈值
    );

    /**
     * @brief 核心功能3：单目反推 (补漏)
     * 针对YOLO多出来的框，利用地平面假设反推3D位置
     */
    std::vector<drd25_msgs::msg::Cone> recover_missing_cones(
        const std::vector<int>& unmatched_indices,
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr& camera_msg,
        const CalibrationParams& params
    );

} // namespace fs_fusion_box

#endif // FS_FUSION_BOX_MATH_HPP
