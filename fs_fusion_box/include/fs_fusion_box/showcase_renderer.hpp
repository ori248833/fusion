#ifndef FS_FUSION_BOX_SHOWCASE_RENDERER_HPP
#define FS_FUSION_BOX_SHOWCASE_RENDERER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <lidar_cone_detector/msg/three_d_cone_array.hpp>

namespace fs_fusion_box {

struct ShowcaseRenderOptions {
    int image_width{1920};
    int image_height{1080};
    std::size_t point_stride{2};
    double forward_min{-5.0};
    double forward_max{35.0};
    double lateral_limit{15.0};
    double height_min{-3.0};
    double height_max{5.0};
};

struct ShowcaseRenderStats {
    std::size_t input_points{0};
    std::size_t rendered_points{0};
    std::size_t colored_points{0};
    std::size_t rendered_cones{0};
};

bool render_showcase_image(
    const sensor_msgs::msg::PointCloud2& cloud,
    const lidar_cone_detector::msg::ThreeDConeArray& lidar_cones,
    const std::vector<uint8_t>& final_colors,
    const ShowcaseRenderOptions& options,
    const std::filesystem::path& output_path,
    ShowcaseRenderStats& stats,
    std::string& error_message);

}  // namespace fs_fusion_box

#endif  // FS_FUSION_BOX_SHOWCASE_RENDERER_HPP
