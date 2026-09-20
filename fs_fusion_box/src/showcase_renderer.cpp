#include "fs_fusion_box/showcase_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <drd25_msgs/msg/cone.hpp>

namespace fs_fusion_box {

namespace {

struct ConeGeometry {
    Eigen::Vector3d center;
    Eigen::Vector3d half_size;
    double yaw{0.0};
    uint8_t color{drd25_msgs::msg::Cone::UNKNOWN};
};

struct PointSample {
    Eigen::Vector3d position;
    int cone_index{-1};
    double depth{0.0};
    cv::Point pixel;
};

struct ProjectedCorner {
    cv::Point pixel;
    double depth{0.0};
    bool valid{false};
};

struct CameraModel {
    Eigen::Vector3d eye;
    Eigen::Vector3d forward;
    Eigen::Vector3d right;
    Eigen::Vector3d up;
    double focal{1.0};
    double offset_x{0.0};
    double offset_y{0.0};
};

constexpr std::array<std::array<int, 2>, 12> kBoxEdges{{
    {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
    {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
    {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
}};

constexpr std::array<std::array<int, 4>, 6> kBoxFaces{{
    {{0, 1, 2, 3}}, {{4, 5, 6, 7}},
    {{0, 1, 5, 4}}, {{1, 2, 6, 5}},
    {{2, 3, 7, 6}}, {{3, 0, 4, 7}},
}};

cv::Scalar cone_color(uint8_t color) {
    switch (color) {
        case drd25_msgs::msg::Cone::BLUE:
            return cv::Scalar(255, 145, 35);
        case drd25_msgs::msg::Cone::RED:
            return cv::Scalar(45, 55, 255);
        case drd25_msgs::msg::Cone::YELLOW_BIG:
            return cv::Scalar(20, 190, 255);
        case drd25_msgs::msg::Cone::YELLOW_SMALL:
            return cv::Scalar(45, 235, 255);
        default:
            return cv::Scalar(165, 165, 165);
    }
}

std::array<Eigen::Vector3d, 8> box_corners(const ConeGeometry& cone) {
    const double cosine = std::cos(cone.yaw);
    const double sine = std::sin(cone.yaw);
    const Eigen::Matrix3d rotation =
        (Eigen::Matrix3d() <<
            cosine, -sine, 0.0,
            sine, cosine, 0.0,
            0.0, 0.0, 1.0).finished();

    std::array<Eigen::Vector3d, 8> corners;
    const std::array<Eigen::Vector3d, 8> local{{
        Eigen::Vector3d(
            -cone.half_size.x(), -cone.half_size.y(), -cone.half_size.z()),
        Eigen::Vector3d(
             cone.half_size.x(), -cone.half_size.y(), -cone.half_size.z()),
        Eigen::Vector3d(
             cone.half_size.x(),  cone.half_size.y(), -cone.half_size.z()),
        Eigen::Vector3d(
            -cone.half_size.x(),  cone.half_size.y(), -cone.half_size.z()),
        Eigen::Vector3d(
            -cone.half_size.x(), -cone.half_size.y(),  cone.half_size.z()),
        Eigen::Vector3d(
             cone.half_size.x(), -cone.half_size.y(),  cone.half_size.z()),
        Eigen::Vector3d(
             cone.half_size.x(),  cone.half_size.y(),  cone.half_size.z()),
        Eigen::Vector3d(
            -cone.half_size.x(),  cone.half_size.y(),  cone.half_size.z()),
    }};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        corners[i] = cone.center + rotation * local[i];
    }
    return corners;
}

bool point_inside_box(
    const Eigen::Vector3d& point,
    const ConeGeometry& cone) {
    const Eigen::Vector3d delta = point - cone.center;
    const double cosine = std::cos(cone.yaw);
    const double sine = std::sin(cone.yaw);
    const double local_x = cosine * delta.x() + sine * delta.y();
    const double local_y = -sine * delta.x() + cosine * delta.y();
    const double bottom_guard = std::min(0.03, cone.half_size.z() * 0.10);
    return std::abs(local_x) <= cone.half_size.x() &&
           std::abs(local_y) <= cone.half_size.y() &&
           delta.z() >= -cone.half_size.z() + bottom_guard &&
           delta.z() <= cone.half_size.z();
}

CameraModel make_camera(
    const std::vector<PointSample>& points,
    const std::vector<ConeGeometry>& cones,
    int image_width,
    int image_height) {
    Eigen::Vector3d target(8.0, 0.0, 0.0);
    if (!cones.empty()) {
        target.setZero();
        for (const auto& cone : cones) {
            target += cone.center;
        }
        target /= static_cast<double>(cones.size());
        target.z() = std::max(0.0, target.z());
    } else if (!points.empty()) {
        target.setZero();
        const std::size_t sample_count =
            std::min<std::size_t>(points.size(), 5000);
        const std::size_t step = std::max<std::size_t>(
            1, points.size() / sample_count);
        std::size_t used = 0;
        for (std::size_t i = 0; i < points.size(); i += step) {
            target += points[i].position;
            ++used;
        }
        if (used > 0) {
            target /= static_cast<double>(used);
        }
    }

    double horizontal_extent = 12.0;
    for (const auto& cone : cones) {
        horizontal_extent = std::max(
            horizontal_extent,
            std::hypot(
                cone.center.x() - target.x(),
                cone.center.y() - target.y()) + 5.0);
    }

    const Eigen::Vector3d eye_direction =
        Eigen::Vector3d(-0.95, -0.75, 0.58).normalized();
    CameraModel camera;
    camera.eye = target + eye_direction * (horizontal_extent * 2.2);
    camera.forward = (target - camera.eye).normalized();
    const Eigen::Vector3d world_up(0.0, 0.0, 1.0);
    camera.right = camera.forward.cross(world_up).normalized();
    camera.up = camera.right.cross(camera.forward).normalized();

    double min_u = std::numeric_limits<double>::infinity();
    double max_u = -std::numeric_limits<double>::infinity();
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    auto include_point = [&](const Eigen::Vector3d& point) {
        const Eigen::Vector3d relative = point - camera.eye;
        const double depth = camera.forward.dot(relative);
        if (depth <= 0.05) {
            return;
        }
        const double normalized_u = camera.right.dot(relative) / depth;
        const double normalized_v = -camera.up.dot(relative) / depth;
        min_u = std::min(min_u, normalized_u);
        max_u = std::max(max_u, normalized_u);
        min_v = std::min(min_v, normalized_v);
        max_v = std::max(max_v, normalized_v);
    };

    for (const auto& point : points) {
        include_point(point.position);
    }
    for (const auto& cone : cones) {
        for (const auto& corner : box_corners(cone)) {
            include_point(corner);
        }
    }

    if (!std::isfinite(min_u) || !std::isfinite(max_u) ||
        !std::isfinite(min_v) || !std::isfinite(max_v)) {
        throw std::runtime_error("showcase camera has no visible geometry");
    }
    const double range_u = std::max(1e-6, max_u - min_u);
    const double range_v = std::max(1e-6, max_v - min_v);
    const double margin = 0.90;
    camera.focal = margin * std::min(
        static_cast<double>(image_width) / range_u,
        static_cast<double>(image_height) / range_v);
    camera.offset_x =
        (static_cast<double>(image_width) -
         camera.focal * (min_u + max_u)) /
        2.0;
    camera.offset_y =
        (static_cast<double>(image_height) -
         camera.focal * (min_v + max_v)) /
        2.0;
    return camera;
}

ProjectedCorner project_point(
    const Eigen::Vector3d& point,
    const CameraModel& camera,
    int image_width,
    int image_height) {
    const Eigen::Vector3d relative = point - camera.eye;
    ProjectedCorner result;
    result.depth = camera.forward.dot(relative);
    if (result.depth <= 0.05) {
        return result;
    }
    const double normalized_u = camera.right.dot(relative) / result.depth;
    const double normalized_v = -camera.up.dot(relative) / result.depth;
    result.pixel.x = cvRound(camera.offset_x + camera.focal * normalized_u);
    result.pixel.y = cvRound(camera.offset_y + camera.focal * normalized_v);
    result.valid = result.pixel.x >= 0 && result.pixel.x < image_width &&
                   result.pixel.y >= 0 && result.pixel.y < image_height;
    return result;
}

}  // namespace

bool render_showcase_image(
    const sensor_msgs::msg::PointCloud2& cloud,
    const lidar_cone_detector::msg::ThreeDConeArray& lidar_cones,
    const std::vector<uint8_t>& final_colors,
    const ShowcaseRenderOptions& options,
    const std::filesystem::path& output_path,
    ShowcaseRenderStats& stats,
    std::string& error_message) {
    try {
        if (options.image_width < 320 || options.image_height < 240) {
            throw std::runtime_error("showcase image dimensions are too small");
        }
        if (cloud.width == 0 || cloud.height == 0 || cloud.data.empty()) {
            throw std::runtime_error("showcase point cloud is empty");
        }

        std::vector<ConeGeometry> cones;
        cones.reserve(lidar_cones.cones.size());
        for (std::size_t i = 0; i < lidar_cones.cones.size(); ++i) {
            const auto& source = lidar_cones.cones[i];
            ConeGeometry cone;
            cone.center = Eigen::Vector3d(
                source.center.x, source.center.y, source.center.z);
            cone.half_size = Eigen::Vector3d(
                std::max(0.03, source.size.x / 2.0),
                std::max(0.03, source.size.y / 2.0),
                std::max(0.03, source.size.z / 2.0));
            cone.yaw = source.yaw;
            cone.color = i < final_colors.size()
                ? final_colors[i]
                : drd25_msgs::msg::Cone::UNKNOWN;
            cones.push_back(cone);
        }

        std::vector<PointSample> points;
        const std::size_t raw_point_count =
            static_cast<std::size_t>(cloud.width) * cloud.height;
        stats.input_points = raw_point_count;
        const std::size_t stride = std::max<std::size_t>(1, options.point_stride);
        points.reserve(raw_point_count / stride + 1);

        sensor_msgs::PointCloud2ConstIterator<float> iterator_x(cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iterator_y(cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iterator_z(cloud, "z");
        std::size_t point_index = 0;
        for (; iterator_x != iterator_x.end();
             ++iterator_x, ++iterator_y, ++iterator_z, ++point_index) {
            const double x = *iterator_x;
            const double y = *iterator_y;
            const double z = *iterator_z;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                x < options.forward_min || x > options.forward_max ||
                std::abs(y) > options.lateral_limit ||
                z < options.height_min || z > options.height_max) {
                continue;
            }

            PointSample sample;
            sample.position = Eigen::Vector3d(x, y, z);
            for (std::size_t cone_index = 0;
                 cone_index < cones.size(); ++cone_index) {
                if (point_inside_box(sample.position, cones[cone_index])) {
                    sample.cone_index = static_cast<int>(cone_index);
                    break;
                }
            }
            if (sample.cone_index < 0 && point_index % stride != 0) {
                continue;
            }
            points.push_back(sample);
        }
        if (points.empty()) {
            throw std::runtime_error(
                "no finite points remain inside the showcase view range");
        }

        const CameraModel camera = make_camera(
            points, cones, options.image_width, options.image_height);
        for (auto& point : points) {
            const ProjectedCorner projected = project_point(
                point.position,
                camera,
                options.image_width,
                options.image_height);
            point.pixel = projected.pixel;
            point.depth = projected.depth;
        }
        std::stable_sort(
            points.begin(),
            points.end(),
            [](const PointSample& lhs, const PointSample& rhs) {
                return lhs.depth > rhs.depth;
            });

        cv::Mat image = cv::Mat::zeros(
            options.image_height, options.image_width, CV_8UC3);

        for (const auto& point : points) {
            if (point.cone_index >= 0 || point.depth <= 0.05 ||
                point.pixel.x < 0 || point.pixel.x >= image.cols ||
                point.pixel.y < 0 || point.pixel.y >= image.rows) {
                continue;
            }
            const double fade = std::clamp(
                1.0 - point.depth / 80.0, 0.25, 1.0);
            const uint8_t gray = static_cast<uint8_t>(45.0 + 75.0 * fade);
            image.at<cv::Vec3b>(point.pixel) = cv::Vec3b(gray, gray, gray);
            ++stats.rendered_points;
        }

        cv::Mat box_overlay = image.clone();
        std::vector<std::array<ProjectedCorner, 8>> projected_boxes;
        projected_boxes.reserve(cones.size());
        for (const auto& cone : cones) {
            std::array<ProjectedCorner, 8> projected;
            const auto corners = box_corners(cone);
            bool all_valid = true;
            for (std::size_t i = 0; i < corners.size(); ++i) {
                projected[i] = project_point(
                    corners[i], camera, image.cols, image.rows);
                all_valid = all_valid && projected[i].valid;
            }
            projected_boxes.push_back(projected);
            if (!all_valid) {
                continue;
            }
            const cv::Scalar color = cone_color(cone.color);
            for (const auto& face : kBoxFaces) {
                std::vector<cv::Point> polygon;
                polygon.reserve(face.size());
                for (const int corner_index : face) {
                    polygon.push_back(projected[corner_index].pixel);
                }
                cv::fillConvexPoly(
                    box_overlay, polygon, color, cv::LINE_AA);
            }
            ++stats.rendered_cones;
        }
        cv::addWeighted(box_overlay, 0.18, image, 0.82, 0.0, image);

        cv::Mat glow_overlay = image.clone();
        for (const auto& point : points) {
            if (point.cone_index < 0 || point.depth <= 0.05 ||
                point.pixel.x < 0 || point.pixel.x >= image.cols ||
                point.pixel.y < 0 || point.pixel.y >= image.rows) {
                continue;
            }
            const auto cone_index = static_cast<std::size_t>(point.cone_index);
            const cv::Scalar color = cone_color(cones[cone_index].color);
            cv::circle(
                glow_overlay, point.pixel, 4, color, cv::FILLED, cv::LINE_AA);
        }
        cv::addWeighted(glow_overlay, 0.16, image, 0.84, 0.0, image);
        for (const auto& point : points) {
            if (point.cone_index < 0 || point.depth <= 0.05 ||
                point.pixel.x < 0 || point.pixel.x >= image.cols ||
                point.pixel.y < 0 || point.pixel.y >= image.rows) {
                continue;
            }
            const auto cone_index = static_cast<std::size_t>(point.cone_index);
            const cv::Scalar color = cone_color(cones[cone_index].color);
            cv::circle(image, point.pixel, 2, color, cv::FILLED, cv::LINE_AA);
            ++stats.colored_points;
            ++stats.rendered_points;
        }

        for (std::size_t cone_index = 0;
             cone_index < cones.size(); ++cone_index) {
            const auto& projected = projected_boxes[cone_index];
            if (!std::all_of(
                    projected.begin(), projected.end(),
                    [](const ProjectedCorner& corner) {
                        return corner.valid;
                    })) {
                continue;
            }
            const cv::Scalar color = cone_color(cones[cone_index].color);
            for (const auto& edge : kBoxEdges) {
                cv::line(
                    image,
                    projected[edge[0]].pixel,
                    projected[edge[1]].pixel,
                    color,
                    2,
                    cv::LINE_AA);
            }
        }

        const std::filesystem::path parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        const std::vector<int> png_parameters{
            cv::IMWRITE_PNG_COMPRESSION, 3};
        if (!cv::imwrite(output_path.string(), image, png_parameters)) {
            throw std::runtime_error("OpenCV failed to write showcase PNG");
        }
        return true;
    } catch (const std::exception& error) {
        error_message = error.what();
        return false;
    }
}

}  // namespace fs_fusion_box
