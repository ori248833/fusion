#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "cone_interfaces/msg/cone.hpp"
#include "cone_interfaces/msg/cone_array.hpp"
#include "drd25_msgs/msg/cone.hpp"
#include "drd25_msgs/msg/map.hpp"
#include "fs_fusion_box/fs_fusion_box_math.hpp"
#include "fs_fusion_box/visualizer.hpp"
#include "lidar_cone_detector/msg/three_d_cone.hpp"
#include "lidar_cone_detector/msg/three_d_cone_array.hpp"

using namespace std::chrono_literals;

namespace fs_fusion_box {

namespace {

double stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<double>(stamp.sec) +
           static_cast<double>(stamp.nanosec) * 1e-9;
}

template<typename T>
void atomic_update_max(std::atomic<T>& target, T value) {
    T current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(
               current, value, std::memory_order_relaxed)) {
    }
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double raw_index = fraction * static_cast<double>(values.size() - 1);
    const size_t index = static_cast<size_t>(std::ceil(raw_index));
    return values[std::min(index, values.size() - 1)];
}

bool is_unknown_map_color(uint8_t color) {
    return color == drd25_msgs::msg::Cone::UNKNOWN ||
           color == drd25_msgs::msg::Cone::UNKNOWN_BIG ||
           color == drd25_msgs::msg::Cone::UNKNOWN_SMALL;
}

bool is_confirmed_map_color(uint8_t color) {
    return color == drd25_msgs::msg::Cone::BLUE ||
           color == drd25_msgs::msg::Cone::RED ||
           color == drd25_msgs::msg::Cone::YELLOW_BIG ||
           color == drd25_msgs::msg::Cone::YELLOW_SMALL;
}

uint8_t unknown_color_from_lidar_label(int32_t label) {
    if (label == 1) {
        return drd25_msgs::msg::Cone::UNKNOWN_SMALL;
    }
    if (label == 2) {
        return drd25_msgs::msg::Cone::UNKNOWN_BIG;
    }
    return drd25_msgs::msg::Cone::UNKNOWN;
}

}  // namespace

class FusionNode : public rclcpp::Node {
public:
    using LidarMsg = lidar_cone_detector::msg::ThreeDConeArray;
    using CameraMsg = cone_interfaces::msg::ConeArray;

    FusionNode()
        : Node("fusion_box_node"),
          start_wall_time_(std::chrono::steady_clock::now()),
          last_health_wall_time_(start_wall_time_) {
        declare_parameters();
        load_runtime_parameters();

        params_ = load_params();
        const std::string frame_id = get_parameter("lidar_frame").as_string();
        visualizer_ = std::make_shared<Visualizer>(this, frame_id);

        // Keep the existing SLAM-facing topic, message type and queue depth.
        map_pub_ = create_publisher<drd25_msgs::msg::Map>(
            "/perception/fusion/map", 5);

        lidar_sub_ = create_subscription<LidarMsg>(
            "/detected_cones_bbox",
            10,
            std::bind(&FusionNode::lidar_callback, this, std::placeholders::_1));

        camera_sub_ = create_subscription<CameraMsg>(
            "/perception/camera/cones_custom",
            // The YOLO publisher is latest-only BEST_EFFORT.  The subscription
            // must request an equal-or-lower reliability level or DDS will not
            // connect the two endpoints.
            rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&FusionNode::camera_callback, this, std::placeholders::_1));

        parameter_callback_handle_ = add_on_set_parameters_callback(
            std::bind(
                &FusionNode::on_parameters_changed,
                this,
                std::placeholders::_1));

        initialize_csv_logging();
        fusion_thread_ = std::thread(&FusionNode::fusion_loop, this);
        visualization_thread_ = std::thread(&FusionNode::visualization_loop, this);

        const double health_interval =
            std::max(0.5, get_parameter("health_log_interval").as_double());
        health_timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::duration<double>(health_interval)),
            std::bind(&FusionNode::log_health, this));

        RCLCPP_INFO(
            get_logger(),
            "Fusion node started: LiDAR-driven output, asynchronous camera color updates");
    }

    ~FusionNode() override {
        stop();
        print_final_summary();
    }

private:
    struct Track {
        uint64_t id{0};
        double x{0.0};
        double y{0.0};
        double last_lidar_stamp{0.0};
        uint8_t color{drd25_msgs::msg::Cone::UNKNOWN};
        uint8_t pending_color{drd25_msgs::msg::Cone::UNKNOWN};
        unsigned int pending_color_count{0};
        double last_color_stamp{0.0};
    };

    struct LidarHistoryFrame {
        double stamp{0.0};
        LidarMsg::ConstSharedPtr msg;
        std::vector<uint64_t> track_ids;
    };

    struct FusionTask {
        LidarHistoryFrame lidar_frame;
        CameraMsg::ConstSharedPtr camera_msg;
        double camera_age_sec{0.0};
        double sync_diff_sec{0.0};
    };

    struct VisualizationData {
        LidarMsg::ConstSharedPtr lidar_msg;
        CameraMsg::ConstSharedPtr camera_msg;
        std::vector<ProjectedBox> projected_boxes;
        std::vector<FusionMatch> matches;
        std::vector<uint64_t> track_ids;
        std::vector<uint8_t> final_colors;
        std::vector<std::string> decisions;
    };

    struct TrackCandidate {
        double distance_squared{0.0};
        size_t cone_index{0};
        uint64_t track_id{0};
    };

    struct ColorUpdateSnapshot {
        std::vector<uint8_t> colors;
        std::vector<std::string> decisions;
    };

    struct CsvConeRow {
        size_t cone_index{0};
        double x{0.0};
        double y{0.0};
        double z{0.0};
        uint8_t color{drd25_msgs::msg::Cone::UNKNOWN};
        std::string decision;
        std::optional<double> iou;
        std::optional<double> confidence;
        std::string camera_color;
    };

    struct CsvFrame {
        size_t frame_index{0};
        builtin_interfaces::msg::Time lidar_stamp;
        builtin_interfaces::msg::Time camera_stamp;
        double sync_diff_ms{0.0};
        size_t lidar_count{0};
        size_t camera_count{0};
        size_t matched_count{0};
        size_t unmatched_camera_count{0};
        std::vector<CsvConeRow> cones;
    };

    void declare_parameters() {
        // Existing externally configured parameters are preserved.
        declare_parameter("sync_window", 0.04);
        declare_parameter("lidar_frame", "hesai_lidar");
        declare_parameter("overlap_threshold", 0.60);
        declare_parameter("image_width", 640);
        declare_parameter("image_height", 480);
        declare_parameter("camera_matrix.fx", 500.0);
        declare_parameter("camera_matrix.fy", 500.0);
        declare_parameter("camera_matrix.cx", 320.0);
        declare_parameter("camera_matrix.cy", 240.0);
        declare_parameter(
            "dist_coeffs", std::vector<double>({0, 0, 0, 0, 0}));
        declare_parameter("lidar_to_camera_matrix", std::vector<double>());
        declare_parameter("lidar_window_size", 10);
        declare_parameter("camera_window_size", 10);
        declare_parameter("max_sync_diff", 0.05);

        // New internal scheduling/tracking parameters. They do not alter the
        // SLAM-facing message or topic contract.
        declare_parameter("lidar_history_duration", 0.60);
        declare_parameter("enable_camera_age_gate", true);
        declare_parameter("max_camera_result_age", 0.50);
        declare_parameter("camera_age_warn_threshold", 0.40);
        declare_parameter("track_match_distance", 1.50);
        declare_parameter("track_timeout", 1.00);
        declare_parameter("color_conflict_confirmations", 2);
        declare_parameter("enable_visualization", false);
        declare_parameter("visualization_every_n", 1);
        declare_parameter("health_log_interval", 5.0);

        // CSV debug output is disabled by default and written asynchronously.
        declare_parameter("enable_visual_csv", false);
        declare_parameter(
            "visual_csv_output_directory", "~/.ros/fusion_debug");
        declare_parameter("visual_csv_every_n", 1);
        declare_parameter("visual_csv_max_frames", 1000);
        declare_parameter("visual_csv_include_unknown", true);
        declare_parameter("visual_csv_queue_size", 1024);
    }

    void load_runtime_parameters() {
        overlap_threshold_ = get_parameter("overlap_threshold").as_double();
        max_sync_diff_ = get_parameter("max_sync_diff").as_double();
        lidar_history_duration_ =
            get_parameter("lidar_history_duration").as_double();
        enable_camera_age_gate_ =
            get_parameter("enable_camera_age_gate").as_bool();
        max_camera_result_age_ =
            get_parameter("max_camera_result_age").as_double();
        camera_age_warn_threshold_ =
            get_parameter("camera_age_warn_threshold").as_double();
        track_match_distance_ =
            get_parameter("track_match_distance").as_double();
        track_timeout_ = get_parameter("track_timeout").as_double();
        color_conflict_confirmations_ = static_cast<unsigned int>(
            std::max<int64_t>(
                1, get_parameter("color_conflict_confirmations").as_int()));
        enable_visualization_.store(
            get_parameter("enable_visualization").as_bool());
        visualization_every_n_.store(static_cast<size_t>(
            std::max<int64_t>(1, get_parameter("visualization_every_n").as_int())));

        enable_visual_csv_ = get_parameter("enable_visual_csv").as_bool();
        visual_csv_output_directory_ =
            get_parameter("visual_csv_output_directory").as_string();
        visual_csv_every_n_ = static_cast<size_t>(std::max<int64_t>(
            1, get_parameter("visual_csv_every_n").as_int()));
        visual_csv_max_frames_ = static_cast<size_t>(std::max<int64_t>(
            0, get_parameter("visual_csv_max_frames").as_int()));
        visual_csv_include_unknown_ =
            get_parameter("visual_csv_include_unknown").as_bool();
        visual_csv_queue_size_ = static_cast<size_t>(std::max<int64_t>(
            1, get_parameter("visual_csv_queue_size").as_int()));
    }

    rcl_interfaces::msg::SetParametersResult on_parameters_changed(
        const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto& parameter : parameters) {
            if (parameter.get_name() == "enable_visualization") {
                enable_visualization_.store(parameter.as_bool());
                if (!parameter.as_bool()) {
                    std::lock_guard<std::mutex> lock(visualization_mutex_);
                    latest_visualization_.reset();
                }
                visualization_cv_.notify_all();
            } else if (parameter.get_name() == "visualization_every_n") {
                const int64_t value = parameter.as_int();
                if (value < 1) {
                    result.successful = false;
                    result.reason = "visualization_every_n must be >= 1";
                    return result;
                }
                visualization_every_n_.store(static_cast<size_t>(value));
            }
        }
        return result;
    }

    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) {
            return;
        }

        fusion_cv_.notify_all();
        visualization_cv_.notify_all();

        if (fusion_thread_.joinable()) {
            fusion_thread_.join();
        }
        if (visualization_thread_.joinable()) {
            visualization_thread_.join();
        }

        csv_running_.store(false, std::memory_order_relaxed);
        csv_cv_.notify_all();
        if (csv_thread_.joinable()) {
            csv_thread_.join();
        }
    }

    void lidar_callback(const LidarMsg::ConstSharedPtr& msg) {
        lidar_received_.fetch_add(1, std::memory_order_relaxed);
        const double lidar_stamp = stamp_to_sec(msg->header.stamp);

        std::vector<drd25_msgs::msg::Cone> output_cones;
        std::vector<uint64_t> track_ids;
        output_cones.reserve(msg->cones.size());

        size_t colored_count = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            prune_tracks_locked(lidar_stamp);
            track_ids = associate_tracks_locked(msg, lidar_stamp);

            for (size_t i = 0; i < msg->cones.size(); ++i) {
                const auto& lidar_cone = msg->cones[i];
                drd25_msgs::msg::Cone output;
                output.x = lidar_cone.center.x;
                output.y = lidar_cone.center.y;

                const auto track_it = tracks_.find(track_ids[i]);
                const uint8_t lidar_unknown_color =
                    unknown_color_from_lidar_label(lidar_cone.label);
                output.color = track_it != tracks_.end() &&
                        is_confirmed_map_color(track_it->second.color)
                    ? track_it->second.color
                    : lidar_unknown_color;
                if (is_confirmed_map_color(output.color)) {
                    ++colored_count;
                }
                output_cones.push_back(output);
            }

            lidar_history_.push_back(
                LidarHistoryFrame{lidar_stamp, msg, track_ids});
            prune_lidar_history_locked(lidar_stamp);
        }

        // LiDAR is the authoritative, real-time output trigger. No camera work,
        // projection, visualization, or worker queue can block this publish.
        drd25_msgs::msg::Map map_msg;
        map_msg.header = msg->header;
        map_msg.track = std::move(output_cones);
        map_pub_->publish(map_msg);

        lidar_published_.fetch_add(1, std::memory_order_relaxed);
        lidar_cones_published_.fetch_add(
            msg->cones.size(), std::memory_order_relaxed);
        colored_cones_published_.fetch_add(
            colored_count, std::memory_order_relaxed);
    }

    void camera_callback(const CameraMsg::ConstSharedPtr& msg) {
        camera_received_.fetch_add(1, std::memory_order_relaxed);

        const double camera_stamp = stamp_to_sec(msg->header.stamp);
        double camera_age = 0.0;
        if (enable_camera_age_gate_) {
            const double now_sec =
                static_cast<double>(get_clock()->now().nanoseconds()) * 1e-9;
            camera_age = now_sec - camera_stamp;
            if (camera_age > max_camera_result_age_) {
                camera_expired_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (camera_age > camera_age_warn_threshold_) {
                camera_high_latency_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        std::optional<LidarHistoryFrame> best_lidar;
        double min_diff = std::numeric_limits<double>::max();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (const auto& frame : lidar_history_) {
                const double diff = std::abs(frame.stamp - camera_stamp);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_lidar = frame;
                }
            }
        }

        if (!best_lidar || min_diff > max_sync_diff_) {
            camera_without_lidar_history_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (enable_camera_age_gate_) {
            record_timing_sample(camera_age_samples_ms_, camera_age * 1000.0);
        }
        record_timing_sample(sync_diff_samples_ms_, min_diff * 1000.0);

        FusionTask task{
            std::move(*best_lidar), msg, camera_age, min_diff};
        {
            std::lock_guard<std::mutex> lock(fusion_mutex_);
            if (!fusion_queue_.empty()) {
                camera_tasks_superseded_.fetch_add(
                    fusion_queue_.size(), std::memory_order_relaxed);
                fusion_queue_.clear();
            }
            fusion_queue_.push_back(std::move(task));
            atomic_update_max(max_fusion_queue_depth_, fusion_queue_.size());
        }
        camera_accepted_.fetch_add(1, std::memory_order_relaxed);
        fusion_cv_.notify_one();
    }

    std::vector<uint64_t> associate_tracks_locked(
        const LidarMsg::ConstSharedPtr& msg,
        double lidar_stamp) {
        std::vector<uint64_t> assignments(msg->cones.size(), 0);
        std::vector<TrackCandidate> candidates;
        const double max_distance_squared =
            track_match_distance_ * track_match_distance_;

        for (size_t cone_index = 0; cone_index < msg->cones.size(); ++cone_index) {
            const auto& cone = msg->cones[cone_index];
            for (const auto& entry : tracks_) {
                const Track& track = entry.second;
                const double dx = cone.center.x - track.x;
                const double dy = cone.center.y - track.y;
                const double distance_squared = dx * dx + dy * dy;
                if (distance_squared <= max_distance_squared) {
                    candidates.push_back(
                        TrackCandidate{distance_squared, cone_index, track.id});
                }
            }
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const TrackCandidate& lhs, const TrackCandidate& rhs) {
                return lhs.distance_squared < rhs.distance_squared;
            });

        std::unordered_map<uint64_t, bool> used_tracks;
        for (const auto& candidate : candidates) {
            if (assignments[candidate.cone_index] != 0 ||
                used_tracks[candidate.track_id]) {
                continue;
            }
            assignments[candidate.cone_index] = candidate.track_id;
            used_tracks[candidate.track_id] = true;
        }

        for (size_t i = 0; i < msg->cones.size(); ++i) {
            if (assignments[i] == 0) {
                const uint64_t id = next_track_id_++;
                Track track;
                track.id = id;
                track.x = msg->cones[i].center.x;
                track.y = msg->cones[i].center.y;
                track.last_lidar_stamp = lidar_stamp;
                tracks_.emplace(id, track);
                assignments[i] = id;
            } else {
                Track& track = tracks_.at(assignments[i]);
                track.x = msg->cones[i].center.x;
                track.y = msg->cones[i].center.y;
                track.last_lidar_stamp = lidar_stamp;
            }
        }
        return assignments;
    }

    void prune_tracks_locked(double current_stamp) {
        for (auto it = tracks_.begin(); it != tracks_.end();) {
            const double age = current_stamp - it->second.last_lidar_stamp;
            if (age > track_timeout_) {
                it = tracks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void prune_lidar_history_locked(double current_stamp) {
        while (!lidar_history_.empty()) {
            const double age = current_stamp - lidar_history_.front().stamp;
            if (age <= lidar_history_duration_) {
                break;
            }
            lidar_history_.pop_front();
        }
    }

    void fusion_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            FusionTask task;
            {
                std::unique_lock<std::mutex> lock(fusion_mutex_);
                fusion_cv_.wait(lock, [&] {
                    return !fusion_queue_.empty() ||
                           !running_.load(std::memory_order_relaxed);
                });
                if (!running_.load(std::memory_order_relaxed)) {
                    break;
                }
                task = std::move(fusion_queue_.back());
                fusion_queue_.clear();
            }

            const auto processing_start = std::chrono::steady_clock::now();
            auto standard_lidar = convert_lidar(task.lidar_frame.msg);
            auto standard_camera = convert_camera(task.camera_msg);
            auto projected_boxes = project_3d_boxes_to_2d(standard_lidar, params_);

            // Camera colors were explicitly converted to drd25 Map colors in
            // convert_camera(). The math function remains LiDAR-authoritative.
            auto fusion_result = fuse_measurements(
                projected_boxes,
                standard_lidar,
                standard_camera,
                overlap_threshold_);

            ColorUpdateSnapshot color_snapshot = update_track_colors(
                task.lidar_frame,
                fusion_result.fused_cones,
                stamp_to_sec(task.camera_msg->header.stamp));

            fusion_processed_.fetch_add(1, std::memory_order_relaxed);
            const size_t matched_count = fusion_result.matches.size();
            camera_detections_.fetch_add(
                standard_camera->detections.size(), std::memory_order_relaxed);
            matched_camera_detections_.fetch_add(
                matched_count, std::memory_order_relaxed);

            const auto processing_end = std::chrono::steady_clock::now();
            const double processing_ms =
                std::chrono::duration<double, std::milli>(
                    processing_end - processing_start)
                    .count();
            record_timing_sample(fusion_time_samples_ms_, processing_ms);

            maybe_queue_csv_frame(
                task, fusion_result, color_snapshot);
            maybe_queue_visualization(
                task,
                std::move(projected_boxes),
                std::move(fusion_result.matches),
                std::move(color_snapshot.colors),
                std::move(color_snapshot.decisions));
        }
    }

    ColorUpdateSnapshot update_track_colors(
        const LidarHistoryFrame& frame,
        const std::vector<drd25_msgs::msg::Cone>& fused_cones,
        double camera_stamp) {
        const size_t count = std::min(frame.track_ids.size(), fused_cones.size());
        ColorUpdateSnapshot snapshot;
        snapshot.colors.resize(
            fused_cones.size(), drd25_msgs::msg::Cone::UNKNOWN);
        snapshot.decisions.resize(fused_cones.size(), "雷达框未匹配");
        std::lock_guard<std::mutex> lock(state_mutex_);

        for (size_t i = 0; i < fused_cones.size(); ++i) {
            const uint8_t new_color = fused_cones[i].color;
            if (i >= count) {
                snapshot.colors[i] = new_color;
                if (is_confirmed_map_color(new_color)) {
                    snapshot.decisions[i] = "匹配成功，跟踪信息缺失";
                }
                continue;
            }

            auto track_it = tracks_.find(frame.track_ids[i]);
            if (track_it == tracks_.end()) {
                snapshot.colors[i] = new_color;
                if (is_confirmed_map_color(new_color)) {
                    snapshot.decisions[i] = "匹配成功，跟踪已失效";
                }
                continue;
            }
            Track& track = track_it->second;

            // UNKNOWN_SMALL and UNKNOWN_BIG carry LiDAR size only.  They must
            // not overwrite the color memory, but should be emitted when no
            // camera-confirmed track color is available.
            if (!is_confirmed_map_color(new_color)) {
                snapshot.colors[i] = is_confirmed_map_color(track.color)
                    ? track.color
                    : new_color;
                if (is_confirmed_map_color(track.color)) {
                    snapshot.decisions[i] = "沿用历史颜色";
                }
                continue;
            }

            if (!is_confirmed_map_color(track.color)) {
                track.color = new_color;
                track.pending_color = drd25_msgs::msg::Cone::UNKNOWN;
                track.pending_color_count = 0;
                track.last_color_stamp = camera_stamp;
                color_updates_.fetch_add(1, std::memory_order_relaxed);
                snapshot.colors[i] = track.color;
                snapshot.decisions[i] = "本帧匹配成功";
                continue;
            }

            if (track.color == new_color) {
                track.pending_color = drd25_msgs::msg::Cone::UNKNOWN;
                track.pending_color_count = 0;
                track.last_color_stamp = camera_stamp;
                color_updates_.fetch_add(1, std::memory_order_relaxed);
                snapshot.colors[i] = track.color;
                snapshot.decisions[i] = "本帧颜色确认";
                continue;
            }

            color_conflicts_.fetch_add(1, std::memory_order_relaxed);
            if (track.pending_color == new_color) {
                ++track.pending_color_count;
            } else {
                track.pending_color = new_color;
                track.pending_color_count = 1;
            }

            if (track.pending_color_count >= color_conflict_confirmations_) {
                track.color = new_color;
                track.pending_color = drd25_msgs::msg::Cone::UNKNOWN;
                track.pending_color_count = 0;
                track.last_color_stamp = camera_stamp;
                color_updates_.fetch_add(1, std::memory_order_relaxed);
                snapshot.decisions[i] = "颜色冲突，已修改";
            } else {
                snapshot.decisions[i] = "颜色冲突，保持原色";
            }
            snapshot.colors[i] = track.color;
        }
        return snapshot;
    }

    static std::string map_color_name(uint8_t color) {
        switch (color) {
            case drd25_msgs::msg::Cone::BLUE:
                return "蓝色";
            case drd25_msgs::msg::Cone::RED:
                return "红色";
            case drd25_msgs::msg::Cone::YELLOW_BIG:
                return "大黄锥";
            case drd25_msgs::msg::Cone::YELLOW_SMALL:
                return "小黄锥";
            case drd25_msgs::msg::Cone::UNKNOWN_BIG:
                return "未知颜色-大锥";
            case drd25_msgs::msg::Cone::UNKNOWN_SMALL:
                return "未知颜色-小锥";
            case drd25_msgs::msg::Cone::UNKNOWN:
                return "未知";
            default:
                return "无效颜色值";
        }
    }

    static std::string camera_color_name(uint8_t color) {
        switch (color) {
            case cone_interfaces::msg::Cone::BLUE:
                return "蓝色";
            case cone_interfaces::msg::Cone::RED:
                return "红色";
            case cone_interfaces::msg::Cone::YELLOW_BIG:
                return "大黄锥";
            case cone_interfaces::msg::Cone::YELLOW_SMALL:
                return "小黄锥";
            default:
                return "未知";
        }
    }

    static std::filesystem::path expand_user_path(const std::string& value) {
        if (value == "~" || value.rfind("~/", 0) == 0 ||
            value.rfind("~\\", 0) == 0) {
            const char* home = std::getenv("HOME");
            if (home == nullptr || *home == '\0') {
                home = std::getenv("USERPROFILE");
            }
            if (home != nullptr && *home != '\0') {
                if (value.size() == 1) {
                    return std::filesystem::path(home);
                }
                return std::filesystem::path(home) / value.substr(2);
            }
        }
        return std::filesystem::path(value);
    }

    static void write_utf8_bom(std::ofstream& stream) {
        constexpr char bom[] = {'\xEF', '\xBB', '\xBF'};
        stream.write(bom, sizeof(bom));
    }

    void initialize_csv_logging() {
        if (!enable_visual_csv_) {
            return;
        }

        try {
            const std::filesystem::path base_directory =
                expand_user_path(visual_csv_output_directory_);
            std::filesystem::create_directories(base_directory);

            const auto now = std::chrono::system_clock::now();
            const std::time_t time_value =
                std::chrono::system_clock::to_time_t(now);
            const std::tm* local_time_pointer = std::localtime(&time_value);
            if (local_time_pointer == nullptr) {
                throw std::runtime_error("无法生成 CSV 运行目录时间");
            }
            const std::tm local_time = *local_time_pointer;
            std::ostringstream run_name;
            run_name << "fusion_debug_"
                     << std::put_time(&local_time, "%Y%m%d_%H%M%S");

            csv_run_directory_ = base_directory / run_name.str();
            size_t suffix = 1;
            while (std::filesystem::exists(csv_run_directory_)) {
                csv_run_directory_ =
                    base_directory /
                    (run_name.str() + "_" + std::to_string(suffix++));
            }
            std::filesystem::create_directories(csv_run_directory_);

            csv_runtime_enabled_ = true;
            csv_running_.store(true, std::memory_order_relaxed);
            csv_thread_ = std::thread(&FusionNode::csv_writer_loop, this);
            RCLCPP_INFO(
                get_logger(),
                "CSV debug enabled: %s",
                csv_run_directory_.string().c_str());
        } catch (const std::exception& error) {
            csv_runtime_enabled_ = false;
            csv_running_.store(false, std::memory_order_relaxed);
            RCLCPP_ERROR(
                get_logger(),
                "CSV debug disabled because initialization failed: %s",
                error.what());
        }
    }

    void maybe_queue_csv_frame(
        const FusionTask& task,
        const FusionResult& fusion_result,
        const ColorUpdateSnapshot& color_snapshot) {
        if (!csv_runtime_enabled_ || fusion_result.matches.empty()) {
            return;
        }

        const size_t candidate_index = ++csv_successful_candidate_count_;
        if ((candidate_index - 1) % visual_csv_every_n_ != 0) {
            return;
        }
        if (visual_csv_max_frames_ > 0 &&
            csv_frames_enqueued_ >= visual_csv_max_frames_) {
            if (!csv_limit_reported_) {
                csv_limit_reported_ = true;
                RCLCPP_INFO(
                    get_logger(),
                    "CSV debug reached visual_csv_max_frames=%zu; "
                    "fusion continues without additional CSV frames",
                    visual_csv_max_frames_);
            }
            csv_runtime_enabled_ = false;
            return;
        }

        CsvFrame frame;
        frame.lidar_stamp = task.lidar_frame.msg->header.stamp;
        frame.camera_stamp = task.camera_msg->header.stamp;
        frame.sync_diff_ms = task.sync_diff_sec * 1000.0;
        frame.lidar_count = task.lidar_frame.msg->cones.size();
        frame.camera_count = task.camera_msg->cones.size();
        frame.matched_count = fusion_result.matches.size();
        frame.unmatched_camera_count =
            fusion_result.unmatched_camera_indices.size();
        frame.cones.reserve(frame.lidar_count);

        std::unordered_map<size_t, const FusionMatch*> matches_by_lidar;
        matches_by_lidar.reserve(fusion_result.matches.size());
        for (const auto& match : fusion_result.matches) {
            matches_by_lidar.emplace(match.lidar_index, &match);
        }

        for (size_t i = 0; i < task.lidar_frame.msg->cones.size(); ++i) {
            const auto& cone = task.lidar_frame.msg->cones[i];
            CsvConeRow row;
            row.cone_index = i;
            row.x = cone.center.x;
            row.y = cone.center.y;
            row.z = cone.center.z;
            if (i < color_snapshot.colors.size()) {
                row.color = color_snapshot.colors[i];
                row.decision = color_snapshot.decisions[i];
            } else {
                row.color = drd25_msgs::msg::Cone::UNKNOWN;
                row.decision = "雷达框未匹配";
            }

            const auto match_it = matches_by_lidar.find(i);
            if (match_it != matches_by_lidar.end()) {
                const FusionMatch& match = *match_it->second;
                row.iou = match.iou;
                if (match.camera_index < task.camera_msg->cones.size()) {
                    const auto& camera_cone =
                        task.camera_msg->cones[match.camera_index];
                    row.confidence = camera_cone.confidence;
                    row.camera_color = camera_color_name(camera_cone.color);
                }
            }
            frame.cones.push_back(std::move(row));
        }

        {
            std::lock_guard<std::mutex> lock(csv_mutex_);
            if (csv_queue_.size() >= visual_csv_queue_size_) {
                const size_t dropped =
                    csv_dropped_frames_.fetch_add(
                        1, std::memory_order_relaxed) + 1;
                if (dropped == 1 || dropped % 100 == 0) {
                    RCLCPP_WARN(
                        get_logger(),
                        "CSV queue full; dropped debug frames=%zu",
                        dropped);
                }
                return;
            }
            frame.frame_index = ++csv_frames_enqueued_;
            csv_queue_.push_back(std::move(frame));
        }
        csv_cv_.notify_one();
    }

    void csv_writer_loop() {
        while (true) {
            CsvFrame frame;
            {
                std::unique_lock<std::mutex> lock(csv_mutex_);
                csv_cv_.wait(lock, [&] {
                    return !csv_queue_.empty() ||
                           !csv_running_.load(std::memory_order_relaxed);
                });
                if (csv_queue_.empty() &&
                    !csv_running_.load(std::memory_order_relaxed)) {
                    break;
                }
                frame = std::move(csv_queue_.front());
                csv_queue_.pop_front();
            }

            try {
                write_csv_frame(frame);
                csv_frames_written_.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception& error) {
                csv_write_errors_.fetch_add(1, std::memory_order_relaxed);
                RCLCPP_ERROR(
                    get_logger(),
                    "Failed to write CSV frame %zu: %s",
                    frame.frame_index,
                    error.what());
            }
        }
    }

    void write_csv_frame(const CsvFrame& frame) {
        std::ostringstream folder_name;
        folder_name << "frame_" << std::setw(6) << std::setfill('0')
                    << frame.frame_index;
        const std::filesystem::path frame_directory =
            csv_run_directory_ / folder_name.str();
        std::filesystem::create_directories(frame_directory);

        std::ofstream cones_csv(
            frame_directory / "cones.csv",
            std::ios::binary | std::ios::out | std::ios::trunc);
        if (!cones_csv.is_open()) {
            throw std::runtime_error("无法创建逐帧 cones.csv");
        }
        write_utf8_bom(cones_csv);
        cones_csv
            << "帧号,锥桶编号,X坐标(米),Y坐标(米),Z坐标(米),"
               "最终颜色,判断,实际IoU,YOLO颜色,YOLO置信度\n";
        cones_csv << std::fixed << std::setprecision(3);
        for (const auto& cone : frame.cones) {
            if (!visual_csv_include_unknown_ &&
                is_unknown_map_color(cone.color)) {
                continue;
            }
            cones_csv << frame.frame_index << ',' << cone.cone_index + 1
                      << ',' << cone.x << ',' << cone.y << ',' << cone.z
                      << ',' << map_color_name(cone.color) << ','
                      << cone.decision << ',';
            if (cone.iou) {
                cones_csv << *cone.iou;
            }
            cones_csv << ',' << cone.camera_color << ',';
            if (cone.confidence) {
                cones_csv << *cone.confidence;
            }
            cones_csv << '\n';
        }
        cones_csv.flush();
        if (!cones_csv.good()) {
            throw std::runtime_error("写入 cones.csv 失败");
        }

        size_t blue_count = 0;
        size_t red_count = 0;
        size_t yellow_big_count = 0;
        size_t yellow_small_count = 0;
        size_t unknown_count = 0;
        for (const auto& cone : frame.cones) {
            switch (cone.color) {
                case drd25_msgs::msg::Cone::BLUE:
                    ++blue_count;
                    break;
                case drd25_msgs::msg::Cone::RED:
                    ++red_count;
                    break;
                case drd25_msgs::msg::Cone::YELLOW_BIG:
                    ++yellow_big_count;
                    break;
                case drd25_msgs::msg::Cone::YELLOW_SMALL:
                    ++yellow_small_count;
                    break;
                default:
                    ++unknown_count;
                    break;
            }
        }

        std::ofstream fusion_csv(
            frame_directory / "fusion.csv",
            std::ios::binary | std::ios::out | std::ios::trunc);
        if (!fusion_csv.is_open()) {
            throw std::runtime_error("无法创建逐帧 fusion.csv");
        }
        write_utf8_bom(fusion_csv);
        fusion_csv
            << "帧号,雷达时间戳,相机时间戳,时间差毫秒,雷达锥桶数,"
               "YOLO框数,成功匹配数,YOLO未匹配数,蓝色数,红色数,"
               "大黄锥数,小黄锥数,未知数,IoU阈值\n";
        fusion_csv << frame.frame_index << ',' << std::fixed
                   << std::setprecision(9)
                   << stamp_to_sec(frame.lidar_stamp) << ','
                   << stamp_to_sec(frame.camera_stamp) << ','
                   << std::setprecision(3) << frame.sync_diff_ms << ','
                   << frame.lidar_count << ',' << frame.camera_count << ','
                   << frame.matched_count << ','
                   << frame.unmatched_camera_count << ',' << blue_count << ','
                   << red_count << ',' << yellow_big_count << ','
                   << yellow_small_count << ',' << unknown_count << ','
                   << overlap_threshold_ << '\n';
        fusion_csv.flush();
        if (!fusion_csv.good()) {
            throw std::runtime_error("写入 fusion.csv 失败");
        }
    }

    void maybe_queue_visualization(
        const FusionTask& task,
        std::vector<ProjectedBox> projected_boxes,
        std::vector<FusionMatch> matches,
        std::vector<uint8_t> final_colors,
        std::vector<std::string> decisions) {
        if (!enable_visualization_.load(std::memory_order_relaxed)) {
            return;
        }

        const size_t frame_number =
            visualization_candidate_count_.fetch_add(
                1, std::memory_order_relaxed) + 1;
        const size_t every_n =
            visualization_every_n_.load(std::memory_order_relaxed);
        if (frame_number % every_n != 0) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(visualization_mutex_);
            latest_visualization_ = VisualizationData{
                task.lidar_frame.msg,
                task.camera_msg,
                std::move(projected_boxes),
                std::move(matches),
                task.lidar_frame.track_ids,
                std::move(final_colors),
                std::move(decisions)};
        }
        visualization_cv_.notify_one();
    }

    void visualization_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            std::optional<VisualizationData> data;
            {
                std::unique_lock<std::mutex> lock(visualization_mutex_);
                visualization_cv_.wait(lock, [&] {
                    return latest_visualization_.has_value() ||
                           !running_.load(std::memory_order_relaxed);
                });
                if (!running_.load(std::memory_order_relaxed)) {
                    break;
                }
                data = std::move(latest_visualization_);
                latest_visualization_.reset();
            }

            if (!enable_visualization_.load(std::memory_order_relaxed) || !data) {
                continue;
            }

            // 3D visualization was intentionally removed. Only the existing
            // synthetic 2D view remains.
            visualizer_->publishSyntheticView(
                data->camera_msg,
                data->lidar_msg,
                params_,
                data->projected_boxes,
                data->matches,
                data->track_ids,
                data->final_colors,
                data->decisions);
        }
    }

    uint8_t map_camera_color(uint8_t camera_color) {
        switch (camera_color) {
            case cone_interfaces::msg::Cone::BLUE:
                return drd25_msgs::msg::Cone::BLUE;
            case cone_interfaces::msg::Cone::RED:
                return drd25_msgs::msg::Cone::RED;
            case cone_interfaces::msg::Cone::YELLOW_SMALL:
                return drd25_msgs::msg::Cone::YELLOW_SMALL;
            case cone_interfaces::msg::Cone::YELLOW_BIG:
                return drd25_msgs::msg::Cone::YELLOW_BIG;
            default:
                camera_unknown_colors_.fetch_add(1, std::memory_order_relaxed);
                return drd25_msgs::msg::Cone::UNKNOWN;
        }
    }

    vision_msgs::msg::Detection3DArray::SharedPtr convert_lidar(
        const LidarMsg::ConstSharedPtr& msg) {
        auto output = std::make_shared<vision_msgs::msg::Detection3DArray>();
        output->header = msg->header;
        output->detections.reserve(msg->cones.size());

        for (const auto& cone : msg->cones) {
            vision_msgs::msg::Detection3D detection;
            detection.bbox.center.position = cone.center;
            detection.bbox.size = cone.size;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            const Eigen::Quaterniond quaternion(
                Eigen::AngleAxisd(cone.yaw, Eigen::Vector3d::UnitZ()));
            hypothesis.pose.pose.orientation.w = quaternion.w();
            hypothesis.pose.pose.orientation.x = quaternion.x();
            hypothesis.pose.pose.orientation.y = quaternion.y();
            hypothesis.pose.pose.orientation.z = quaternion.z();

            // LiDAR label semantics are intentionally unchanged:
            // label 1 = small cone, label 2 = big cone.
            hypothesis.hypothesis.class_id = std::to_string(cone.label);
            detection.results.push_back(hypothesis);
            output->detections.push_back(detection);
        }
        return output;
    }

    vision_msgs::msg::Detection2DArray::SharedPtr convert_camera(
        const CameraMsg::ConstSharedPtr& msg) {
        auto output = std::make_shared<vision_msgs::msg::Detection2DArray>();
        output->header = msg->header;
        output->detections.reserve(msg->cones.size());

        for (const auto& cone : msg->cones) {
            vision_msgs::msg::Detection2D detection;
            detection.bbox.center.position.x = cone.center.x;
            detection.bbox.center.position.y = cone.center.y;
            detection.bbox.size_x = cone.size.x;
            detection.bbox.size_y = cone.size.y;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.score = cone.confidence;
            const uint8_t mapped_color = map_camera_color(cone.color);
            hypothesis.hypothesis.class_id = std::to_string(mapped_color);
            detection.results.push_back(hypothesis);
            output->detections.push_back(detection);
        }
        return output;
    }

    CalibrationParams load_params() {
        CalibrationParams params;
        const std::vector<double> matrix =
            get_parameter("lidar_to_camera_matrix").as_double_array();
        if (matrix.size() == 16) {
            params.T_l2c = Eigen::Map<
                const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(
                matrix.data());
        } else {
            RCLCPP_WARN(
                get_logger(),
                "lidar_to_camera_matrix has %zu values; expected 16",
                matrix.size());
        }

        params.img_w = get_parameter("image_width").as_int();
        params.img_h = get_parameter("image_height").as_int();
        params.K = cv::Mat::eye(3, 3, CV_64F);
        params.K.at<double>(0, 0) = get_parameter("camera_matrix.fx").as_double();
        params.K.at<double>(1, 1) = get_parameter("camera_matrix.fy").as_double();
        params.K.at<double>(0, 2) = get_parameter("camera_matrix.cx").as_double();
        params.K.at<double>(1, 2) = get_parameter("camera_matrix.cy").as_double();
        params.D = cv::Mat(
            get_parameter("dist_coeffs").as_double_array()).clone();
        return params;
    }

    void record_timing_sample(
        std::deque<double>& samples,
        double value) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        samples.push_back(value);
        constexpr size_t kMaxSamples = 2048;
        if (samples.size() > kMaxSamples) {
            samples.pop_front();
        }
    }

    struct TimingSnapshot {
        double age_p50{0.0};
        double age_p95{0.0};
        double age_max{0.0};
        double sync_p95{0.0};
        double fusion_p95{0.0};
    };

    TimingSnapshot timing_snapshot() {
        std::vector<double> age;
        std::vector<double> sync;
        std::vector<double> fusion;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            age.assign(
                camera_age_samples_ms_.begin(), camera_age_samples_ms_.end());
            sync.assign(
                sync_diff_samples_ms_.begin(), sync_diff_samples_ms_.end());
            fusion.assign(
                fusion_time_samples_ms_.begin(), fusion_time_samples_ms_.end());
        }

        TimingSnapshot result;
        result.age_p50 = percentile(age, 0.50);
        result.age_p95 = percentile(age, 0.95);
        result.age_max = percentile(age, 1.00);
        result.sync_p95 = percentile(sync, 0.95);
        result.fusion_p95 = percentile(fusion, 0.95);
        return result;
    }

    void log_health() {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - last_health_wall_time_).count();
        if (elapsed <= 0.0) {
            return;
        }

        const size_t lidar_received = lidar_received_.load();
        const size_t lidar_published = lidar_published_.load();
        const size_t camera_received = camera_received_.load();

        const double lidar_receive_hz =
            static_cast<double>(lidar_received - last_lidar_received_) / elapsed;
        const double lidar_publish_hz =
            static_cast<double>(lidar_published - last_lidar_published_) / elapsed;
        const double camera_receive_hz =
            static_cast<double>(camera_received - last_camera_received_) / elapsed;

        last_lidar_received_ = lidar_received;
        last_lidar_published_ = lidar_published;
        last_camera_received_ = camera_received;
        last_health_wall_time_ = now;

        const size_t total_cones = lidar_cones_published_.load();
        const size_t colored_cones = colored_cones_published_.load();
        const double color_ratio = total_cones > 0
            ? 100.0 * static_cast<double>(colored_cones) /
                  static_cast<double>(total_cones)
            : 0.0;
        const TimingSnapshot timing = timing_snapshot();

        size_t track_count = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            track_count = tracks_.size();
        }

        RCLCPP_INFO(
            get_logger(),
            "Health | lidar rx/pub %.1f/%.1f Hz (%zu/%zu), camera %.1f Hz "
            "accepted=%zu expired=%zu no_history=%zu | age p50/p95/max "
            "%.1f/%.1f/%.1f ms sync_p95=%.1f ms fusion_p95=%.1f ms | "
            "tracks=%zu colored=%.1f%% updates=%zu conflicts=%zu",
            lidar_receive_hz,
            lidar_publish_hz,
            lidar_received,
            lidar_published,
            camera_receive_hz,
            camera_accepted_.load(),
            camera_expired_.load(),
            camera_without_lidar_history_.load(),
            timing.age_p50,
            timing.age_p95,
            timing.age_max,
            timing.sync_p95,
            timing.fusion_p95,
            track_count,
            color_ratio,
            color_updates_.load(),
            color_conflicts_.load());

        if (lidar_received != lidar_published) {
            RCLCPP_WARN(
                get_logger(),
                "LiDAR receive/publish mismatch: received=%zu published=%zu",
                lidar_received,
                lidar_published);
        }
        if (camera_unknown_colors_.load() > 0) {
            RCLCPP_WARN(
                get_logger(),
                "%zu camera detections mapped to UNKNOWN; verify message enums",
                camera_unknown_colors_.load());
        }
    }

    void print_final_summary() {
        const double runtime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_wall_time_).count();
        const size_t total_cones = lidar_cones_published_.load();
        const size_t colored_cones = colored_cones_published_.load();
        const double color_ratio = total_cones > 0
            ? 100.0 * static_cast<double>(colored_cones) /
                  static_cast<double>(total_cones)
            : 0.0;
        const TimingSnapshot timing = timing_snapshot();

        std::cout << "\n======================================================\n"
                  << " Fusion Node final health summary\n"
                  << "======================================================\n"
                  << std::fixed << std::setprecision(2)
                  << "Runtime:                    " << runtime << " s\n"
                  << "LiDAR received/published:   " << lidar_received_.load()
                  << " / " << lidar_published_.load() << "\n"
                  << "LiDAR internal loss:        "
                  << (lidar_received_.load() - lidar_published_.load()) << "\n"
                  << "Camera received/accepted:   " << camera_received_.load()
                  << " / " << camera_accepted_.load() << "\n"
                  << "Camera expired/no history:  " << camera_expired_.load()
                  << " / " << camera_without_lidar_history_.load() << "\n"
                  << "Camera high latency:        "
                  << camera_high_latency_.load() << "\n"
                  << "Camera tasks superseded:    "
                  << camera_tasks_superseded_.load() << "\n"
                  << "Camera age p50/p95/max:      " << timing.age_p50 << " / "
                  << timing.age_p95 << " / " << timing.age_max << " ms\n"
                  << "Sync difference p95:        " << timing.sync_p95 << " ms\n"
                  << "Fusion processing p95:      " << timing.fusion_p95 << " ms\n"
                  << "Max fusion queue depth:      "
                  << max_fusion_queue_depth_.load() << "\n"
                  << "Colored cone ratio:         " << color_ratio << " %\n"
                  << "Color updates/conflicts:    " << color_updates_.load()
                  << " / " << color_conflicts_.load() << "\n"
                  << "Unknown color mappings:     "
                  << camera_unknown_colors_.load() << "\n"
                  << "CSV frames written/dropped: "
                  << csv_frames_written_.load() << " / "
                  << csv_dropped_frames_.load() << "\n"
                  << "CSV write errors:           "
                  << csv_write_errors_.load() << "\n"
                  << "======================================================\n"
                  << std::flush;
    }

    CalibrationParams params_;
    double overlap_threshold_{0.60};
    double max_sync_diff_{0.05};
    double lidar_history_duration_{0.60};
    bool enable_camera_age_gate_{true};
    double max_camera_result_age_{0.50};
    double camera_age_warn_threshold_{0.40};
    double track_match_distance_{1.50};
    double track_timeout_{1.00};
    unsigned int color_conflict_confirmations_{2};

    rclcpp::Publisher<drd25_msgs::msg::Map>::SharedPtr map_pub_;
    rclcpp::Subscription<LidarMsg>::SharedPtr lidar_sub_;
    rclcpp::Subscription<CameraMsg>::SharedPtr camera_sub_;
    rclcpp::TimerBase::SharedPtr health_timer_;
    OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
    std::shared_ptr<Visualizer> visualizer_;

    std::mutex state_mutex_;
    std::unordered_map<uint64_t, Track> tracks_;
    std::deque<LidarHistoryFrame> lidar_history_;
    uint64_t next_track_id_{1};

    std::atomic<bool> running_{true};
    std::thread fusion_thread_;
    std::mutex fusion_mutex_;
    std::condition_variable fusion_cv_;
    std::deque<FusionTask> fusion_queue_;

    std::thread visualization_thread_;
    std::mutex visualization_mutex_;
    std::condition_variable visualization_cv_;
    std::optional<VisualizationData> latest_visualization_;
    std::atomic<bool> enable_visualization_{false};
    std::atomic<size_t> visualization_every_n_{1};
    std::atomic<size_t> visualization_candidate_count_{0};

    std::mutex stats_mutex_;
    std::deque<double> camera_age_samples_ms_;
    std::deque<double> sync_diff_samples_ms_;
    std::deque<double> fusion_time_samples_ms_;

    std::atomic<size_t> lidar_received_{0};
    std::atomic<size_t> lidar_published_{0};
    std::atomic<size_t> lidar_cones_published_{0};
    std::atomic<size_t> colored_cones_published_{0};
    std::atomic<size_t> camera_received_{0};
    std::atomic<size_t> camera_accepted_{0};
    std::atomic<size_t> camera_expired_{0};
    std::atomic<size_t> camera_high_latency_{0};
    std::atomic<size_t> camera_without_lidar_history_{0};
    std::atomic<size_t> camera_tasks_superseded_{0};
    std::atomic<size_t> camera_unknown_colors_{0};
    std::atomic<size_t> fusion_processed_{0};
    std::atomic<size_t> camera_detections_{0};
    std::atomic<size_t> matched_camera_detections_{0};
    std::atomic<size_t> color_updates_{0};
    std::atomic<size_t> color_conflicts_{0};
    std::atomic<size_t> max_fusion_queue_depth_{0};

    bool enable_visual_csv_{false};
    std::string visual_csv_output_directory_{"~/.ros/fusion_debug"};
    size_t visual_csv_every_n_{1};
    size_t visual_csv_max_frames_{1000};
    bool visual_csv_include_unknown_{true};
    size_t visual_csv_queue_size_{1024};
    bool csv_runtime_enabled_{false};
    bool csv_limit_reported_{false};
    size_t csv_successful_candidate_count_{0};
    size_t csv_frames_enqueued_{0};
    std::filesystem::path csv_run_directory_;
    std::atomic<bool> csv_running_{false};
    std::thread csv_thread_;
    std::mutex csv_mutex_;
    std::condition_variable csv_cv_;
    std::deque<CsvFrame> csv_queue_;
    std::atomic<size_t> csv_frames_written_{0};
    std::atomic<size_t> csv_dropped_frames_{0};
    std::atomic<size_t> csv_write_errors_{0};

    const std::chrono::steady_clock::time_point start_wall_time_;
    std::chrono::steady_clock::time_point last_health_wall_time_;
    size_t last_lidar_received_{0};
    size_t last_lidar_published_{0};
    size_t last_camera_received_{0};
};

}  // namespace fs_fusion_box

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<fs_fusion_box::FusionNode>();
    rclcpp::spin(node);
    node.reset();
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}
