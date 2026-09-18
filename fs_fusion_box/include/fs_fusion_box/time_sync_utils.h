#ifndef TIME_SYNC_UTILS_H
#define TIME_SYNC_UTILS_H

#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace fs_fusion_box {
namespace time_sync {

/**
 * @brief 动态时间偏移计算器（滑动窗口+3σ异常值过滤）
 *        用于计算雷达-相机之间的固定时间偏移，自动过滤异常值
 */
class DynamicTimeOffsetCalculator {
public:
    /**
     * @brief 构造函数
     * @param window_size 滑动窗口大小（建议：30~100帧）
     * @param sigma_threshold 异常值过滤阈值（建议：2.0~3.0）
     * @param min_valid_samples 计算有效偏移所需的最小样本数（建议：10~20）
     * @param initial_offset 初始偏移值（单位：秒），未计算出有效偏移时使用
     */
    explicit DynamicTimeOffsetCalculator(
        size_t window_size = 50,
        double sigma_threshold = 3.0,
        size_t min_valid_samples = 15,
        double initial_offset = 0.75)
        : window_size_(window_size),
          sigma_threshold_(sigma_threshold),
          min_valid_samples_(min_valid_samples),
          initial_offset_(initial_offset),
          current_offset_(initial_offset) {}

    /**
     * @brief 添加一帧的原始时间差，并更新动态偏移
     * @param raw_time_diff 原始时间差 = 雷达时间 - 相机原始时间（单位：秒）
     */
    void add_sample(double raw_time_diff) {
        // 1. 将新样本加入窗口
        time_diff_queue_.push_back(raw_time_diff);
        
        // 2. 窗口满则剔除最旧样本
        if (time_diff_queue_.size() > window_size_) {
            time_diff_queue_.pop_front();
        }

        // 3. 样本数足够时，计算过滤后的有效偏移
        if (time_diff_queue_.size() >= min_valid_samples_) {
            current_offset_ = calculate_filtered_offset();
        }
        // 样本数不足时，保持初始偏移不变
    }

    /**
     * @brief 获取当前最优时间偏移值
     * @return 动态计算的时间偏移（单位：秒）
     */
    double get_current_offset() const {
        return current_offset_;
    }

    /**
     * @brief 重置计算器，清空所有历史数据
     */
    void reset() {
        time_diff_queue_.clear();
        current_offset_ = initial_offset_;
    }

    /**
     * @brief 获取当前窗口内的有效样本数
     */
    size_t get_sample_count() const {
        return time_diff_queue_.size();
    }

    /**
     * @brief 检查是否已经计算出有效偏移（样本数达到最小值）
     */
    bool is_valid() const {
        return time_diff_queue_.size() >= min_valid_samples_;
    }

private:
    /**
     * @brief 核心：计算过滤异常值后的平均时间偏移
     * @return 过滤后的均值（作为当前最优偏移）
     */
    double calculate_filtered_offset() const {
        // 1. 计算原始数据的均值和标准差
        double mean = std::accumulate(
            time_diff_queue_.begin(), time_diff_queue_.end(), 0.0
        ) / time_diff_queue_.size();

        double sum_sq = 0.0;
        for (double diff : time_diff_queue_) {
            sum_sq += (diff - mean) * (diff - mean);
        }
        double std_dev = std::sqrt(sum_sq / time_diff_queue_.size());

        // 2. 3σ异常值过滤
        std::vector<double> valid_samples;
        double lower_bound = mean - sigma_threshold_ * std_dev;
        double upper_bound = mean + sigma_threshold_ * std_dev;

        for (double diff : time_diff_queue_) {
            if (diff >= lower_bound && diff <= upper_bound) {
                valid_samples.push_back(diff);
            }
        }

        // 3. 过滤后样本数仍足够，计算新均值；否则返回原始均值
        if (valid_samples.size() >= min_valid_samples_) {
            return std::accumulate(
                valid_samples.begin(), valid_samples.end(), 0.0
            ) / valid_samples.size();
        } else {
            return mean;
        }
    }

    // 配置参数
    size_t window_size_;          // 滑动窗口大小
    double sigma_threshold_;      // 异常值过滤阈值（σ倍数）
    size_t min_valid_samples_;    // 最小有效样本数
    double initial_offset_;       // 初始偏移值

    // 运行时数据
    std::deque<double> time_diff_queue_;  // 历史时间差队列
    double current_offset_;               // 当前最优偏移值
};

} // namespace time_sync
} // namespace fs_fusion_box

#endif // TIME_SYNC_UTILS_H