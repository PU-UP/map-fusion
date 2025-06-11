#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <vector>

#include "common/data_type.hpp"
#include "common/slam_logger.hpp"

using namespace common;

namespace map_fusion {

inline int64_t encodeKey(int gi, int gj) { return ((int64_t(gi) << 32) | uint32_t(gj)); }

inline std::pair<int, int> decodeKey(int64_t key) {
    // 先右移 32 位，低 32 位自动丢弃，高 32 位保留 gi 的有符号位扩展
    int32_t gi = int32_t(key >> 32);
    // 再截取低 32 位，直接当作有符号 int32_t
    int32_t gj = int32_t(key & 0xFFFFFFFF);
    return {gi, gj};
}

// Helpers
inline double logOdds(double p) { return std::log(p / (1.0 - p)); }

inline double clampLogOdds(double l) {
    const double Lmax = 20.0;
    return std::max(-Lmax, std::min(Lmax, l));
}

inline double invLogOdds(double l) {
    // exp(l)/(1+exp(l))，数值稳定写法：
    return 1.0 / (1.0 + std::exp(-l));
}

enum Percep_Label2_ {
    UNKNOWN_OCC,
    OBSTACLE_AND_BACKGROUND,  // 背景或其他障碍物
    ADULT,                    // 成人
    CHILD,                    // 小孩
    PET,                      // 宠物
    HEDGEHOG,                 // 刺猬
    CHARGING_STATION,         // 充电站
    SKY,                      // 天空
    GREEN_PLANTS,             // 绿植
    GRASS,                    // 草地
    MANHOLE_COVER,            // 井盖
    HARD_ROAD,                // 硬质路面
    HARD_WALL,                // 硬质墙面
    STOOL,                    // 粪便
    MUD,                      // 泥土
    PIT,                      // 土坑
    SPRINKLER,                // 平式灌溉头
    HIGH_GRASS,               // 高草
    LEAVES,                   // 叶子
    PIPELINE,                 // 管线
    FRUIT,                    // 果实
    FENCE,                    // 一般栅栏
    STEPPING_STONE,           // 脚踏石
    BODY_PARTS,               // 人体部分
    LOOSE_SOIL,               // 松软泥土
    ARTIFICIAL_GRASS          // 人工草坪
};

static inline int colorToLabel(const cv::Vec3b &bgr) {
    if (bgr == cv::Vec3b(255, 191, 0)) return SKY;
    if (bgr == cv::Vec3b(0, 128, 0)) return GRASS;
    if (bgr == cv::Vec3b(0, 64, 0)) return ARTIFICIAL_GRASS;
    if (bgr == cv::Vec3b(255, 255, 153)) return HEDGEHOG;
    if (bgr == cv::Vec3b(0, 128, 128)) return CHARGING_STATION;
    if (bgr == cv::Vec3b(0, 0, 128)) return MANHOLE_COVER;
    if (bgr == cv::Vec3b(255, 255, 0)) return STOOL;
    if (bgr == cv::Vec3b(255, 128, 0)) return SPRINKLER;
    if (bgr == cv::Vec3b(128, 128, 0)) return LEAVES;
    if (bgr == cv::Vec3b(255, 69, 0)) return PIPELINE;
    if (bgr == cv::Vec3b(255, 204, 153)) return FRUIT;

    // if (bgr == cv::Vec3b(0, 0, 0))       return PIPELINE;  // 背景 - 黑色(Black)
    if (bgr == cv::Vec3b(255, 0, 0)) return ADULT;      // 成人 - 红色(Red)
    if (bgr == cv::Vec3b(255, 153, 153)) return ADULT;  // 小孩 - 淡红色(Light Red)
    if (bgr == cv::Vec3b(0, 0, 255)) return ADULT;      // 宠物 - 蓝色(Blue)
    // if (bgr == cv::Vec3b(255, 255, 153)) return PIPELINE;  // 刺猬 - 淡黄色(Light Yellow)
    // if (bgr == cv::Vec3b(0, 128, 128))   return CHARGING_STATION;  // 充电站 - 青色(Teal, 调整后)
    // if (bgr == cv::Vec3b(0, 191, 255))   return SKY;  // 天空 - 深天蓝色(Deep Sky Blue)
    // if (bgr == cv::Vec3b(153, 204, 153)) return PIPELINE;  // 绿植 - 淡绿色(Light Green)
    // if (bgr == cv::Vec3b(0, 128, 0))     return GRASS;  // 草 - 绿色(Green)
    // if (bgr == cv::Vec3b(0, 0, 128))     return PIPELINE;  // 井盖 - 深蓝色(Dark Blue)
    if (bgr == cv::Vec3b(192, 192, 192)) return HARD_ROAD;  // 硬质路面 - 银色(Silver)
    if (bgr == cv::Vec3b(178, 34, 34))   return HARD_WALL;  // 硬质墙面 - 火砖色(Firebrick)
    // if (bgr == cv::Vec3b(255, 255, 0))   return PIPELINE;  // 粪便 - 黄色(Yellow)
    if (bgr == cv::Vec3b(139, 69, 19)) return MUD;  // 泥土 - 马鞍棕色(Saddle Brown)
    if (bgr == cv::Vec3b(64, 64, 64)) return PIT;   // 土坑 - 深灰色(Dark Gray)
    // if (bgr == cv::Vec3b(255, 128, 0))   return PIPELINE;  // 平式喷灌头 - 橙色(Orange)
    // if (bgr == cv::Vec3b(0, 255, 0))     return PIPELINE;  // 高草 - 青绿色(Lime Green)
    // if (bgr == cv::Vec3b(128, 128, 0))   return LEAVES;  // 叶子 - 青色(Teal, 调整后)
    // if (bgr == cv::Vec3b(255, 204, 153)) return PIPELINE;  // 果实 - 橙色(Orange)
    // if (bgr == cv::Vec3b(204, 153, 204)) return PIPELINE;  // 网状栅栏 - 淡紫色(Light Purple, 调整后)
    // if (bgr == cv::Vec3b(128, 0, 128))   return PIPELINE;  // 一般栅栏 - 紫色(Purple, 调整后)
    // default to obstacle/background
    return OBSTACLE_AND_BACKGROUND;
}

static inline cv::Vec3b labelToColor(int label) {
    switch (label) {
        case SKY:
            return cv::Vec3b(255, 191, 0);
        case GRASS:
            return cv::Vec3b(0, 128, 0);
        case ARTIFICIAL_GRASS:
            return cv::Vec3b(0, 64, 0);
        case HEDGEHOG:
            return cv::Vec3b(255, 255, 153);
        case CHARGING_STATION:
            return cv::Vec3b(0, 128, 128);
        case MANHOLE_COVER:
            return cv::Vec3b(0, 0, 128);
        case STOOL:
            return cv::Vec3b(255, 255, 0);
        case SPRINKLER:
            return cv::Vec3b(255, 128, 0);
        case LEAVES:
            return cv::Vec3b(128, 128, 0);
        case PIPELINE:
            return cv::Vec3b(255, 69, 0);
        case FRUIT:
            return cv::Vec3b(255, 204, 153);
        case ADULT:
            return cv::Vec3b(255, 0, 0);
        case MUD:
            return cv::Vec3b(139, 69, 19);
        case PIT:
            return cv::Vec3b(64, 64, 64);
        case UNKNOWN_OCC:
            return cv::Vec3b(128, 128, 128);
        case HARD_ROAD:
            return cv::Vec3b(192, 192, 192);
        case HARD_WALL:
            return cv::Vec3b(178, 34, 34);
        default:
            return cv::Vec3b(0, 0, 0);
    }
}

// 尽量在map_fusion_component.cpp的LoadConfiguration中修改配置参数，节省编译时间
// TODO，读取配置文件
struct BEVParams {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // ------------- 相机参数 -------------
    Eigen::Matrix3d K;
    Eigen::Matrix4d wheel_T_cam0;
    // 双目间距
    double eye_distance = 0.06;
    // 深度图转换因子（与相机内参和双目间距有关）
    double disparity_depth_coef = 0;

    // ------------- Local Map参数 -------------
    // 局部地图下采样比例
    double local_res = 0.05;
    int downsample = 1;
    double max_depth = 2.0;
    // local map长宽（影响储存大小）
    int grid_w = 80;
    int grid_h = 40;
    // 栅格更新
    double p_occ_update = 0.7;
    double p_free_update = 0.4;

    // 障碍物高度：暂时未使用
    double obstacle_height = -0.5;

    // ------------- Submap参数 -----------------
    // 子图关键帧数量
    int submap_size = 200;  // 50
    double submap_res = 0.05;

    // ------------- global map参数 -----------------
    // 地图分辨率
    double global_res = 0.1;

    // ------------- 其他 -----------------
    // 用于可视化放大倍数
    int display_scale = 2;

    BEVParams(const Eigen::Matrix3d &_K, const Eigen::Matrix4d _wheel_T_cam0) : K(_K), wheel_T_cam0(_wheel_T_cam0) {
        disparity_depth_coef = eye_distance * K(0, 0) * 256;
    }
};

// move following code to common
#include <sys/resource.h>
#include <sys/time.h>

#include <chrono>
#include <iostream>
#include <string>

// ResourceProfiler: measures wall-clock time, CPU time, and peak memory usage
class ResourceProfiler {
   public:
    using clock = std::chrono::high_resolution_clock;

    // Constructor: records the start time and CPU usage
    ResourceProfiler() : start_wall_(clock::now()), start_cpu_(getCPUTime()) {}

    // Reset the start markers
    void reset() {
        start_wall_ = clock::now();
        start_cpu_ = getCPUTime();
    }

    // Elapsed wall-clock time in seconds
    double elapsedWallSec() const {
        auto end = clock::now();
        return std::chrono::duration<double>(end - start_wall_).count();
    }

    // Elapsed CPU time in seconds
    double elapsedCPUSec() const { return getCPUTime() - start_cpu_; }

    // Current peak memory usage (resident set size) in kilobytes
    long memoryUsageKB() const { return getMemoryUsage(); }

    // Print a report with an optional label
    void report(const std::string &label = "") const {
        SLAM_LOG_HIGHLIGHT() << (label.empty() ? "" : label + ": ") << "Wall Time = " << elapsedWallSec() << " s, "
                             << "CPU Time = " << elapsedCPUSec() << " s, "
                             << "Peak RSS = " << memoryUsageKB() << " KB";
    }

   private:
    clock::time_point start_wall_;
    double start_cpu_;

    // Helper: get cumulative CPU time (user + system) in seconds
    static double getCPUTime() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        double sys = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
        return user + sys;
    }

    // Helper: get peak resident set size (kilobytes)
    static long getMemoryUsage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss;
    }
};

// Example usage:
// int main() {
//     ResourceProfiler profiler;
//     // ... code to measure ...
//     profiler.report("MyTask");
//     return 0;
// }

};  // namespace map_fusion