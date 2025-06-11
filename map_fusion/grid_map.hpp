#pragma once

#include "map_fusion_common.hpp"

using namespace common;

namespace map_fusion {

class GridMap {
   public:
    GridMap() = default;

    virtual ~GridMap() = default;
    std::unordered_map<int64_t, double> occ_map;
    std::unordered_map<int64_t, std::unordered_map<int, int>> label_map;
    std::unordered_map<int64_t, double> height_map;
    int min_i = 0;
    int min_j = 0;
    int max_i = 0;
    int max_j = 0;

    // 更新占用栅格
    void updateOcc(int i, int j, double p_meas);
    // 更新标签
    void updateLabel(int i, int j, const std::unordered_map<int, int>& vote_map);
    void updateLabel(int i, int j, int label);
    // 更新高度
    void updateHeight(int i, int j, double height);

    // 转eigen map
    Eigen::MatrixXi labelEigenMap() const;
    Eigen::MatrixXd heightEigenMap() const;
    Eigen::MatrixXd occupancyEigenMap() const;
    int submapWidth() const { return (max_j - min_j + 1); }

    const std::unordered_map<int64_t, double> &getOccMap() const { return occ_map; }

    void updateBounds(int i, int j) {
        min_i = std::min(min_i, i);
        max_i = std::max(max_i, i);
        min_j = std::min(min_j, j);
        max_j = std::max(max_j, j);
    }

    void getBounds(int &mi, int &ma_i, int &mj, int &ma_j) const {
        mi = min_i;
        ma_i = max_i;
        mj = min_j;
        ma_j = max_j;
    }
};

// TODO,考虑在激活子图中保存所有local map，并在成为完整子图前可以优化他们的pose
class ActiveSubmap : public GridMap {
   public:
    int start_idx = 0;
    int fused_count = 0;
    double ts = 0.0;
    Eigen::Matrix4d first_pose;

    ActiveSubmap() = default;
    ~ActiveSubmap() override = default;
};

class FinishedSubmap : public GridMap {
   public:
    int submap_id = 0;
    Eigen::Matrix4d first_pose;
    double ts;

    FinishedSubmap() = default;
    FinishedSubmap(const ActiveSubmap &as, const int &_submap_id) {
        first_pose = as.first_pose;
        ts = as.ts;
        // 注意把active转成finish时，原本在as中的map将不复存在！
        occ_map = std::move(as.occ_map);
        label_map = std::move(as.label_map);
        height_map = std::move(as.height_map);
        as.getBounds(min_i, max_i, min_j, max_j);
        submap_id = _submap_id;
    }
    ~FinishedSubmap() override = default;

    // 只允许finished submap被保存或加载
    bool save(const std::string &dir_path) const;
    bool load(const std::string &full_path);
};

};  // namespace map_fusion