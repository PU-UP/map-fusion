/**
****************************************************************************************

 * @CopyRight: 2020-2030, Positec Tech. CO.,LTD. All Rights Reserved.
 * @FilePath: local_map.hpp
 * @Author: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Date: 2025-06-09 17:58:30
 * @Version: 0.1
 * @LastEditTime: 2025-06-09 17:58:30
 * @LastEditors: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Description:

****************************************************************************************
*/

#pragma once

#include "map_fusion_common.hpp"

using namespace common;

namespace map_fusion {

class LocalMap {
   public:
    LocalMap() = default;

    LocalMap(double h, double w, double local_res, double p_free, double p_occ) {
        grid_h = h;
        grid_w = w;
        category_map.setZero(grid_h, grid_w);
        height_map.setZero(grid_h, grid_w);
        occupancy_map.setZero(grid_h, grid_w);
        local_resolution = local_res;
        l_free_update = logOdds(p_free);
        l_occ_update = logOdds(p_occ);
    }

    // Local per-frame BEV
    Eigen::MatrixXi category_map;   // each cell holds category label
    Eigen::MatrixXd height_map;     // max height per cell
    Eigen::MatrixXd occupancy_map;  // using log_odds

    Eigen::Matrix4d world_T_wheel;

    double local_resolution = 0.05;
    double l_free_update = 0.85;
    double l_occ_update = -0.85;
    int grid_w = 80;
    int grid_h = 40;
    double distance_threshold = 2;

    // local用不到pose，这里只是便于保存local_map对应的pose
    void projectLocalPointCloud(const std::vector<Eigen::Vector3d> &pts_wheel, const std::vector<int> &pts_label,
                                const Eigen::Matrix4d &_world_T_wheel);

    bool labelFiltered(int label, double distance);
    void updateLocalOcc(int i, int j, int label, double distance);
    int selectBestLabel(const std::unordered_map<int, int> &vote_count);

    bool saveFrame(const std::string &path) const;
    bool loadFrame(const std::string &path);
};

};  // namespace map_fusion