
#pragma once

#include "grid_map.hpp"
#include "local_map.hpp"
#include "map_fusion_common.hpp"

using namespace common;

namespace map_fusion {

class MapFusion {
   public:
    // Constructor takes intrinsic and extrinsic params
    MapFusion(const BEVParams &params);

    // Single-frame processing: builds local BEV category and height maps
    void processFrame(const PackageData &pkg);

    // Visualization: shows original, segmentation, BEV maps
    void visualize(const PackageData &pkg) const;

   private:
    BEVParams params_;

    // 局部地图（单帧）
    LocalMap local_map_;

    // 全局地图
    GridMap global_map_;

    int frame_count_ = 0;
    std::vector<ActiveSubmap> active_submaps_;
    std::vector<FinishedSubmap> finished_submaps_;

    // 深度图&分割图转点云
    void generatePointCloudWithLabels(const cv::Mat &depth, const cv::Mat &seg, std::vector<Eigen::Vector3d> &pts,
                                      std::vector<int> &labels);
    // local融合子图
    void fuseLocalToSubmap(ActiveSubmap &as, const LocalMap &local_map, const Eigen::Matrix4d &world_T_wheel);
    // submap融合全局图
    void fuseSubmapToGlobal(const FinishedSubmap &fs);
    // local融合全局图（目前不会使用）
    void fuseLocalToGlobal(const Eigen::MatrixXi &local_cat, const Eigen::MatrixXd &local_hgt,
                           const Eigen::MatrixXd &local_occ, const Eigen::Matrix4d &world_T_wheel);

    // 可视化
    cv::Mat visualizeHeightMap(const Eigen::MatrixXd &height_map, double min_h, double max_h) const;
    cv::Mat visualizeCategoryMap(const Eigen::MatrixXi &cat_map) const;
    cv::Mat visualizeOccupancyMap(const Eigen::MatrixXd &occ_map, double p_free, double p_occ) const;
    void showMaps(const Eigen::MatrixXi &cat_map, const Eigen::MatrixXd &hgt_map, const Eigen::MatrixXd &occ_map,
                  const std::string &name = "", const bool &save = false) const;
    void showMaps(const Eigen::MatrixXd &occ_map, const std::string &name = "", const bool &save = false) const;
};

};  // namespace map_fusion