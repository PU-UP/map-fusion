#include "map_fusion_component.hpp"

namespace map_fusion_component {

void MapFusionComponent::AddData(const std::shared_ptr<common::Data>& data) {
    if (auto package_data = std::dynamic_pointer_cast<common::PackageData>(data)) {
        ProcessPackageData(package_data);
    }
}

void MapFusionComponent::ProcessPackageData(const std::shared_ptr<common::PackageData>& _data) {
    if (_data->type != common::PackageType::IMAGE) return;
    SLAM_LOG_INFO() << "[ MAP FUSION PACKAGE ] : map fusion received image package data";
    std::unique_lock<std::mutex> buf_lock(buf_mutex_);
    if (queues_[package_data_topic_].size() > 100) {
        SLAM_LOG_ERROR() << "[ MAP FUSION PACKAGE ] : size > 100, clear all";
        queues_[package_data_topic_].clear();
        queues_[package_data_topic_].shrink_to_fit();
    }
    queues_[package_data_topic_].emplace_back(_data);
    buf_cv_.notify_all();
}

void MapFusionComponent::Evaluate() {
    std::shared_ptr<common::PackageData> package_data_msg = NULL;

    std::unique_lock<std::mutex> buf_lock(buf_mutex_);

    if (queues_[package_data_topic_].empty()) {
        buf_cv_.wait(buf_lock, [&] { return (!queues_[package_data_topic_].empty() || stop_thread_); });
    }

    if (stop_thread_) {
        return;
    }
    if (queues_[package_data_topic_].empty()) {
        return;
    }

    SLAM_LOG_INFO() << "[ MAP FUSION PACKAGE ] !!!!!!!!!  queues_[package_data_topic_] size: "
                    << queues_[package_data_topic_].size();

    package_data_msg = std::dynamic_pointer_cast<common::PackageData>(queues_[package_data_topic_].front());
    queues_[package_data_topic_].pop_front();

    buf_lock.unlock();

    if (package_data_msg == NULL || package_data_msg->type != common::PackageType::IMAGE) {
        return;
    }

    map_fusion::ResourceProfiler profiler;
    mapper_->processFrame(*package_data_msg);
    profiler.report("processFrame");
    profiler.reset();
    mapper_->visualize(*package_data_msg);
    profiler.report("visualize");
}

void MapFusionComponent::LoadConfiguration() {
    SLAM_LOG_INFO() << "integration_configure_path_ " << integration_configure_path_;

    YAML::Node config = YAML::LoadFile(integration_configure_path_);

    package_data_topic_ = config["package_data_topic"].as<std::string>();

    Eigen::Matrix3d tmp_K;
    tmp_K << calibration_data_.intrinsic_camera.projection_parameters(0), 0,
        calibration_data_.intrinsic_camera.projection_parameters(2), 0,
        calibration_data_.intrinsic_camera.projection_parameters(1),
        calibration_data_.intrinsic_camera.projection_parameters(3), 0, 0, 1;

    map_fusion::BEVParams bev_params = map_fusion::BEVParams(tmp_K, calibration_data_.extrinsic_wheel_T_cam0.transform);

    // you can adjust bev params here to save more compile time.
    // bev_params.local_res = 0.1;

    // ------------- Local Map参数 -------------
    // 局部地图下采样比例
    bev_params.local_res = 0.05;
    bev_params.downsample = 1;
    bev_params.max_depth = 2.0;
    // local map长宽（影响储存大小）
    bev_params.grid_w = 80;
    bev_params.grid_h = 40;
    // 栅格更新
    bev_params.p_occ_update = 0.8;
    bev_params.p_free_update = 0.43;

    // 障碍物高度：暂时未使用
    bev_params.obstacle_height = -0.5;

    // ------------- Submap参数 -----------------
    // 子图关键帧数量
    bev_params.submap_size = 200;  // 50
    bev_params.submap_res = 0.05;

    // ------------- global map参数 -----------------
    // 地图分辨率
    bev_params.global_res = 0.1;

    // ------------- 其他 -----------------
    // 用于可视化放大倍数
    bev_params.display_scale = 2;

    mapper_ = std::make_unique<map_fusion::MapFusion>(bev_params);
}

}  // namespace map_fusion_component
