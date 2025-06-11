#pragma once

#include <opencv2/opencv.hpp>

#include "base_algorithm.hpp"
#include "common/calibration_data.hpp"
#include "common/slam_logger.hpp"
#include "map_fusion/map_fusion.hpp"

namespace map_fusion_component {

class MapFusionComponent : public Algorithm {
   public:
    MapFusionComponent(const std::string &config_file, const std::string &integration_config_file,
                       CalibrationData _calibration_data,
                       std::function<void(const std::shared_ptr<common::Data> &)> _callback, std::string _function_name)
        : Algorithm(config_file, integration_config_file, _calibration_data, _callback, _function_name) {
        SLAM_LOG_HIGHLIGHT() << "----------- This is the beginning of Map Fusion Component -----------";

        // construct MapFusion in LoadConfiguration()
        LoadConfiguration();

        is_running_.store(true);
        worker_thread_ = std::thread(std::bind(&MapFusionComponent::ThreadFunction, this));
    }

    ~MapFusionComponent() {
        SLAM_LOG_HIGHLIGHT() << "This is the end of Map Fusion Component ";
        is_running_.store(false);
        stop_thread_ = true;
        data_condition_.notify_one();
        buf_cv_.notify_one();
        worker_thread_.join();
    }

    void AddData(const std::shared_ptr<common::Data> &data) override;

    void Evaluate() override;

   public:
    std::atomic<bool> is_running_;

   private:
    std::unique_ptr<map_fusion::MapFusion> mapper_;

    std::mutex buf_mutex_;            // 队列缓存互斥锁
    std::condition_variable buf_cv_;  // 队列缓存条件变量

    std::string grid_map_topic_;
    std::string package_data_topic_;

   private:
    void LoadConfiguration() override;

    void ProcessPackageData(const std::shared_ptr<common::PackageData> &_data);

    std::condition_variable_any data_condition_;
};

}  // namespace map_fusion_component