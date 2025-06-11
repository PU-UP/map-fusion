#include "map_fusion.hpp"

using namespace map_fusion;

MapFusion::MapFusion(const BEVParams &params) : params_(params) {
    local_map_ =
        LocalMap(params_.grid_h, params_.grid_w, params_.local_res, params_.p_free_update, params_.p_occ_update);

    // test loading local map
    // for (int i = 0; i < 5000; i++) {
    //     SLAM_LOG_HIGHLIGHT() << "load " << i << " local map";
    //     std::string path = "/work/develop_gitlab/slam-core/outputs/local/" + std::to_string(i) + "_occ.bin";
    //     Eigen::Matrix4d pose;
    //     Eigen::MatrixXd occ_map;
    //     LocalMap tmp_lm(params_.grid_h, params_.grid_w, params_.local_res, params_.p_free_update,
    //     params_.p_occ_update); tmp_lm.loadFrame(path);
    // }

    // // test loading submap
    // for (int i = 0; i < 180; i++) {
    //     SLAM_LOG_HIGHLIGHT() << "Load " << i << " submap";
    //     std::string path = "/home/user-f295/Documents/result/Gridmap/0609/submap/submap_" + std::to_string(i) +
    //     ".bin";
    //     // std::string path = "/work/develop_gitlab/slam-core/outputs/submap/submap_" + std::to_string(i) + ".bin";
    //     FinishedSubmap fs;
    //     if (fs.load(path)) {
    //         // showMaps(fs.occupancyEigenMap(), "tmp", true);
    //         // cv::waitKey(100);
    //         SLAM_LOG_HIGHLIGHT() << "Load " << i << ", ts: " << std::fixed << std::setprecision(3) << fs.ts;
    //         finished_submaps_.push_back(std::move(fs));
    //     } else {
    //         SLAM_LOG_ERROR() << "Load " << i << " submap failed";
    //     }
    // }
    // SLAM_LOG_HIGHLIGHT() << "Total " << finished_submaps_.size() << " loaded";

    // ResourceProfiler profiler;
    // for (int i = 0; i < finished_submaps_.size(); ++i) {
    //     auto fs = finished_submaps_[i];
    //     fuseSubmapToGlobal(fs);
    // }
    // profiler.report("fuse to global");
    // showMaps(global_map_.occupancyEigenMap(), "global");
    // cv::waitKey(0);
}

void MapFusion::generatePointCloudWithLabels(const cv::Mat &depth, const cv::Mat &seg,
                                             std::vector<Eigen::Vector3d> &pts, std::vector<int> &labels) {
    const double fx = params_.K(0, 0), fy = params_.K(1, 1);
    const double cx = params_.K(0, 2), cy = params_.K(1, 2);
    // 清空输出
    pts.clear();
    labels.clear();

    int rows = depth.rows;
    int cols = depth.cols;
    int ds = params_.downsample;

    for (int v = 0; v < rows; v += ds) {
        for (int u = 0; u < cols; u += ds) {
            // 1) 读取原始视差并换算深度
            uint16_t d_raw = depth.at<uint16_t>(v, u);
            if (d_raw <= 0) continue;  // 防止除零或无效值
            float d = static_cast<float>(params_.disparity_depth_coef / d_raw);
            if (d <= 0 || d > params_.max_depth) continue;

            // 2) 反投影到相机坐标系
            double z = d;
            double x = (u - cx) * z / fx;
            double y = (v - cy) * z / fy;
            pts.emplace_back(x, y, z);

            // 3) 从分割图取颜色→转换为类别
            cv::Vec3b bgr = seg.at<cv::Vec3b>(v, u);
            cv::Vec3b rgb{bgr[2], bgr[1], bgr[0]};
            int label = colorToLabel(rgb);
            labels.push_back(label);
        }
    }
}

void MapFusion::processFrame(const PackageData &pkg) {
    int idx = frame_count_;
    frame_count_++;

    std::vector<Eigen::Vector3d> pts_cam;
    std::vector<int> pts_label;
    generatePointCloudWithLabels(pkg.disparity_data->image, pkg.segmentation_data->image, pts_cam, pts_label);

    std::vector<Eigen::Vector3d> pts_wheel;
    pts_wheel.reserve(pts_cam.size());
    for (const auto &p : pts_cam) {
        Eigen::Vector4d ph(p.x(), p.y(), p.z(), 1.0);
        auto pw = params_.wheel_T_cam0 * ph;
        pts_wheel.emplace_back(pw.x(), pw.y(), pw.z());
    }

    Eigen::Matrix4d world_T_wheel = Eigen::Matrix4d::Identity();
    world_T_wheel.block<3, 3>(0, 0) = pkg.odom_data->rotation.toRotationMatrix();
    world_T_wheel.block<3, 1>(0, 3) = pkg.odom_data->position;

    // Build local map（local用不到pose，这里只是便于保存local_map对应的pose）
    local_map_.projectLocalPointCloud(pts_wheel, pts_label, world_T_wheel);

    // test saving local map
    // std::string path = "/work/develop_gitlab/slam-core/outputs/local/" + std::to_string(frame_count_);
    // local_map_.saveFrame(path);

    // 构建子图
    int half_submap_size = params_.submap_size / 2;

    // 每有一个子图构建到一半时会新建一个子图
    if (idx % half_submap_size == 0) {
        ActiveSubmap as;
        as.start_idx = idx;
        as.first_pose = world_T_wheel;
        as.occ_map.clear();
        as.fused_count = 0;
        as.ts = pkg.time;
        active_submaps_.push_back(std::move(as));
    }

    // 对每个正在构造的子图，把本帧数据融合进去
    std::vector<int> to_remove;
    for (int a = 0; a < int(active_submaps_.size()); ++a) {
        ActiveSubmap &as = active_submaps_[a];
        int start = as.start_idx;
        int offset = idx - start;

        // 只有当 offset ∈ [0,49] 时，才是该子图的“第 offset 帧”
        if (offset < 0 || offset >= params_.submap_size) continue;

        // 把本帧 local_occ 融入 as
        fuseLocalToSubmap(as, local_map_, world_T_wheel);

        // 如果 as 融合满 50 帧，就把它“结算”为 FinishedSubmap”
        if (as.fused_count >= params_.submap_size) {
            // 构造完成子图
            FinishedSubmap fs(as, finished_submaps_.size());

            // test saving submap
            fs.save("/work/develop_gitlab/slam-core/outputs/submap/");
            // fs.save("/home/user-f295/Documents/result/Gridmap/0609/submap/");

            // 通过子图融合全局图
            fuseSubmapToGlobal(fs);

            finished_submaps_.push_back(std::move(fs));
            to_remove.push_back(a);
        }
    }

    // 从 active_submaps_ 中移除已完成的子图
    for (int k = int(to_remove.size()) - 1; k >= 0; --k) {
        active_submaps_.erase(active_submaps_.begin() + to_remove[k]);
    }

    SLAM_LOG_HIGHLIGHT() << "[SUBMAP] active submap: " << active_submaps_.size()
                         << " finished submap: " << finished_submaps_.size();
}

void MapFusion::fuseLocalToGlobal(const Eigen::MatrixXi &local_cat, const Eigen::MatrixXd &local_hgt,
                                  const Eigen::MatrixXd &local_occ, const Eigen::Matrix4d &world_T_wheel) {
    int H = local_cat.rows();
    int W = local_cat.cols();
    double res = params_.global_res;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int label = local_cat(i, j);
            // 如果这个栅格在本地图中是空的（label=0 可视为背景），这里可以选择跳过
            if (label == Percep_Label2_::SKY) continue;

            double h = local_hgt(i, j);
            double p_meas = local_occ(i, j);

            // 1) 计算这个栅格中心在 local 坐标系下的 (x,y,0)
            double x_local = i * res;
            double y_local = (j - W / 2) * res;
            Eigen::Vector4d pt_local(x_local, y_local, 0, 1);

            // 2) 变换到世界坐标系
            Eigen::Vector4d pt_world = world_T_wheel * pt_local;
            int gi = int(std::floor(pt_world.x() / res));
            int gj = int(std::floor(pt_world.y() / res));

            // 3) 占用概率融合（Bayes odds 方式）
            global_map_.updateOcc(gi, gj, p_meas);

            // 4) 合并投票
            global_map_.updateLabel(gi, gj, label);

            // 5) 合并高度（取最高）
            global_map_.updateHeight(gi, gj, h);
        }
    }
}

void MapFusion::fuseSubmapToGlobal(const FinishedSubmap &fs) {
    double global_res = params_.global_res;
    double submap_res = params_.submap_res;
    // 1) 遍历子图稀疏概率哈希
    for (auto const &kv : fs.occ_map) {
        int64_t key_sub = kv.first;  // 子图内部打包的 (i_sub,j_sub)
        double p_meas = kv.second;   // 观测到的概率 ∈ [0,1]

        auto [sub_i, sub_j] = decodeKey(key_sub);

        // 3) 在子图坐标系下的物理点 (x_sub, y_sub, 0)
        double x_sub = double(sub_i) * submap_res;
        double y_sub = double(sub_j) * submap_res;

        Eigen::Vector4d pt_sub(x_sub, y_sub, 0.0, 1.0);

        // 4) 变换到世界坐标系
        Eigen::Vector4d pt_w = fs.first_pose * pt_sub;
        double x_w = pt_w.x();
        double y_w = pt_w.y();

        // 5) “量化”到全局网格索引，取 floor
        int gi_glob = int(std::floor(x_w / global_res));
        int gj_glob = int(std::floor(y_w / global_res));

        // 6) 进行贝叶斯 Odds 乘法更新：
        global_map_.updateOcc(gi_glob, gj_glob, p_meas);

        // 7）更新LABEL和height
        auto vote_map = fs.label_map.at(key_sub);
        global_map_.updateLabel(gi_glob, gj_glob, vote_map);
        
        double h = fs.height_map.at(key_sub);
        global_map_.updateHeight(gi_glob, gj_glob, h);
    }
}

void MapFusion::fuseLocalToSubmap(ActiveSubmap &as, const LocalMap &local_map, const Eigen::Matrix4d &world_T_wheel) {
    const int H = params_.grid_h;
    const int W = params_.grid_w;
    const double res = params_.submap_res;

    // 1) 计算相对子图首帧的变换 rel = first_pose⁻¹ * world_T_wheel
    Eigen::Matrix4d rel = as.first_pose.inverse() * world_T_wheel;

    // 2) 遍历 local_occ 的每个格子 (i,j)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double p_meas = local_map.occupancy_map(i, j);
            int label = local_map.category_map(i, j);
            double height = local_map.height_map(i, j);
            if (p_meas == 0.5) {
                continue;
            }
            // 计算 (i,j) 在子图坐标系下的物理坐标
            double x_local = i * res;
            double y_local = (j - W / 2) * res;
            Eigen::Vector4d pt_local(x_local, y_local, 0.0, 1.0);

            // 投影到子图坐标系：pt_sub = rel * pt_local
            Eigen::Vector4d pt_sub = rel * pt_local;
            int gi = int(std::floor(pt_sub.x() / res));
            int gj = int(std::floor(pt_sub.y() / res));

            as.updateOcc(gi, gj, p_meas);
            as.updateLabel(gi, gj, label);
            as.updateHeight(gi, gj, height);
        }
    }

    // 3) 增加该子图的 fused_count
    as.fused_count++;
}

void MapFusion::visualize(const PackageData &pkg) const {
    if (pkg.image_data && !pkg.image_data->image.empty() && pkg.segmentation_data &&
        !pkg.segmentation_data->image.empty()) {
        cv::Mat combined;
        cv::hconcat(pkg.image_data->image, pkg.segmentation_data->image, combined);
        cv::imshow("Original", combined);
    }

    // display local map
    showMaps(local_map_.category_map, local_map_.height_map, local_map_.occupancy_map, "local");

    // display last submap
    if (finished_submaps_.size() > 0) {
        auto fs = finished_submaps_[finished_submaps_.size() - 1];
        SLAM_LOG_INFO() << "submap id: " << fs.submap_id;
        // showMaps(fs.occupancyEigenMap(), "finished_submap", false);
        showMaps(fs.labelEigenMap(), fs.heightEigenMap(), fs.occupancyEigenMap(), "finished_submap", false);
    }

    // display global map
    // showMaps(global_map_.occupancyEigenMap(), "global");
    showMaps(global_map_.labelEigenMap(), global_map_.heightEigenMap(), global_map_.occupancyEigenMap(), "global");

    // TODO，未叠label和height图
    // showMaps(global_map_.labelEigenMap(), global_map_.heightEigenMap(), global_map_.occupancyEigenMap(), "global");

    cv::waitKey(1);
}

// 可视化：直接根据 cat_map 中的 label 调用 labelToColor
cv::Mat MapFusion::visualizeCategoryMap(const Eigen::MatrixXi &cat_map) const {
    int H = cat_map.rows(), W = cat_map.cols();
    cv::Mat vis(H, W, CV_8UC3);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            cv::Vec3b rgb = labelToColor(cat_map(i, j));
            cv::Vec3b bgr{rgb[2], rgb[1], rgb[0]};
            vis.at<cv::Vec3b>(i, j) = bgr;
        }
    }
    return vis;
}

// 可视化高度图保持不变
cv::Mat MapFusion::visualizeHeightMap(const Eigen::MatrixXd &height_map, double min_h, double max_h) const {
    int H = height_map.rows(), W = height_map.cols();
    cv::Mat gray(H, W, CV_8UC1);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double h = height_map(i, j);
            int v = 0;
            if (h > min_h) {
                v = cv::saturate_cast<uchar>(255 * (h - min_h) / (max_h - min_h));
            }
            gray.at<uchar>(i, j) = v;
        }
    }
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);
    return color;
}

cv::Mat MapFusion::visualizeOccupancyMap(const Eigen::MatrixXd &occ_map, double p_free, double p_occ) const {
    cv::Mat occ_vis(occ_map.rows(), occ_map.cols(), CV_8UC3);
    for (int i = 0; i < occ_map.rows(); ++i) {
        for (int j = 0; j < occ_map.cols(); ++j) {
            double p = occ_map(i, j);
            if (p < p_free) {
                occ_vis.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            } else if (p <= p_occ) {
                occ_vis.at<cv::Vec3b>(i, j) = cv::Vec3b(128, 128, 128);
            } else {
                occ_vis.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return occ_vis;
}

void MapFusion::showMaps(const Eigen::MatrixXi &cat_map, const Eigen::MatrixXd &hgt_map, const Eigen::MatrixXd &occ_map,
                         const std::string &name, const bool &save) const {
    // 1. 计算高度范围
    double min_h = std::numeric_limits<double>::max();
    double max_h = -std::numeric_limits<double>::max();
    for (int i = 0; i < hgt_map.rows(); ++i) {
        for (int j = 0; j < hgt_map.cols(); ++j) {
            double h = hgt_map(i, j);
            if (h > std::numeric_limits<double>::lowest()) {
                min_h = std::min(min_h, h);
                max_h = std::max(max_h, h);
            }
        }
    }

    // 2. 生成原始可视化图
    cv::Mat cat_vis = visualizeCategoryMap(cat_map);
    cv::Mat hgt_vis = visualizeHeightMap(hgt_map, min_h, max_h);
    // 三值可视化
    cv::Mat occ_vis = visualizeOccupancyMap(occ_map, 0.4, 0.6);

    // 3. 缩放参数：等比放大 N 倍
    const double display_scale = params_.display_scale;  // 放大 x 倍
    cv::Mat cat_vis_s, hgt_vis_s, occ_vis_s;
    // 类别图用最近邻插值，保证颜色块不模糊
    cv::resize(cat_vis, cat_vis_s, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);
    // 高度图用双线性插值，平滑渐变
    cv::resize(hgt_vis, hgt_vis_s, cv::Size(), display_scale, display_scale, cv::INTER_LINEAR);

    cv::resize(occ_vis, occ_vis_s, cv::Size(), display_scale, display_scale, cv::INTER_LINEAR);

    // 4. 将三张图水平拼接到同一个大图
    cv::Mat top;
    cv::hconcat(cat_vis_s, hgt_vis_s, top);
    cv::Mat combined;
    cv::hconcat(top, occ_vis_s, combined);

    cv::imshow(name, combined);

    cv::waitKey(1);

    // 5. 可选：保存放大后的图
    if (save) {
        static int _i = 0;
        std::string window_name_cat = name + "_cat";
        std::string window_name_hgt = name + "_hgt";
        std::string window_name_occ = name + "_occ";
        window_name_cat += ("_" + std::to_string(_i) + ".png");
        window_name_hgt += ("_" + std::to_string(_i) + ".png");
        window_name_occ += ("_" + std::to_string(_i) + ".png");
        cv::imwrite(window_name_cat, cat_vis_s);
        cv::imwrite(window_name_hgt, hgt_vis_s);
        cv::imwrite(window_name_occ, occ_vis_s);
        _i++;
    }
}

void MapFusion::showMaps(const Eigen::MatrixXd &occ_map, const std::string &name, const bool &save) const {
    // 三值可视化
    cv::Mat occ_vis = visualizeOccupancyMap(occ_map, 0.4, 0.6);

    // 3. 缩放参数：等比放大 N 倍
    const double display_scale = params_.display_scale;
    cv::Mat occ_vis_s;
    cv::resize(occ_vis, occ_vis_s, cv::Size(), display_scale, display_scale, cv::INTER_LINEAR);

    cv::imshow(name, occ_vis_s);

    cv::waitKey(1);

    // 5. 可选：保存放大后的图
    if (save) {
        static int _i = 0;
        // std::string path = "/work/develop_gitlab/slam-core/outputs/submap/";
        std::string path = "/work/develop_gitlab/slam-core/outputs/s2/";
        std::string window_name_occ = name + "_occ";
        window_name_occ += ("_" + std::to_string(_i) + ".png");
        path += window_name_occ;
        cv::imwrite(path, occ_vis_s);
        _i++;
    }
}
