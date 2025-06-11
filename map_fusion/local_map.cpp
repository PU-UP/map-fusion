
#include "local_map.hpp"

using namespace map_fusion;

void LocalMap::projectLocalPointCloud(const std::vector<Eigen::Vector3d> &pts_wheel, const std::vector<int> &pts_label,
                                      const Eigen::Matrix4d &_world_T_wheel) {
    world_T_wheel = _world_T_wheel;
    // 尺寸
    const int H = category_map.rows();
    const int W = category_map.cols();

    // 清空
    category_map.setZero();
    height_map.setConstant(std::numeric_limits<double>::lowest());
    occupancy_map.setConstant(0.5);

    // 投票容器：H*W 个 map<label,count>
    std::vector<std::unordered_map<int, int>> vote_counts(H * W);
    std::vector<double> distances(H * W);

    // 遍历每个点
    for (size_t idx = 0; idx < pts_wheel.size() && idx < pts_label.size(); ++idx) {
        const auto &p = pts_wheel[idx];
        const int label = pts_label[idx];

        // 栅格索引
        int i = int(std::floor(p.x() / local_resolution));
        int j = int(std::floor(p.y() / local_resolution)) + W / 2;
        if (i < 0 || i >= H || j < 0 || j >= W) continue;
        
        // 线性索引
        int lin_idx = i * W + j;
        double dis = std::sqrt(p.x() * p.x() + p.y() * p.y());
        distances[lin_idx] = dis;
        // 过滤距离大于2m的深度点
        if (dis > distance_threshold) continue;
        // 投票：同一个格子里，每个 label 自增
        vote_counts[lin_idx][label]++;

        // 高度改为该格最高
        height_map(i, j) = std::max(height_map(i, j), p.z());
    }

    // 根据票数最多的 label 填充 category_map
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            auto &mp = vote_counts[i * W + j];
            if (mp.empty()) continue;
            double cur_dis = distances.at(i * W + j);
            int best_label = selectBestLabel(mp);
            if(!labelFiltered(best_label, cur_dis)) {
                category_map(i, j) = best_label;
                updateLocalOcc(i, j, best_label, cur_dis);
            }
        }
    }
}

bool LocalMap::labelFiltered(int label, double distance) {
    const std::unordered_set<int> conditional_ignore_labels = {Percep_Label2_::OBSTACLE_AND_BACKGROUND,
                                                               Percep_Label2_::CHARGING_STATION,
                                                               Percep_Label2_::GRASS,
                                                               Percep_Label2_::HARD_ROAD,
                                                               Percep_Label2_::HARD_WALL,
                                                               Percep_Label2_::PIT,
                                                               Percep_Label2_::PIPELINE};
    const std::unordered_set<int> ignore_labels = {Percep_Label2_::UNKNOWN_OCC, Percep_Label2_::ADULT, Percep_Label2_::CHILD};

    if (ignore_labels.count(label) > 0) {
        return true;
    }
    if (distance > distance_threshold / 2 && conditional_ignore_labels.count(label) > 0) {
        return true;
    }
    if (distance > distance_threshold) {
        return true;
    }

    return false;
}

void LocalMap::updateLocalOcc(int i, int j, int label, double distance) {
    const std::unordered_set<int> free_labels = {Percep_Label2_::GRASS, Percep_Label2_::LEAVES, Percep_Label2_::MUD,
                                                 Percep_Label2_::PIPELINE};
    // 1) 取出旧概率并防止 p=0/1
    double p_old = std::clamp(occupancy_map(i, j), 1e-3, 1.0 - 1e-3);
    // 2) 变成 log-odds 并 clamp 到 [-Lmax, Lmax]
    double l_old = clampLogOdds(logOdds(p_old));
    // 3) 选增量：空闲 or 占用
    double delta = free_labels.count(label) ? l_free_update : l_occ_update;
    // 4) 累加 & clamp
    double w = distance < 1 ? 1 : 0.5;
    double l_new = clampLogOdds(l_old + delta * w);
    // 5) 反变换成概率 [0,1]
    double p_new = invLogOdds(l_new);
    // 6) 写回
    occupancy_map(i, j) = p_new;
}

int LocalMap::selectBestLabel(const std::unordered_map<int, int> &vote_count) {
    // 找出票数最高的 label
    int best_label = 0;
    int best_count = 0;
    int second_best_label = 0;
    int second_best_count = 0;
    for (auto &kv : vote_count) {
        int label = kv.first;
        int cnt   = kv.second;
        if (cnt > best_count) {
            // 新的最大值出现，原 best 下沉为 second_best
            second_best_label = best_label;
            second_best_count = best_count;

            best_label = label;
            best_count = cnt;
        } else if (label != best_label && cnt > second_best_count) {
            // 更新次大值（排除与当前 best_label 相同的标签）
            second_best_label = label;
            second_best_count = cnt;
        }
    }
    const std::unordered_set<int> conditional_labels = {Percep_Label2_::OBSTACLE_AND_BACKGROUND, Percep_Label2_::GRASS};
    if (best_count > 0) {
        if (conditional_labels.count(best_label) > 0) {
            if (second_best_count > 0.5 * best_count) {
                best_label = second_best_label;
            }
        }
    }
    return best_label;
}

bool LocalMap::saveFrame(const std::string &path) const {
    std::ofstream ofs(path + "_occ.bin", std::ios::binary);
    if (!ofs) {
        // if (!ofs) throw std::runtime_error("无法打开文件写入: " + path + "_occ.bin");
        SLAM_LOG_ERROR() << "Save occupancy map failed, map-fusion fail to access " << path;
        return false;
    }
    // occ_map 大小固定为 grid_h×grid_w
    // 直接写入内存块 sizeof(double)×rows×cols
    ofs.write(reinterpret_cast<const char *>(occupancy_map.data()), sizeof(double) * occupancy_map.size());
    ofs.write(reinterpret_cast<const char *>(world_T_wheel.data()), sizeof(double) * world_T_wheel.size());
    return true;
}

/// 从二进制文件加载指定帧的 occ_map 与 pose，
/// 返回 true 表示成功，false 表示文件不存在或读取失败。
bool LocalMap::loadFrame(const std::string &path) {
    // 确保 occ_map 已经被 resize 到 (grid_h, grid_w)
    occupancy_map.resize(grid_h, grid_w);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        SLAM_LOG_WARN() << "Load occupancy map failed, map-fusion fail to access " << path;
        return false;
    }
    ifs.read(reinterpret_cast<char *>(occupancy_map.data()), sizeof(double) * occupancy_map.size());
    ifs.read(reinterpret_cast<char *>(world_T_wheel.data()), sizeof(double) * world_T_wheel.size());

    return true;
}