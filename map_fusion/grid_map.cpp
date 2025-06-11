
#include "grid_map.hpp"

using namespace map_fusion;

void GridMap::updateOcc(int i, int j, double p_meas) {
    int64_t key = encodeKey(i, j);
    double eps = 1e-3;
    // —— 1. 拿到旧概率 p_old，如果 key 不存在，则当作先验 p_old = 0.5 ——
    double p_old = 0.5;
    auto it = occ_map.find(key);
    if (it != occ_map.end()) {
        // clamp 到 [eps, 1 - eps]
        p_old = std::clamp(it->second, eps, 1.0 - eps);
    }

    p_meas = std::clamp(p_meas, eps, 1.0 - eps);

    // —— 2. 计算 l_old, l_meas, 再相加并 clamp ——
    double l_old = logOdds(p_old);
    double l_meas = logOdds(p_meas);
    double l_new = clampLogOdds(l_old + l_meas);

    // —— 3. 把新的 log‐odds 转回概率 p_new ——
    double p_new = invLogOdds(l_new);

    // —— 4. 更新（或插入）occ_map[key] = p_new ——
    bool was_present = (it != occ_map.end());
    occ_map[key] = p_new;

    // —— 5. 如果这是第一次插入该 key，就取 decodeKey 更新边界 ——
    if (!was_present) {
        if (occ_map.size() == 1) {
            // 刚好插入的第 1 个元素，用它初始化所有边界
            min_i = max_i = i;
            min_j = max_j = j;
        } else {
            // 否则只要更新四个方向的上下界
            updateBounds(i, j);
        }
    }
}

// TODO. 未考虑扩张边界（默认会先调用updateOcc）
void GridMap::updateLabel(int i, int j, const std::unordered_map<int, int>& vote_map) {
    const std::unordered_set<int> conditional_labels = {Percep_Label2_::OBSTACLE_AND_BACKGROUND, Percep_Label2_::GRASS};
    int best_label = 0; int best_count = 0;
    int second_best_label = 0; int second_best_count = 0;

    for (auto const &lc : vote_map) {
        int label = lc.first;
        int count = lc.second;

        if (count > best_count) {
            // 当前成为新的最大值，原来的最大值落到次大值
            second_best_label = best_label;
            second_best_count = best_count;

            best_label = label;
            best_count = count;
        }
        else if (count > second_best_count && label != best_label) {
            // 只更新次大值（且排除与当前 best_label 相同的标签）
            second_best_label = label;
            second_best_count = count;
        }
    }
    if (best_count > 0 && best_label != Percep_Label2_::UNKNOWN_OCC) {
        if (conditional_labels.count(best_label) > 0) {
            if (second_best_count > 0.5 * best_count) {
                best_label = second_best_label;
            }
        }
        int64_t key = encodeKey(i, j);
        label_map[key][best_label]++;
    }
}

void GridMap::updateLabel(int i, int j, int label) {
    int64_t key = encodeKey(i, j);
    label_map[key][label]++;
}

void GridMap::updateHeight(int i, int j, double height) {
    int64_t key = encodeKey(i, j);
    double eps = 1e-3;
    auto it_h = height_map.find(key);
    if (it_h == height_map.end()) {
        height_map[key] = height;
    } else {
        it_h->second = std::max(it_h->second, height);
    }
}

Eigen::MatrixXi GridMap::labelEigenMap() const {
    int Gi = max_i - min_i + 1;
    int Gj = max_j - min_j + 1;

    Eigen::MatrixXi out_label_map = Eigen::MatrixXi::Zero(Gi, Gj);

    for (auto const &kv : label_map) {
        int64_t key = kv.first;
        auto [gi, gj] = decodeKey(key);
        int ii = gi - min_i;
        int jj = gj - min_j;
        const auto &vote_map = kv.second;

        if (ii >= 0 && ii < Gi && jj >= 0 && jj < Gj) {
            int best_label = 0, best_count = 0;
            for (auto const &lc : vote_map) {
                if (lc.second > best_count) {
                    best_label = lc.first;
                    best_count = lc.second;
                }
            }
            out_label_map(ii, jj) = best_label;
        } else {
            SLAM_LOG_WARN() << "Get out of bound (Label), Gi: " << Gi << " Gj: " << Gj << "; ii: " << ii
                            << " jj: " << jj;
        }
    }
    return out_label_map;
}

Eigen::MatrixXd GridMap::heightEigenMap() const {
    int Gi = max_i - min_i + 1;
    int Gj = max_j - min_j + 1;

    Eigen::MatrixXd out_heigh_map = Eigen::MatrixXd::Constant(Gi, Gj, std::numeric_limits<double>::lowest());

    for (auto const &kv : height_map) {
        int64_t key = kv.first;
        auto [gi, gj] = decodeKey(key);
        int ii = gi - min_i;
        int jj = gj - min_j;
        double h = kv.second;
        // 边界检查
        if (ii >= 0 && ii < Gi && jj >= 0 && jj < Gj) {
            out_heigh_map(ii, jj) = h;
        } else {
            SLAM_LOG_WARN() << "Get out of bound (Height), Gi: " << Gi << " Gj: " << Gj << "; ii: " << ii
                            << " jj: " << jj;
        }
    }
    return out_heigh_map;
}

Eigen::MatrixXd GridMap::occupancyEigenMap() const {
    int Gi = max_i - min_i + 1;
    int Gj = max_j - min_j + 1;
    // 默认填充为 0.5
    Eigen::MatrixXd out_occ_map = Eigen::MatrixXd::Constant(Gi, Gj, 0.5);

    for (auto const &kv : occ_map) {
        int64_t key = kv.first;
        auto [gi, gj] = decodeKey(key);
        int ii = gi - min_i;
        int jj = gj - min_j;
        double p = kv.second;
        // 边界检查
        if (ii >= 0 && ii < Gi && jj >= 0 && jj < Gj) {
            out_occ_map(ii, jj) = p;
        } else {
            SLAM_LOG_WARN() << "Get out of bound (occ), Gi: " << Gi << " Gj: " << Gj << "; ii: " << ii << " jj: " << jj;
        }
    }
    return out_occ_map;
}
