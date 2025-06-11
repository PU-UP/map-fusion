#include "grid_map.hpp"

using namespace map_fusion;

bool FinishedSubmap::save(const std::string &dir_path) const {
    // 1. 构造完整文件名
    //    C++17 可以直接用 std::filesystem::path 拼接，也可手动拼接字符串
    std::filesystem::path out_dir(dir_path);
    if (!std::filesystem::exists(out_dir)) {
        // 不存在就尝试创建
        std::filesystem::create_directories(out_dir);
    }
    std::string filename = "submap_" + std::to_string(submap_id) + ".bin";
    std::filesystem::path full_path = out_dir / filename;

    // 2. 以二进制模式打开输出流
    std::ofstream ofs(full_path, std::ios::binary);
    if (!ofs.is_open()) {
        SLAM_LOG_ERROR() << "Failed to open file for writing: " << full_path;
        return false;
    }

    // 3. 写入 submap_id
    ofs.write(reinterpret_cast<const char *>(&submap_id), sizeof(submap_id));

    // zy 20250609
    ofs.write(reinterpret_cast<const char *>(&ts), sizeof(ts));

    // 4. 写入 first_pose（16 个 double）
    //    Eigen 默认按列主序存储 data() 指向连续的 16 个 double
    ofs.write(reinterpret_cast<const char *>(first_pose.data()), sizeof(double) * 16);

    // 5. 写入四个边界值：min_i, max_i, min_j, max_j
    ofs.write(reinterpret_cast<const char *>(&min_i), sizeof(min_i));
    ofs.write(reinterpret_cast<const char *>(&max_i), sizeof(max_i));
    ofs.write(reinterpret_cast<const char *>(&min_j), sizeof(min_j));
    ofs.write(reinterpret_cast<const char *>(&max_j), sizeof(max_j));

    // 6. 写入 occ_map 的大小（size_t 或 uint64_t）
    uint64_t map_size = static_cast<uint64_t>(occ_map.size());
    ofs.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));

    // 7. 遍历 occ_map，把每个 (key, value) 写进去
    //    key 是 int64_t，value 是 double
    for (const auto &kv : occ_map) {
        int64_t key = kv.first;
        double prob = kv.second;
        ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
        ofs.write(reinterpret_cast<const char *>(&prob), sizeof(prob));
    }

    ofs.close();
    // 如果需要，也可以在这里 flush 一下：ofs.flush();
    return true;
}

bool FinishedSubmap::load(const std::string &full_path) {
    // 打开二进制输入流
    std::ifstream ifs(full_path, std::ios::binary);
    if (!ifs.is_open()) {
        SLAM_LOG_ERROR() << "Failed to open file for writing: " << full_path;
        return false;
    }

    // 读取 submap_id
    ifs.read(reinterpret_cast<char *>(&submap_id), sizeof(submap_id));

    // zy 20250609
    ifs.read(reinterpret_cast<char *>(&ts), sizeof(ts));

    // 读取 first_pose（16 个 double）
    ifs.read(reinterpret_cast<char *>(first_pose.data()), sizeof(double) * 16);

    // 读取边界值 min_i, max_i, min_j, max_j
    ifs.read(reinterpret_cast<char *>(&min_i), sizeof(min_i));
    ifs.read(reinterpret_cast<char *>(&max_i), sizeof(max_i));
    ifs.read(reinterpret_cast<char *>(&min_j), sizeof(min_j));
    ifs.read(reinterpret_cast<char *>(&max_j), sizeof(max_j));

    // 读取 occ_map 大小
    uint64_t map_size = 0;
    ifs.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));

    // 清空旧数据并预分配空间
    occ_map.clear();
    occ_map.reserve(static_cast<size_t>(map_size));

    // 逐对读取 (key, prob) 并 emplace 到 occ_map
    for (uint64_t i = 0; i < map_size; ++i) {
        int64_t key;
        double prob;
        ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
        ifs.read(reinterpret_cast<char *>(&prob), sizeof(prob));
        occ_map.emplace(key, prob);
    }

    ifs.close();
    return true;
}