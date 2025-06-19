#!/usr/bin/env python3
"""
子图融合实现
功能：
1. 加载子图
2. 融合到全局地图
3. 可视化和保存结果
"""

import os
import sys
import struct
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from typing import Optional
from scipy.spatial.transform import Rotation as R

# 配置 matplotlib 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def decode_key(key: int) -> tuple:
    """完全匹配C++实现的key解码
    C++: 
    int32_t gi = int32_t(key >> 32);
    int32_t gj = int32_t(key & 0xFFFFFFFF);
    """
    # 提取高32位作为有符号整数 → gi
    gi = np.int32(key >> 32).item()
    # 提取低32位作为无符号数，然后转换为有符号整数 → gj
    low = key & 0xFFFFFFFF
    if low >= (1 << 31):
        gj = int(low - (1 << 32))
    else:
        gj = int(low)
    return int(gi), int(gj)

def encode_key(gi: int, gj: int) -> int:
    """完全匹配C++实现的key编码
    C++: return (int64_t(gi) << 32) | uint32_t(gj);
    """
    return (int(gi) << 32) | (int(gj) & 0xFFFFFFFF)

def log_odds(p: float) -> float:
    """完全匹配C++实现"""
    return np.log(p / (1.0 - p))

def clamp_log_odds(l: float) -> float:
    """完全匹配C++实现"""
    L_MAX = 20.0
    return np.clip(l, -L_MAX, L_MAX)

def inv_log_odds(l: float) -> float:
    """完全匹配C++实现"""
    return 1.0 / (1.0 + np.exp(-l))

def ternarize_probability(p: float) -> float:
    """将概率三值化：空p < 0.4 = 0，占用p > 0.6 = 1，未知else p = 0.5"""
    if p < 0.4:
        return 0.0  # 空闲
    elif p > 0.6:
        return 1.0  # 占用
    else:
        return 0.5  # 未知

class GridMap:
    """完全匹配C++的GridMap类"""
    def __init__(self):
        self.occ_map = {}  # Dict[int, float] - key: encoded (gi,gj), value: probability
        self.min_i = 0
        self.max_i = 0
        self.min_j = 0
        self.max_j = 0
        self.initialized = False
    
    def update_bounds(self, i: int, j: int):
        """更新地图边界"""
        if not self.initialized:
            self.min_i = i
            self.max_i = i
            self.min_j = j
            self.max_j = j
            self.initialized = True
        else:
            self.min_i = min(self.min_i, i)
            self.max_i = max(self.max_i, i)
            self.min_j = min(self.min_j, j)
            self.max_j = max(self.max_j, j)
    
    def update_occ(self, i: int, j: int, p_meas: float):
        """完全匹配C++的概率更新实现"""
        key = encode_key(i, j)
        eps = 1e-3
        
        # 1. 获取旧概率
        p_old = 0.5  # 默认先验
        if key in self.occ_map:
            p_old = np.clip(self.occ_map[key], eps, 1.0 - eps)
        
        # 限制测量概率范围
        p_meas = np.clip(p_meas, eps, 1.0 - eps)
        
        # 2. 计算log odds并相加
        l_old = log_odds(p_old)
        l_meas = log_odds(p_meas)
        l_new = clamp_log_odds(l_old + l_meas)
        
        # 3. 转回概率
        p_new = inv_log_odds(l_new)
        
        # 4. 更新地图
        was_present = key in self.occ_map
        self.occ_map[key] = p_new
        
        # 5. 更新边界
        if not was_present:
            if len(self.occ_map) == 1:
                self.min_i = self.max_i = i
                self.min_j = self.max_j = j
                self.initialized = True
            else:
                self.update_bounds(i, j)
    
    def set_occ_direct(self, i: int, j: int, p: float):
        """直接设置占用概率，不经过log odds更新"""
        key = encode_key(i, j)
        was_present = key in self.occ_map
        self.occ_map[key] = p
        
        # 更新边界
        if not was_present:
            if len(self.occ_map) == 1:
                self.min_i = self.max_i = i
                self.min_j = self.max_j = j
                self.initialized = True
            else:
                self.update_bounds(i, j)
    
    def to_matrix(self) -> np.ndarray:
        """转换为矩阵形式，用于可视化"""
        if not self.initialized:
            return np.full((1, 1), 0.5)
        
        h = self.max_i - self.min_i + 1
        w = self.max_j - self.min_j + 1
        grid = np.full((h, w), 0.5)
        
        for key, p in self.occ_map.items():
            gi, gj = decode_key(key)
            ii = gi - self.min_i
            jj = gj - self.min_j
            if 0 <= ii < h and 0 <= jj < w:
                grid[ii, jj] = p
        
        return grid

def load_submap(bin_path: str) -> tuple:
    """完全匹配C++的子图加载实现"""
    with open(bin_path, 'rb') as f:
        # 1) submap_id (int32)
        raw = f.read(4)
        if len(raw) < 4:
            raise RuntimeError(f"{bin_path} is too short (missing submap_id)")
        submap_id = struct.unpack('i', raw)[0]

        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError(f"{bin_path} is too short (missing ts).")
        ts = struct.unpack('d', raw)[0]
        
        # 2) first_pose (16 doubles)
        raw = f.read(8 * 16)
        if len(raw) < 8 * 16:
            raise RuntimeError(f"{bin_path} is too short (missing first_pose)")
        pose_vals = struct.unpack('d' * 16, raw)
        # 使用列优先顺序(order='F')来匹配Eigen的存储方式
        first_pose = np.array(pose_vals, dtype=np.float64).reshape((4, 4), order='F')
        
        # 3) bounds: min_i, max_i, min_j, max_j (4×int32)
        raw = f.read(4 * 4)
        if len(raw) < 16:
            raise RuntimeError(f"{bin_path} is too short (missing bounds)")
        min_i, max_i, min_j, max_j = struct.unpack('i' * 4, raw)
        
        # 4) occ_map size (uint64)
        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError(f"{bin_path} is too short (missing map_size)")
        map_size = struct.unpack('Q', raw)[0]
        
        # 5) Read map_size entries of (int64 key, double prob)
        occ_map = {}
        for _ in range(map_size):
            entry = f.read(8 + 8)
            if len(entry) < 16:
                raise RuntimeError(f"{bin_path} is too short (incomplete map entries)")
            key, prob = struct.unpack('q d', entry)
            occ_map[key] = prob
            
    return submap_id, ts,first_pose, min_i, max_i, min_j, max_j, occ_map

def read_gt_poses(folder_path: str) -> dict:
    """从path_pg_rtk.txt读取地面真值姿态，并转换为时间戳到4x4变换矩阵的映射。
    文件格式: timestamp tx ty tz qx qy qz qw
    """
    gt_poses = {}
    gt_file_path = os.path.join(folder_path, 'path_pg_rtk.txt')
    if not os.path.exists(gt_file_path):
        print(f"警告：地面真值文件 {gt_file_path} 不存在。将不使用地面真值姿态。")
        return gt_poses

    print(f"正在读取地面真值文件: {gt_file_path}")
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                # 创建旋转矩阵
                rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

                # 创建4x4变换矩阵
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [tx, ty, tz]
                
                gt_poses[timestamp] = transform_matrix
    print(f"已加载 {len(gt_poses)} 个地面真值姿态。")
    return gt_poses

def visualize_map(occ_map: np.ndarray, p_free: float = 0.3, p_occ: float = 0.7):
    """可视化占用栅格地图"""
    vis = np.zeros((*occ_map.shape, 3), dtype=np.uint8)
    
    # 对于三值化地图，使用精确的阈值
    # 空闲区域显示为白色 (p = 0.0)
    free_mask = np.isclose(occ_map, 0.0, rtol=1e-5)
    vis[free_mask] = [255, 255, 255]
    
    # 占用区域显示为黑色 (p = 1.0)
    occ_mask = np.isclose(occ_map, 1.0, rtol=1e-5)
    vis[occ_mask] = [0, 0, 0]
    
    # 未知区域显示为灰色 (p = 0.5 或其他值)
    unknown_mask = ~(free_mask | occ_mask)
    vis[unknown_mask] = [128, 128, 128]
    
    return vis

def fuse_submaps(folder_path: str, save_path: Optional[str] = None, use_gt: bool = False, gt_poses: Optional[dict] = None, global_res: float = 0.1):
    """融合所有子图"""
    # 创建全局地图
    global_map = GridMap()
    submap_res = 0.05  # 子图分辨率
    
    # 获取所有子图文件并创建ID到文件路径的映射
    submap_files = glob.glob(os.path.join(folder_path, 'submap_*.bin'))
    if not submap_files:
        print(f"错误：在{folder_path}中没有找到子图文件")
        return None, None
    
    # 创建submap_id到文件路径的映射
    id_to_file = {}
    for bin_path in submap_files:
        try:
            submap_id = int(os.path.basename(bin_path).split('_')[1].split('.')[0])
            id_to_file[submap_id] = bin_path
        except (IndexError, ValueError):
            print(f"警告：无法从{bin_path}提取子图ID")
            continue
    
    if not id_to_file:
        print("错误：没有有效的子图文件")
        return None, None
    
    print(f"找到{len(id_to_file)}个子图文件")
    
    # 按ID顺序处理子图
    max_id = max(id_to_file.keys())
    for submap_id in range(max_id + 1):
        if submap_id not in id_to_file:
            print(f"警告：缺少submap_{submap_id}")
            continue
        
        bin_path = id_to_file[submap_id]
        print(f"\nProcessing submap_{submap_id}...")
        
        # 加载子图
        _, ts, first_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(bin_path)
        
        if use_gt and gt_poses and ts in gt_poses:
            print(f"使用地面真值姿态更新submap_{submap_id}的姿态 (ts: {ts})")
            first_pose = gt_poses[ts]
        elif use_gt and ts not in gt_poses:
            print(f"警告：submap_{submap_id}的时间戳 {ts} 未在地面真值文件中找到。跳过此子图的融合。")
            continue

        print(f"子图边界: i[{min_i}, {max_i}], j[{min_j}, {max_j}]")
        print(f"栅格数量: {len(occ_map)}")
        
        # 遍历子图中的每个占用栅格
        for key, p_meas in occ_map.items():
            # 1) 解码子图栅格索引
            sub_i, sub_j = decode_key(key)
            
            # 2) 计算子图坐标系下的物理坐标
            p_s = np.array([
                sub_i * submap_res,
                sub_j * submap_res,
                0.0
            ])
            
            # 3) 转换到世界坐标系
            p_w = first_pose[:3, :3] @ p_s + first_pose[:3, 3]
            
            # 4) 计算全局地图栅格索引
            gi_glob = int(np.floor(p_w[0] / global_res))
            gj_glob = int(np.floor(p_w[1] / global_res))
            
            # 5) 更新全局地图
            global_map.update_occ(gi_glob, gj_glob, p_meas)
    
    # 打印全局地图信息
    print(f"\n全局地图信息 (分辨率: {global_res}m):")
    print(f"栅格数量: {len(global_map.occ_map)} cells")
    print(f"边界: i[{global_map.min_i}, {global_map.max_i}], j[{global_map.min_j}, {global_map.max_j}]")
    print(f"物理范围: x[{global_map.min_i*global_res:.2f}, {global_map.max_i*global_res:.2f}]")
    print(f"          y[{global_map.min_j*global_res:.2f}, {global_map.max_j*global_res:.2f}]")
    
    # 转换为矩阵形式
    grid = global_map.to_matrix()
    vis = visualize_map(grid) # 无论是否保存都生成可视化图像

    # 保存全局地图
    if save_path:
        # 保存为二进制文件
        save_global_map(global_map, save_path + '.bin')
        # 保存可视化结果
        plt.imsave(save_path + '.png', vis, cmap='gray')
        print(f"全局地图已保存到: {save_path}.bin 和 {save_path}.png")
    
    return global_map, vis # 返回地图和可视化图像

def save_global_map(global_map: GridMap, save_path: str):
    """保存全局地图为二进制文件"""
    with open(save_path, 'wb') as f:
        # 1. 保存地图边界
        f.write(struct.pack('i' * 4, global_map.min_i, global_map.max_i, 
                          global_map.min_j, global_map.max_j))
        
        # 2. 保存地图大小
        f.write(struct.pack('Q', len(global_map.occ_map)))
        
        # 3. 保存地图数据
        for key, prob in global_map.occ_map.items():
            f.write(struct.pack('q d', key, prob))

def load_global_map(load_path: str) -> GridMap:
    """加载全局地图"""
    global_map = GridMap()
    
    with open(load_path, 'rb') as f:
        # 1. 读取地图边界
        raw = f.read(4 * 4)
        min_i, max_i, min_j, max_j = struct.unpack('i' * 4, raw)
        global_map.min_i = min_i
        global_map.max_i = max_i
        global_map.min_j = min_j
        global_map.max_j = max_j
        global_map.initialized = True
        
        # 2. 读取地图大小
        raw = f.read(8)
        map_size = struct.unpack('Q', raw)[0]
        
        # 3. 读取地图数据
        for _ in range(map_size):
            raw = f.read(16)
            key, prob = struct.unpack('q d', raw)
            global_map.occ_map[key] = prob
    
    return global_map

def downsample_map(high_res_map: GridMap, high_res: float, low_res: float) -> GridMap:
    """将高分辨率地图降采样为低分辨率地图
    
    降采样规则：
    - 如果其中有一个是占用，就认为这个格子是占用
    - 如果没有占用，但有一个是空，则认为是空  
    - 如果全是未知，才认为是未知
    """
    low_res_map = GridMap()
    
    print(f"开始降采样：从 {high_res}m 到 {low_res}m")
    print(f"高分辨率地图栅格数量: {len(high_res_map.occ_map)}")
    
    # 将高分辨率地图的概率先进行三值化
    ternarized_map = {}
    ternary_stats = {'occupied': 0, 'free': 0, 'unknown': 0}
    
    for key, prob in high_res_map.occ_map.items():
        ternary_prob = ternarize_probability(prob)
        ternarized_map[key] = ternary_prob
        
        if ternary_prob == 1.0:
            ternary_stats['occupied'] += 1
        elif ternary_prob == 0.0:
            ternary_stats['free'] += 1
        else:
            ternary_stats['unknown'] += 1
    
    print(f"三值化统计 - 占用: {ternary_stats['occupied']}, 空闲: {ternary_stats['free']}, 未知: {ternary_stats['unknown']}")
    
    # 按照低分辨率栅格对高分辨率栅格进行分组
    low_res_groups = {}
    
    for key, ternary_prob in ternarized_map.items():
        gi, gj = decode_key(key)
        
        # 转换为物理坐标
        x = gi * high_res
        y = gj * high_res
        
        # 计算对应的低分辨率栅格索引
        low_gi = int(np.floor(x / low_res))
        low_gj = int(np.floor(y / low_res))
        low_key = encode_key(low_gi, low_gj)
        
        if low_key not in low_res_groups:
            low_res_groups[low_key] = []
        low_res_groups[low_key].append(ternary_prob)
    
    print(f"低分辨率栅格组数量: {len(low_res_groups)}")
    
    # 对每个低分辨率栅格应用融合规则
    fusion_stats = {'occupied': 0, 'free': 0, 'unknown': 0}
    
    for i, (low_key, probs) in enumerate(low_res_groups.items()):
        # 统计不同状态的数量
        occupied_count = sum(1 for p in probs if p == 1.0)
        free_count = sum(1 for p in probs if p == 0.0)
        unknown_count = sum(1 for p in probs if p == 0.5)
        
        # 添加调试信息到前几个栅格
        if i < 5:
            print(f"栅格 {i}: 占用={occupied_count}, 空闲={free_count}, 未知={unknown_count}")
        
        # 应用融合规则
        if occupied_count > 0:
            final_prob = 1.0  # 有占用就是占用
            fusion_stats['occupied'] += 1
        elif free_count > 0:
            final_prob = 0.0  # 没有占用但有空闲就是空闲
            fusion_stats['free'] += 1
        else:
            final_prob = 0.5  # 全是未知才是未知
            fusion_stats['unknown'] += 1
        
        # 添加到低分辨率地图
        low_gi, low_gj = decode_key(low_key)
        low_res_map.set_occ_direct(low_gi, low_gj, final_prob)
    
    print(f"融合统计 - 占用: {fusion_stats['occupied']}, 空闲: {fusion_stats['free']}, 未知: {fusion_stats['unknown']}")
    
    print(f"降采样完成，低分辨率地图栅格数量: {len(low_res_map.occ_map)}")
    
    # 打印状态统计
    occupied_total = sum(1 for p in low_res_map.occ_map.values() if p == 1.0)
    free_total = sum(1 for p in low_res_map.occ_map.values() if p == 0.0)
    unknown_total = sum(1 for p in low_res_map.occ_map.values() if p == 0.5)
    
    print(f"状态统计 - 占用: {occupied_total}, 空闲: {free_total}, 未知: {unknown_total}")
    
    return low_res_map

def ternarize_map(grid_map: GridMap) -> GridMap:
    """将地图中的所有概率进行三值化"""
    ternarized_map = GridMap()
    ternarized_map.min_i = grid_map.min_i
    ternarized_map.max_i = grid_map.max_i
    ternarized_map.min_j = grid_map.min_j
    ternarized_map.max_j = grid_map.max_j
    ternarized_map.initialized = grid_map.initialized
    
    for key, prob in grid_map.occ_map.items():
        ternarized_map.occ_map[key] = ternarize_probability(prob)
    
    return ternarized_map

# 添加子图匹配相关的函数
def add_noise_to_pose(pose: np.ndarray, translation_noise: float = 0.1, rotation_noise_deg: float = 5.0) -> np.ndarray:
    """给位姿添加噪声
    Args:
        pose: 4x4变换矩阵
        translation_noise: 平移噪声（米）
        rotation_noise_deg: 旋转噪声（度）
    """
    noisy_pose = pose.copy()
    
    # 添加平移噪声
    noisy_pose[:3, 3] += np.random.normal(0, translation_noise, 3)
    
    # 添加旋转噪声（简单起见，只在yaw方向添加）
    theta = np.random.normal(0, np.deg2rad(rotation_noise_deg))
    c, s = np.cos(theta), np.sin(theta)
    R_noise = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    noisy_pose[:3, :3] = R_noise @ pose[:3, :3]
    
    return noisy_pose

def compute_score(submap: GridMap, global_map: GridMap, pose: np.ndarray, 
                 submap_res: float = 0.05, global_res: float = 0.1) -> float:
    """计算子图在给定位姿下与全局地图的匹配得分"""
    score = 0
    count = 0
    
    # 遍历子图中的每个占用栅格
    for key, p_sub in submap.occ_map.items():
        if p_sub < 0.7:  # 只考虑占用概率高的点
            continue
            
        # 解码子图栅格索引
        sub_i, sub_j = decode_key(key)
        
        # 计算子图坐标系下的物理坐标
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 计算全局地图栅格索引
        gi_glob = int(np.floor(p_w[0] / global_res))
        gj_glob = int(np.floor(p_w[1] / global_res))
        
        # 在全局地图中查找对应栅格
        key_glob = encode_key(gi_glob, gj_glob)
        if key_glob in global_map.occ_map:
            p_glob = global_map.occ_map[key_glob]
            score += abs(p_glob - p_sub)
            count += 1
    
    return score / count if count > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description='子图融合工具')
    parser.add_argument('--folder', type=str, required=True, help='包含子图bin文件的文件夹路径')
    parser.add_argument('--save', type=str, default='global_map', help='保存融合后全局地图图像的文件名 (默认: global_map)')
    parser.add_argument('--use-gt', action='store_true', help='如果启用，将从path_pg_rtk.txt读取地面真值姿态并更新子图姿态')
    parser.add_argument('--multi-res', action='store_true', help='如果启用，将生成5个不同分辨率的全局地图 (0.1m, 0.2m, 0.4m, 0.8m, 1.6m)')
    
    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.exists(args.folder):
        print(f"错误：文件夹 {args.folder} 不存在")
        sys.exit(1)

    folder_path = args.folder
    
    # 读取地面真值姿态（如果需要）
    gt_poses = read_gt_poses(folder_path) if args.use_gt else None

    if args.multi_res:
        # 多分辨率模式：生成5个不同分辨率的地图
        resolutions = [0.1, 0.2, 0.4, 0.8, 1.6]
        resolution_names = ['01', '02', '04', '08', '16']
        
        print("开启多分辨率模式，将生成5个不同分辨率的全局地图...")
        
        # 首先生成最高分辨率地图（0.1m）
        print(f"\n生成最高分辨率 0.1m 的全局地图...")
        base_save_path = os.path.join(folder_path, 'global_map_01')
        high_res_map, high_res_vis = fuse_submaps(folder_path, base_save_path, args.use_gt, gt_poses, global_res=0.1)
        
        if high_res_map is None:
            print("错误：无法生成基础高分辨率地图")
            return
        
        # 三值化最高分辨率地图
        print(f"三值化前栅格数量: {len(high_res_map.occ_map)}")
        high_res_map = ternarize_map(high_res_map)
        print(f"三值化后栅格数量: {len(high_res_map.occ_map)}")
        
        # 打印三值化后的状态统计
        occupied_count = sum(1 for p in high_res_map.occ_map.values() if p == 1.0)
        free_count = sum(1 for p in high_res_map.occ_map.values() if p == 0.0)
        unknown_count = sum(1 for p in high_res_map.occ_map.values() if p == 0.5)
        print(f"三值化状态统计 - 占用: {occupied_count}, 空闲: {free_count}, 未知: {unknown_count}")
        
        # 重新保存三值化后的高分辨率地图
        if base_save_path:
            save_global_map(high_res_map, base_save_path + '.bin')
            grid = high_res_map.to_matrix()
            vis = visualize_map(grid)
            plt.imsave(base_save_path + '.png', vis, cmap='gray')
        
        # 显示最高分辨率地图
        if high_res_vis is not None:
            plt.figure(figsize=(10, 10))
            grid = high_res_map.to_matrix()
            vis = visualize_map(grid)
            plt.imshow(vis, origin='upper')
            plt.title(f'全局地图 - 分辨率: 0.1m (三值化)')
            plt.axis('off')
            
            info_text = f'栅格数量: {len(high_res_map.occ_map)}, 分辨率: 0.1m'
            plt.text(0.5, 0.02, info_text, transform=plt.gca().transAxes, 
                    ha='center', va='bottom', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.show()
            print(f"分辨率 0.1m 的地图已保存: {base_save_path}.bin 和 {base_save_path}.png")
        
        # 基于高分辨率地图生成其他分辨率的地图
        for i, (res, name) in enumerate(zip(resolutions[1:], resolution_names[1:]), 1):
            print(f"\n通过降采样生成分辨率 {res}m 的全局地图...")
            
            # 降采样生成低分辨率地图
            low_res_map = downsample_map(high_res_map, 0.1, res)
            
            # 构建保存路径
            save_path = os.path.join(folder_path, f'global_map_{name}')
            
            # 保存地图
            save_global_map(low_res_map, save_path + '.bin')
            grid = low_res_map.to_matrix()
            vis = visualize_map(grid)
            plt.imsave(save_path + '.png', vis, cmap='gray')
            
            # 显示当前分辨率的地图
            plt.figure(figsize=(10, 10))
            plt.imshow(vis, origin='upper')
            plt.title(f'全局地图 - 分辨率: {res}m (降采样)')
            plt.axis('off')
            
            # 添加地图信息到标题
            info_text = f'栅格数量: {len(low_res_map.occ_map)}, 分辨率: {res}m'
            plt.text(0.5, 0.02, info_text, transform=plt.gca().transAxes, 
                    ha='center', va='bottom', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.show()
            
            print(f"分辨率 {res}m 的地图已保存: {save_path}.bin 和 {save_path}.png")
    else:
        # 单分辨率模式（原有逻辑）
        save_path = None
        if args.save:
            save_path = os.path.join(folder_path, args.save)

        # 融合子图并保存
        global_map, vis = fuse_submaps(folder_path, save_path, args.use_gt, gt_poses)
        
        if global_map and vis is not None:
            # 三值化地图
            global_map = ternarize_map(global_map)
            
            # 重新保存三值化后的地图
            if save_path:
                save_global_map(global_map, save_path + '.bin')
                grid = global_map.to_matrix()
                vis = visualize_map(grid)
                plt.imsave(save_path + '.png', vis, cmap='gray')
                print(f"三值化全局地图已保存到: {save_path}.bin 和 {save_path}.png")
            
            # 显示三值化后的地图
            plt.figure(figsize=(10, 10))
            grid = global_map.to_matrix()
            vis = visualize_map(grid)
            plt.imshow(vis, origin='upper')
            plt.title('全局地图 (三值化)')
            plt.axis('off')
            
            # 添加地图信息
            info_text = f'栅格数量: {len(global_map.occ_map)}, 分辨率: 0.1m'
            plt.text(0.5, 0.02, info_text, transform=plt.gca().transAxes, 
                    ha='center', va='bottom', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.show()

if __name__ == '__main__':
    main() 