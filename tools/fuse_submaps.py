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
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def decode_key(key: int) -> tuple:
    """完全匹配C++实现的key解码"""
    gi = np.int32(key >> 32).item()
    low = key & 0xFFFFFFFF
    if low >= (1 << 31):
        gj = int(low - (1 << 32))
    else:
        gj = int(low)
    return int(gi), int(gj)

def encode_key(gi: int, gj: int) -> int:
    """完全匹配C++实现的key编码"""
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
        return 0.0
    elif p > 0.6:
        return 1.0
    else:
        return 0.5

class GridMap:
    """完全匹配C++的GridMap类"""
    def __init__(self):
        self.occ_map = {}
        self.min_i = 0
        self.max_i = 0
        self.min_j = 0
        self.max_j = 0
        self.initialized = False
    
    def update_bounds(self, i: int, j: int):
        """更新地图边界"""
        if not self.initialized:
            self.min_i = self.max_i = i
            self.min_j = self.max_j = j
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
        
        p_old = 0.5
        if key in self.occ_map:
            p_old = np.clip(self.occ_map[key], eps, 1.0 - eps)
        
        p_meas = np.clip(p_meas, eps, 1.0 - eps)
        
        l_old = log_odds(p_old)
        l_meas = log_odds(p_meas)
        l_new = clamp_log_odds(l_old + l_meas)
        
        p_new = inv_log_odds(l_new)
        
        was_present = key in self.occ_map
        self.occ_map[key] = p_new
        
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
            
    return submap_id, ts, first_pose, min_i, max_i, min_j, max_j, occ_map

def read_gt_poses(folder_path: str) -> dict:
    """从path_pg_rtk.txt读取地面真值姿态"""
    gt_poses = {}
    gt_file_path = os.path.join(folder_path, 'path_pg_rtk.txt')
    if not os.path.exists(gt_file_path):
        print(f"警告：地面真值文件 {gt_file_path} 不存在。")
        return gt_poses

    print(f"正在读取地面真值文件: {gt_file_path}")
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [tx, ty, tz]
                
                gt_poses[timestamp] = transform_matrix
    print(f"已加载 {len(gt_poses)} 个地面真值姿态。")
    return gt_poses

def visualize_map(occ_map: np.ndarray) -> np.ndarray:
    """可视化占用栅格地图"""
    vis = np.zeros((*occ_map.shape, 3), dtype=np.uint8)
    
    # 空闲区域显示为白色
    free_mask = np.isclose(occ_map, 0.0, rtol=1e-5)
    vis[free_mask] = [255, 255, 255]
    
    # 占用区域显示为黑色
    occ_mask = np.isclose(occ_map, 1.0, rtol=1e-5)
    vis[occ_mask] = [0, 0, 0]
    
    # 未知区域显示为灰色
    unknown_mask = ~(free_mask | occ_mask)
    vis[unknown_mask] = [128, 128, 128]
    
    return vis

def fuse_submaps(folder_path: str, save_path: Optional[str] = None, 
                use_gt: bool = False, gt_poses: Optional[dict] = None, 
                global_res: float = 0.1):
    """融合所有子图"""
    global_map = GridMap()
    submap_res = 0.05
    
    # 获取所有子图文件
    submap_files = glob.glob(os.path.join(folder_path, 'submap_*.bin'))
    if not submap_files:
        print(f"错误：在{folder_path}中没有找到子图文件")
        return None, None
    
    # 创建ID到文件的映射
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
    for submap_id in range(max(id_to_file.keys()) + 1):
        if submap_id not in id_to_file:
            print(f"警告：缺少submap_{submap_id}")
            continue
        
        bin_path = id_to_file[submap_id]
        print(f"\nProcessing submap_{submap_id}...")
        
        # 加载子图
        _, ts, first_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(bin_path)
        
        if use_gt and gt_poses and ts in gt_poses:
            print(f"使用地面真值姿态更新submap_{submap_id}的姿态")
            first_pose = gt_poses[ts]
        elif use_gt and ts not in gt_poses:
            print(f"警告：submap_{submap_id}的时间戳未在真值文件中找到，跳过")
            continue

        # 融合子图到全局地图
        for key, p_meas in occ_map.items():
            sub_i, sub_j = decode_key(key)
            p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
            p_w = first_pose[:3, :3] @ p_s + first_pose[:3, 3]
            
            gi_glob = int(np.floor(p_w[0] / global_res))
            gj_glob = int(np.floor(p_w[1] / global_res))
            
            global_map.update_occ(gi_glob, gj_glob, p_meas)
    
    # 打印全局地图信息
    print(f"\n全局地图信息 (分辨率: {global_res}m):")
    print(f"栅格数量: {len(global_map.occ_map)} cells")
    print(f"边界: i[{global_map.min_i}, {global_map.max_i}], j[{global_map.min_j}, {global_map.max_j}]")
    
    # 转换为矩阵并可视化
    grid = global_map.to_matrix()
    vis = visualize_map(grid)

    # 保存结果
    if save_path:
        save_global_map(global_map, save_path + '.bin')
        plt.imsave(save_path + '.png', vis, cmap='gray')
        print(f"全局地图已保存到: {save_path}.bin 和 {save_path}.png")
    
    return global_map, vis

def save_global_map(global_map: GridMap, save_path: str):
    """保存全局地图为二进制文件"""
    with open(save_path, 'wb') as f:
        f.write(struct.pack('i' * 4, global_map.min_i, global_map.max_i, 
                          global_map.min_j, global_map.max_j))
        f.write(struct.pack('Q', len(global_map.occ_map)))
        
        for key, prob in global_map.occ_map.items():
            f.write(struct.pack('q d', key, prob))

def load_global_map(load_path: str) -> GridMap:
    """加载全局地图"""
    global_map = GridMap()
    
    with open(load_path, 'rb') as f:
        raw = f.read(4 * 4)
        min_i, max_i, min_j, max_j = struct.unpack('i' * 4, raw)
        global_map.min_i = min_i
        global_map.max_i = max_i
        global_map.min_j = min_j
        global_map.max_j = max_j
        global_map.initialized = True
        
        raw = f.read(8)
        map_size = struct.unpack('Q', raw)[0]
        
        for _ in range(map_size):
            raw = f.read(16)
            key, prob = struct.unpack('q d', raw)
            global_map.occ_map[key] = prob
    
    return global_map

def downsample_map(high_res_map: GridMap, high_res: float, low_res: float) -> GridMap:
    """将高分辨率地图降采样为低分辨率地图"""
    low_res_map = GridMap()
    
    print(f"降采样：从 {high_res}m 到 {low_res}m")
    
    # 三值化高分辨率地图
    ternarized_map = {}
    for key, prob in high_res_map.occ_map.items():
        ternarized_map[key] = ternarize_probability(prob)
    
    # 按低分辨率栅格分组
    low_res_groups = {}
    for key, ternary_prob in ternarized_map.items():
        gi, gj = decode_key(key)
        x, y = gi * high_res, gj * high_res
        
        low_gi = int(np.floor(x / low_res))
        low_gj = int(np.floor(y / low_res))
        low_key = encode_key(low_gi, low_gj)
        
        if low_key not in low_res_groups:
            low_res_groups[low_key] = []
        low_res_groups[low_key].append(ternary_prob)
    
    # 应用融合规则
    for low_key, probs in low_res_groups.items():
        occupied_count = sum(1 for p in probs if p == 1.0)
        free_count = sum(1 for p in probs if p == 0.0)
        
        if occupied_count > 0:
            final_prob = 1.0
        elif free_count > 0:
            final_prob = 0.0
        else:
            final_prob = 0.5
        
        low_gi, low_gj = decode_key(low_key)
        low_res_map.set_occ_direct(low_gi, low_gj, final_prob)
    
    print(f"降采样完成，栅格数量: {len(low_res_map.occ_map)}")
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

def add_noise_to_pose(pose: np.ndarray, translation_noise: float = 0.1, 
                     rotation_noise_deg: float = 5.0) -> np.ndarray:
    """给位姿添加噪声"""
    noisy_pose = pose.copy()
    
    # 添加平移噪声
    noisy_pose[:3, 3] += np.random.normal(0, translation_noise, 3)
    
    # 添加旋转噪声
    theta = np.random.normal(0, np.deg2rad(rotation_noise_deg))
    c, s = np.cos(theta), np.sin(theta)
    R_noise = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    noisy_pose[:3, :3] = R_noise @ pose[:3, :3]
    
    return noisy_pose

def display_map_with_info(grid_map: GridMap, title: str, resolution: float):
    """显示地图并添加信息"""
    plt.figure(figsize=(10, 10))
    grid = grid_map.to_matrix()
    vis = visualize_map(grid)
    plt.imshow(vis, origin='upper')
    plt.title(title)
    plt.axis('off')
    
    info_text = f'栅格数量: {len(grid_map.occ_map)}, 分辨率: {resolution}m'
    plt.text(0.5, 0.02, info_text, transform=plt.gca().transAxes, 
            ha='center', va='bottom', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='子图融合工具')
    parser.add_argument('--folder', type=str, required=True, help='包含子图bin文件的文件夹路径')
    parser.add_argument('--save', type=str, default='global_map', help='保存文件名')
    parser.add_argument('--use-gt', action='store_true', help='使用地面真值姿态')
    parser.add_argument('--multi-res', action='store_true', help='生成多分辨率地图')
    
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"错误：文件夹 {args.folder} 不存在")
        sys.exit(1)

    folder_path = args.folder
    gt_poses = read_gt_poses(folder_path) if args.use_gt else None

    if args.multi_res:
        resolutions = [0.1, 0.2, 0.4, 0.8, 1.6]
        resolution_names = ['01', '02', '04', '08', '16']
        
        print("多分辨率模式：生成5个不同分辨率的全局地图...")
        
        # 生成最高分辨率地图
        base_save_path = os.path.join(folder_path, 'global_map_01')
        high_res_map, _ = fuse_submaps(folder_path, base_save_path, args.use_gt, gt_poses, 0.1)
        
        if high_res_map is None:
            print("错误：无法生成基础高分辨率地图")
            return
        
        # 三值化并保存
        high_res_map = ternarize_map(high_res_map)
        save_global_map(high_res_map, base_save_path + '.bin')
        grid = high_res_map.to_matrix()
        vis = visualize_map(grid)
        plt.imsave(base_save_path + '.png', vis, cmap='gray')
        
        display_map_with_info(high_res_map, '全局地图 - 分辨率: 0.1m (三值化)', 0.1)
        
        # 生成其他分辨率
        for res, name in zip(resolutions[1:], resolution_names[1:]):
            print(f"\n生成分辨率 {res}m 的地图...")
            low_res_map = downsample_map(high_res_map, 0.1, res)
            
            save_path = os.path.join(folder_path, f'global_map_{name}')
            save_global_map(low_res_map, save_path + '.bin')
            grid = low_res_map.to_matrix()
            vis = visualize_map(grid)
            plt.imsave(save_path + '.png', vis, cmap='gray')
            
            display_map_with_info(low_res_map, f'全局地图 - 分辨率: {res}m', res)
    else:
        # 单分辨率模式
        save_path = os.path.join(folder_path, args.save) if args.save else None
        global_map, _ = fuse_submaps(folder_path, save_path, args.use_gt, gt_poses)
        
        if global_map:
            global_map = ternarize_map(global_map)
            
            if save_path:
                save_global_map(global_map, save_path + '.bin')
                grid = global_map.to_matrix()
                vis = visualize_map(grid)
                plt.imsave(save_path + '.png', vis, cmap='gray')
            
            display_map_with_info(global_map, '全局地图 (三值化)', 0.1)

if __name__ == '__main__':
    main() 