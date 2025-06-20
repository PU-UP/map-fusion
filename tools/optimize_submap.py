#!/usr/bin/env python3
"""
子图位姿优化脚本
功能：
1. 加载已保存的全局地图
2. 随机选择一个子图并添加位姿扰动
3. 使用scan-to-map方式优化位姿
4. 可视化优化前后的结果

多分辨率优化策略（--multi-res选项）：
1. 加载5个不同分辨率的全局地图(0.1m, 0.2m, 0.4m, 0.8m, 1.6m)
2. 对子图生成对应的5个分辨率版本
3. 从低分辨率(1.6m)开始匹配，逐步提升到高分辨率(0.1m)
4. 每层的匹配结果作为下一层的初始值
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from fuse_submaps import (
    load_submap, load_global_map, GridMap, decode_key,
    add_noise_to_pose, downsample_map
)
from particle_filter_matcher import encode_key, match_submap_with_particle_filter
from typing import Optional
import argparse
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from io import StringIO

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def optimize_submap_pose(submap: GridMap, 
                        global_map: GridMap,
                        init_pose: np.ndarray,
                        visualize: bool = False) -> tuple:
    """使用粒子滤波优化子图位姿"""
    print("使用粒子滤波进行优化...")
    
    optimized_pose, final_error = match_submap_with_particle_filter(
        submap, global_map, init_pose,
        n_particles=100,
        n_iterations=200,
        visualize=visualize,
        spread=(1.0, 1.0, np.deg2rad(15.0)),
        submap_res=0.05,
        global_res=0.1
    )
    return optimized_pose, final_error

def compute_matching_error(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1,
                         debug: bool = False) -> float:
    """计算子图与全局地图的匹配误差"""
    total_error = 0.0
    count = 0
    matched_count = 0

    cells = [(key, p) for key, p in submap.occ_map.items() if p > 0.6 or p < 0.4]
    if debug:
        print(f"子图占用/空闲栅格数量: {len(cells)}")

    for key, p_cell in cells:
        sub_i, sub_j = decode_key(key)
        p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        key_glob = encode_key(gi_glob, gj_glob)
        
        p_glob = global_map.occ_map.get(key_glob, 0.5)
        diff = p_glob - p_cell
        if abs(diff) < 0.4:
            matched_count += 1
        total_error += diff * diff

        count += 1
    
    if debug:
        unmatched_count = count - matched_count
        print(f"匹配统计: 总计{count}, 匹配{matched_count}, 不匹配{unmatched_count}")
        print(f"匹配率: {matched_count/max(count,1)*100:.1f}%")
    
    return total_error / max(count, 1)

def angle_diff(a: float, b: float) -> float:
    """Return normalized difference between two angles."""
    diff = a - b
    return np.arctan2(np.sin(diff), np.cos(diff))

def visualize_optimization(global_map: GridMap,
                         submap: GridMap,
                         true_pose: np.ndarray,
                         init_pose: np.ndarray,
                         opt_pose: np.ndarray,
                         save_path: Optional[str] = None,
                         submap_id: int = -1,
                         submap_res: float = 0.05,
                         global_res: float = 0.1):
    """可视化优化结果"""
    global_grid = global_map.to_matrix()
    
    def transform_submap_vis(pose: np.ndarray) -> np.ndarray:
        grid = np.zeros_like(global_grid)
        for key, p_meas in submap.occ_map.items():
            if p_meas < 0.6:
                continue
            sub_i, sub_j = decode_key(key)
            p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
            p_w = pose[:3, :3] @ p_s + pose[:3, 3]
            gi_glob = int(np.round(p_w[0] / global_res))
            gj_glob = int(np.round(p_w[1] / global_res))
            if 0 <= gi_glob - global_map.min_i < grid.shape[0] and \
               0 <= gj_glob - global_map.min_j < grid.shape[1]:
                grid[gi_glob - global_map.min_i, gj_glob - global_map.min_j] = 1
        return grid
    
    true_grid = transform_submap_vis(true_pose)
    init_grid = transform_submap_vis(init_pose)
    opt_grid = transform_submap_vis(opt_pose)
    
    def compute_error(pred_grid, true_grid):
        pred_points = np.sum(pred_grid > 0)
        if pred_points == 0:
            return 1.0
        mismatch = np.sum((pred_grid > 0) & (true_grid == 0))
        return mismatch / pred_points
    
    init_error = compute_error(init_grid, true_grid)
    opt_error = compute_error(opt_grid, true_grid)
    
    # 创建可视化图像
    vis = np.zeros((*global_grid.shape, 3))
    background = np.zeros_like(global_grid)
    background[global_grid <= 0.4] = 0.7
    background[global_grid >= 0.6] = 0.0
    background[np.logical_and(global_grid > 0.4, global_grid < 0.6)] = 0.3
    
    for i in range(3):
        vis[..., i] = background
    
    vis[init_grid > 0] = [0, 0, 1]  # 蓝色
    vis[opt_grid > 0] = [1, 0, 0]   # 红色
    vis[true_grid > 0] = [0, 1, 0]  # 绿色
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(1, 2, width_ratios=[4, 1])
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')
    
    ax.imshow(vis)
    ax.set_title(f'子图ID: {submap_id} 优化结果\n栅格占用不匹配率: 优化前: {init_error*100:.1f}%, 优化后: {opt_error*100:.1f}%')
    
    legend_elements = [ 
        Rectangle((0, 0), 1, 1, fc=[0, 0, 1], label='优化前'),
        Rectangle((0, 0), 1, 1, fc=[1, 0, 0], label='优化后'),
        Rectangle((0, 0), 1, 1, fc=[0, 1, 0], label='真值'),
        Rectangle((0, 0), 1, 1, fc=[0.7, 0.7, 0.7], label='空闲区域'),
        Rectangle((0, 0), 1, 1, fc=[0.3, 0.3, 0.3], label='未知区域'),
        Rectangle((0, 0), 1, 1, fc=[0, 0, 0], label='占用区域'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center left', fontsize=12)
    
    # 计算位姿误差
    init_trans_error = np.linalg.norm(init_pose[:2, 3] - true_pose[:2, 3])
    init_rot_error = abs(angle_diff(
        np.arctan2(init_pose[1, 0], init_pose[0, 0]),
        np.arctan2(true_pose[1, 0], true_pose[0, 0])
    ))
    opt_trans_error = np.linalg.norm(opt_pose[:2, 3] - true_pose[:2, 3])
    opt_rot_error = abs(angle_diff(
        np.arctan2(opt_pose[1, 0], opt_pose[0, 0]),
        np.arctan2(true_pose[1, 0], true_pose[0, 0])
    ))
    
    fig.subplots_adjust(bottom=0.18)
    info_text = (
        f'位姿误差 (相对于真值)：\n'
        f'优化前: {init_trans_error:.3f} 米, {np.rad2deg(init_rot_error):.1f}°    '
        f'优化后: {opt_trans_error:.3f} 米, {np.rad2deg(opt_rot_error):.1f}°'
    )
    fig.text(0.5, 0.05, info_text, ha='center', va='center', fontsize=13, 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path)
        print(f"优化结果已保存到: {save_path}")
    else:
        plt.show()

def load_gt_pose_from_file(gt_file_path: str, timestamp: float) -> Optional[np.ndarray]:
    """从path_pg_rtk.txt文件中加载指定时间戳的真值位姿"""
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            file_ts_float = float(parts[0])
            
            if math.isclose(file_ts_float, timestamp, rel_tol=0, abs_tol=0.01):
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [x, y, z]
                
                return transform_matrix
    
    return None

def load_multi_resolution_global_maps(folder_path: str) -> dict:
    """加载多分辨率全局地图"""
    resolutions = [0.1, 0.2, 0.4, 0.8, 1.6]
    resolution_names = ['01', '02', '04', '08', '16']
    global_maps = {}
    
    print("加载多分辨率全局地图...")
    
    for res, name in zip(resolutions, resolution_names):
        map_path = os.path.join(folder_path, f'global_map_{name}.bin')
        if os.path.exists(map_path):
            try:
                global_map = load_global_map(map_path)
                global_maps[res] = global_map
                print(f"已加载分辨率 {res}m 的全局地图，栅格数量: {len(global_map.occ_map)}")
            except Exception as e:
                print(f"警告：无法加载分辨率 {res}m 的全局地图: {e}")
        else:
            print(f"警告：未找到分辨率 {res}m 的全局地图文件: {map_path}")
    
    if not global_maps:
        raise RuntimeError("未找到任何多分辨率全局地图文件")
    
    return global_maps

def generate_multi_resolution_submap(submap: GridMap) -> dict:
    """生成多分辨率子图"""
    resolutions = [0.1, 0.2, 0.4, 0.8, 1.6]
    submaps = {}
    
    print("生成多分辨率子图...")
    
    # 临时重定向输出保持界面清洁
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        base_submap = downsample_map(submap, 0.05, 0.1)
        submaps[0.1] = base_submap

        for res in resolutions[1:]:
            downsampled = downsample_map(base_submap, 0.1, res)
            submaps[res] = downsampled
    finally:
        sys.stdout = old_stdout
    
    print("多分辨率子图生成完成:")
    for res in resolutions:
        if res in submaps:
            total_cells = len(submaps[res].occ_map)
            occupied_cells = sum(1 for p in submaps[res].occ_map.values() if p > 0.6)
            free_cells = sum(1 for p in submaps[res].occ_map.values() if p < 0.4)
            unknown_cells = total_cells - occupied_cells - free_cells
            print(f"  分辨率 {res}m: 总计{total_cells} (占用:{occupied_cells}, 空闲:{free_cells}, 未知:{unknown_cells})")
    
    return submaps

def multi_resolution_optimization(multi_res_submaps: dict,
                                multi_res_global_maps: dict,
                                init_pose: np.ndarray,
                                true_pose: Optional[np.ndarray] = None,
                                visualize: bool = False) -> tuple:
    """多分辨率位姿优化"""
    resolutions = [1.6, 0.8, 0.4, 0.2, 0.1]
    current_pose = init_pose.copy()
    
    print("开始多分辨率优化...")
    
    # 参数配置
    param_config = {
        1.6: {'spread': (4.0, 4.0, np.deg2rad(30.0)), 'particles': 80, 'iterations': 80},
        0.8: {'spread': (2.0, 2.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 100},
        0.4: {'spread': (1.0, 1.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 120},
        0.2: {'spread': (0.5, 0.5, np.deg2rad(10.0)), 'particles': 100, 'iterations': 150},
        0.1: {'spread': (0.3, 0.3, np.deg2rad(5.0)), 'particles': 120, 'iterations': 200}
    }
    
    successful_optimizations = 0
    
    for res in resolutions:
        if res not in multi_res_submaps or res not in multi_res_global_maps:
            continue
        
        print(f"\n===== 分辨率 {res}m 优化 =====")
        
        submap = multi_res_submaps[res]
        global_map = multi_res_global_maps[res]
        
        submap_occupied = sum(1 for p in submap.occ_map.values() if p > 0.6)
        submap_occ_ratio = submap_occupied / max(len(submap.occ_map), 1)
        
        if submap_occ_ratio > 0.85 or submap_occ_ratio < 0.1 or submap_occupied == 0:
            print(f"跳过分辨率 {res}m (占用比例不合理或无占用栅格)")
            continue

        config = param_config[res]
        spread_x, spread_y, spread_theta = config['spread']
        n_particles = config['particles']
        n_iterations = config['iterations']
        
        initial_error = compute_matching_error(submap, global_map, current_pose, 
                                             submap_res=res, global_res=res, debug=True)
        
        try:
            optimized_pose, final_error = match_submap_with_particle_filter(
                submap, global_map, current_pose,
                n_particles=n_particles,
                n_iterations=n_iterations,
                visualize=visualize and res <= 0.2,  # 只在高分辨率层显示可视化
                spread=(spread_x, spread_y, spread_theta),
                submap_res=res,
                global_res=res
            )
            
            pose_diff = np.linalg.norm(optimized_pose[:2, 3] - current_pose[:2, 3])
            error_improvement = (initial_error - final_error) / max(initial_error, 1e-6)
            
            print(f"误差变化: {initial_error:.6f} → {final_error:.6f} (改善: {error_improvement*100:.1f}%)")
            
            # 智能位姿更新策略
            if (error_improvement > 0.05 and pose_diff > 0.02) or \
               (final_error < initial_error and pose_diff > 0.01) or \
               (res >= 0.8 and pose_diff > 0.02) or \
               (res <= 0.2 and error_improvement > 0.01):
                current_pose = optimized_pose.copy()
                successful_optimizations += 1
                print("✓ 更新位姿")
            else:
                print("× 保持原位姿")
                
        except Exception as e:
            print(f"分辨率 {res}m 优化失败: {e}")
    
    # 计算最终误差
    final_error = 0.0
    if 0.1 in multi_res_submaps and 0.1 in multi_res_global_maps:
        final_error = compute_matching_error(
            multi_res_submaps[0.1], 
            multi_res_global_maps[0.1], 
            current_pose,
            submap_res=0.1,
            global_res=0.1
        )
    
    print(f"\n========== 多分辨率优化总结 ==========")
    print(f"成功优化层数: {successful_optimizations}")
    print(f"最终误差: {final_error:.6f}")
    
    return current_pose, final_error

def main():
    parser = argparse.ArgumentParser(description="子图位姿优化脚本")
    parser.add_argument("folder_path", type=str, help="包含子图和全局地图的文件夹路径")
    parser.add_argument("--plot", action="store_true", help="显示粒子滤波中间过程的可视化")
    parser.add_argument("--use-gt", action="store_true", help="使用path_pg_rtk.txt中的真值作为初始位姿和参考真值")
    parser.add_argument("--submap", type=int, help="指定要优化的子图ID，默认为随机选择")
    parser.add_argument("--multi-res", action="store_true",
                       help="使用多分辨率匹配策略：从低分辨率到高分辨率逐层优化")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"错误：文件夹 {args.folder_path} 不存在")
        return

    # 1. 加载全局地图
    if args.multi_res:
        try:
            multi_res_global_maps = load_multi_resolution_global_maps(args.folder_path)
            global_map = multi_res_global_maps[0.1]
        except Exception as e:
            print(f"多分辨率地图加载失败: {e}")
            print("回退到单分辨率模式...")
            args.multi_res = False
            global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
    else:
        global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
    
    # 2. 选择子图
    submap_files = [f for f in os.listdir(args.folder_path) 
                    if f.startswith('submap_') and f.endswith('.bin')]
    if not submap_files:
        print("错误：没有找到子图文件")
        return
    
    if args.submap is not None:
        target_file = f'submap_{args.submap}.bin'
        if target_file not in submap_files:
            print(f"错误：未找到指定的子图文件: {target_file}")
            sys.exit(1)
        submap_id = args.submap
    else:
        target_file = np.random.choice(submap_files)
        submap_id = int(target_file.split('_')[1].split('.')[0])
    
    print(f"选择子图 {submap_id} 进行优化")
    
    # 3. 加载子图
    submap_path = os.path.join(args.folder_path, target_file)
    _, ts, loaded_true_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(submap_path)
    
    submap = GridMap()
    for key, prob in occ_map.items():
        gi, gj = decode_key(key)
        submap.update_occ(gi, gj, prob)
    
    # 4. 设置真值和初始位姿
    if args.use_gt:
        gt_file_path = os.path.join(args.folder_path, 'path_pg_rtk.txt')
        if not os.path.exists(gt_file_path):
            print(f"错误：未找到真值文件: {gt_file_path}")
            sys.exit(1)
        
        gt_pose = load_gt_pose_from_file(gt_file_path, ts)
        if gt_pose is None:
            print(f"错误：未找到时间戳 {ts} 对应的真值位姿")
            sys.exit(1)
        
        true_pose = gt_pose
        init_pose = loaded_true_pose
        print("使用path_pg_rtk.txt中的真值位姿")
    else:
        true_pose = loaded_true_pose
        init_pose = add_noise_to_pose(loaded_true_pose, 0.5, 10.0)
        print("已添加噪声作为初始位姿")
    
    # 5. 优化位姿
    if args.multi_res:
        print("使用多分辨率优化策略...")
        multi_res_submaps = generate_multi_resolution_submap(submap)
        opt_pose, error = multi_resolution_optimization(
            multi_res_submaps, 
            multi_res_global_maps, 
            init_pose,
            true_pose,
            visualize=args.plot
        )
        final_submap = multi_res_submaps[0.1] if 0.1 in multi_res_submaps else submap
        visualize_optimization(global_map, final_submap, true_pose, init_pose, opt_pose, 
                             None, submap_id, submap_res=0.1, global_res=0.1)
    else:
        print("使用单分辨率优化...")
        opt_pose, error = optimize_submap_pose(submap, global_map, init_pose, visualize=args.plot)
        visualize_optimization(global_map, submap, true_pose, init_pose, opt_pose, 
                             None, submap_id, submap_res=0.05, global_res=0.1)
    
    print(f"优化完成，最终误差: {error:.6f}")

if __name__ == '__main__':
    main() 
