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

似然地图优化策略（--likelihood选项）：
1. 使用原始概率地图而非三值化地图进行匹配
2. 采用梯度下降优化位姿参数
3. 多层分辨率策略：从粗到细逐步优化
4. 目标：提高匹配精度并降低计算资源消耗

批量处理功能（--submap -1选项）：
1. 自动处理文件夹中的所有子图
2. 将所有结果绘制在一张综合图中
3. 提供详细的统计信息和误差分析
4. 支持所有优化策略（粒子滤波、似然优化、多分辨率）
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from fuse_submaps import (
    load_submap, load_global_map, GridMap, decode_key,
    add_noise_to_pose, downsample_map, ternarize_map
)
from particle_filter_matcher import encode_key, match_submap_with_particle_filter
from likelihood_optimizer import (
    multi_resolution_likelihood_optimization,
    match_submap_with_likelihood
)
from typing import Optional
import argparse
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from io import StringIO
import time

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def optimize_submap_pose(submap: GridMap, 
                        global_map: GridMap,
                        init_pose: np.ndarray,
                        visualize: bool = False,
                        debug: bool = False) -> tuple:
    """使用粒子滤波优化子图位姿"""
    print("使用粒子滤波进行优化...")
    
    # 开始计时
    start_time = time.time()
    
    optimized_pose, final_error = match_submap_with_particle_filter(
        submap, global_map, init_pose,
        n_particles=100,
        n_iterations=200,
        visualize=visualize,
        spread=(1.0, 1.0, np.deg2rad(15.0)),
        submap_res=0.05,
        global_res=0.1,
        debug=debug
    )
    
    # 结束计时
    end_time = time.time()
    total_time = end_time - start_time
    
    if debug:
        print(f"单分辨率优化总耗时: {total_time:.4f}s")
    else:
        print(f"优化耗时: {total_time:.4f}s")
    
    return optimized_pose, final_error

def compute_matching_error(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1,
                         debug: bool = False) -> float:
    """计算子图与全局地图的匹配误差"""
    total_error = 0
    count = 0
    matched_count = 0
    
    occupied_cells = [(key, p) for key, p in submap.occ_map.items() if p > 0.6]
    if debug:
        print(f"子图占用栅格数量: {len(occupied_cells)}")
    
    for key, _ in occupied_cells:
        sub_i, sub_j = decode_key(key)
        p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        key_glob = encode_key(gi_glob, gj_glob)
        
        if key_glob in global_map.occ_map and global_map.occ_map[key_glob] > 0.6:
            matched_count += 1
        else:
            total_error += 1
        
        count += 1
    
    if debug:
        unmatched_count = count - matched_count
        print(f"匹配统计: 总计{count}, 匹配{matched_count}, 不匹配{unmatched_count}")
        print(f"匹配率: {matched_count/max(count,1)*100:.1f}%")
    
    return total_error / max(count, 1)

def format_improvement(before, after, is_rate=False):
    """格式化改善率，is_rate=True表示越高越好"""
    if abs(before) < 1e-9:
        return "N/A"
    
    improvement = (after - before) / abs(before) * 100
    if not is_rate: # Error, the lower the better
        improvement = -improvement
        
    if improvement > 0.1:
        return f"↑ {improvement:.1f}%"
    elif improvement < -0.1:
        return f"↓ {-improvement:.1f}%"
    else:
        return "→ 0.0%"

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

def load_multi_resolution_global_maps(folder_path: str, num_layers: int = 5) -> dict:
    """加载多分辨率全局地图"""
    # 所有可能的分辨率，从粗到细
    all_resolutions = [1.6, 0.8, 0.4, 0.2, 0.1]
    all_resolution_names = ['16', '08', '04', '02', '01']
    
    # 根据指定层数选择分辨率
    if num_layers > len(all_resolutions) or num_layers <= 0:
        print(f"警告：请求的层数 {num_layers} 无效，将使用所有 {len(all_resolutions)} 个可用层数")
        num_layers = len(all_resolutions)
    
    # 选择N个最高分辨率的地图层。例如 num_layers=2 会选择 [0.2, 0.1]
    resolutions = all_resolutions[-num_layers:]
    resolution_names = all_resolution_names[-num_layers:]
    global_maps = {}
    
    print(f"加载 {num_layers} 层多分辨率全局地图...")
    
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

def generate_multi_resolution_submap(submap: GridMap, use_likelihood: bool = False, num_layers: int = 5) -> dict:
    """生成多分辨率子图"""
    # 所有可能的分辨率，从粗到细
    all_resolutions = [1.6, 0.8, 0.4, 0.2, 0.1]
    
    # 根据指定层数选择分辨率
    if num_layers > len(all_resolutions) or num_layers <= 0:
        print(f"警告：请求的层数 {num_layers} 无效，将使用所有 {len(all_resolutions)} 个可用层数")
        num_layers = len(all_resolutions)
    
    # 选择N个最高分辨率的地图层。例如 num_layers=2 会选择 [0.2, 0.1]
    resolutions = all_resolutions[-num_layers:]
    submaps = {}
    
    print(f"生成 {num_layers} 层多分辨率子图...")
    
    # 临时重定向输出保持界面清洁
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        base_submap = downsample_map(submap, 0.05, 0.1)
        if not use_likelihood:
            base_submap = ternarize_map(base_submap)
        submaps[0.1] = base_submap
        
        for res in resolutions[1:]:
            downsampled = downsample_map(base_submap, 0.1, res)
            if not use_likelihood:
                downsampled = ternarize_map(downsampled)
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
                                visualize: bool = False,
                                debug: bool = False) -> tuple:
    """多分辨率位姿优化"""
    # 获取可用的分辨率列表，从粗到细排序
    available_resolutions = sorted(multi_res_submaps.keys(), reverse=True)
    current_pose = init_pose.copy()
    
    print("开始多分辨率优化...")
    
    # 参数配置 - 根据可用分辨率动态调整
    param_config = {
        1.6: {'spread': (4.0, 4.0, np.deg2rad(30.0)), 'particles': 80, 'iterations': 80},
        0.8: {'spread': (2.0, 2.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 100},
        0.4: {'spread': (1.0, 1.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 120},
        0.2: {'spread': (0.5, 0.5, np.deg2rad(10.0)), 'particles': 100, 'iterations': 150},
        0.1: {'spread': (0.3, 0.3, np.deg2rad(5.0)), 'particles': 120, 'iterations': 200}
    }
    
    successful_optimizations = 0
    resolution_times = {}  # 记录每个分辨率的耗时
    
    for res in available_resolutions:
        if res not in multi_res_global_maps:
            continue
        
        print(f"\n===== 分辨率 {res}m 优化 =====")
        
        # 开始计时
        res_start_time = time.time()
        
        submap = multi_res_submaps[res]
        global_map = multi_res_global_maps[res]
        
        submap_occupied = sum(1 for p in submap.occ_map.values() if p > 0.6)
        submap_occ_ratio = submap_occupied / max(len(submap.occ_map), 1)
        
        if submap_occ_ratio > 0.85 or submap_occ_ratio < 0.1 or submap_occupied == 0:
            print(f"跳过分辨率 {res}m (占用比例不合理或无占用栅格)")
            continue

        # 获取参数配置，如果不存在则使用默认值
        config = param_config.get(res, {'spread': (1.0, 1.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 150})
        spread_x, spread_y, spread_theta = config['spread']
        n_particles = config['particles']
        n_iterations = config['iterations']
        
        initial_error = compute_matching_error(submap, global_map, current_pose, 
                                             submap_res=res, global_res=res, debug=debug)
        
        try:
            optimized_pose, final_error = match_submap_with_particle_filter(
                submap, global_map, current_pose,
                n_particles=n_particles,
                n_iterations=n_iterations,
                visualize=visualize and res <= 0.2,  # 只在高分辨率层显示可视化
                spread=(spread_x, spread_y, spread_theta),
                submap_res=res,
                global_res=res,
                debug=debug
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
        
        # 结束计时并记录
        res_end_time = time.time()
        res_time = res_end_time - res_start_time
        resolution_times[res] = res_time
        
        if debug:
            print(f"分辨率 {res}m 优化耗时: {res_time:.4f}s")
    
    # 计算最终误差 - 使用最高分辨率
    final_error = 0.0
    highest_res = min(multi_res_submaps.keys()) if multi_res_submaps else 0.1
    if highest_res in multi_res_submaps and highest_res in multi_res_global_maps:
        final_error = compute_matching_error(
            multi_res_submaps[highest_res], 
            multi_res_global_maps[highest_res], 
            current_pose,
            submap_res=highest_res,
            global_res=highest_res
        )
    
    print(f"\n========== 多分辨率优化总结 ==========")
    print(f"成功优化层数: {successful_optimizations}")
    print(f"最终误差: {final_error:.6f}")
    
    # 显示各分辨率层耗时统计
    if resolution_times:
        total_time = sum(resolution_times.values())
        if debug:
            print(f"\n========== 各分辨率层耗时统计 ==========")
            for res in available_resolutions:
                if res in resolution_times:
                    time_percent = (resolution_times[res] / total_time) * 100
                    print(f"分辨率 {res}m: {resolution_times[res]:.4f}s ({time_percent:.1f}%)")
        print(f"总耗时: {total_time:.4f}s")
    
    return current_pose, final_error

def visualize_all_optimization_results(global_map: GridMap,
                                     all_results: list,
                                     save_path: Optional[str] = None,
                                     submap_res: float = 0.05,
                                     global_res: float = 0.1):
    """可视化所有子图的优化结果"""
    global_grid = global_map.to_matrix()
    
    def transform_submap_vis(submap: GridMap, pose: np.ndarray) -> np.ndarray:
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
    
    # --- 图形 1: 主图 ---
    vis = np.zeros((*global_grid.shape, 3))
    background = np.zeros_like(global_grid)
    background[global_grid <= 0.4] = 0.7
    background[global_grid >= 0.6] = 0.0
    background[np.logical_and(global_grid > 0.4, global_grid < 0.6)] = 0.3
    for i in range(3):
        vis[..., i] = background

    for result in all_results:
        submap = result['submap_for_vis']
        true_pose = result['true_pose']
        true_grid = transform_submap_vis(submap, true_pose)
        vis[true_grid > 0] = [0, 1, 0]
    for result in all_results:
        submap = result['submap_for_vis']
        opt_pose = result['opt_pose']
        opt_grid = transform_submap_vis(submap, opt_pose)
        vis[opt_grid > 0] = [1, 0, 0]
    for result in all_results:
        submap = result['submap_for_vis']
        init_pose = result['init_pose']
        init_grid = transform_submap_vis(submap, init_pose)
        vis[init_grid > 0] = [0, 0, 1]

    fig1, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [4, 1]})
    ax_main.imshow(vis)
    ax_main.set_title(f'所有子图优化结果 (共{len(all_results)}个子图)', fontsize=16)
    ax_main.axis('off')
    
    ax_legend.axis('off')
    legend_elements = [ 
        Rectangle((0, 0), 1, 1, fc=[0, 0, 1], label='初始位置'),
        Rectangle((0, 0), 1, 1, fc=[1, 0, 0], label='优化后'),
        Rectangle((0, 0), 1, 1, fc=[0, 1, 0], label='真值'),
        Rectangle((0, 0), 1, 1, fc=[0.7, 0.7, 0.7], label='空闲区域'),
        Rectangle((0, 0), 1, 1, fc=[0.3, 0.3, 0.3], label='未知区域'),
        Rectangle((0, 0), 1, 1, fc=[0, 0, 0], label='占用区域'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center left', fontsize=12)
    fig1.tight_layout()

    # --- 图形 2: 误差统计 ---
    # 准备表格数据
    table_data = [['子图ID', '优化前位姿误差\n(m/°)', '优化后位姿误差\n(m/°)', '位姿改善率', '栅格匹配率', '匹配耗时(s)']]
    
    for res in all_results:
        init_pose_str = f"{res['init_pos_error']:.2f} / {res['init_rot_error']:.1f}"
        opt_pose_str = f"{res['opt_pos_error']:.2f} / {res['opt_rot_error']:.1f}"
        
        # Match rate: 1 - match_error
        init_match_rate = 1.0 - res['init_match_error']
        final_match_rate = 1.0 - res['final_match_error']
        match_rate_str = f"{init_match_rate:.1%} → {final_match_rate:.1%}"
        
        pose_improvement_str = format_improvement(res['init_pos_error'], res['opt_pos_error'])
        
        table_data.append([
            f"{res['submap_id']}",
            init_pose_str,
            opt_pose_str,
            pose_improvement_str,
            match_rate_str,
            f"{res['matching_time']:.2f}"
        ])

    # 根据结果数量动态计算图形高度
    num_rows = len(all_results) + 1 # +1 for header
    table_height_inch = num_rows * 0.5
    stats_height_inch = 5
    fig2_height = table_height_inch + stats_height_inch

    fig2 = plt.figure(figsize=(18, max(10, fig2_height)))
    gs2 = GridSpec(2, 1, height_ratios=[table_height_inch, stats_height_inch], hspace=0.3)
    
    ax_table = fig2.add_subplot(gs2[0])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 位姿误差统计
    ax_pose_stats = fig2.add_subplot(gs2[1])
    ax_pose_stats.axis('off')
    ax_pose_stats.set_title('位姿与耗时统计', fontsize=16)

    # 提取数据
    init_pos_errors = [r['init_pos_error'] for r in all_results]
    opt_pos_errors = [r['opt_pos_error'] for r in all_results]
    init_rot_errors = [r['init_rot_error'] for r in all_results]
    opt_rot_errors = [r['opt_rot_error'] for r in all_results]
    match_times = [r['matching_time'] for r in all_results]
    
    pose_improvement_count = sum(1 for r in all_results if r['opt_pos_error'] < r['init_pos_error'])
    
    total_batch_time = sum(match_times)
    avg_match_time = total_batch_time / len(all_results) if all_results else 0

    pose_stats_text = (
        f'位姿误差统计 (相对于真值)\n\n'
        f'位置误差: 初始平均 {np.mean(init_pos_errors):.3f}±{np.std(init_pos_errors):.3f} m  →  '
        f'优化后平均 {np.mean(opt_pos_errors):.3f}±{np.std(opt_pos_errors):.3f} m\n\n'
        f'角度误差: 初始平均 {np.mean(init_rot_errors):.1f}±{np.std(init_rot_errors):.1f}°  →  '
        f'优化后平均 {np.mean(opt_rot_errors):.1f}±{np.std(opt_rot_errors):.1f}°\n\n'
        f'位姿改善情况: {pose_improvement_count}/{len(all_results)} 个子图的位姿得到改善 ({pose_improvement_count/len(all_results)*100:.1f}%)\n\n'
        f'耗时统计: 平均匹配耗时 {avg_match_time:.2f}s, 总耗时 {total_batch_time:.2f}s'
    )
    ax_pose_stats.text(0.5, 0.45, pose_stats_text, ha='center', va='center', fontsize=12,
                      bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='none'),
                      transform=ax_pose_stats.transAxes)
    
    fig2.suptitle('批量优化结果统计', fontsize=20)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        main_save_path = save_path.replace('.png', '_main.png')
        fig1.savefig(main_save_path, dpi=300, bbox_inches='tight')
        print(f"主图已保存到: {main_save_path}")
        stats_save_path = save_path.replace('.png', '_stats.png')
        fig2.savefig(stats_save_path, dpi=300, bbox_inches='tight')
        print(f"统计图已保存到: {stats_save_path}")
        plt.close(fig1)
        plt.close(fig2)
    else:
        print("显示两个窗口：主图和误差统计。关闭窗口后程序将继续。")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="子图位姿优化脚本")
    parser.add_argument("folder_path", type=str, help="包含子图和全局地图的文件夹路径")
    parser.add_argument("--plot", action="store_true", help="显示粒子滤波中间过程的可视化")
    parser.add_argument("--use-gt", action="store_true", help="使用path_pg_rtk.txt中的真值作为初始位姿和参考真值")
    parser.add_argument("--submap", type=int, help="指定要优化的子图ID，默认为随机选择。使用-1处理所有子图")
    parser.add_argument("--multi-res", nargs='?', const=5, type=int, default=None,
                       help="使用多分辨率匹配策略：从低分辨率到高分辨率逐层优化。可指定层数(默认5层，包含0.1m分辨率)")
    parser.add_argument("--likelihood", action="store_true",
                       help="使用似然地图优化：基于概率地图的梯度下降优化，提高精度并降低资源消耗")
    parser.add_argument("--candidates", type=int, default=3,
                       help="多候选位姿策略的候选数量，默认3个（仅对--likelihood有效）")
    parser.add_argument("--debug", action="store_true", help="开启调试模式：详细耗时打印和误差变化图")
    parser.add_argument("--add-noise", nargs=2, type=float, metavar=('DIS', 'DEG'),
                       help="为初始位姿添加噪声：DIS(米)为位置噪声，DEG(度)为角度噪声")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"错误：文件夹 {args.folder_path} 不存在")
        return

    # 1. 加载全局地图
    use_multi_res = args.multi_res is not None
    
    if use_multi_res:
        try:
            # 确定层数
            num_layers = args.multi_res
            multi_res_global_maps = load_multi_resolution_global_maps(args.folder_path, num_layers)
            
            # 检查是否成功加载了地图
            if not multi_res_global_maps:
                print("多分辨率地图加载失败，回退到单分辨率模式...")
                use_multi_res = False
                global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
            else:
                # 使用最高分辨率的地图作为默认全局地图
                highest_res = min(multi_res_global_maps.keys())
                global_map = multi_res_global_maps[highest_res]
                
        except Exception as e:
            print(f"多分辨率地图加载失败: {e}")
            print("回退到单分辨率模式...")
            use_multi_res = False
            global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
    else:
        global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
    
    # 2. 选择子图
    submap_files = [f for f in os.listdir(args.folder_path) 
                    if f.startswith('submap_') and f.endswith('.bin')]
    if not submap_files:
        print("错误：没有找到子图文件")
        return
    
    # 处理所有子图的情况
    if args.submap == -1:
        print(f"处理所有子图，共 {len(submap_files)} 个")
        # 按子图ID进行数值排序
        submap_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        all_results = []
        
        for i, target_file in enumerate(submap_files):
            submap_id = int(target_file.split('_')[1].split('.')[0])
            print(f"\n{'='*50}")
            print(f"处理子图 {submap_id} ({i+1}/{len(submap_files)})")
            print(f"{'='*50}")
            
            # 3. 加载子图
            submap_path = os.path.join(args.folder_path, target_file)
            _, ts, loaded_true_pose, _, _, _, _, occ_map = load_submap(submap_path)
            
            submap = GridMap()
            for key, prob in occ_map.items():
                gi, gj = decode_key(key)
                submap.update_occ(gi, gj, prob)
            
            # 4. 设置真值和初始位姿
            if args.use_gt:
                gt_file_path = os.path.join(args.folder_path, 'path_pg_rtk.txt')
                if not os.path.exists(gt_file_path):
                    print(f"错误：未找到真值文件: {gt_file_path}")
                    continue
                
                gt_pose = load_gt_pose_from_file(gt_file_path, ts)
                if gt_pose is None:
                    print(f"警告：未找到时间戳 {ts} 对应的真值位姿，跳过子图 {submap_id}")
                    continue
                
                true_pose = gt_pose
            else:
                true_pose = loaded_true_pose

            if args.add_noise:
                dis_noise, deg_noise = args.add_noise
                init_pose = add_noise_to_pose(loaded_true_pose, dis_noise, deg_noise)
            else:
                init_pose = loaded_true_pose

            # 5. 优化位姿
            try:
                matching_time = 0.0
                final_error = 0.0
                
                match_start_time = time.time()
                if args.likelihood:
                    # 似然优化策略
                    if use_multi_res:
                        multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=True, num_layers=args.multi_res)
                        opt_pose, _ = multi_resolution_likelihood_optimization(
                            multi_res_submaps, multi_res_global_maps, init_pose,
                            true_pose, False, args.debug, args.candidates)
                    else:
                        opt_pose, _ = match_submap_with_likelihood(
                            submap, global_map, init_pose, 0.05, 0.1, 100, 1e-6,
                            args.debug, 'Nelder-Mead', False)
                else:
                    # 粒子滤波优化策略
                    if use_multi_res:
                        multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=False, num_layers=args.multi_res)
                        opt_pose, _ = multi_resolution_optimization(
                            multi_res_submaps, multi_res_global_maps, init_pose,
                            true_pose, False, args.debug)
                    else:
                        opt_pose, _ = optimize_submap_pose(
                            submap, global_map, init_pose, False, args.debug)
                matching_time = time.time() - match_start_time

                # 计算初始和最终误差
                if use_multi_res:
                    highest_res = min(multi_res_submaps.keys())
                    submap_for_vis = multi_res_submaps[highest_res]
                    init_match_error = compute_matching_error(
                        submap_for_vis,
                        multi_res_global_maps[highest_res], 
                        init_pose,
                        submap_res=highest_res,
                        global_res=highest_res
                    )
                    final_match_error = compute_matching_error(
                        submap_for_vis,
                        multi_res_global_maps[highest_res],
                        opt_pose,
                        submap_res=highest_res,
                        global_res=highest_res
                    )
                else:
                    submap_for_vis = submap
                    init_match_error = compute_matching_error(submap, global_map, init_pose, 
                                                      submap_res=0.05, global_res=0.1)
                    final_match_error = compute_matching_error(submap, global_map, opt_pose, 
                                                       submap_res=0.05, global_res=0.1)

                # 计算位姿误差
                init_pos_error = np.linalg.norm(init_pose[:2, 3] - true_pose[:2, 3])
                init_rot_error = np.rad2deg(abs(angle_diff(np.arctan2(init_pose[1, 0], init_pose[0, 0]), np.arctan2(true_pose[1, 0], true_pose[0, 0]))))
                opt_pos_error = np.linalg.norm(opt_pose[:2, 3] - true_pose[:2, 3])
                opt_rot_error = np.rad2deg(abs(angle_diff(np.arctan2(opt_pose[1, 0], opt_pose[0, 0]), np.arctan2(true_pose[1, 0], true_pose[0, 0]))))

                all_results.append({
                    'submap_id': submap_id,
                    'submap_for_vis': submap_for_vis,
                    'true_pose': true_pose, 'init_pose': init_pose, 'opt_pose': opt_pose,
                    'init_pos_error': init_pos_error, 'init_rot_error': init_rot_error,
                    'opt_pos_error': opt_pos_error, 'opt_rot_error': opt_rot_error,
                    'init_match_error': init_match_error, 'final_match_error': final_match_error,
                    'matching_time': matching_time
                })
                print(f"子图 {submap_id} 优化完成，耗时 {matching_time:.2f}s")
                
            except Exception as e:
                print(f"子图 {submap_id} 优化失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 6. 可视化所有结果
        if all_results:
            print(f"\n{'='*50}")
            print(f"所有子图处理完成，共成功处理 {len(all_results)} 个子图")
            print(f"{'='*50}")
            
            # 使用最高分辨率的地图进行可视化
            if use_multi_res:
                highest_res = min(multi_res_global_maps.keys())
                visualize_all_optimization_results(global_map, all_results, 
                                                 None, submap_res=highest_res, global_res=highest_res)
            else:
                visualize_all_optimization_results(global_map, all_results, 
                                                 None, submap_res=0.05, global_res=0.1)
        else:
            print("没有成功处理任何子图")
        
        return
    
    # 处理单个子图的情况（原有逻辑）
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
        print("使用path_pg_rtk.txt中的真值位姿")
    else:
        true_pose = loaded_true_pose
        print("使用原始位姿作为真实值")

    if args.add_noise:
        dis_noise, deg_noise = args.add_noise
        init_pose = add_noise_to_pose(loaded_true_pose, dis_noise, deg_noise)
        print(f"已添加噪声作为初始位姿 (位置噪声: {dis_noise}m, 角度噪声: {deg_noise}°)")
    else:
        init_pose = loaded_true_pose
        print("使用原始位姿作为初始位姿")

    
    # 5. 优化位姿
    if args.likelihood:
        # 似然优化策略
        if use_multi_res:
            print(f"使用多分辨率似然优化策略 ({args.multi_res} 层, 候选数量: {args.candidates})...")
            multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=True, num_layers=args.multi_res)
            highest_res = min(multi_res_submaps.keys())
            opt_pose, final_score = multi_resolution_likelihood_optimization(
                multi_res_submaps, 
                multi_res_global_maps, 
                init_pose,
                true_pose,
                visualize=args.plot,
                debug=args.debug,
                n_candidates=args.candidates
            )
            
            # 计算最终匹配误差（使用三值化方式计算匹配误差）- 使用最高分辨率
            final_error = 0.0
            if highest_res in multi_res_submaps and highest_res in multi_res_global_maps:
                final_error = compute_matching_error(
                    multi_res_submaps[highest_res], 
                    multi_res_global_maps[highest_res], 
                    opt_pose,
                    submap_res=highest_res,
                    global_res=highest_res
                )
            
            # 使用最高分辨率的地图进行可视化
            final_submap = multi_res_submaps[highest_res] if multi_res_submaps else submap
            visualize_optimization(global_map, final_submap, true_pose, init_pose, opt_pose, 
                                 None, submap_id, submap_res=highest_res, global_res=highest_res)
        else:
            print("使用单分辨率似然优化...")
            opt_pose, final_score = match_submap_with_likelihood(
                submap, global_map, init_pose,
                submap_res=0.05, global_res=0.1,
                max_iterations=100,
                tolerance=1e-6,
                debug=args.debug,
                method='Nelder-Mead',
                visualize=args.plot
            )
            
            # 计算最终匹配误差
            final_error = compute_matching_error(submap, global_map, opt_pose, 
                                               submap_res=0.05, global_res=0.1)
            
            visualize_optimization(global_map, submap, true_pose, init_pose, opt_pose, 
                                 None, submap_id, submap_res=0.05, global_res=0.1)
        
        print(f"优化完成，最终似然分数: {final_score:.6f}, 最终匹配误差: {final_error:.6f}")
        
    else:
        # 粒子滤波优化策略
        if use_multi_res:
            print(f"使用多分辨率粒子滤波优化策略 ({args.multi_res} 层)...")
            multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=False, num_layers=args.multi_res)
            opt_pose, error = multi_resolution_optimization(
                multi_res_submaps, 
                multi_res_global_maps, 
                init_pose,
                true_pose,
                visualize=args.plot,
                debug=args.debug
            )
            # 使用最高分辨率的地图进行可视化
            highest_res = min(multi_res_submaps.keys()) if multi_res_submaps else 0.1
            final_submap = multi_res_submaps[highest_res] if multi_res_submaps else submap
            visualize_optimization(global_map, final_submap, true_pose, init_pose, opt_pose, 
                                 None, submap_id, submap_res=highest_res, global_res=highest_res)
        else:
            print("使用单分辨率粒子滤波优化...")
            opt_pose, error = optimize_submap_pose(submap, global_map, init_pose, 
                                                  visualize=args.plot, debug=args.debug)
            visualize_optimization(global_map, submap, true_pose, init_pose, opt_pose, 
                                 None, submap_id, submap_res=0.05, global_res=0.1)
        
        print(f"优化完成，最终误差: {error:.6f}")

if __name__ == '__main__':
    main() 
