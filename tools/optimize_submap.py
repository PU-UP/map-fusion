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
from typing import Optional
import argparse
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from io import StringIO
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

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
    if num_layers > len(all_resolutions):
        print(f"警告：请求的层数 {num_layers} 超过可用层数 {len(all_resolutions)}，使用单层地图匹配")
        return {}
    
    # 选择最细的指定层数分辨率（从后往前选择）
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
    if num_layers > len(all_resolutions):
        print(f"警告：请求的层数 {num_layers} 超过可用层数 {len(all_resolutions)}，使用单层地图匹配")
        return {}
    
    # 选择最细的指定层数分辨率（从后往前选择）
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

def compute_likelihood_score(submap: GridMap,
                           global_map: GridMap,
                           pose: np.ndarray,
                           submap_res: float = 0.05,
                           global_res: float = 0.1,
                           debug: bool = False) -> float:
    """计算子图与全局地图的似然匹配分数（负对数似然）"""
    total_score = 0.0
    count = 0
    
    # 获取子图中所有非未知的栅格
    valid_cells = [(key, p) for key, p in submap.occ_map.items() 
                   if p < 0.4 or p > 0.6]  # 排除未知区域
    
    if debug:
        print(f"子图有效栅格数量: {len(valid_cells)}")
    
    for key, sub_prob in valid_cells:
        sub_i, sub_j = decode_key(key)
        p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        key_glob = encode_key(gi_glob, gj_glob)
        
        # 获取全局地图中对应位置的概率
        global_prob = global_map.occ_map.get(key_glob, 0.5)  # 未知区域默认为0.5
        
        # 改进的似然计算：使用更敏感的评分函数
        if sub_prob > 0.6:  # 占用
            # 占用栅格应该匹配到高概率区域
            if global_prob > 0.5:
                # 匹配成功：奖励
                score = -np.log(global_prob + 1e-6)
            else:
                # 匹配失败：惩罚
                score = -np.log(1.0 - global_prob + 1e-6) + 2.0  # 额外惩罚
        else:  # 空闲
            # 空闲栅格应该匹配到低概率区域
            if global_prob < 0.5:
                # 匹配成功：奖励
                score = -np.log(1.0 - global_prob + 1e-6)
            else:
                # 匹配失败：惩罚
                score = -np.log(global_prob + 1e-6) + 2.0  # 额外惩罚
        
        total_score += score
        count += 1
    
    return total_score / max(count, 1)

def pose_to_params(pose: np.ndarray) -> np.ndarray:
    """将4x4变换矩阵转换为优化参数 [x, y, theta]"""
    x = pose[0, 3]
    y = pose[1, 3]
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([x, y, theta])

def params_to_pose(params: np.ndarray) -> np.ndarray:
    """将优化参数 [x, y, theta] 转换为4x4变换矩阵"""
    x, y, theta = params
    pose = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose[:3, 3] = [x, y, 0]
    return pose

def likelihood_objective(params: np.ndarray,
                        submap: GridMap,
                        global_map: GridMap,
                        submap_res: float,
                        global_res: float) -> float:
    """似然优化的目标函数"""
    pose = params_to_pose(params)
    return compute_likelihood_score(submap, global_map, pose, submap_res, global_res)

def gradient_optimization(submap: GridMap,
                         global_map: GridMap,
                         init_pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1,
                         max_iterations: int = 100,
                         tolerance: float = 1e-6,
                         debug: bool = False,
                         method: str = 'Nelder-Mead',
                         visualize: bool = False) -> tuple:
    print(f"使用{method}进行似然优化...")
    start_time = time.time()
    init_params = pose_to_params(init_pose)
    init_score = compute_likelihood_score(submap, global_map, init_pose, submap_res, global_res)
    if debug:
        print(f"初始似然分数: {init_score:.6f}")
        print(f"初始位姿参数: {init_params}")
    # 可视化相关
    if visualize:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        score_history = []
        pose_history = []
    def callback(xk):
        if visualize:
            score = likelihood_objective(xk, submap, global_map, submap_res, global_res)
            score_history.append(score)
            pose_history.append(xk.copy())
            ax1.clear()
            ax2.clear()
            # 可视化当前子图叠加到全局地图
            global_grid = global_map.to_matrix()
            ax1.imshow(global_grid, cmap='gray', origin='upper')
            # 画当前子图
            submap_vis = transform_submap_for_visualization(submap, params_to_pose(xk), submap_res, global_res, global_map)
            ax1.imshow(submap_vis, cmap='Reds', alpha=0.5, origin='upper')
            ax1.set_title(f'当前匹配  Score: {score:.4f}')
            # 画分数曲线
            ax2.plot(score_history, 'b-')
            ax2.set_title('似然分数变化')
            ax2.set_xlabel('迭代')
            ax2.set_ylabel('Score')
            plt.pause(0.1)
    if method == 'L-BFGS-B':
        result = minimize(
            likelihood_objective,
            init_params,
            args=(submap, global_map, submap_res, global_res),
            method='L-BFGS-B',
            bounds=[
                (init_params[0] - 2.0, init_params[0] + 2.0),
                (init_params[1] - 2.0, init_params[1] + 2.0),
                (init_params[2] - np.pi/2, init_params[2] + np.pi/2)
            ],
            options={
                'maxiter': max_iterations,
                'gtol': tolerance,
                'eps': 0.05,
                'disp': debug
            },
            callback=callback if visualize else None
        )
    else:
        result = minimize(
            likelihood_objective,
            init_params,
            args=(submap, global_map, submap_res, global_res),
            method='Nelder-Mead',
            options={
                'maxiter': max_iterations,
                'xatol': 1e-3,
                'fatol': 1e-6,
                'disp': debug
            },
            callback=callback if visualize else None
        )
    optimized_pose = params_to_pose(result.x)
    final_score = result.fun
    end_time = time.time()
    optimization_time = end_time - start_time
    if debug:
        print(f"优化耗时: {optimization_time:.4f}s")
        print(f"优化状态: {result.message}")
        print(f"迭代次数: {result.nit}")
        print(f"最终似然分数: {final_score:.6f}")
        print(f"最终位姿参数: {result.x}")
        print(f"分数改善: {(init_score - final_score) / max(init_score, 1e-6) * 100:.1f}%")
    if visualize:
        plt.ioff()
        plt.show()
    return optimized_pose, final_score

def transform_submap_for_visualization(submap: GridMap, pose: np.ndarray, submap_res: float, global_res: float, global_map: GridMap) -> np.ndarray:
    # 生成与全局地图同shape的0/1矩阵
    grid = np.zeros_like(global_map.to_matrix())
    for key, p_meas in submap.occ_map.items():
        if p_meas < 0.6:
            continue
        sub_i, sub_j = decode_key(key)
        p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        if 0 <= gi_glob - global_map.min_i < grid.shape[0] and 0 <= gj_glob - global_map.min_j < grid.shape[1]:
            grid[gi_glob - global_map.min_i, gj_glob - global_map.min_j] = 1
    return grid

def check_pose_robustness(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float,
                         global_res: float,
                         debug: bool = False) -> tuple:
    """检查位姿的鲁棒性，通过多次扰动测试"""
    base_score = compute_likelihood_score(submap, global_map, pose, submap_res, global_res)
    
    # 生成多个扰动位姿
    perturbed_scores = []
    perturbations = [
        (0.1, 5.0),   # 小扰动
        (0.2, 10.0),  # 中等扰动
        (0.3, 15.0),  # 大扰动
    ]
    
    for trans_noise, rot_noise in perturbations:
        perturbed_pose = add_noise_to_pose(pose, trans_noise, rot_noise)
        perturbed_score = compute_likelihood_score(submap, global_map, perturbed_pose, submap_res, global_res)
        perturbed_scores.append(perturbed_score)
    
    # 计算鲁棒性指标
    score_variance = np.var(perturbed_scores)
    score_increase = max(perturbed_scores) - base_score
    
    # 鲁棒性判断：如果扰动后分数显著增加，说明当前位姿可能不是最优的
    is_robust = score_increase < base_score * 0.1  # 扰动后分数增加不超过10%
    
    if debug:
        print(f"  鲁棒性检查: 基础分数 {base_score:.6f}, 扰动分数 {perturbed_scores}")
        print(f"  分数方差: {score_variance:.6f}, 分数增加: {score_increase:.6f}")
        print(f"  鲁棒性: {'✓' if is_robust else '✗'}")
    
    return is_robust, base_score, perturbed_scores

def multi_resolution_likelihood_optimization(multi_res_submaps: dict,
                                           multi_res_global_maps: dict,
                                           init_pose: np.ndarray,
                                           true_pose: Optional[np.ndarray] = None,
                                           visualize: bool = False,
                                           debug: bool = False,
                                           method: str = 'Nelder-Mead',
                                           n_candidates: int = 3) -> tuple:
    """多分辨率似然优化 - 支持多候选位姿策略"""
    # 获取可用的分辨率列表，从粗到细排序
    available_resolutions = sorted(multi_res_submaps.keys(), reverse=True)
    current_candidates = [(init_pose.copy(), 0.0)]  # (pose, score)
    
    print("开始多分辨率似然优化...")
    
    # 参数配置 - 根据可用分辨率动态调整
    param_config = {
        1.6: {'max_iter': 100, 'tolerance': 1e-8},
        0.8: {'max_iter': 150, 'tolerance': 1e-8},
        0.4: {'max_iter': 200, 'tolerance': 1e-8},
        0.2: {'max_iter': 250, 'tolerance': 1e-8},
        0.1: {'max_iter': 300, 'tolerance': 1e-8}
    }
    
    successful_optimizations = 0
    resolution_times = {}
    score_history = {}
    
    for res_idx, res in enumerate(available_resolutions):
        if res not in multi_res_global_maps:
            continue
        
        print(f"\n===== 分辨率 {res}m 似然优化 =====")
        
        res_start_time = time.time()
        
        submap = multi_res_submaps[res]
        global_map = multi_res_global_maps[res]
        
        # 检查子图质量
        occupied_cells = sum(1 for p in submap.occ_map.values() if p > 0.6)
        free_cells = sum(1 for p in submap.occ_map.values() if p < 0.4)
        total_cells = len(submap.occ_map)
        
        if total_cells == 0 or (occupied_cells + free_cells) / total_cells < 0.2:
            print(f"跳过分辨率 {res}m (有效栅格太少)")
            continue
        
        # 获取参数配置，如果不存在则使用默认值
        config = param_config.get(res, {'max_iter': 200, 'tolerance': 1e-8})
        max_iter = config['max_iter']
        tolerance = config['tolerance']
        
        # 多候选位姿优化
        new_candidates = []
        
        for candidate_idx, (candidate_pose, _) in enumerate(current_candidates):
            if debug:
                print(f"  候选位姿 {candidate_idx + 1}: {pose_to_params(candidate_pose)}")
            
            initial_score = compute_likelihood_score(submap, global_map, candidate_pose, 
                                                   submap_res=res, global_res=res, debug=False)
            
            try:
                optimized_pose, final_score = gradient_optimization(
                    submap, global_map, candidate_pose,
                    submap_res=res, global_res=res,
                    max_iterations=max_iter,
                    tolerance=tolerance,
                    debug=False,
                    method=method,
                    visualize=visualize and res <= 1.6 and candidate_idx == 0  # 只显示第一个候选的可视化
                )
                
                score_improvement = (initial_score - final_score) / max(initial_score, 1e-6)
                pose_diff = np.linalg.norm(optimized_pose[:2, 3] - candidate_pose[:2, 3])
                
                if debug:
                    print(f"  候选 {candidate_idx + 1}: 分数 {initial_score:.6f} → {final_score:.6f} (改善: {score_improvement*100:.1f}%)")
                
                # 在粗分辨率层进行鲁棒性检查
                if res >= 0.4:
                    is_robust, _, _ = check_pose_robustness(submap, global_map, optimized_pose, 
                                                          res, res, debug=debug)
                    if not is_robust and debug:
                        print(f"  警告: 候选 {candidate_idx + 1} 鲁棒性较差，可能匹配错误")
                
                # 保留所有候选位姿，但记录分数
                new_candidates.append((optimized_pose, final_score))
                
            except Exception as e:
                if debug:
                    print(f"  候选 {candidate_idx + 1} 优化失败: {e}")
                # 保留原始位姿
                new_candidates.append((candidate_pose, initial_score))
        
        # 在粗分辨率层，生成额外的候选位姿
        if res >= 0.4 and len(new_candidates) < n_candidates:
            best_pose = min(new_candidates, key=lambda x: x[1])[0]
            
            # 在最佳位姿周围生成扰动候选
            for i in range(n_candidates - len(new_candidates)):
                # 随机扰动
                noise_pose = add_noise_to_pose(best_pose, 0.5, 15.0)
                noise_score = compute_likelihood_score(submap, global_map, noise_pose, 
                                                     submap_res=res, global_res=res)
                
                # 优化扰动位姿
                try:
                    opt_noise_pose, opt_noise_score = gradient_optimization(
                        submap, global_map, noise_pose,
                        submap_res=res, global_res=res,
                        max_iterations=max_iter // 2,  # 减少迭代次数
                        tolerance=tolerance,
                        debug=False,
                        method=method,
                        visualize=False
                    )
                    new_candidates.append((opt_noise_pose, opt_noise_score))
                    if debug:
                        print(f"  额外候选 {i + 1}: 扰动优化后分数 {opt_noise_score:.6f}")
                except:
                    new_candidates.append((noise_pose, noise_score))
        
        # 选择最佳候选位姿
        if res >= 0.2:  # 在中等分辨率以上，只保留最佳候选
            best_candidate = min(new_candidates, key=lambda x: x[1])
            current_candidates = [best_candidate]
            if debug:
                print(f"  选择最佳候选，分数: {best_candidate[1]:.6f}")
        else:  # 在粗分辨率层，保留多个候选
            # 按分数排序，保留前n_candidates个
            new_candidates.sort(key=lambda x: x[1])
            current_candidates = new_candidates[:n_candidates]
            if debug:
                print(f"  保留 {len(current_candidates)} 个候选位姿")
                for i, (pose, score) in enumerate(current_candidates):
                    print(f"    候选 {i + 1}: 分数 {score:.6f}")
        
        # 记录最佳分数
        best_score = min(score for _, score in current_candidates)
        score_history[res] = (initial_score if 'initial_score' in locals() else 0.0, best_score)
        
        res_end_time = time.time()
        res_time = res_end_time - res_start_time
        resolution_times[res] = res_time
        
        if debug:
            print(f"分辨率 {res}m 优化耗时: {res_time:.4f}s")
    
    # 最终选择最佳位姿
    final_pose = current_candidates[0][0]
    final_score = current_candidates[0][1]
    
    # 计算最终误差（使用三值化方式计算匹配误差）- 使用最高分辨率
    final_error = 0.0
    highest_res = min(multi_res_submaps.keys()) if multi_res_submaps else 0.1
    if highest_res in multi_res_submaps and highest_res in multi_res_global_maps:
        final_error = compute_matching_error(
            multi_res_submaps[highest_res], 
            multi_res_global_maps[highest_res], 
            final_pose,
            submap_res=highest_res,
            global_res=highest_res
        )
    
    print(f"\n========== 多分辨率似然优化总结 ==========")
    print(f"最终匹配误差: {final_error:.6f}")
    print(f"最终似然分数: {final_score:.6f}")
    
    if debug and score_history:
        print(f"\n========== 各分辨率层似然分数变化 ==========")
        for res in available_resolutions:
            if res in score_history:
                init_score, final_score = score_history[res]
                improvement = (init_score - final_score) / max(init_score, 1e-6) * 100
                print(f"分辨率 {res}m: {init_score:.6f} → {final_score:.6f} (改善: {improvement:.1f}%)")
    
    if resolution_times:
        total_time = sum(resolution_times.values())
        if debug:
            print(f"\n========== 各分辨率层耗时统计 ==========")
            for res in available_resolutions:
                if res in resolution_times:
                    time_percent = (resolution_times[res] / total_time) * 100
                    print(f"分辨率 {res}m: {resolution_times[res]:.4f}s ({time_percent:.1f}%)")
        print(f"总耗时: {total_time:.4f}s")
    
    return final_pose, final_error

def main():
    parser = argparse.ArgumentParser(description="子图位姿优化脚本")
    parser.add_argument("folder_path", type=str, help="包含子图和全局地图的文件夹路径")
    parser.add_argument("--plot", action="store_true", help="显示粒子滤波中间过程的可视化")
    parser.add_argument("--use-gt", action="store_true", help="使用path_pg_rtk.txt中的真值作为初始位姿和参考真值")
    parser.add_argument("--submap", type=int, help="指定要优化的子图ID，默认为随机选择")
    parser.add_argument("--multi-res", nargs='?', const=5, type=int, default=None,
                       help="使用多分辨率匹配策略：从低分辨率到高分辨率逐层优化。可指定层数(默认5层，包含0.1m分辨率)")
    parser.add_argument("--likelihood", action="store_true",
                       help="使用似然地图优化：基于概率地图的梯度下降优化，提高精度并降低资源消耗")
    parser.add_argument("--candidates", type=int, default=3,
                       help="多候选位姿策略的候选数量，默认3个（仅对--likelihood有效）")
    parser.add_argument("--debug", action="store_true", help="开启调试模式：详细耗时打印和误差变化图")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"错误：文件夹 {args.folder_path} 不存在")
        return

    # 1. 加载全局地图
    if args.multi_res is not None or args.likelihood:
        try:
            # 确定层数
            num_layers = args.multi_res if args.multi_res is not None else 5
            multi_res_global_maps = load_multi_resolution_global_maps(args.folder_path, num_layers)
            
            # 检查是否成功加载了地图
            if not multi_res_global_maps:
                print("多分辨率地图加载失败，回退到单分辨率模式...")
                args.multi_res = None
                args.likelihood = False
                global_map = load_global_map(os.path.join(args.folder_path, 'global_map.bin'))
            else:
                # 使用最高分辨率的地图作为默认全局地图
                highest_res = min(multi_res_global_maps.keys())
                global_map = multi_res_global_maps[highest_res]
                
        except Exception as e:
            print(f"多分辨率地图加载失败: {e}")
            print("回退到单分辨率模式...")
            args.multi_res = None
            args.likelihood = False
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
    if args.likelihood:
        print(f"使用似然地图优化策略 (候选数量: {args.candidates})...")
        num_layers = args.multi_res if args.multi_res is not None else 5
        multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=True, num_layers=num_layers)
        opt_pose, error = multi_resolution_likelihood_optimization(
            multi_res_submaps, 
            multi_res_global_maps, 
            init_pose,
            true_pose,
            visualize=args.plot,
            debug=args.debug,
            n_candidates=args.candidates
        )
        # 使用最高分辨率的地图进行可视化
        highest_res = min(multi_res_submaps.keys()) if multi_res_submaps else 0.1
        final_submap = multi_res_submaps[highest_res] if multi_res_submaps else submap
        visualize_optimization(global_map, final_submap, true_pose, init_pose, opt_pose, 
                             None, submap_id, submap_res=highest_res, global_res=highest_res)
    elif args.multi_res is not None:
        print(f"使用多分辨率优化策略 ({args.multi_res} 层)...")
        num_layers = args.multi_res
        multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=False, num_layers=num_layers)
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
        print("使用单分辨率优化...")
        opt_pose, error = optimize_submap_pose(submap, global_map, init_pose, 
                                              visualize=args.plot, debug=args.debug)
        visualize_optimization(global_map, submap, true_pose, init_pose, opt_pose, 
                             None, submap_id, submap_res=0.05, global_res=0.1)
    
    print(f"优化完成，最终误差: {error:.6f}")

if __name__ == '__main__':
    main() 
