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
5. 自适应参数调整：低分辨率用更大搜索范围和更少粒子
6. 早期收敛检测：如果在低分辨率收敛良好，可跳过中间层
7. 资源优化：总计算量比单一高分辨率匹配更少，收敛更稳定
"""

import os
import sys
import numpy as np
# from matplotlib import font_manager
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fuse_submaps import (
    load_submap, load_global_map, GridMap, decode_key,
    visualize_map, add_noise_to_pose, downsample_map, ternarize_map
)
from particle_filter_matcher import ParticleFilter, encode_key, match_submap_with_particle_filter
from typing import List, Tuple
import glob
import argparse
import math

from scipy.spatial.transform import Rotation as R

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def compute_error_and_jacobian(pose_params: np.ndarray, 
                             submap: GridMap, 
                             global_map: GridMap,
                             submap_res: float = 0.05,
                             global_res: float = 0.1) -> tuple:
    """计算误差和雅可比矩阵
    参数：
        pose_params: [x, y, theta] - SE(2)位姿参数
        submap: 待优化的子图
        global_map: 全局地图
    返回：
        error: 总误差（不匹配栅格数量）
        jacobian: 雅可比矩阵 [3,]
    """
    x, y, theta = pose_params
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = np.array([x, y, 0])
    
    total_error = 0.0
    jacobian = np.zeros(3)
    total_points = 0
    
    # 遍历子图中的占用栅格
    for key, p_sub in submap.occ_map.items():
        # 只考虑占用栅格
        if p_sub < 0.6:  # 非占用栅格
            continue
        else:
            p_sub = 1.0
            
        total_points += 1
        
        # 解码子图栅格索引
        sub_i, sub_j = decode_key(key)
        
        # 计算子图坐标系下的物理坐标
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = R @ p_s + t
        
        # 计算全局地图栅格索引
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        
        # 在7x7邻域内搜索最近的占用栅格
        min_dist = float('inf')
        best_dx = 0
        best_dy = 0
        found_match = False
        
        search_range = 3  # 搜索范围为7x7（中心点±3）
        for di in range(-search_range, search_range + 1):
            for dj in range(-search_range, search_range + 1):
                ni = gi_glob + di
                nj = gj_glob + dj
                
                key_n = (ni << 32) | (nj & 0xFFFFFFFF)
                if key_n in global_map.occ_map:
                    p_n = global_map.occ_map[key_n]
                    if p_n >= 0.6:  # 找到占用栅格
                        dist = di*di + dj*dj
                        if dist < min_dist:
                            min_dist = dist
                            best_dx = di * global_res
                            best_dy = dj * global_res
                            if dist == 0:  # 完全匹配
                                found_match = True
                                break
            if found_match:
                break
        
        if found_match:
            continue  # 如果找到完全匹配，不计算误差和梯度
            
        # 计算误差（直接使用距离）
        if min_dist < float('inf'):
            dist = np.sqrt(min_dist) * global_res
            total_error += dist  # 直接使用距离作为误差
            
            # 计算梯度（使用固定步长）
            dx = best_dx / (global_res * np.sqrt(min_dist))  # 归一化方向
            dy = best_dy / (global_res * np.sqrt(min_dist))
            
            # 使用较大的固定步长
            step = 1.0
            jacobian[0] += step * dx
            jacobian[1] += step * dy
            # 旋转梯度也使用较大的步长
            jacobian[2] += step * (-p_s[0] * s + p_s[1] * c) * (dx * c + dy * s)
        else:
            # 如果在搜索范围内没有找到占用栅格，使用最大误差和固定梯度
            total_error += search_range * global_res  # 最大搜索距离作为误差
            
            # 使用中心位置的梯度
            dx = -1.0 if p_w[0] > gi_glob * global_res + global_res/2 else 1.0
            dy = -1.0 if p_w[1] > gj_glob * global_res + global_res/2 else 1.0
            
            # 使用较大的固定步长
            step = 1.0
            jacobian[0] += step * dx
            jacobian[1] += step * dy
            jacobian[2] += step * (-p_s[0] * s + p_s[1] * c) * (dx * c + dy * s)
    
    # 返回平均误差和归一化的雅可比矩阵
    error = total_error / max(total_points, 1)
    jacobian = jacobian / max(total_points, 1)
    return error, jacobian

def transform_submap_to_size(submap: GridMap, pose: np.ndarray, 
                           target_shape: Tuple[int, int],
                           submap_res: float = 0.05,
                           global_res: float = 0.1) -> np.ndarray:
    """将子图转换到指定尺寸的栅格地图
    
    Args:
        submap: 源子图
        pose: 变换位姿
        target_shape: 目标尺寸 (height, width)
        submap_res: 子图分辨率
        global_res: 全局地图分辨率
    
    Returns:
        转换后的栅格地图，尺寸与target_shape相同
    """
    result = np.full(target_shape, 0.5)  # 默认值0.5表示未知
    
    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        
        # 转换到物理坐标（使用子图分辨率）
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 转换到全局地图栅格坐标（使用全局地图分辨率）
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        
        # 检查是否在目标尺寸范围内
        if 0 <= gi_glob < target_shape[0] and 0 <= gj_glob < target_shape[1]:
            result[gi_glob, gj_glob] = p_sub
    
    return result

def visualize_optimization_step(ax1, ax2, 
                              global_map: GridMap,
                              submap: GridMap,
                              particles: List['Particle'],
                              current_pose: np.ndarray,
                              iter_num: int,
                              error: float):
    """优化过程的可视化"""
    ax1.clear()
    ax2.clear()
    
    # 1. 获取全局地图
    global_grid = global_map.to_matrix()
    
    # 计算全局地图的物理范围
    global_res = 0.1  # 这里保持0.1，因为这个函数主要用于单分辨率可视化
    x_min = global_map.min_i * global_res
    x_max = global_map.max_i * global_res
    y_min = global_map.min_j * global_res
    y_max = global_map.max_j * global_res
    
    # 将概率值转换为更清晰的显示
    vis_global = np.zeros_like(global_grid)
    vis_global[global_grid > 0.6] = 1.0  # 占用栅格
    vis_global[global_grid < 0.4] = 0.3  # 空闲栅格
    
    # 显示全局地图
    ax1.imshow(vis_global, cmap='gray', origin='upper',
               extent=[y_min, y_max, x_max, x_min])  # 注意这里x和y的顺序
    
    # 绘制粒子，使用渐变色表示权重
    weights = np.array([p.weight for p in particles])
    max_weight = weights.max()
    if max_weight > 0:
        weights = weights / max_weight
    
    # 直接使用物理坐标绘制粒子
    xs = []
    ys = []
    dirs = []  # 方向
    valid_weights = []
    for p, w in zip(particles, weights):
        # 检查是否在显示范围内
        if x_min <= p.x <= x_max and y_min <= p.y <= y_max:
            xs.append(p.y)  # matplotlib中x对应y坐标
            ys.append(p.x)  # matplotlib中y对应x坐标
            dirs.append([np.cos(p.theta), np.sin(p.theta)])
            valid_weights.append(w)
    
    if xs:  # 如果有有效的粒子
        # 绘制粒子位置
        scatter = ax1.scatter(xs, ys, 
                            c=valid_weights,
                            cmap='hot',
                            s=30,
                            alpha=0.6)
        
        # 绘制粒子方向
        for x, y, d, w in zip(xs, ys, dirs, valid_weights):
            if w > 0.5:  # 只显示权重较大的粒子的方向
                ax1.arrow(x, y, 
                         d[1]*0.3, d[0]*0.3,  # 缩小箭头长度
                         head_width=0.1, 
                         head_length=0.1,
                         fc='red', 
                         ec='red',
                         alpha=0.6)
    
    # 2. 右图：叠加显示
    # 转换子图到全局地图尺寸
    submap_grid = transform_submap_to_size(submap, current_pose, global_grid.shape)
    
    # 创建RGB图像用于叠加显示
    overlay = np.zeros((*global_grid.shape, 3))
    
    # 设置全局地图为灰度背景
    overlay[..., 0] = vis_global
    overlay[..., 1] = vis_global
    overlay[..., 2] = vis_global
    
    # 将变换后的子图叠加为红色
    valid_mask = submap_grid > 0.6
    overlay[valid_mask, 0] = 1.0  # 红色通道
    overlay[valid_mask, 1] = 0.0
    overlay[valid_mask, 2] = 0.0
    
    # 显示叠加结果，使用相同的坐标范围
    ax2.imshow(overlay, origin='upper',
               extent=[y_min, y_max, x_max, x_min])
    
    # 设置标题
    ax1.set_title(f'粒子分布 (迭代次数 {iter_num})')
    ax2.set_title(f'匹配结果 (误差 {error:.3f})')
    
    # 添加图例
    ax1.text(0.02, 0.98, '粒子权重:', transform=ax1.transAxes, 
         verticalalignment='top', color='white')
    ax2.text(0.02, 0.98, '红色: 当前子图\n灰色: 全局地图',
         transform=ax2.transAxes, verticalalignment='top', color='white')

    
    # 设置坐标轴标签
    ax1.set_xlabel('Y (米)')
    ax1.set_ylabel('X (米)')
    ax2.set_xlabel('Y (米)')
    ax2.set_ylabel('X (米)')
    
    # 保持两个子图的显示范围一致
    ax1.set_xlim([y_min, y_max])
    ax1.set_ylim([x_max, x_min])  # 注意y轴方向
    ax2.set_xlim([y_min, y_max])
    ax2.set_ylim([x_max, x_min])
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

def optimize_submap_pose(submap: GridMap, 
                        global_map: GridMap,
                        init_pose: np.ndarray,
                        max_iter: int = 100,
                        use_particle_filter: bool = True,
                        visualize: bool = False) -> tuple:
    """使用梯度下降或粒子滤波优化子图位姿"""
    if use_particle_filter:
        print("使用粒子滤波进行优化...")
        # 初始散布：x, y方向±1m, 角度±15度
        spread_x_m = 1.0
        spread_y_m = 1.0
        spread_theta_rad = np.deg2rad(15.0)

        optimized_pose, final_error = match_submap_with_particle_filter(
            submap, global_map, init_pose,
            n_particles=100,
            n_iterations=200,
            visualize=visualize,
            spread=(spread_x_m, spread_y_m, spread_theta_rad),
            submap_res=0.05,
            global_res=0.1
        )
        return optimized_pose, final_error
    else:
        # 使用原有的ICP方法
        best_pose = init_pose.copy()
        min_error = float('inf')
        no_improvement_count = 0
        max_no_improvement = 10
        
        current_pose = init_pose.copy()
        
        # ICP迭代
        for iter in range(max_iter):
            # 1. 转换子图特征到全局坐标系
            R = current_pose[:3, :3]
            t = current_pose[:3, 3]
            
            total_error = 0.0
            total_weight = 0.0
            H = np.zeros((2, 2))
            b = np.zeros(2)
            
            point_match_count = 0
            line_match_count = 0
            
            # 2. 处理点特征
            transformed_features = submap_features.copy()
            transformed_features[:, :2] = (R[:2, :2] @ submap_features[:, :2].T).T + t[:2]
            
            # 找到特征点匹配
            point_matches = find_feature_matches(
                transformed_features.tolist(),
                global_features.tolist(),
                max_dist=1.0  # 增大匹配距离
            )
            
            point_match_count = len(point_matches)
            
            if point_match_count >= 3:
                # 提取匹配点对
                p = np.array([m[3][:2] for m in point_matches])  # 源点
                q = np.array([m[4][:2] for m in point_matches])  # 目标点
                w = np.array([m[5] for m in point_matches])      # 权重
                
                # 计算误差和权重
                point_error = sum(d * w for _, _, d, _, _, w in point_matches)
                total_error += point_error
                total_weight += sum(w)
                
                # 计算加权质心
                p_mean = np.average(p, axis=0, weights=w)
                q_mean = np.average(q, axis=0, weights=w)
                
                # 累积H矩阵
                for i in range(len(p)):
                    p_centered = p[i] - p_mean
                    q_centered = q[i] - q_mean
                    H += w[i] * np.outer(p_centered, q_centered)
            
            # 3. 处理线段特征
            if len(submap_lines) > 0 and len(global_lines) > 0:
                # 转换子图线段
                transformed_lines = []
                for line in submap_lines:
                    # 转换线段端点
                    start = np.array([line[0], line[1], 0])
                    end = np.array([line[2], line[3], 0])
                    
                    t_start = (R @ start) + t
                    t_end = (R @ end) + t
                    
                    # 计算新的角度和长度
                    dx = t_end[0] - t_start[0]
                    dy = t_end[1] - t_start[1]
                    angle = np.arctan2(dy, dx)
                    length = np.sqrt(dx*dx + dy*dy)
                    
                    transformed_lines.append([
                        t_start[0], t_start[1],
                        t_end[0], t_end[1],
                        angle, length
                    ])
                
                # 匹配线段
                line_matches = match_line_segments(
                    transformed_lines,
                    global_lines,
                    max_dist=0.5,
                    max_angle=np.pi/4
                )
                
                line_match_count = len(line_matches)
                
                if line_match_count > 0:
                    # 计算线段匹配的误差和贡献
                    for i, j, dist, angle_diff in line_matches:
                        src = transformed_lines[i]
                        target = global_lines[j]
                        
                        # 线段中点
                        src_mid = np.array([(src[0] + src[2])/2, (src[1] + src[3])/2])
                        target_mid = np.array([(target[0] + target[2])/2, 
                                             (target[1] + target[3])/2])
                        
                        # 使用距离和角度差异计算权重
                        w = 3.0 * np.exp(-dist/0.5) * np.exp(-angle_diff/(np.pi/4))  # 增加线段权重
                        
                        # 累积误差
                        total_error += (dist + angle_diff * 0.5) * w
                        total_weight += w
                        
                        # 累积H矩阵（使用线段中点）
                        H += w * np.outer(src_mid - np.mean(src_mid), 
                                        target_mid - np.mean(target_mid))
            
            print(f"迭代 {iter}: 点匹配数={point_match_count}, 线段匹配数={line_match_count}, " 
                  f"总误差={total_error:.3f}, 总权重={total_weight:.3f}")
            
            if total_weight == 0:
                print("警告: 没有有效的特征匹配!")
                break
            
            # 计算平均误差
            avg_error = total_error / total_weight
            
            # 保存最佳结果
            if avg_error < min_error:
                min_error = avg_error
                best_pose = current_pose.copy()
                print(f"更新最佳位姿, 误差={min_error:.3f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= max_no_improvement:
                print("连续多次没有改进，提前结束优化")
                break
            
            # 计算最优旋转
            U, S, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T
            
            # 确保是正交矩阵
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = Vt.T @ U.T
            
            # 计算最优平移（使用所有特征的平均偏移）
            if total_weight > 0:
                t_opt = b / total_weight
            else:
                break
            
            # 更新位姿
            new_pose = np.eye(4)
            new_pose[:2, :2] = R_opt
            new_pose[:2, 3] = t_opt
            
            # 将新位姿与当前位姿组合
            current_pose = new_pose @ init_pose
        
        return best_pose, min_error

def compute_matching_error(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1,
                         debug: bool = False) -> float:
    """Compute matching error between submap and global map"""
    total_error = 0
    count = 0
    matched_count = 0
    unmatched_count = 0
    
    occupied_cells = [(key, p) for key, p in submap.occ_map.items() if p > 0.6]
    if debug:
        print(f"子图占用栅格数量: {len(occupied_cells)}")
    
    for key, p_sub_raw in occupied_cells:
        # 对子图概率进行二值化
        p_sub = 1.0
            
        # Get submap grid coordinates
        sub_i, sub_j = decode_key(key)
        
        # Convert to physical coordinates using submap resolution
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # Transform to world coordinates
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # Convert to global map grid coordinates using global resolution
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        
        # Check occupancy in global map
        key_glob = encode_key(gi_glob, gj_glob)
        
        p_glob = 0.0 # 默认全局地图该位置非占用
        if key_glob in global_map.occ_map:
            p_glob_raw = global_map.occ_map[key_glob]
            p_glob = 1.0 if p_glob_raw > 0.6 else 0.0
            if p_glob == 1.0:
                matched_count += 1
            else:
                unmatched_count += 1
        else:
            unmatched_count += 1
            
        # 计算残差（这里直接是二值化后的差异）
        error = abs(p_glob - p_sub)
        total_error += error
        count += 1
    
    if debug:
        print(f"匹配统计: 总计{count}, 匹配{matched_count}, 不匹配{unmatched_count}")
        print(f"匹配率: {matched_count/max(count,1)*100:.1f}%")
    
    return total_error / max(count, 1)

def transform_submap(submap: GridMap, pose: np.ndarray, 
                    submap_res: float = 0.05, global_res: float = 0.1) -> GridMap:
    """使用给定位姿变换子图"""
    transformed = GridMap()
    
    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        
        # 转换到物理坐标（使用子图分辨率）
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 转换到全局地图栅格坐标（使用全局地图分辨率）
        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))
        
        transformed.update_occ(gi_glob, gj_glob, p_sub)
    
    return transformed

def visualize_optimization(global_map: GridMap,
                         submap: GridMap,
                         true_pose: np.ndarray,
                         init_pose: np.ndarray,
                         opt_pose: np.ndarray,
                         save_path: str = None,
                         submap_id: int = -1,
                         submap_res: float = 0.05,
                         global_res: float = 0.1):
    """可视化优化结果"""
    # 准备全局地图和子图
    global_grid = global_map.to_matrix()
    
    # 转换子图到全局坐标系
    def transform_submap_vis(pose: np.ndarray) -> np.ndarray:
        grid = np.zeros_like(global_grid)
        for key, p_meas in submap.occ_map.items():
            if p_meas < 0.6:  # 只显示占用栅格
                continue
            sub_i, sub_j = decode_key(key)
            p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
            p_w = pose[:3, :3] @ p_s + pose[:3, 3]
            gi_glob = int(np.round(p_w[0] / global_res))
            gj_glob = int(np.round(p_w[1] / global_res))
            if 0 <= gi_glob - global_map.min_i < grid.shape[0] and \
               0 <= gj_glob - global_map.min_j < grid.shape[1]:
                grid[gi_glob - global_map.min_i, 
                     gj_glob - global_map.min_j] = 1
        return grid
    
    # 生成三个位置的子图
    true_grid = transform_submap_vis(true_pose)
    init_grid = transform_submap_vis(init_pose)
    opt_grid = transform_submap_vis(opt_pose)
    
    # 计算误差（占用栅格的不匹配率）
    def compute_error(pred_grid, true_grid):
        pred_points = np.sum(pred_grid > 0)
        if pred_points == 0:
            return 1.0  # 如果没有预测点，返回100%错误
        mismatch = np.sum((pred_grid > 0) & (true_grid == 0))
        return mismatch / pred_points
    
    init_error = compute_error(init_grid, true_grid)
    opt_error = compute_error(opt_grid, true_grid)
    
    # 创建可视化图像
    vis = np.zeros((*global_grid.shape, 3))
    
    # 1. 设置全局地图的灰度背景
    background = np.zeros_like(global_grid)
    background[global_grid <= 0.4] = 0.7  # 空闲为灰色
    background[global_grid >= 0.6] = 0.0  # 占用为黑色
    background[np.logical_and(global_grid > 0.4, global_grid < 0.6)] = 0.3  # 未知为深灰色
    
    # 2. 将灰度背景复制到三个通道
    for i in range(3):
        vis[..., i] = background
    
    # 3. 在背景上叠加彩色标记
    # 蓝色表示初始位置
    vis[init_grid > 0] = [0, 0, 1]  # 蓝色
    # 红色表示优化后位置
    vis[opt_grid > 0] = [1, 0, 0]   # 红色
    # 绿色表示真值位置
    vis[true_grid > 0] = [0, 1, 0]  # 绿色
    
    # 创建figure和axes，为图例留出空间
    fig = plt.figure(figsize=(15, 10))  # 加宽图形以容纳图例
    gs = plt.GridSpec(1, 2, width_ratios=[4, 1])  # 创建网格，左侧4份，右侧1份
    ax = fig.add_subplot(gs[0])  # 主图在左侧
    ax_legend = fig.add_subplot(gs[1])  # 图例在右侧
    ax_legend.axis('off')  # 关闭图例区域的坐标轴
    
    # 显示主图
    ax.imshow(vis)
    ax.set_title(f'子图ID: {submap_id} 优化结果\n栅格占用不匹配率: 优化前: {init_error*100:.1f}%, 优化后: {opt_error*100:.1f}%')
    
    # 添加图例到右侧
    legend_elements = [ 
        plt.Rectangle((0, 0), 1, 1, fc=[0, 0, 1], label='优化前'),
        plt.Rectangle((0, 0), 1, 1, fc=[1, 0, 0], label='优化后'),
        plt.Rectangle((0, 0), 1, 1, fc=[0, 1, 0], label='真值'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.7, 0.7, 0.7], label='空闲区域'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.3, 0.3, 0.3], label='未知区域'),
        plt.Rectangle((0, 0), 1, 1, fc=[0, 0, 0], label='占用区域'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center left', fontsize=12)
    
    # 计算位姿误差
    init_trans_error = np.linalg.norm(init_pose[:2, 3] - true_pose[:2, 3])
    init_rot_error = np.abs(np.arctan2(init_pose[1, 0], init_pose[0, 0]) - 
                           np.arctan2(true_pose[1, 0], true_pose[0, 0]))
    opt_trans_error = np.linalg.norm(opt_pose[:2, 3] - true_pose[:2, 3])
    opt_rot_error = np.abs(np.arctan2(opt_pose[1, 0], opt_pose[0, 0]) - 
                          np.arctan2(true_pose[1, 0], true_pose[0, 0]))
    
    # 用figtext在图片下方显示误差信息，不遮挡地图
    fig.subplots_adjust(bottom=0.18)  # 给下方留出空间
    info_text = (
        f'位姿误差 (相对于真值)：\n'
        f'优化前: {init_trans_error:.3f} 米, {np.rad2deg(init_rot_error):.1f}°    '
        f'优化后: {opt_trans_error:.3f} 米, {np.rad2deg(opt_rot_error):.1f}°'
    )
    fig.text(0.5, 0.05, info_text, ha='center', va='center', fontsize=13, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path)
        print(f"优化结果已保存到: {save_path}")
    else:
        plt.show()

def load_gt_pose_from_file(gt_file_path: str, timestamp: float) -> np.ndarray:
    """
    从path_pg_rtk.txt文件中加载指定时间戳的真值位姿。
    文件格式：timestamp tx ty tz qx qy qz qw
    """
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            file_ts_float = float(parts[0])
            
            # 使用math.isclose进行浮点数比较，只使用绝对误差容忍度
            if math.isclose(file_ts_float, timestamp, rel_tol=0, abs_tol=0.01):
                print(f"[DEBUG] 匹配到时间戳: 文件 {file_ts_float}, 查找 {timestamp}")
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) # z轴在2D优化中通常不考虑
                
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
                # 创建旋转矩阵
                rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

                # 创建4x4变换矩阵
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [x, y, z]
                print(f"[DEBUG] 从path_pg_rtk.txt加载位姿: x={x:.3f}, y={y:.3f}, z={z:.3f}\n旋转矩阵:\n{rot_matrix}\n变换矩阵:\n{transform_matrix}")
                print(f"[DEBUG] 找到的时间戳: {file_ts_float}, 查询的时间戳: {timestamp}")

                return transform_matrix

    return None # 没有找到匹配的时间戳

def visualize_multi_res_intermediate(submap: GridMap,
                                   global_map: GridMap,
                                   true_pose: np.ndarray,
                                   init_pose: np.ndarray,
                                   current_pose: np.ndarray,
                                   resolution: float,
                                   layer_index: int,
                                   error: float):
    """可视化多分辨率匹配的中间结果"""
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'多分辨率匹配 - 第{layer_index+1}层 (分辨率: {resolution}m)', fontsize=16)
    
    # 获取全局地图
    global_grid = global_map.to_matrix()
    
    # 计算物理范围
    x_min = global_map.min_i * resolution
    x_max = global_map.max_i * resolution
    y_min = global_map.min_j * resolution
    y_max = global_map.max_j * resolution
    
    # 1. 左上：全局地图概览
    vis_global = np.zeros_like(global_grid)
    vis_global[global_grid > 0.6] = 1.0  # 占用
    vis_global[global_grid < 0.4] = 0.3  # 空闲
    
    ax1.imshow(vis_global, cmap='gray', origin='upper',
               extent=[y_min, y_max, x_max, x_min])
    ax1.set_title(f'全局地图 (分辨率: {resolution}m)')
    ax1.set_xlabel('Y (米)')
    ax1.set_ylabel('X (米)')
    
    # 2. 右上：真值位置叠加
    overlay1 = np.zeros((*global_grid.shape, 3))
    overlay1[..., 0] = vis_global
    overlay1[..., 1] = vis_global 
    overlay1[..., 2] = vis_global
    
    # 叠加真值子图（绿色）- 使用当前分辨率的子图和分辨率
    true_submap_grid = transform_submap_to_size(submap, true_pose, global_grid.shape, 
                                               submap_res=resolution, global_res=resolution)
    true_mask = true_submap_grid > 0.6
    overlay1[true_mask] = [0, 1, 0]  # 绿色
    
    ax2.imshow(overlay1, origin='upper', extent=[y_min, y_max, x_max, x_min])
    ax2.set_title('真值位置 (绿色)')
    ax2.set_xlabel('Y (米)')
    ax2.set_ylabel('X (米)')
    
    # 3. 左下：初始位置叠加
    overlay2 = np.zeros((*global_grid.shape, 3))
    overlay2[..., 0] = vis_global
    overlay2[..., 1] = vis_global
    overlay2[..., 2] = vis_global
    
    # 叠加初始位置子图（蓝色）- 使用当前分辨率
    init_submap_grid = transform_submap_to_size(submap, init_pose, global_grid.shape,
                                               submap_res=resolution, global_res=resolution)
    init_mask = init_submap_grid > 0.6
    overlay2[init_mask] = [0, 0, 1]  # 蓝色
    
    ax3.imshow(overlay2, origin='upper', extent=[y_min, y_max, x_max, x_min])
    ax3.set_title('初始位置 (蓝色)')
    ax3.set_xlabel('Y (米)')
    ax3.set_ylabel('X (米)')
    
    # 4. 右下：当前优化结果叠加
    overlay3 = np.zeros((*global_grid.shape, 3))
    overlay3[..., 0] = vis_global
    overlay3[..., 1] = vis_global
    overlay3[..., 2] = vis_global
    
    # 叠加当前位置子图（红色）- 使用当前分辨率
    current_submap_grid = transform_submap_to_size(submap, current_pose, global_grid.shape,
                                                  submap_res=resolution, global_res=resolution)
    current_mask = current_submap_grid > 0.6
    overlay3[current_mask] = [1, 0, 0]  # 红色
    
    ax4.imshow(overlay3, origin='upper', extent=[y_min, y_max, x_max, x_min])
    ax4.set_title(f'优化后位置 (红色) - 误差: {error:.4f}')
    ax4.set_xlabel('Y (米)')
    ax4.set_ylabel('X (米)')
    
    # 计算并显示误差信息
    trans_error = np.linalg.norm(current_pose[:2, 3] - true_pose[:2, 3])
    rot_error_rad = abs(np.arctan2(current_pose[1, 0], current_pose[0, 0]) - 
                       np.arctan2(true_pose[1, 0], true_pose[0, 0]))
    rot_error_deg = np.rad2deg(rot_error_rad)
    
    # 在图的底部添加误差信息
    error_text = f'位姿误差: {trans_error:.3f}m, {rot_error_deg:.1f}° | 子图栅格数: {len(submap.occ_map)}'
    fig.text(0.5, 0.02, error_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # 为底部文字留出空间
    plt.show()

def load_multi_resolution_global_maps(folder_path: str) -> dict:
    """加载多分辨率全局地图
    
    Args:
        folder_path: 包含多分辨率地图文件的文件夹路径
        
    Returns:
        字典，键为分辨率，值为对应的GridMap对象
    """
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
    """生成多分辨率子图
    
    Args:
        submap: 原始高分辨率子图 (0.05m)
        
    Returns:
        字典，键为分辨率，值为对应的GridMap对象
    """
    resolutions = [0.1, 0.2, 0.4, 0.8, 1.6]
    submaps = {}
    
    print("生成多分辨率子图...")
    
    # 临时重定向downsample_map的输出
    import sys
    from io import StringIO
    
    # 首先将原始0.05m子图转换为0.1m
    old_stdout = sys.stdout
    sys.stdout = StringIO()  # 暂时禁用输出
    try:
        base_submap = downsample_map(submap, 0.05, 0.1)
        base_submap = ternarize_map(base_submap)
        submaps[0.1] = base_submap
        
        # 基于0.1m子图生成其他分辨率
        for res in resolutions[1:]:
            downsampled = downsample_map(base_submap, 0.1, res)
            submaps[res] = downsampled
    finally:
        sys.stdout = old_stdout  # 恢复输出
    
    # 详细显示结果，包括占用栅格统计
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
                                true_pose: np.ndarray = None,
                                visualize: bool = False) -> tuple:
    """多分辨率位姿优化
    
    Args:
        multi_res_submaps: 多分辨率子图字典
        multi_res_global_maps: 多分辨率全局地图字典
        init_pose: 初始位姿
        true_pose: 真值位姿（用于中间结果可视化）
        visualize: 是否可视化中间过程
        
    Returns:
        (优化后的位姿, 最终误差)
    """
    resolutions = [1.6, 0.8, 0.4, 0.2, 0.1]  # 从低分辨率到高分辨率
    current_pose = init_pose.copy()
    
    print("开始多分辨率优化...")
    print(f"可用分辨率层: {sorted([r for r in resolutions if r in multi_res_submaps and r in multi_res_global_maps])}")
    if visualize and true_pose is not None:
        print("将显示每层的中间匹配结果")
    
    # 自适应参数配置 - 优化后的参数
    param_config = {
        # 低分辨率：更大搜索范围，更多粒子，更少迭代（粗略定位）
        1.6: {'spread': (4.0, 4.0, np.deg2rad(30.0)), 'particles': 20, 'iterations': 80},
        0.8: {'spread': (2.0, 2.0, np.deg2rad(15.0)), 'particles': 120, 'iterations': 100},
        # 中分辨率：平衡搜索范围和精度
        0.4: {'spread': (1.0, 1.0, np.deg2rad(15.0)), 'particles': 100, 'iterations': 120},
        0.2: {'spread': (0.5, 0.5, np.deg2rad(10.0)), 'particles': 100, 'iterations': 150},
        # 高分辨率：小搜索范围，高精度
        0.1: {'spread': (0.3, 0.3, np.deg2rad(5.0)), 'particles': 120, 'iterations': 200}
    }
    
    successful_optimizations = 0
    layer_results = {}  # 记录每层的结果
    
    for i, res in enumerate(resolutions):
        if res not in multi_res_submaps or res not in multi_res_global_maps:
            print(f"跳过分辨率 {res}m (缺少数据)")
            continue
        
        print(f"\n===== 分辨率 {res}m 优化 =====")
        
        submap = multi_res_submaps[res]
        global_map = multi_res_global_maps[res]
        
        # 统计子图和全局地图的特征信息
        submap_occupied = sum(1 for p in submap.occ_map.values() if p > 0.6)
        submap_free = sum(1 for p in submap.occ_map.values() if p < 0.4) 
        submap_unknown = len(submap.occ_map) - submap_occupied - submap_free
        
        global_occupied = sum(1 for p in global_map.occ_map.values() if p > 0.6)
        global_free = sum(1 for p in global_map.occ_map.values() if p < 0.4)
        global_unknown = len(global_map.occ_map) - global_occupied - global_free
        
        print(f"子图特征: 总计{len(submap.occ_map)} (占用:{submap_occupied}, 空闲:{submap_free}, 未知:{submap_unknown})")
        print(f"全局地图特征: 总计{len(global_map.occ_map)} (占用:{global_occupied}, 空闲:{global_free}, 未知:{global_unknown})")
        
        # 检查子图占用比例是否在合理范围内
        submap_occ_ratio = submap_occupied / max(len(submap.occ_map), 1)
        if submap_occ_ratio > 0.85 or submap_occ_ratio < 0.1:
            print(f"跳过分辨率 {res}m (占用比例{submap_occ_ratio*100:.1f}% 超出范围[10%,85%])")
            continue

        # 获取当前分辨率的参数配置
        config = param_config[res]
        spread_x, spread_y, spread_theta = config['spread']
        n_particles = config['particles']
        n_iterations = config['iterations']
        
        print(f"搜索范围: ±{spread_x}m, ±{spread_y}m, ±{np.rad2deg(spread_theta):.1f}°")
        print(f"粒子数: {n_particles}, 迭代数: {n_iterations}")
        
        # 计算优化前的误差
        print(f"计算优化前误差...")
        initial_error = compute_matching_error(submap, global_map, current_pose, 
                                             submap_res=res, global_res=res, debug=True)
        print(f"优化前误差: {initial_error:.6f}")
        
        # 检查子图是否有足够的占用栅格进行匹配
        if submap_occupied == 0:
            print(f"警告：分辨率 {res}m 的子图没有占用栅格，跳过优化")
            continue
            
        # 显示当前位姿信息
        current_x, current_y = current_pose[:2, 3]
        current_theta = np.arctan2(current_pose[1, 0], current_pose[0, 0])
        print(f"当前位姿: x={current_x:.3f}, y={current_y:.3f}, theta={np.rad2deg(current_theta):.1f}°")
        
        try:
            # 执行当前分辨率的优化
            # 注意：这里关闭粒子滤波的实时可视化，改用层级完成后的中间结果可视化
            print(f"开始粒子滤波优化...")
            optimized_pose, final_error = match_submap_with_particle_filter(
                submap, global_map, current_pose,
                n_particles=n_particles,
                n_iterations=n_iterations,
                visualize=True,  # 关闭粒子滤波实时可视化，避免干扰
                spread=(spread_x, spread_y, spread_theta),
                submap_res=res,
                global_res=res
            )
            print(f"粒子滤波优化完成，最终误差: {final_error:.6f}")
            
            # 显示优化后位姿信息
            opt_x, opt_y = optimized_pose[:2, 3]
            opt_theta = np.arctan2(optimized_pose[1, 0], optimized_pose[0, 0])
            print(f"优化后位姿: x={opt_x:.3f}, y={opt_y:.3f}, theta={np.rad2deg(opt_theta):.1f}°")
            
            # 验证优化是否真的返回了新的位姿
            if np.allclose(optimized_pose, current_pose, atol=1e-6):
                print("警告：粒子滤波返回的位姿与输入位姿完全相同！")
            
            # 计算位姿变化
            pose_diff = np.linalg.norm(optimized_pose[:2, 3] - current_pose[:2, 3])
            angle_diff = abs(np.arctan2(optimized_pose[1, 0], optimized_pose[0, 0]) - 
                           np.arctan2(current_pose[1, 0], current_pose[0, 0]))
            
            # 计算误差改善
            error_improvement = (initial_error - final_error) / max(initial_error, 1e-6)
            
            print(f"误差变化: {initial_error:.6f} → {final_error:.6f} (改善: {error_improvement*100:.1f}%)")
            print(f"位姿变化: {pose_diff:.4f}m, {np.rad2deg(angle_diff):.2f}°")
            
            # 记录当前层的结果
            layer_results[res] = {
                'initial_error': initial_error,
                'final_error': final_error,
                'error_improvement': error_improvement,
                'pose_diff': pose_diff,
                'angle_diff': angle_diff,
                'optimized_pose': optimized_pose.copy()
            }
            
            # 智能位姿更新策略
            update_pose = False
            update_reason = ""
            
            if error_improvement > 0.05 and pose_diff > 0.02:  # 显著改善
                update_pose = True
                update_reason = "显著改善"
                successful_optimizations += 1
            elif final_error < initial_error and pose_diff > 0.01:  # 有改善且有位姿变化
                update_pose = True
                update_reason = "有效改善"
                successful_optimizations += 1
            elif res >= 0.8 and pose_diff > 0.02:  # 低分辨率层，有一定位姿变化
                update_pose = True
                update_reason = "低分辨率层接受"
            elif res <= 0.2 and error_improvement > 0.01:  # 高分辨率层，微小改善也接受
                update_pose = True
                update_reason = "高分辨率层微调"
            
            if update_pose:
                current_pose = optimized_pose.copy()
                print(f"✓ 更新位姿 ({update_reason})")
            else:
                print("× 保持原位姿")
            
            # 如果开启可视化且有真值位姿，显示当前层的中间结果
            if visualize and true_pose is not None:
                # 使用visualize_optimization显示中间结果
                print(f"显示分辨率 {res}m 的匹配结果...")
                visualize_optimization(
                    global_map, multi_res_submaps[res], true_pose, init_pose, current_pose,
                    save_path=None, submap_id=f"MultiRes_{res}m_Layer{i+1}",
                    submap_res=res, global_res=res
                )
                
        except Exception as e:
            print(f"分辨率 {res}m 优化失败: {e}")
            continue
        
        # 智能层级跳跃：如果当前层效果很好，可以考虑跳过下一层
        if res == 1.6 and error_improvement > 0.2 and pose_diff > 0.1:
            print("🚀 1.6m层效果优秀，跳过0.8m层，直接进入0.4m层优化")
            # 这里可以通过修改循环来实现跳跃，但为了代码简洁，我们继续所有层
        elif res == 0.8 and error_improvement > 0.15 and pose_diff > 0.05:
            print("🚀 0.8m层效果良好，可能会获得更好的高分辨率结果")
        elif res == 0.4 and error_improvement > 0.1:
            print("🚀 0.4m层表现良好，高分辨率优化有望取得成功")
    
    # 如果前面的优化效果不好，强制在最高分辨率再次优化
    if successful_optimizations == 0 and 0.1 in multi_res_submaps and 0.1 in multi_res_global_maps:
        print("\n===== 强制最高分辨率优化 =====")
        try:
            config = param_config[0.1]
            spread_x, spread_y, spread_theta = config['spread']
            # 增大搜索范围
            spread_x *= 2
            spread_y *= 2
            spread_theta *= 2
            
            optimized_pose, final_error = match_submap_with_particle_filter(
                multi_res_submaps[0.1], multi_res_global_maps[0.1], current_pose,
                n_particles=120,
                n_iterations=250,
                visualize=visualize,
                spread=(spread_x, spread_y, spread_theta),
                submap_res=0.1,
                global_res=0.1
            )
            current_pose = optimized_pose
        except Exception as e:
            print(f"强制优化也失败: {e}")
    
    # 计算最终误差（使用最高分辨率）
    if 0.1 in multi_res_submaps and 0.1 in multi_res_global_maps:
        final_error = compute_matching_error(
            multi_res_submaps[0.1], 
            multi_res_global_maps[0.1], 
            current_pose,
            submap_res=0.1,
            global_res=0.1
        )
    else:
        final_error = 0.0
    
    # 显示各层优化结果总结
    print(f"\n========== 多分辨率优化总结 ==========")
    print(f"成功优化层数: {successful_optimizations}/{len([r for r in resolutions if r in multi_res_submaps])}")
    print(f"最终误差: {final_error:.6f}")
    
    print("\n各层详细结果:")
    for res in resolutions:
        if res in layer_results:
            result = layer_results[res]
            print(f"  {res}m层: 误差 {result['initial_error']:.4f}→{result['final_error']:.4f} "
                  f"(改善{result['error_improvement']*100:.1f}%), "
                  f"位姿变化 {result['pose_diff']:.3f}m/{np.rad2deg(result['angle_diff']):.1f}°")
    
    # 计算总体改善
    if layer_results:
        first_res = min(layer_results.keys())
        last_res = max(layer_results.keys())
        total_improvement = (layer_results[first_res]['initial_error'] - final_error) / layer_results[first_res]['initial_error']
        print(f"\n总体改善: {total_improvement*100:.1f}% (从{layer_results[first_res]['initial_error']:.4f}到{final_error:.4f})")
    
    print("=" * 45)
    
    return current_pose, final_error

def main():
    parser = argparse.ArgumentParser(description="子图位姿优化脚本")
    parser.add_argument("folder_path", type=str, help="包含子图和全局地图的文件夹路径")
    parser.add_argument("--plot", action="store_true", help="显示粒子滤波中间过程的可视化")
    parser.add_argument("--use-gt", action="store_true", help="使用path_pg_rtk.txt中的真值作为初始位姿和参考真值")
    parser.add_argument("--submap", type=int, help="指定要优化的子图ID，默认为随机选择")
    parser.add_argument("--multi-res", action="store_true", 
                       help="使用多分辨率匹配策略：从低分辨率(1.6m)到高分辨率(0.1m)逐层优化，提高收敛效率和鲁棒性")
    args = parser.parse_args()

    folder_path = args.folder_path
    plot_intermediate = args.plot # 获取--plot参数的值
    use_gt = args.use_gt # 获取--use-gt参数的值
    specified_submap_id = args.submap # 获取--submap参数的值
    use_multi_res = args.multi_res # 获取--multi-res参数的值

    # 1. 加载全局地图（单分辨率或多分辨率）
    if use_multi_res:
        try:
            multi_res_global_maps = load_multi_resolution_global_maps(folder_path)
            global_map = multi_res_global_maps[0.1]  # 使用最高分辨率进行可视化
        except Exception as e:
            print(f"多分辨率地图加载失败: {e}")
            print("回退到单分辨率模式...")
            use_multi_res = False
            global_map_path = os.path.join(folder_path, 'global_map.bin')
            global_map = load_global_map(global_map_path)
    else:
        global_map_path = os.path.join(folder_path, 'global_map.bin')
        print("加载单分辨率全局地图...")
        global_map = load_global_map(global_map_path)
    
    # 2. 选择一个子图
    submap_files = [f for f in os.listdir(folder_path) 
                    if f.startswith('submap_') and f.endswith('.bin')]
    if not submap_files:
        print("错误：没有找到子图文件")
        return
    
    target_file = None
    submap_id = -1

    if specified_submap_id is not None:
        target_file = f'submap_{specified_submap_id}.bin'
        submap_id = specified_submap_id
        if target_file not in submap_files:
            print(f"错误：未找到指定的子图文件: {target_file}")
            sys.exit(1)
        print(f"选择子图 {submap_id} 进行优化 (用户指定)")
    else:
        target_file = np.random.choice(submap_files)
        submap_id = int(target_file.split('_')[1].split('.')[0])
        print(f"选择子图 {submap_id} 进行优化 (随机选择)")
    
    # 3. 加载子图
    submap_path = os.path.join(folder_path, target_file)
    _, ts, loaded_true_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(submap_path)
    
    # 创建子图对象
    submap = GridMap()
    for key, prob in occ_map.items():
        gi, gj = decode_key(key)
        submap.update_occ(gi, gj, prob)
    
    true_pose = loaded_true_pose # 默认从submap加载的为真值
    init_pose = None

    if use_gt:
        gt_file_path = os.path.join(folder_path, 'path_pg_rtk.txt')
        if not os.path.exists(gt_file_path):
            print(f"错误：使用--use-gt选项时，未找到真值文件: {gt_file_path}")
            sys.exit(1) # 退出
        
        gt_pose = load_gt_pose_from_file(gt_file_path, ts)
        if gt_pose is None:
            print(f"错误：在 {gt_file_path} 中未找到时间戳 {ts} 对应的真值位姿，程序退出。")
            sys.exit(1) # 退出
        
        true_pose = gt_pose  # 使用从path_pg_rtk.txt中加载的真值
        init_pose = loaded_true_pose # 初始值用加载submap中的pose
        print("已使用path_pg_rtk.txt中的真值位姿，并以加载子图的位姿作为初始位姿。")
    else:
        # 4. 添加初始噪声
        init_pose = add_noise_to_pose(loaded_true_pose, 0.5, 10.0)  # 0.5m, 10度
        print("已添加噪声作为初始位姿。")
    
    # 5. 优化位姿
    if use_multi_res:
        print("使用多分辨率优化策略...")
        # 生成多分辨率子图
        multi_res_submaps = generate_multi_resolution_submap(submap)
        
        # 执行多分辨率优化
        opt_pose, error = multi_resolution_optimization(
            multi_res_submaps, 
            multi_res_global_maps, 
            init_pose,
            true_pose,  # 传递真值位姿用于可视化
            visualize=plot_intermediate
        )
    else:
        print("使用单分辨率优化...")
        opt_pose, error = optimize_submap_pose(submap, global_map, init_pose, visualize=plot_intermediate)
    
    print(f"优化完成，最终误差: {error:.6f}")
    
    print(f"[DEBUG] 传递给可视化的真值位姿 (true_pose):\n{true_pose}")
    print(f"[DEBUG] 传递给可视化的初始位姿 (init_pose):\n{init_pose}")
    print(f"[DEBUG] 传递给可视化的优化位姿 (opt_pose):\n{opt_pose}")
    
    # 6. 直接显示结果
    if use_multi_res:
        # 多分辨率模式，使用最高分辨率进行最终可视化
        final_submap = multi_res_submaps[0.1] if 0.1 in multi_res_submaps else submap
        visualize_optimization(
            global_map, final_submap, true_pose, init_pose, opt_pose, None, submap_id,
            submap_res=0.1, global_res=0.1
        )
    else:
        # 单分辨率模式
        visualize_optimization(
            global_map, submap, true_pose, init_pose, opt_pose, None, submap_id,
            submap_res=0.05, global_res=0.1
        )
    plt.show()  # 确保图像显示出来

if __name__ == '__main__':
    main() 
