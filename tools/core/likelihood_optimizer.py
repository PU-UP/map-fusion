#!/usr/bin/env python3
"""
似然优化模块
功能：
1. 基于概率地图的梯度下降优化位姿参数
2. 支持多分辨率似然优化策略
3. 提供多候选位姿优化功能
4. 位姿鲁棒性检查
"""

import numpy as np
import time
from typing import Optional, Tuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from fuse_submaps import GridMap, decode_key, encode_key, add_noise_to_pose

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

def transform_submap_for_visualization(submap: GridMap, pose: np.ndarray, 
                                     submap_res: float, global_res: float, 
                                     global_map: GridMap) -> np.ndarray:
    """生成与全局地图同shape的0/1矩阵用于可视化"""
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

def gradient_optimization(submap: GridMap,
                         global_map: GridMap,
                         init_pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1,
                         max_iterations: int = 100,
                         tolerance: float = 1e-6,
                         debug: bool = False,
                         method: str = 'Nelder-Mead',
                         visualize: bool = False) -> Tuple[np.ndarray, float]:
    """梯度优化位姿"""
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

def check_pose_robustness(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float,
                         global_res: float,
                         debug: bool = False) -> Tuple[bool, float, list]:
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
                                           n_candidates: int = 3) -> Tuple[np.ndarray, float]:
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
    
    print(f"\n========== 多分辨率似然优化总结 ==========")
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
    
    return final_pose, final_score

def match_submap_with_likelihood(submap: GridMap,
                                global_map: GridMap,
                                init_pose: np.ndarray,
                                submap_res: float = 0.05,
                                global_res: float = 0.1,
                                max_iterations: int = 100,
                                tolerance: float = 1e-6,
                                debug: bool = False,
                                method: str = 'Nelder-Mead',
                                visualize: bool = False) -> Tuple[np.ndarray, float]:
    """使用似然优化匹配子图与全局地图"""
    print("使用似然优化进行匹配...")
    
    start_time = time.time()
    
    optimized_pose, final_score = gradient_optimization(
        submap, global_map, init_pose,
        submap_res=submap_res,
        global_res=global_res,
        max_iterations=max_iterations,
        tolerance=tolerance,
        debug=debug,
        method=method,
        visualize=visualize
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if debug:
        print(f"似然优化总耗时: {total_time:.4f}s")
    else:
        print(f"优化耗时: {total_time:.4f}s")
    
    return optimized_pose, final_score 