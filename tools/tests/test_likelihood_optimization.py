#!/usr/bin/env python3
"""
测试似然优化功能的脚本
"""

import os
import sys
import numpy as np
from optimize_submap import (
    compute_likelihood_score, pose_to_params, params_to_pose,
    gradient_optimization, multi_resolution_likelihood_optimization,
    load_multi_resolution_global_maps, generate_multi_resolution_submap,
    load_submap, GridMap, decode_key, add_noise_to_pose,
    likelihood_objective
)
from scipy.optimize import minimize

def test_likelihood_score():
    """测试似然分数计算"""
    print("=== 测试似然分数计算 ===")
    
    # 创建简单的测试地图
    submap = GridMap()
    global_map = GridMap()
    
    # 在子图中添加一些占用和空闲栅格
    for i in range(-5, 6):
        for j in range(-5, 6):
            if abs(i) <= 2 and abs(j) <= 2:  # 中心区域占用
                submap.set_occ_direct(i, j, 0.9)
            else:  # 外围区域空闲
                submap.set_occ_direct(i, j, 0.1)
    
    # 在全局地图中添加类似的模式，但稍微偏移
    for i in range(-4, 7):
        for j in range(-4, 7):
            if abs(i-1) <= 2 and abs(j-1) <= 2:  # 偏移的占用区域
                global_map.set_occ_direct(i, j, 0.8)
            else:
                global_map.set_occ_direct(i, j, 0.2)
    
    # 测试不同位姿的似然分数
    poses = [
        np.eye(4),  # 原始位姿
        add_noise_to_pose(np.eye(4), 0.1, 5.0),  # 轻微噪声
        add_noise_to_pose(np.eye(4), 0.5, 15.0),  # 较大噪声
    ]
    
    for i, pose in enumerate(poses):
        score = compute_likelihood_score(submap, global_map, pose, 0.1, 0.1)
        print(f"位姿 {i}: 似然分数 = {score:.6f}")

def test_gradient_optimization():
    """测试梯度优化"""
    print("\n=== 测试梯度优化 ===")
    
    # 创建测试数据
    submap = GridMap()
    global_map = GridMap()
    
    # 创建简单的测试模式
    for i in range(-3, 4):
        for j in range(-3, 4):
            if i*i + j*j <= 4:  # 圆形占用区域
                submap.set_occ_direct(i, j, 0.9)
                global_map.set_occ_direct(i+1, j+1, 0.8)  # 偏移的全局地图
            else:
                submap.set_occ_direct(i, j, 0.1)
                global_map.set_occ_direct(i+1, j+1, 0.2)
    
    # 初始位姿（有噪声）
    init_pose = add_noise_to_pose(np.eye(4), 0.3, 10.0)
    
    print("初始位姿:", pose_to_params(init_pose))
    
    # 进行Nelder-Mead优化
    try:
        result_nm = minimize(
            likelihood_objective,
            pose_to_params(init_pose),
            args=(submap, global_map, 0.1, 0.1),
            method='Nelder-Mead',
            options={
                'maxiter': 50,
                'xatol': 1e-3,
                'fatol': 1e-6,
                'disp': True
            }
        )
        opt_pose_nm = params_to_pose(result_nm.x)
        final_score_nm = compute_likelihood_score(submap, global_map, opt_pose_nm, 0.1, 0.1)
        print("[Nelder-Mead] 优化后位姿:", pose_to_params(opt_pose_nm))
        print("[Nelder-Mead] 最终似然分数:", final_score_nm)
        init_params = pose_to_params(init_pose)
        opt_params_nm = pose_to_params(opt_pose_nm)
        pose_diff_nm = np.linalg.norm(opt_params_nm[:2] - init_params[:2])
        angle_diff_nm = abs(opt_params_nm[2] - init_params[2])
        print(f"[Nelder-Mead] 位姿变化: 平移 {pose_diff_nm:.3f}m, 旋转 {np.rad2deg(angle_diff_nm):.1f}°")
    except Exception as e:
        print(f"Nelder-Mead优化失败: {e}")
    # 进行Powell优化
    try:
        result_pow = minimize(
            likelihood_objective,
            pose_to_params(init_pose),
            args=(submap, global_map, 0.1, 0.1),
            method='Powell',
            options={
                'maxiter': 50,
                'xtol': 1e-3,
                'ftol': 1e-6,
                'disp': True
            }
        )
        opt_pose_pow = params_to_pose(result_pow.x)
        final_score_pow = compute_likelihood_score(submap, global_map, opt_pose_pow, 0.1, 0.1)
        print("[Powell] 优化后位姿:", pose_to_params(opt_pose_pow))
        print("[Powell] 最终似然分数:", final_score_pow)
        opt_params_pow = pose_to_params(opt_pose_pow)
        pose_diff_pow = np.linalg.norm(opt_params_pow[:2] - init_params[:2])
        angle_diff_pow = abs(opt_params_pow[2] - init_params[2])
        print(f"[Powell] 位姿变化: 平移 {pose_diff_pow:.3f}m, 旋转 {np.rad2deg(angle_diff_pow):.1f}°")
    except Exception as e:
        print(f"Powell优化失败: {e}")

def test_multi_resolution_likelihood():
    """测试多分辨率似然优化"""
    print("\n=== 测试多分辨率似然优化 ===")
    
    # 检查是否有测试数据
    test_data_path = "../data/1_rtk"
    if not os.path.exists(test_data_path):
        print(f"测试数据路径不存在: {test_data_path}")
        print("请确保在tools目录下运行此脚本，且data目录包含测试数据")
        return
    
    try:
        # 加载多分辨率全局地图
        multi_res_global_maps = load_multi_resolution_global_maps(test_data_path)
        print(f"成功加载 {len(multi_res_global_maps)} 个分辨率的全局地图")
        
        # 加载一个子图
        submap_files = [f for f in os.listdir(test_data_path) 
                       if f.startswith('submap_') and f.endswith('.bin')]
        if not submap_files:
            print("未找到子图文件")
            return
        
        submap_path = os.path.join(test_data_path, submap_files[0])
        _, ts, true_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(submap_path)
        
        submap = GridMap()
        for key, prob in occ_map.items():
            gi, gj = decode_key(key)
            submap.update_occ(gi, gj, prob)
        
        print(f"加载子图: {submap_files[0]}, 栅格数量: {len(submap.occ_map)}")
        
        # 生成多分辨率子图
        multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=True)
        print(f"生成 {len(multi_res_submaps)} 个分辨率的子图")
        
        # 添加噪声作为初始位姿
        init_pose = add_noise_to_pose(true_pose, 0.5, 10.0)
        
        # 进行多分辨率似然优化
        opt_pose, final_error = multi_resolution_likelihood_optimization(
            multi_res_submaps,
            multi_res_global_maps,
            init_pose,
            true_pose,
            visualize=False,
            debug=True
        )
        
        print(f"多分辨率似然优化完成，最终误差: {final_error:.6f}")
        
        # 计算位姿误差
        init_params = pose_to_params(init_pose)
        opt_params = pose_to_params(opt_pose)
        true_params = pose_to_params(true_pose)
        
        init_error = np.linalg.norm(init_params[:2] - true_params[:2])
        opt_error = np.linalg.norm(opt_params[:2] - true_params[:2])
        
        print(f"位姿误差改善: {init_error:.3f}m → {opt_error:.3f}m")
        
    except Exception as e:
        print(f"多分辨率似然优化测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("开始测试似然优化功能...")
    
    test_likelihood_score()
    test_gradient_optimization()
    test_multi_resolution_likelihood()
    
    print("\n测试完成！") 