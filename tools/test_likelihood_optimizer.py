#!/usr/bin/env python3
"""
测试likelihood_optimizer模块的功能
"""

import numpy as np
import os
import sys
from fuse_submaps import GridMap, decode_key, encode_key, add_noise_to_pose
from likelihood_optimizer import (
    compute_likelihood_score,
    pose_to_params,
    params_to_pose,
    gradient_optimization,
    check_pose_robustness,
    match_submap_with_likelihood
)

def create_test_maps():
    """创建测试用的子图和全局地图"""
    # 创建子图
    submap = GridMap()
    # 添加一些占用栅格
    for i in range(-5, 6):
        for j in range(-5, 6):
            if abs(i) <= 3 and abs(j) <= 3:  # 内部区域
                submap.update_occ(i, j, 0.8)  # 占用
            else:  # 外部区域
                submap.update_occ(i, j, 0.2)  # 空闲
    
    # 创建全局地图
    global_map = GridMap()
    # 添加一些占用栅格，稍微偏移
    for i in range(-10, 11):
        for j in range(-10, 11):
            if abs(i-1) <= 3 and abs(j-1) <= 3:  # 偏移的内部区域
                global_map.update_occ(i, j, 0.8)  # 占用
            else:  # 外部区域
                global_map.update_occ(i, j, 0.2)  # 空闲
    
    return submap, global_map

def test_basic_functions():
    """测试基本函数"""
    print("=== 测试基本函数 ===")
    
    # 测试位姿转换
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 0.0]
    pose[:3, :3] = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])  # 30度旋转
    
    params = pose_to_params(pose)
    print(f"原始位姿参数: {params}")
    
    restored_pose = params_to_pose(params)
    restored_params = pose_to_params(restored_pose)
    print(f"恢复后位姿参数: {restored_params}")
    
    # 检查是否一致
    if np.allclose(params, restored_params, atol=1e-6):
        print("✓ 位姿转换函数工作正常")
    else:
        print("✗ 位姿转换函数有问题")

def test_likelihood_score():
    """测试似然分数计算"""
    print("\n=== 测试似然分数计算 ===")
    
    submap, global_map = create_test_maps()
    
    # 测试不同位姿的似然分数
    poses = [
        np.eye(4),  # 原始位姿
        np.eye(4),  # 稍微偏移的位姿
    ]
    poses[1][:3, 3] = [0.5, 0.5, 0.0]
    
    for i, pose in enumerate(poses):
        score = compute_likelihood_score(submap, global_map, pose, 0.05, 0.1)
        print(f"位姿 {i+1} 似然分数: {score:.6f}")

def test_gradient_optimization():
    """测试梯度优化"""
    print("\n=== 测试梯度优化 ===")
    
    submap, global_map = create_test_maps()
    
    # 创建初始位姿（有噪声）
    true_pose = np.eye(4)
    true_pose[:3, 3] = [1.0, 1.0, 0.0]  # 真值位姿
    
    init_pose = add_noise_to_pose(true_pose, 0.3, 10.0)  # 添加噪声
    
    print("开始梯度优化...")
    try:
        optimized_pose, final_score = gradient_optimization(
            submap, global_map, init_pose,
            submap_res=0.05, global_res=0.1,
            max_iterations=50,
            tolerance=1e-6,
            debug=True,
            method='Nelder-Mead',
            visualize=False
        )
        
        # 计算位姿误差
        trans_error = np.linalg.norm(optimized_pose[:2, 3] - true_pose[:2, 3])
        print(f"优化完成，平移误差: {trans_error:.3f}m")
        print(f"最终似然分数: {final_score:.6f}")
        
    except Exception as e:
        print(f"梯度优化失败: {e}")

def test_robustness_check():
    """测试鲁棒性检查"""
    print("\n=== 测试鲁棒性检查 ===")
    
    submap, global_map = create_test_maps()
    
    # 创建一个好的位姿
    good_pose = np.eye(4)
    good_pose[:3, 3] = [1.0, 1.0, 0.0]
    
    is_robust, base_score, perturbed_scores = check_pose_robustness(
        submap, global_map, good_pose, 0.05, 0.1, debug=True
    )
    
    print(f"鲁棒性检查结果: {'✓' if is_robust else '✗'}")

def test_likelihood_matcher():
    """测试似然匹配器"""
    print("\n=== 测试似然匹配器 ===")
    
    submap, global_map = create_test_maps()
    
    # 创建初始位姿（有噪声）
    true_pose = np.eye(4)
    true_pose[:3, 3] = [1.0, 1.0, 0.0]  # 真值位姿
    
    init_pose = add_noise_to_pose(true_pose, 0.5, 15.0)  # 添加较大噪声
    
    print("开始似然匹配...")
    try:
        optimized_pose, final_score = match_submap_with_likelihood(
            submap, global_map, init_pose,
            submap_res=0.05, global_res=0.1,
            max_iterations=100,
            tolerance=1e-6,
            debug=True,
            method='Nelder-Mead',
            visualize=False
        )
        
        # 计算位姿误差
        trans_error = np.linalg.norm(optimized_pose[:2, 3] - true_pose[:2, 3])
        print(f"匹配完成，平移误差: {trans_error:.3f}m")
        print(f"最终似然分数: {final_score:.6f}")
        
    except Exception as e:
        print(f"似然匹配失败: {e}")

def main():
    """主测试函数"""
    print("开始测试likelihood_optimizer模块...")
    
    try:
        test_basic_functions()
        test_likelihood_score()
        test_gradient_optimization()
        test_robustness_check()
        test_likelihood_matcher()
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 