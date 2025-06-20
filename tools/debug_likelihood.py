#!/usr/bin/env python3
"""
调试似然优化问题的脚本
"""

import os
import sys
import numpy as np
from optimize_submap import (
    compute_likelihood_score, pose_to_params, params_to_pose,
    gradient_optimization, load_multi_resolution_global_maps, 
    generate_multi_resolution_submap, load_submap, GridMap, decode_key, add_noise_to_pose
)

def debug_likelihood_score():
    """调试似然分数计算"""
    print("=== 调试似然分数计算 ===")
    
    # 创建简单的测试地图
    submap = GridMap()
    global_map = GridMap()
    
    # 在子图中添加一些占用和空闲栅格
    for i in range(-3, 4):
        for j in range(-3, 4):
            if abs(i) <= 1 and abs(j) <= 1:  # 中心区域占用
                submap.set_occ_direct(i, j, 0.9)
            else:  # 外围区域空闲
                submap.set_occ_direct(i, j, 0.1)
    
    # 在全局地图中添加类似的模式，但稍微偏移
    for i in range(-2, 5):
        for j in range(-2, 5):
            if abs(i-1) <= 1 and abs(j-1) <= 1:  # 偏移的占用区域
                global_map.set_occ_direct(i, j, 0.8)
            else:
                global_map.set_occ_direct(i, j, 0.2)
    
    print(f"子图栅格数量: {len(submap.occ_map)}")
    print(f"全局地图栅格数量: {len(global_map.occ_map)}")
    
    # 测试不同位姿的似然分数
    poses = [
        np.eye(4),  # 原始位姿
        add_noise_to_pose(np.eye(4), 0.1, 5.0),  # 轻微噪声
        add_noise_to_pose(np.eye(4), 0.5, 15.0),  # 较大噪声
    ]
    
    for i, pose in enumerate(poses):
        score = compute_likelihood_score(submap, global_map, pose, 0.1, 0.1, debug=True)
        print(f"位姿 {i}: 似然分数 = {score:.6f}")
        print(f"位姿参数: {pose_to_params(pose)}")
        print()

def debug_gradient_optimization():
    """调试梯度优化"""
    print("\n=== 调试梯度优化 ===")
    
    # 创建测试数据
    submap = GridMap()
    global_map = GridMap()
    
    # 创建简单的测试模式 - 圆形占用区域
    for i in range(-5, 6):
        for j in range(-5, 6):
            if i*i + j*j <= 4:  # 圆形占用区域
                submap.set_occ_direct(i, j, 0.9)
            else:
                submap.set_occ_direct(i, j, 0.1)
    
    # 在全局地图中创建偏移的圆形
    for i in range(-4, 7):
        for j in range(-4, 7):
            if (i-1)*(i-1) + (j-1)*(j-1) <= 4:  # 偏移的圆形占用区域
                global_map.set_occ_direct(i, j, 0.8)
            else:
                global_map.set_occ_direct(i, j, 0.2)
    
    print(f"子图栅格数量: {len(submap.occ_map)}")
    print(f"全局地图栅格数量: {len(global_map.occ_map)}")
    
    # 初始位姿（有噪声）
    init_pose = add_noise_to_pose(np.eye(4), 0.3, 10.0)
    
    print("初始位姿参数:", pose_to_params(init_pose))
    
    # 测试似然分数的梯度
    init_params = pose_to_params(init_pose)
    init_score = compute_likelihood_score(submap, global_map, init_pose, 0.1, 0.1, debug=True)
    print(f"初始似然分数: {init_score:.6f}")
    
    # 测试小的参数变化
    delta = 0.01
    for i in range(3):
        test_params = init_params.copy()
        test_params[i] += delta
        test_pose = params_to_pose(test_params)
        test_score = compute_likelihood_score(submap, global_map, test_pose, 0.1, 0.1)
        gradient = (test_score - init_score) / delta
        print(f"参数 {i} 梯度: {gradient:.6f}")
    
    # 进行梯度优化
    try:
        opt_pose, final_score = gradient_optimization(
            submap, global_map, init_pose,
            submap_res=0.1, global_res=0.1,
            max_iterations=100,
            tolerance=1e-8,  # 降低容差
            debug=True
        )
        
        print("优化后位姿参数:", pose_to_params(opt_pose))
        print("最终似然分数:", final_score)
        
        # 计算位姿变化
        opt_params = pose_to_params(opt_pose)
        pose_diff = np.linalg.norm(opt_params[:2] - init_params[:2])
        angle_diff = abs(opt_params[2] - init_params[2])
        
        print(f"位姿变化: 平移 {pose_diff:.3f}m, 旋转 {np.rad2deg(angle_diff):.1f}°")
        
    except Exception as e:
        print(f"梯度优化失败: {e}")
        import traceback
        traceback.print_exc()

def debug_real_data():
    """调试真实数据"""
    print("\n=== 调试真实数据 ===")
    
    # 检查是否有测试数据
    test_data_path = "../data/1_rtk"
    if not os.path.exists(test_data_path):
        print(f"测试数据路径不存在: {test_data_path}")
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
        
        # 测试不同分辨率
        for res in [1.6, 0.8, 0.4, 0.2, 0.1]:
            if res in multi_res_submaps and res in multi_res_global_maps:
                print(f"\n--- 分辨率 {res}m ---")
                submap_res = multi_res_submaps[res]
                global_map_res = multi_res_global_maps[res]
                
                # 统计栅格信息
                occupied_cells = sum(1 for p in submap_res.occ_map.values() if p > 0.6)
                free_cells = sum(1 for p in submap_res.occ_map.values() if p < 0.4)
                total_cells = len(submap_res.occ_map)
                
                print(f"子图: 总计{total_cells}, 占用{occupied_cells}, 空闲{free_cells}")
                print(f"全局地图: {len(global_map_res.occ_map)} 个栅格")
                
                # 测试真值位姿的似然分数
                true_score = compute_likelihood_score(submap_res, global_map_res, true_pose, 
                                                    submap_res=res, global_res=res, debug=True)
                print(f"真值位姿似然分数: {true_score:.6f}")
                
                # 测试噪声位姿
                noisy_pose = add_noise_to_pose(true_pose, 0.5, 10.0)
                noisy_score = compute_likelihood_score(submap_res, global_map_res, noisy_pose, 
                                                     submap_res=res, global_res=res)
                print(f"噪声位姿似然分数: {noisy_score:.6f}")
                
                # 测试梯度优化
                try:
                    opt_pose, final_score = gradient_optimization(
                        submap_res, global_map_res, noisy_pose,
                        submap_res=res, global_res=res,
                        max_iterations=50,
                        tolerance=1e-8,
                        debug=True
                    )
                    
                    score_improvement = (noisy_score - final_score) / max(noisy_score, 1e-6)
                    pose_diff = np.linalg.norm(opt_pose[:2, 3] - noisy_pose[:2, 3])
                    
                    print(f"优化结果: 分数改善 {score_improvement*100:.1f}%, 位姿变化 {pose_diff:.3f}m")
                    
                except Exception as e:
                    print(f"优化失败: {e}")
        
    except Exception as e:
        print(f"真实数据调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("开始调试似然优化问题...")
    
    debug_likelihood_score()
    debug_gradient_optimization()
    debug_real_data()
    
    print("\n调试完成！") 