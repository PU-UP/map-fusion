#!/usr/bin/env python3
"""
测试批量处理保存功能改进
验证当使用 --submap -1 时：
1. 不保存每个子图的单独结果
2. 只保存批量结果，包含两张可视化图像
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def test_batch_save_improvement():
    """测试批量处理保存功能改进"""
    print("=" * 60)
    print("测试批量处理保存功能改进")
    print("=" * 60)
    
    # 测试数据路径
    test_data_path = "../data/mapping_0506"
    save_path = "../results/test_batch_save"
    
    # 清理之前的测试结果
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    print(f"使用测试数据: {test_data_path}")
    print(f"保存路径: {save_path}")
    
    # 运行批量优化命令
    cmd = [
        sys.executable, "../core/optimize_submap.py",
        test_data_path,
        "--submap", "-1",
        "--save", save_path,
        "--multi-res", "2",  # 使用2层多分辨率以加快测试
        "--debug"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ 批量优化执行成功")
            print("输出:")
            print(result.stdout)
        else:
            print("✗ 批量优化执行失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 批量优化超时")
        return False
    except Exception as e:
        print(f"✗ 执行命令时出错: {e}")
        return False
    
    # 检查保存结果
    print("\n" + "=" * 60)
    print("检查保存结果")
    print("=" * 60)
    
    if not os.path.exists(save_path):
        print("✗ 保存目录不存在")
        return False
    
    # 查找批量优化结果目录
    batch_dirs = [d for d in os.listdir(save_path) if d.startswith("batch_optimization_")]
    if not batch_dirs:
        print("✗ 未找到批量优化结果目录")
        return False
    
    batch_dir = os.path.join(save_path, batch_dirs[0])
    print(f"找到批量优化结果目录: {batch_dir}")
    
    # 检查文件结构
    expected_files = [
        "detailed_results.json",
        "statistics.json", 
        "config.json",
        "batch_summary.txt",
        "batch_main_visualization.png",
        "batch_statistics_visualization.png"
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(batch_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ 缺少文件: {missing_files}")
        return False
    
    # 检查是否没有单个子图的结果目录
    submap_dirs = [d for d in os.listdir(save_path) if d.startswith("submap_")]
    if submap_dirs:
        print(f"✗ 发现单个子图结果目录: {submap_dirs}")
        print("  批量处理时不应该保存单个子图结果")
        return False
    else:
        print("✓ 没有单个子图结果目录（符合预期）")
    
    # 检查JSON文件内容
    print("\n检查JSON文件内容:")
    
    # 检查详细结果
    with open(os.path.join(batch_dir, "detailed_results.json"), 'r', encoding='utf-8') as f:
        detailed_results = json.load(f)
        print(f"✓ detailed_results.json: {len(detailed_results)} 个子图结果")
    
    # 检查统计信息
    with open(os.path.join(batch_dir, "statistics.json"), 'r', encoding='utf-8') as f:
        statistics = json.load(f)
        print(f"✓ statistics.json: 总子图数 {statistics['summary']['total_submaps']}")
        print(f"  改善率: {statistics['summary']['improvement_rate']:.1f}%")
    
    # 检查配置
    with open(os.path.join(batch_dir, "config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)
        print(f"✓ config.json: 多分辨率层数 {config['optimization_settings']['multi_resolution_layers']}")
    
    # 检查图像文件大小
    main_image_path = os.path.join(batch_dir, "batch_main_visualization.png")
    stats_image_path = os.path.join(batch_dir, "batch_statistics_visualization.png")
    
    main_size = os.path.getsize(main_image_path) / 1024  # KB
    stats_size = os.path.getsize(stats_image_path) / 1024  # KB
    
    print(f"✓ 主图大小: {main_size:.1f} KB")
    print(f"✓ 统计图大小: {stats_size:.1f} KB")
    
    print("\n" + "=" * 60)
    print("✓ 批量处理保存功能改进测试通过")
    print("=" * 60)
    
    return True

def test_single_submap_save():
    """测试单个子图处理时的保存功能（确保向后兼容）"""
    print("\n" + "=" * 60)
    print("测试单个子图处理保存功能（向后兼容性）")
    print("=" * 60)
    
    # 测试数据路径
    test_data_path = "../data/mapping_0506"
    save_path = "../results/test_single_save"
    
    # 清理之前的测试结果
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    print(f"使用测试数据: {test_data_path}")
    print(f"保存路径: {save_path}")
    
    # 运行单个子图优化命令
    cmd = [
        sys.executable, "../core/optimize_submap.py",
        test_data_path,
        "--submap", "0",  # 处理子图0
        "--save", save_path,
        "--debug"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ 单个子图优化执行成功")
        else:
            print("✗ 单个子图优化执行失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 单个子图优化超时")
        return False
    except Exception as e:
        print(f"✗ 执行命令时出错: {e}")
        return False
    
    # 检查保存结果
    if not os.path.exists(save_path):
        print("✗ 保存目录不存在")
        return False
    
    # 查找单个子图结果目录
    submap_dirs = [d for d in os.listdir(save_path) if d.startswith("submap_0_")]
    if not submap_dirs:
        print("✗ 未找到单个子图结果目录")
        return False
    
    submap_dir = os.path.join(save_path, submap_dirs[0])
    print(f"找到单个子图结果目录: {submap_dir}")
    
    # 检查文件结构
    expected_files = [
        "poses.json",
        "performance.json",
        "config.json",
        "summary.txt",
        "optimization_visualization.png",
        "optimization_summary.png"
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(submap_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ 缺少文件: {missing_files}")
        return False
    
    print("✓ 单个子图处理保存功能正常")
    return True

if __name__ == "__main__":
    print("开始测试批量处理保存功能改进...")
    
    # 测试批量处理保存功能改进
    batch_success = test_batch_save_improvement()
    
    # 测试单个子图处理保存功能（向后兼容性）
    single_success = test_single_submap_save()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if batch_success and single_success:
        print("✓ 所有测试通过")
        print("✓ 批量处理时不再保存单个子图结果")
        print("✓ 批量结果包含两张可视化图像")
        print("✓ 单个子图处理功能保持向后兼容")
    else:
        print("✗ 部分测试失败")
        if not batch_success:
            print("  - 批量处理保存功能改进测试失败")
        if not single_success:
            print("  - 单个子图处理保存功能测试失败")
    
    print("=" * 60) 