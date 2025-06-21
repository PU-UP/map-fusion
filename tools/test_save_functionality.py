#!/usr/bin/env python3
"""
测试保存功能的脚本
"""

import os
import sys
import subprocess
import tempfile
import shutil

def test_save_functionality():
    """测试保存功能"""
    print("测试保存功能...")
    
    # 检查是否有可用的数据文件夹
    data_folders = [
        "data/1_rtk",
        "data/2_rtk", 
        "data/localization_0506",
        "data/mapping_0506"
    ]
    
    test_folder = None
    for folder in data_folders:
        if os.path.exists(folder):
            test_folder = folder
            break
    
    if not test_folder:
        print("错误：未找到可用的数据文件夹进行测试")
        return False
    
    print(f"使用测试文件夹: {test_folder}")
    
    # 测试单个子图优化并保存
    print("\n1. 测试单个子图优化并保存...")
    cmd = [
        sys.executable, "optimize_submap.py", 
        test_folder,
        "--submap", "0",
        "--save", "results/test_single"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ 单个子图优化保存测试成功")
        else:
            print(f"✗ 单个子图优化保存测试失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 单个子图优化保存测试超时")
        return False
    except Exception as e:
        print(f"✗ 单个子图优化保存测试异常: {e}")
        return False
    
    # 测试批量优化并保存
    print("\n2. 测试批量优化并保存...")
    cmd = [
        sys.executable, "optimize_submap.py", 
        test_folder,
        "--submap", "-1",
        "--save", "results/test_batch"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ 批量优化保存测试成功")
        else:
            print(f"✗ 批量优化保存测试失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 批量优化保存测试超时")
        return False
    except Exception as e:
        print(f"✗ 批量优化保存测试异常: {e}")
        return False
    
    # 检查保存的文件
    print("\n3. 检查保存的文件...")
    if os.path.exists("results/test_single"):
        print("✓ 单个子图结果文件夹存在")
        files = os.listdir("results/test_single")
        print(f"  包含文件: {files}")
    else:
        print("✗ 单个子图结果文件夹不存在")
        return False
    
    if os.path.exists("results/test_batch"):
        print("✓ 批量优化结果文件夹存在")
        files = os.listdir("results/test_batch")
        print(f"  包含文件: {files}")
    else:
        print("✗ 批量优化结果文件夹不存在")
        return False
    
    print("\n✓ 所有保存功能测试通过！")
    return True

def cleanup_test_files():
    """清理测试文件"""
    test_dirs = ["results/test_single", "results/test_batch"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"已清理: {test_dir}")

if __name__ == "__main__":
    print("开始测试保存功能...")
    
    try:
        success = test_save_functionality()
        if success:
            print("\n测试完成！保存功能正常工作。")
        else:
            print("\n测试失败！保存功能存在问题。")
            sys.exit(1)
    finally:
        # 询问是否清理测试文件
        response = input("\n是否清理测试文件？(y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_test_files()
            print("测试文件已清理。")
        else:
            print("测试文件保留在 results/ 目录中。") 