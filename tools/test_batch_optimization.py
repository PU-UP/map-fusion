#!/usr/bin/env python3
"""
测试批量优化功能的脚本
"""

import os
import sys
import subprocess
import tempfile
import shutil

def test_batch_optimization():
    """测试批量优化功能"""
    
    # 检查是否有数据文件夹
    data_folders = [
        "data/1_rtk",
        "data/2_rtk", 
        "data/localization_0506",
        "data/mapping_0506",
        "data/submap_200_visual"
    ]
    
    available_folders = []
    for folder in data_folders:
        if os.path.exists(folder):
            # 检查是否有子图文件
            submap_files = [f for f in os.listdir(folder) 
                           if f.startswith('submap_') and f.endswith('.bin')]
            if len(submap_files) > 0:
                available_folders.append((folder, len(submap_files)))
    
    if not available_folders:
        print("错误：未找到包含子图文件的数据文件夹")
        return
    
    print("可用的数据文件夹：")
    for folder, count in available_folders:
        print(f"  {folder}: {count} 个子图")
    
    # 选择第一个可用的文件夹进行测试
    test_folder = available_folders[0][0]
    print(f"\n使用文件夹 {test_folder} 进行测试...")
    
    # 测试命令
    test_commands = [
        # 基本批量测试
        f"python optimize_submap.py {test_folder} --submap -1",
        
        # 多分辨率批量测试
        f"python optimize_submap.py {test_folder} --submap -1 --multi-res 3",
        
        # 似然优化批量测试
        f"python optimize_submap.py {test_folder} --submap -1 --likelihood",
        
        # 添加噪声的批量测试
        f"python optimize_submap.py {test_folder} --submap -1 --add-noise 0.5 10",
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {cmd}")
        print(f"{'='*60}")
        
        try:
            # 运行命令，设置超时时间为300秒（5分钟）
            result = subprocess.run(cmd.split(), 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            if result.returncode == 0:
                print("✓ 测试成功")
                # 打印最后几行输出
                lines = result.stdout.strip().split('\n')
                if lines:
                    print("最后几行输出：")
                    for line in lines[-5:]:
                        print(f"  {line}")
            else:
                print("✗ 测试失败")
                print("错误输出：")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("✗ 测试超时（5分钟）")
        except Exception as e:
            print(f"✗ 测试异常: {e}")

def test_batch_optimization_no_visualization():
    """测试批量处理时不保存中间可视化图片"""
    print("测试批量处理时不保存中间图片的功能...")
    
    # 检查是否有可用的数据文件夹
    data_folders = [
        "../data/1_rtk",
        "../data/2_rtk", 
        "../data/localization_0506",
        "../data/mapping_0506"
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
    
    # 测试批量优化并保存
    print("\n1. 测试批量优化并保存（不保存中间图片）...")
    cmd = [
        sys.executable, "optimize_submap.py", 
        test_folder,
        "--submap", "-1",
        "--save", "../results/test_batch_no_vis"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ 批量优化测试成功")
        else:
            print(f"✗ 批量优化测试失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 批量优化测试超时")
        return False
    except Exception as e:
        print(f"✗ 批量优化测试异常: {e}")
        return False
    
    # 检查保存的文件
    print("\n2. 检查保存的文件...")
    if os.path.exists("../results/test_batch_no_vis"):
        print("✓ 批量结果文件夹存在")
        
        # 查找批量优化结果文件夹
        batch_folders = [f for f in os.listdir("../results/test_batch_no_vis") 
                        if f.startswith("batch_optimization_")]
        
        if batch_folders:
            batch_folder = batch_folders[0]
            batch_path = os.path.join("../results/test_batch_no_vis", batch_folder)
            print(f"✓ 找到批量结果文件夹: {batch_folder}")
            
            # 检查批量结果文件
            batch_files = os.listdir(batch_path)
            print(f"  批量结果文件: {batch_files}")
            
            # 检查是否有单个子图的文件夹
            submap_folders = [f for f in os.listdir("../results/test_batch_no_vis") 
                            if f.startswith("submap_")]
            
            if submap_folders:
                print(f"✓ 找到 {len(submap_folders)} 个单个子图结果文件夹")
                
                # 检查第一个子图文件夹的内容
                first_submap = submap_folders[0]
                submap_path = os.path.join("../results/test_batch_no_vis", first_submap)
                submap_files = os.listdir(submap_path)
                print(f"  子图 {first_submap} 的文件: {submap_files}")
                
                # 检查是否没有可视化图片
                vis_files = [f for f in submap_files if f.endswith('.png')]
                if not vis_files:
                    print("✓ 确认：单个子图文件夹中没有可视化图片（符合预期）")
                else:
                    print(f"✗ 意外：单个子图文件夹中有可视化图片: {vis_files}")
                    return False
                
                # 检查是否有数据文件
                data_files = [f for f in submap_files if f.endswith('.json') or f.endswith('.txt')]
                if data_files:
                    print(f"✓ 确认：单个子图文件夹中有数据文件: {data_files}")
                else:
                    print("✗ 错误：单个子图文件夹中没有数据文件")
                    return False
            else:
                print("✗ 错误：没有找到单个子图结果文件夹")
                return False
        else:
            print("✗ 错误：没有找到批量优化结果文件夹")
            return False
    else:
        print("✗ 批量结果文件夹不存在")
        return False
    
    print("\n✓ 批量处理不保存中间图片的功能测试通过！")
    return True

def cleanup_test_files():
    """清理测试文件"""
    test_dir = "../results/test_batch_no_vis"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"已清理: {test_dir}")

if __name__ == '__main__':
    print("开始测试批量处理不保存中间图片的功能...")
    
    try:
        success = test_batch_optimization_no_visualization()
        if success:
            print("\n测试完成！批量处理不保存中间图片的功能正常工作。")
        else:
            print("\n测试失败！功能存在问题。")
            sys.exit(1)
    finally:
        # 询问是否清理测试文件
        response = input("\n是否清理测试文件？(y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_test_files()
            print("测试文件已清理。")
        else:
            print("测试文件保留在 results/ 目录中。") 