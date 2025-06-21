#!/usr/bin/env python3
"""
测试保存功能的脚本
"""

import os
import sys
import subprocess

def test_save_functionality():
    """测试保存功能"""
    
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
    
    # 创建输出目录
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 测试保存功能
    test_commands = [
        # 基本批量测试并保存
        f"python optimize_submap.py {test_folder} --submap -1 --multi-res 2",
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
                    for line in lines[-10:]:
                        print(f"  {line}")
            else:
                print("✗ 测试失败")
                print("错误输出：")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("✗ 测试超时（5分钟）")
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n测试完成！")
    print(f"注意：当前版本会显示两个窗口，但不会自动保存图像。")
    print(f"如需保存图像，可以手动截图或修改代码添加保存功能。")

if __name__ == '__main__':
    test_save_functionality() 