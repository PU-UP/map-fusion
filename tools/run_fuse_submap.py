#!/usr/bin/env python3
"""
子图融合工具运行脚本
简化用户使用体验
"""

import sys
import os
import subprocess

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python run_fuse_submap.py <数据路径> [选项]")
        print("\n示例:")
        print("  python run_fuse_submap.py ../data/1_rtk")
        print("  python run_fuse_submap.py ../data/1_rtk --use-gt")
        print("  python run_fuse_submap.py ../data/1_rtk --multi-res")
        print("  python run_fuse_submap.py ../data/1_rtk --save my_global_map")
        print("\n选项:")
        print("  --use-gt          使用地面真值姿态")
        print("  --multi-res       生成多分辨率地图")
        print("  --save <文件名>    指定保存文件名（默认为global_map）")
        print("\n说明:")
        print("  - 数据路径应包含submap_*.bin文件")
        print("  - 使用--use-gt时会读取path_pg_rtk.txt作为真值")
        print("  - 使用--multi-res会生成0.1m到1.6m的5种分辨率地图")
        print("  - 输出文件包括.bin（二进制）和.png（可视化）")
        return
    
    # 构建命令
    cmd = [sys.executable, "core/fuse_submaps.py", "--folder"] + sys.argv[1:]
    
    try:
        # 运行融合程序
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 