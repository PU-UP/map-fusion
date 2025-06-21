#!/usr/bin/env python3
"""
地图融合工具运行脚本
简化用户使用体验
"""

import sys
import os
import subprocess

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python run_optimization.py <数据路径> [选项]")
        print("\n示例:")
        print("  python run_optimization.py ../data/1_rtk --submap 0")
        print("  python run_optimization.py ../data/1_rtk --submap -1")
        print("  python run_optimization.py ../data/1_rtk --submap 0 --likelihood")
        print("\n选项:")
        print("  --submap <ID>      子图ID，-1表示批量处理")
        print("  --likelihood       使用似然优化")
        print("  --multi-res <N>    多分辨率层数")
        print("  --save <路径>      保存结果")
        print("  --add-noise <pos> <angle>  添加噪声")
        return
    
    # 构建命令
    cmd = [sys.executable, "core/optimize_submap.py"] + sys.argv[1:]
    
    try:
        # 运行优化程序
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