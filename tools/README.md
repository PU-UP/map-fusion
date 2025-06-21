# Map Fusion Tools

这个文件夹包含了地图融合项目的各种工具和脚本。

## 文件夹结构

```
tools/
├── core/                    # 核心功能模块
│   ├── optimize_submap.py   # 子图位姿优化主程序
│   ├── likelihood_optimizer.py  # 似然优化器
│   ├── particle_filter_matcher.py  # 粒子滤波匹配器
│   └── fuse_submaps.py      # 子图融合工具
├── examples/                # 使用示例
│   ├── example_batch_usage.md      # 批量使用示例
│   ├── example_batch_save_usage.md # 批量保存使用示例
│   └── example_save_usage.md       # 保存功能使用示例
├── tests/                   # 测试文件
│   ├── test_batch_optimization.py      # 批量优化测试
│   ├── test_batch_save_improvement.py  # 批量保存改进测试
│   ├── test_likelihood_optimizer.py    # 似然优化器测试
│   ├── test_likelihood_optimization.py # 似然优化测试
│   └── test_save_functionality.py      # 保存功能测试
├── docs/                    # 文档
│   ├── README_batch_optimization.md    # 批量优化说明
│   ├── README_likelihood_optimizer.md  # 似然优化器说明
│   └── README_save_functionality.md    # 保存功能说明
├── debug/                   # 调试工具
│   └── debug_likelihood.py  # 似然调试工具
├── requirements.txt         # Python依赖
├── run_optimization.py      # 子图优化简化运行脚本
├── run_fuse_submap.py       # 子图融合简化运行脚本
├── .gitignore              # Git忽略文件
└── README.md               # 本文件
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 子图位姿优化

**方法1：使用简化运行脚本（推荐）**

```bash
# 优化单个子图
python run_optimization.py ../data/1_rtk --submap 0

# 批量优化所有子图
python run_optimization.py ../data/1_rtk --submap -1

# 使用似然优化
python run_optimization.py ../data/1_rtk --submap 0 --likelihood

# 保存结果
python run_optimization.py ../data/1_rtk --submap 0 --save results/my_test
```

**方法2：直接运行核心程序**

```bash
# 优化单个子图
python core/optimize_submap.py ../data/1_rtk --submap 0

# 批量优化所有子图
python core/optimize_submap.py ../data/1_rtk --submap -1

# 使用似然优化
python core/optimize_submap.py ../data/1_rtk --submap 0 --likelihood
```

#### 子图融合

**方法1：使用简化运行脚本（推荐）**

```bash
# 基本融合
python run_fuse_submap.py ../data/1_rtk

# 使用地面真值姿态
python run_fuse_submap.py ../data/1_rtk --use-gt

# 生成多分辨率地图
python run_fuse_submap.py ../data/1_rtk --multi-res

# 指定保存文件名
python run_fuse_submap.py ../data/1_rtk --save my_global_map
```

**方法2：直接运行核心程序**

```bash
# 基本融合
python core/fuse_submaps.py --folder ../data/1_rtk

# 使用地面真值姿态
python core/fuse_submaps.py --folder ../data/1_rtk --use-gt

# 生成多分辨率地图
python core/fuse_submaps.py --folder ../data/1_rtk --multi-res

# 指定保存文件名
python core/fuse_submaps.py --folder ../data/1_rtk --save my_global_map
```

## 功能模块

### Core 模块
- **optimize_submap.py**: 主要的子图位姿优化程序，支持多种优化策略
- **likelihood_optimizer.py**: 基于似然地图的优化器
- **particle_filter_matcher.py**: 粒子滤波匹配算法
- **fuse_submaps.py**: 子图融合工具，将多个子图融合成全局地图

### 优化策略
1. **粒子滤波优化**: 默认策略，适用于大多数场景
2. **似然优化**: 基于似然地图的优化，适合高精度要求
3. **多分辨率优化**: 提高优化效率和鲁棒性
4. **批量优化**: 自动处理多个子图

### 融合功能
- **基本融合**: 将多个子图融合成单一全局地图
- **多分辨率融合**: 生成0.1m到1.6m的5种分辨率地图
- **真值融合**: 使用地面真值姿态进行融合
- **可视化输出**: 自动生成.bin和.png格式的输出文件

### 保存功能
- 支持保存优化结果到指定路径
- 生成详细的性能报告和可视化图表
- 支持批量保存和统计汇总

## 测试

运行所有测试：

```bash
# 运行批量优化测试
python tests/test_batch_optimization.py

# 运行保存功能测试
python tests/test_save_functionality.py

# 运行似然优化测试
python tests/test_likelihood_optimizer.py
```

## 调试

使用调试工具分析优化过程：

```bash
python debug/debug_likelihood.py
```

## 工作流程示例

### 完整的子图处理流程

1. **子图位姿优化**：
   ```bash
   # 优化所有子图的位姿
   python run_optimization.py ../data/1_rtk --submap -1 --save results/optimized
   ```

2. **子图融合**：
   ```bash
   # 将优化后的子图融合成全局地图
   python run_fuse_submap.py ../data/1_rtk --multi-res
   ```

3. **结果查看**：
   - 优化结果保存在 `results/optimized/` 文件夹中
   - 融合结果保存在数据文件夹中，包括：
     - `global_map.bin` 和 `global_map.png`（单分辨率）
     - `global_map_01.bin` 到 `global_map_16.bin`（多分辨率）

## 文档

详细的使用说明请参考 `docs/` 文件夹中的文档：

- [批量优化说明](docs/README_batch_optimization.md)
- [似然优化器说明](docs/README_likelihood_optimizer.md)
- [保存功能说明](docs/README_save_functionality.md)

## 示例

查看 `examples/` 文件夹中的使用示例：

- [批量使用示例](examples/example_batch_usage.md)
- [批量保存示例](examples/example_batch_save_usage.md)
- [保存功能示例](examples/example_save_usage.md)

## 注意事项

1. 确保数据文件夹结构正确
2. 批量处理可能需要较长时间
3. 保存功能会生成大量文件，注意磁盘空间
4. 建议在虚拟环境中运行 