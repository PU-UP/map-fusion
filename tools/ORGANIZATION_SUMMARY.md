# Tools 文件夹整理总结

## 整理前的问题

原始的 `tools/` 文件夹存在以下问题：
- 所有文件都混在一个文件夹中，难以找到特定功能
- 缺乏清晰的文件分类和组织结构
- 测试文件、文档、示例代码混杂在一起
- 没有统一的使用说明和入口点

## 整理后的结构

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
├── README.md               # 主说明文档
└── ORGANIZATION_SUMMARY.md # 整理总结
```

## 主要改进

### 1. 清晰的模块分类
- **core/**: 核心功能模块，包含主要的算法实现
- **examples/**: 使用示例，帮助用户快速上手
- **tests/**: 测试文件，确保功能正确性
- **docs/**: 详细文档，说明各种功能的使用方法
- **debug/**: 调试工具，帮助开发者排查问题

### 2. 简化的使用体验
- 创建了 `run_optimization.py` 简化运行脚本
- 创建了 `run_fuse_submap.py` 简化融合脚本
- 更新了主 README.md 文件，提供清晰的使用指南
- 保持了向后兼容性，原有的直接调用方式仍然有效

### 3. 路径引用更新
- 更新了所有测试文件中的路径引用
- 确保测试能够正确运行
- 保持了相对路径的灵活性

### 4. 版本控制优化
- 添加了 `.gitignore` 文件
- 忽略 Python 缓存文件和虚拟环境
- 忽略结果文件和临时文件

## 使用方式

### 子图位姿优化

**推荐方式（简化）**：
```bash
# 优化单个子图
python run_optimization.py ../data/1_rtk --submap 0

# 批量优化所有子图
python run_optimization.py ../data/1_rtk --submap -1

# 使用似然优化
python run_optimization.py ../data/1_rtk --submap 0 --likelihood
```

**直接方式（保持兼容）**：
```bash
# 优化单个子图
python core/optimize_submap.py ../data/1_rtk --submap 0

# 批量优化所有子图
python core/optimize_submap.py ../data/1_rtk --submap -1
```

### 子图融合

**推荐方式（简化）**：
```bash
# 基本融合
python run_fuse_submap.py ../data/1_rtk

# 使用地面真值姿态
python run_fuse_submap.py ../data/1_rtk --use-gt

# 生成多分辨率地图
python run_fuse_submap.py ../data/1_rtk --multi-res
```

**直接方式（保持兼容）**：
```bash
# 基本融合
python core/fuse_submaps.py --folder ../data/1_rtk

# 使用地面真值姿态
python core/fuse_submaps.py --folder ../data/1_rtk --use-gt

# 生成多分辨率地图
python core/fuse_submaps.py --folder ../data/1_rtk --multi-res
```

## 测试验证

所有测试文件都已更新路径引用，可以正常运行：

```bash
# 运行批量优化测试
python tests/test_batch_optimization.py

# 运行保存功能测试
python tests/test_save_functionality.py

# 运行似然优化测试
python tests/test_likelihood_optimizer.py
```

## 完整工作流程

### 子图处理完整流程

1. **子图位姿优化**：
   ```bash
   python run_optimization.py ../data/1_rtk --submap -1 --save results/optimized
   ```

2. **子图融合**：
   ```bash
   python run_fuse_submap.py ../data/1_rtk --multi-res
   ```

3. **结果查看**：
   - 优化结果：`results/optimized/` 文件夹
   - 融合结果：数据文件夹中的 `global_map_*.bin` 和 `global_map_*.png`

## 文档结构

- **README.md**: 主要使用指南和快速开始
- **docs/**: 详细的功能说明文档
- **examples/**: 具体的使用示例
- **ORGANIZATION_SUMMARY.md**: 本整理总结文档

## 维护建议

1. **新功能开发**: 将新的核心功能放在 `core/` 文件夹中
2. **测试编写**: 将测试文件放在 `tests/` 文件夹中
3. **文档更新**: 将详细文档放在 `docs/` 文件夹中
4. **示例添加**: 将使用示例放在 `examples/` 文件夹中
5. **调试工具**: 将调试相关工具放在 `debug/` 文件夹中
6. **简化脚本**: 为新的核心功能创建对应的简化运行脚本

这样的组织结构使得项目更加清晰、易于维护和使用。 