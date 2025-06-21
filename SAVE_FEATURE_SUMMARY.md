# 保存功能实现总结

## 概述

成功为 `optimize_submap.py` 脚本添加了 `--save` 选项，实现了完整的优化结果保存功能。该功能可以将所有运行结果保存到指定路径，便于后续分析和对比。

## 实现的功能

### 1. 命令行选项
- 添加了 `--save [path]` 选项
- 默认保存路径为 `results/` 文件夹
- 支持自定义保存路径

### 2. 保存内容

#### 单个子图优化结果
每个子图的优化结果保存在 `submap_{ID}_{timestamp}/` 文件夹中：

- **poses.json** - 位姿数据（真值、初始、优化后位姿及误差对比）
- **performance.json** - 性能数据（匹配误差、匹配率、改善百分比、耗时）
- **config.json** - 配置参数（命令行参数、优化设置）
- **optimization_visualization.png** - 可视化图像（优化前后对比）
- **optimization_summary.png** - 统计图表（误差对比图）
- **summary.txt** - 文本总结报告

#### 批量优化结果
批量优化结果保存在 `batch_optimization_{timestamp}/` 文件夹中：

- **detailed_results.json** - 所有子图的详细结果
- **statistics.json** - 统计摘要（平均值、标准差等）
- **config.json** - 配置参数
- **batch_summary.txt** - 批量总结报告

### 3. 批量处理优化
- 批量处理时不保存每个子图的中间可视化图片
- 只保存数据文件和最终的批量总结图像
- 提高性能并减少磁盘空间占用
- 单个子图处理时仍保存完整的可视化图像

### 4. 文件结构
```
results/
├── submap_0_20241201_143022/
│   ├── poses.json
│   ├── performance.json
│   ├── config.json
│   ├── optimization_visualization.png
│   ├── optimization_summary.png
│   └── summary.txt
└── batch_optimization_20241201_143100/
    ├── detailed_results.json
    ├── statistics.json
    ├── config.json
    └── batch_summary.txt
```

## 技术实现

### 1. 新增函数
- `save_optimization_results()` - 保存单个子图优化结果
- `save_batch_results()` - 保存批量优化结果

### 2. 修改的函数
- `main()` - 添加保存逻辑调用
- `visualize_optimization()` - 支持保存图像到指定路径

### 3. 数据格式
- JSON格式存储结构化数据
- PNG格式保存高质量可视化图像
- TXT格式提供人类可读的总结报告

## 项目配置

### 1. 创建results文件夹
```bash
mkdir results
```

### 2. 更新.gitignore
```
*__pycache__
*.venv
results/
```

确保结果文件不会被提交到版本控制。

## 使用示例

### 基本用法
```bash
# 单个子图优化并保存
python optimize_submap.py data/1_rtk --submap 0 --save

# 批量优化并保存
python optimize_submap.py data/1_rtk --submap -1 --save

# 保存到指定路径
python optimize_submap.py data/1_rtk --submap 0 --save my_experiment
```

### 高级用法
```bash
# 多分辨率优化
python optimize_submap.py data/1_rtk --submap 0 --multi-res 3 --save results/multi_res

# 似然优化
python optimize_submap.py data/1_rtk --submap 0 --likelihood --save results/likelihood

# 完整实验
python optimize_submap.py data/1_rtk --submap 0 \
    --multi-res 3 --likelihood --use-gt --debug \
    --save results/full_experiment
```

## 文档

### 1. 功能说明文档
- `tools/README_save_functionality.md` - 详细的功能说明

### 2. 使用示例文档
- `tools/example_save_usage.md` - 各种使用场景的示例

### 3. 测试脚本
- `tools/test_save_functionality.py` - 功能测试脚本

## 特性

### 1. 时间戳命名
- 每个结果文件夹都包含时间戳
- 避免覆盖之前的结果
- 便于追踪实验历史

### 2. 高分辨率图像
- 可视化图像使用300 DPI
- 确保图像质量
- 适合论文和报告使用

### 3. 结构化数据
- JSON格式便于程序读取
- 支持后续数据分析
- 包含完整的配置信息

### 4. 人类可读报告
- 文本总结报告
- 中文界面
- 清晰的统计信息

## 兼容性

### 1. 向后兼容
- 不添加 `--save` 选项时行为不变
- 所有现有功能保持兼容
- 不影响现有工作流程

### 2. 参数组合
- 支持所有现有参数组合
- 与多分辨率、似然优化等功能完全兼容
- 支持调试模式和可视化选项

## 测试验证

### 1. 语法检查
- 通过Python语法检查
- 无编译错误

### 2. 功能测试
- 提供测试脚本 `test_save_functionality.py`
- 测试单个和批量保存功能
- 验证文件结构正确性

## 注意事项

### 1. 磁盘空间
- 批量处理会生成大量文件
- 建议定期清理旧结果
- 图像文件较大，注意存储空间

### 2. 性能影响
- 保存功能对优化性能影响很小
- 主要开销在图像生成和文件写入
- 建议在需要时使用，不需要时可省略

### 3. 文件管理
- 结果文件不会被Git跟踪
- 需要手动管理文件清理
- 建议建立文件命名规范

## 未来改进

### 1. 可能的扩展
- 添加结果压缩功能
- 支持数据库存储
- 添加结果对比工具

### 2. 优化建议
- 可选择性保存图像文件
- 添加结果清理工具
- 支持结果导出为其他格式

## 总结

成功实现了完整的保存功能，提供了：

1. **完整的命令行支持** - 简单易用的 `--save` 选项
2. **丰富的保存内容** - 位姿数据、性能数据、配置信息、可视化图像
3. **结构化数据格式** - JSON格式便于程序处理
4. **人类可读报告** - 文本总结便于快速了解结果
5. **完善的文档** - 详细的使用说明和示例
6. **测试验证** - 提供测试脚本确保功能正确性

该功能大大提升了实验的可重复性和结果的可追溯性，为后续的算法改进和性能分析提供了强有力的支持。 