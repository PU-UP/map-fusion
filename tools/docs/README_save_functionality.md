# 保存功能说明

## 概述

`optimize_submap.py` 脚本现在支持 `--save` 选项，可以将优化结果保存到指定路径。默认情况下，结果会保存到 `results/` 文件夹中。

## 使用方法

### 基本用法

```bash
# 保存到默认的 results 文件夹
python optimize_submap.py data/1_rtk --submap 0 --save

# 保存到指定路径
python optimize_submap.py data/1_rtk --submap 0 --save my_results

# 批量处理并保存
python optimize_submap.py data/1_rtk --submap -1 --save batch_results
```

### 高级用法

```bash
# 多分辨率优化并保存
python optimize_submap.py data/1_rtk --submap 0 --multi-res 3 --save results/multi_res

# 似然优化并保存
python optimize_submap.py data/1_rtk --submap 0 --likelihood --save results/likelihood

# 添加噪声并保存
python optimize_submap.py data/1_rtk --submap 0 --add-noise 0.5 10 --save results/noise_test
```

## 保存的内容

### 单个子图优化结果

每个子图的优化结果会保存在以 `submap_{ID}_{timestamp}` 命名的文件夹中，包含：

1. **poses.json** - 位姿数据
   - 真值位姿
   - 初始位姿
   - 优化后位姿
   - 位姿误差对比

2. **performance.json** - 性能数据
   - 匹配误差（优化前后）
   - 匹配率（优化前后）
   - 改善百分比
   - 优化耗时

3. **config.json** - 配置参数
   - 命令行参数
   - 优化设置
   - 额外信息

4. **optimization_visualization.png** - 可视化图像
   - 优化前后的子图位置对比
   - 全局地图背景
   - 图例说明

5. **optimization_summary.png** - 统计图表
   - 位置误差对比
   - 角度误差对比
   - 匹配误差对比
   - 匹配率对比

6. **summary.txt** - 文本总结报告
   - 优化策略说明
   - 误差改善情况
   - 性能统计
   - 配置参数

### 批量优化结果

批量优化结果会保存在以 `batch_optimization_{timestamp}` 命名的文件夹中，包含：

1. **detailed_results.json** - 所有子图的详细结果
   - 每个子图的位姿误差
   - 每个子图的匹配误差
   - 每个子图的性能数据

2. **statistics.json** - 统计摘要
   - 总体统计信息
   - 位置误差统计
   - 角度误差统计
   - 性能统计

3. **config.json** - 配置参数
   - 命令行参数
   - 优化设置

4. **batch_summary.txt** - 批量总结报告
   - 处理子图数量
   - 成功改善数量
   - 平均误差统计
   - 总耗时统计

## 文件结构示例

```
results/
├── submap_0_20241201_143022/
│   ├── poses.json
│   ├── performance.json
│   ├── config.json
│   ├── optimization_visualization.png
│   ├── optimization_summary.png
│   └── summary.txt
├── submap_1_20241201_143045/
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

## 数据格式

### poses.json 示例

```json
{
  "submap_id": 0,
  "timestamp": "20241201_143022",
  "true_pose": [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
  "initial_pose": [[1.0, 0.0, 0.0, 10.5], [0.0, 1.0, 0.0, 20.3], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
  "optimized_pose": [[1.0, 0.0, 0.0, 10.1], [0.0, 1.0, 0.0, 20.1], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
  "pose_differences": {
    "translation_error_before": 0.583,
    "translation_error_after": 0.141,
    "rotation_error_before": 2.5,
    "rotation_error_after": 0.8
  }
}
```

### performance.json 示例

```json
{
  "matching_errors": {
    "initial": 0.234,
    "final": 0.089,
    "improvement_percentage": 61.9
  },
  "performance": {
    "matching_time_seconds": 2.45,
    "matching_rate_before": 0.766,
    "matching_rate_after": 0.911
  }
}
```

## 注意事项

1. **默认路径**: 如果不指定 `--save` 参数，结果会保存到 `results/` 文件夹
2. **时间戳**: 每个结果文件夹都包含时间戳，避免覆盖之前的结果
3. **Git忽略**: `results/` 文件夹已添加到 `.gitignore`，不会被提交到版本控制
4. **磁盘空间**: 批量处理会生成大量文件，请确保有足够的磁盘空间
5. **图像质量**: 保存的图像使用高分辨率（300 DPI），文件较大但质量好
6. **批量处理优化**: 当使用 `--submap -1` 进行批量处理时，不会保存每个子图的中间可视化图片，只保存数据文件和最终的批量总结图像，以提高性能并减少磁盘空间占用

## 测试

可以使用提供的测试脚本验证保存功能：

```bash
python test_save_functionality.py
```

测试脚本会：
1. 测试单个子图优化并保存
2. 测试批量优化并保存
3. 检查保存的文件结构
4. 提供清理选项 