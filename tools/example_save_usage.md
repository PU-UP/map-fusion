# 保存功能使用示例

## 快速开始

### 1. 单个子图优化并保存

```bash
# 优化子图0并保存到默认results文件夹
python optimize_submap.py data/1_rtk --submap 0 --save

# 优化子图0并保存到指定路径
python optimize_submap.py data/1_rtk --submap 0 --save my_experiment
```

### 2. 批量优化并保存

```bash
# 优化所有子图并保存到默认results文件夹
python optimize_submap.py data/1_rtk --submap -1 --save

# 优化所有子图并保存到指定路径
python optimize_submap.py data/1_rtk --submap -1 --save batch_experiment
```

## 高级示例

### 3. 多分辨率优化

```bash
# 使用3层多分辨率优化
python optimize_submap.py data/1_rtk --submap 0 --multi-res 3 --save results/multi_res_3

# 使用5层多分辨率优化（默认）
python optimize_submap.py data/1_rtk --submap 0 --multi-res 5 --save results/multi_res_5
```

### 4. 似然优化

```bash
# 使用似然优化
python optimize_submap.py data/1_rtk --submap 0 --likelihood --save results/likelihood

# 使用似然优化 + 多候选位姿
python optimize_submap.py data/1_rtk --submap 0 --likelihood --candidates 5 --save results/likelihood_5candidates
```

### 5. 添加噪声测试

```bash
# 添加位置噪声0.5米，角度噪声10度
python optimize_submap.py data/1_rtk --submap 0 --add-noise 0.5 10 --save results/noise_test

# 添加较大噪声测试鲁棒性
python optimize_submap.py data/1_rtk --submap 0 --add-noise 2.0 30 --save results/large_noise_test
```

### 6. 使用真值对比

```bash
# 使用path_pg_rtk.txt中的真值
python optimize_submap.py data/1_rtk --submap 0 --use-gt --save results/gt_comparison

# 批量使用真值对比
python optimize_submap.py data/1_rtk --submap -1 --use-gt --save results/gt_batch
```

### 7. 调试模式

```bash
# 开启调试模式，保存详细日志
python optimize_submap.py data/1_rtk --submap 0 --debug --save results/debug_mode

# 显示可视化过程
python optimize_submap.py data/1_rtk --submap 0 --plot --save results/with_visualization
```

## 组合使用示例

### 8. 完整实验

```bash
# 多分辨率 + 似然优化 + 真值对比 + 调试模式
python optimize_submap.py data/1_rtk --submap 0 \
    --multi-res 3 \
    --likelihood \
    --candidates 3 \
    --use-gt \
    --debug \
    --save results/full_experiment
```

### 9. 批量完整实验

```bash
# 批量处理所有子图，使用完整优化策略
python optimize_submap.py data/1_rtk --submap -1 \
    --multi-res 3 \
    --likelihood \
    --candidates 3 \
    --use-gt \
    --save results/batch_full_experiment
```

## 结果查看

运行完成后，可以查看保存的结果：

```bash
# 查看结果文件夹结构
ls -la results/

# 查看单个子图结果
ls -la results/submap_0_20241201_143022/

# 查看批量结果
ls -la results/batch_optimization_20241201_143100/

# 查看总结报告
cat results/submap_0_20241201_143022/summary.txt
cat results/batch_optimization_20241201_143100/batch_summary.txt
```

## 数据分析

保存的JSON文件可以用于进一步的数据分析：

```python
import json

# 读取位姿数据
with open('results/submap_0_20241201_143022/poses.json', 'r') as f:
    pose_data = json.load(f)

# 读取性能数据
with open('results/submap_0_20241201_143022/performance.json', 'r') as f:
    perf_data = json.load(f)

# 分析位姿改善
translation_improvement = pose_data['pose_differences']['translation_error_before'] - \
                         pose_data['pose_differences']['translation_error_after']
print(f"位置误差改善: {translation_improvement:.3f} 米")

# 分析匹配性能
matching_improvement = perf_data['matching_errors']['improvement_percentage']
print(f"匹配误差改善: {matching_improvement:.1f}%")
```

## 注意事项

1. **路径选择**: 建议使用有意义的路径名称，便于后续分析
2. **磁盘空间**: 批量处理会生成大量文件，确保有足够空间
3. **时间戳**: 每次运行都会生成带时间戳的文件夹，不会覆盖之前的结果
4. **图像文件**: 可视化图像文件较大，如不需要可以手动删除
5. **数据格式**: JSON文件使用UTF-8编码，支持中文字符
6. **批量处理优化**: 批量处理时只保存数据文件，不保存每个子图的中间可视化图片，提高性能并节省空间 