# 子图融合使用示例

## 概述

子图融合工具可以将多个子图融合成一个全局地图，支持多种分辨率和不同的融合策略。

## 基本使用

### 1. 基本融合

最简单的使用方式，将文件夹中的所有子图融合成单一全局地图：

```bash
python run_fuse_submap.py ../data/1_rtk
```

这将生成：
- `global_map.bin` - 二进制格式的全局地图
- `global_map.png` - 可视化图像

### 2. 使用地面真值姿态

如果数据文件夹中有 `path_pg_rtk.txt` 文件，可以使用地面真值姿态进行融合：

```bash
python run_fuse_submap.py ../data/1_rtk --use-gt
```

### 3. 生成多分辨率地图

生成5种不同分辨率的全局地图（0.1m, 0.2m, 0.4m, 0.8m, 1.6m）：

```bash
python run_fuse_submap.py ../data/1_rtk --multi-res
```

这将生成：
- `global_map_01.bin` 和 `global_map_01.png` (0.1m分辨率)
- `global_map_02.bin` 和 `global_map_02.png` (0.2m分辨率)
- `global_map_04.bin` 和 `global_map_04.png` (0.4m分辨率)
- `global_map_08.bin` 和 `global_map_08.png` (0.8m分辨率)
- `global_map_16.bin` 和 `global_map_16.png` (1.6m分辨率)

### 4. 指定保存文件名

自定义输出文件名：

```bash
python run_fuse_submap.py ../data/1_rtk --save my_custom_map
```

这将生成：
- `my_custom_map.bin` - 二进制格式的全局地图
- `my_custom_map.png` - 可视化图像

## 高级用法

### 组合使用

可以组合多个选项：

```bash
# 使用地面真值并生成多分辨率地图
python run_fuse_submap.py ../data/1_rtk --use-gt --multi-res

# 使用地面真值并指定保存文件名
python run_fuse_submap.py ../data/1_rtk --use-gt --save optimized_map
```

### 直接调用核心程序

如果需要更精细的控制，可以直接调用核心程序：

```bash
# 基本融合
python core/fuse_submaps.py --folder ../data/1_rtk

# 使用地面真值
python core/fuse_submaps.py --folder ../data/1_rtk --use-gt

# 生成多分辨率地图
python core/fuse_submaps.py --folder ../data/1_rtk --multi-res

# 指定保存文件名
python core/fuse_submaps.py --folder ../data/1_rtk --save my_map
```

## 输出文件说明

### 二进制文件 (.bin)

二进制文件包含以下信息：
- 地图边界信息（min_i, max_i, min_j, max_j）
- 栅格数量
- 每个栅格的占用概率

### 可视化文件 (.png)

PNG文件是地图的可视化表示：
- 白色：空闲区域（概率 < 0.4）
- 灰色：未知区域（0.4 ≤ 概率 ≤ 0.6）
- 黑色：占用区域（概率 > 0.6）

## 数据要求

### 输入数据

数据文件夹应包含：
- `submap_*.bin` - 子图文件（必需）
- `path_pg_rtk.txt` - 地面真值文件（可选，用于 `--use-gt` 选项）

### 子图文件格式

每个子图文件包含：
- 子图ID
- 时间戳
- 初始位姿（4x4矩阵）
- 地图边界
- 占用栅格数据

### 地面真值文件格式

`path_pg_rtk.txt` 文件格式：
```
timestamp x y z roll pitch yaw
```

## 性能考虑

### 内存使用

- 单个子图融合：内存使用较少
- 多分辨率融合：需要更多内存，因为要生成5个不同分辨率的地图

### 处理时间

- 基本融合：取决于子图数量和大小
- 多分辨率融合：需要额外时间生成不同分辨率的地图

## 常见问题

### 1. 找不到子图文件

确保数据文件夹中包含 `submap_*.bin` 文件。

### 2. 地面真值文件不存在

如果使用 `--use-gt` 选项但找不到 `path_pg_rtk.txt` 文件，程序会显示警告并继续使用子图中的位姿。

### 3. 输出文件已存在

如果输出文件已存在，程序会覆盖它们。

### 4. 内存不足

对于大型数据集，可能需要增加系统内存或分批处理。

## 完整工作流程示例

### 从子图优化到融合的完整流程

1. **优化子图位姿**：
   ```bash
   python run_optimization.py ../data/1_rtk --submap -1 --save results/optimized
   ```

2. **融合优化后的子图**：
   ```bash
   python run_fuse_submap.py ../data/1_rtk --multi-res
   ```

3. **查看结果**：
   - 优化结果在 `results/optimized/` 文件夹中
   - 融合结果在数据文件夹中，包括多种分辨率的全局地图

这样的工作流程可以确保子图位姿得到优化后再进行融合，获得更好的全局地图质量。 