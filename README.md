# MapFusion

MapFusion fuses local occupancy grid submaps generated from point clouds and camera data into a global map. It is written in C++ with accompanying Python tools for offline processing.

## Build Requirements
- C++14 compiler
- [Eigen3](https://eigen.tuxfamily.org/)
- [OpenCV](https://opencv.org/) (tested with 3.x or newer)
- [Ceres Solver](http://ceres-solver.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

These packages can be installed via your system package manager. For example on Ubuntu:

```bash
sudo apt-get install build-essential cmake libeigen3-dev libopencv-dev libyaml-cpp-dev
```

## Building with CMake
Create a build directory and compile the library:

```bash
mkdir build
cd build
cmake ..
make -j
```

This builds the `map_fusion` library and `map_fusion_component` shared library.

## Python Tools
The `tools` directory contains several helper scripts written in Python. Install the dependencies with:

```bash
pip install -r tools/requirements.txt
```

### Fusing Submaps
The script `tools/fuse_submaps.py` can fuse a folder of `submap_*.bin` files into a global occupancy grid. A small data set is provided in `data/submap_200_visual`.

Run the following command from the repository root:

```bash
python tools/fuse_submaps.py --folder data/submap_200_visual --save fused
```

The script will create `fused.bin` and `fused.png` in the same folder showing the combined map.

### 子图位姿优化

`tools/optimize_submap.py` 脚本提供了多种子图位姿优化策略：

#### 基本用法
```bash
python tools/optimize_submap.py <数据文件夹路径> [选项]
```

#### 命令行选项详解

**必需参数：**
- `folder_path`：包含子图和全局地图的文件夹路径

**优化策略选项：**

1. **单分辨率粒子滤波优化**（默认）
   ```bash
   python tools/optimize_submap.py data/1_rtk/
   ```

2. **多分辨率粒子滤波优化**
   ```bash
   python tools/optimize_submap.py data/1_rtk/ --multi-res
   python tools/optimize_submap.py data/1_rtk/ --multi-res 3  # 指定3层分辨率
   ```

3. **似然地图梯度优化**
   ```bash
   python tools/optimize_submap.py data/1_rtk/ --likelihood
   ```

**详细选项说明：**

- `--plot`：显示粒子滤波中间过程的可视化
- `--use-gt`：使用path_pg_rtk.txt中的真值作为初始位姿和参考真值
- `--submap <ID>`：指定要优化的子图ID，默认为随机选择
- `--multi-res [层数]`：使用多分辨率匹配策略
  - 不指定层数时默认使用5层（1.6m, 0.8m, 0.4m, 0.2m, 0.1m）
  - 可指定层数，如`--multi-res 3`使用3层分辨率
  - 从低分辨率到高分辨率逐层优化
- `--likelihood`：使用似然地图优化
  - 基于概率地图的梯度下降优化
  - 提高精度并降低资源消耗
  - 使用原始概率地图而非三值化地图
- `--candidates <数量>`：多候选位姿策略的候选数量（默认3个，仅对--likelihood有效）
- `--debug`：开启调试模式
  - 详细耗时打印
  - 误差变化图
  - 各分辨率层耗时统计
- `--add-noise <位置噪声> <角度噪声>`：为初始位姿添加噪声
  - 位置噪声：米为单位
  - 角度噪声：度为单位
  - 例如：`--add-noise 0.5 10`表示添加0.5米位置噪声和10度角度噪声

#### 优化策略详解

**1. 单分辨率粒子滤波优化（默认）**
- 使用固定分辨率（子图0.05m，全局地图0.1m）
- 100个粒子，200次迭代
- 适合快速测试和简单场景

**2. 多分辨率粒子滤波优化**
- 支持1.6m, 0.8m, 0.4m, 0.2m, 0.1m五个分辨率层
- 从粗分辨率开始，逐步细化到高分辨率
- 每层参数动态调整：
  - 1.6m: 80粒子, 80迭代, 大搜索范围
  - 0.8m: 100粒子, 100迭代, 中等搜索范围
  - 0.4m: 100粒子, 120迭代, 中等搜索范围
  - 0.2m: 100粒子, 150迭代, 小搜索范围
  - 0.1m: 120粒子, 200迭代, 精细搜索范围

**3. 似然地图梯度优化**
- 使用原始概率地图而非三值化地图
- 采用L-BFGS-B梯度优化算法
- 多层分辨率策略：从粗到细逐步优化
- 多候选位姿策略：避免过早收敛到错误解
- 鲁棒性检查：通过扰动测试检测位姿稳定性

#### 似然地图优化特性

`--likelihood` 选项实现了基于概率地图的梯度下降优化：

- **精度提升**：使用原始概率地图而非三值化地图，保留更多信息
- **资源优化**：采用L-BFGS-B梯度优化算法，比粒子滤波更高效
- **多层策略**：从粗分辨率(1.6m)到细分辨率(0.1m)逐层优化
- **智能更新**：根据似然分数改善程度智能决定是否更新位姿
- **多候选策略**：在粗分辨率层保留多个候选位姿，避免过早收敛到错误解
- **鲁棒性检查**：通过扰动测试检测位姿的稳定性

#### 多候选位姿策略

为了解决粗分辨率匹配错误导致高分辨率优化陷入局部最优的问题，新增了多候选位姿策略：

- **候选数量控制**：使用 `--candidates <数量>` 参数控制候选位姿数量（默认3个）
- **分层策略**：
  - 粗分辨率层（1.6m, 0.8m, 0.4m）：保留多个候选位姿
  - 中等分辨率层（0.2m）：选择最佳候选
  - 高分辨率层（0.1m）：精细优化
- **鲁棒性检查**：在粗分辨率层检测位姿稳定性，警告可能的匹配错误
- **扰动生成**：在最佳位姿周围生成扰动候选，增加搜索范围

#### 使用示例

**基本测试：**
```bash
# 使用默认设置随机选择子图进行优化
python tools/optimize_submap.py data/1_rtk/

# 指定子图ID进行优化
python tools/optimize_submap.py data/1_rtk/ --submap 18

# 使用真值数据作为参考
python tools/optimize_submap.py data/1_rtk/ --use-gt --submap 18
```

**多分辨率优化：**
```bash
# 使用默认5层多分辨率优化
python tools/optimize_submap.py data/1_rtk/ --multi-res --use-gt --submap 18

# 使用3层多分辨率优化
python tools/optimize_submap.py data/1_rtk/ --multi-res 3 --use-gt --submap 18

# 开启可视化显示
python tools/optimize_submap.py data/1_rtk/ --multi-res --plot --use-gt --submap 18
```

**似然优化：**
```bash
# 使用似然地图优化
python tools/optimize_submap.py data/1_rtk/ --likelihood --use-gt --submap 18

# 多分辨率似然优化
python tools/optimize_submap.py data/1_rtk/ --likelihood --multi-res --use-gt --submap 18

# 调整候选位姿数量
python tools/optimize_submap.py data/1_rtk/ --likelihood --multi-res --candidates 5 --use-gt --submap 18
```

**调试和测试：**
```bash
# 开启调试模式
python tools/optimize_submap.py data/1_rtk/ --debug --use-gt --submap 18

# 添加噪声测试鲁棒性
python tools/optimize_submap.py data/1_rtk/ --add-noise 0.5 10 --use-gt --submap 18

# 完整调试配置
python tools/optimize_submap.py data/1_rtk/ --multi-res --likelihood --debug --plot --use-gt --submap 18
```

#### 默认调试选项
为了快速开始调试和测试匹配功能，建议优先使用以下命令：

```bash
python .\tools\optimize_submap.py ..\data\submap_200_visual\ --use-gt --submap 18 --multi-res
```

这个命令使用真实轨迹（ground truth）数据，针对第18个子地图进行多分辨率优化匹配，是推荐的默认调试选项。

#### 输出说明

脚本会输出以下信息：
- 选择的子图ID
- 使用的优化策略和参数
- 各分辨率层的优化进度和结果
- 最终位姿误差（相对于真值）
- 优化耗时统计
- 可视化结果（如果启用--plot）

#### 文件要求

数据文件夹需要包含：
- `global_map.bin`：全局地图文件
- `global_map_*.bin`：多分辨率全局地图文件（使用--multi-res时）
- `submap_*.bin`：子图文件
- `path_pg_rtk.txt`：真值轨迹文件（使用--use-gt时）

### 测试似然优化功能

运行测试脚本验证新功能：

```bash
python tools/test_likelihood_optimization.py
```

这将测试似然分数计算、梯度优化和多分辨率似然优化功能。

