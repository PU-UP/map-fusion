# 似然优化器模块 (Likelihood Optimizer)

## 概述

`likelihood_optimizer.py` 是一个独立的似然优化模块，提供基于概率地图的梯度下降优化方法，用于子图位姿估计。该模块与粒子滤波优化器并列，都支持单分辨率和多分辨率优化策略。

## 功能特点

### 核心功能
- **似然计算**: 基于概率地图计算位姿似然分数
- **梯度优化**: 使用多种优化算法（Nelder-Mead、BFGS、L-BFGS-B等）
- **多分辨率优化**: 从粗到细的逐层优化策略
- **多候选位姿**: 支持多个初始位姿候选，提高鲁棒性
- **鲁棒性检查**: 自动检测和过滤异常结果

### 优化策略对比

| 特性 | 似然优化 | 粒子滤波优化 |
|------|----------|--------------|
| **优化方法** | 梯度下降 | 粒子滤波 |
| **地图类型** | 概率地图 | 三值化地图 |
| **计算效率** | 高 | 中等 |
| **精度** | 高 | 中等 |
| **鲁棒性** | 中等 | 高 |
| **单分辨率** | ✅ 支持 | ✅ 支持 |
| **多分辨率** | ✅ 支持 | ✅ 支持 |

## 使用方法

### 1. 单分辨率似然优化

```python
from likelihood_optimizer import match_submap_with_likelihood

# 基本使用
opt_pose, final_score = match_submap_with_likelihood(
    submap, global_map, init_pose,
    submap_res=0.05, global_res=0.1
)

# 自定义参数
opt_pose, final_score = match_submap_with_likelihood(
    submap, global_map, init_pose,
    submap_res=0.05, global_res=0.1,
    max_iterations=100,
    tolerance=1e-6,
    method='Nelder-Mead',
    debug=True,
    visualize=False
)
```

### 2. 多分辨率似然优化

```python
from likelihood_optimizer import multi_resolution_likelihood_optimization

# 生成多分辨率子图
multi_res_submaps = generate_multi_resolution_submap(submap, use_likelihood=True, num_layers=5)

# 多分辨率优化
opt_pose, final_score = multi_resolution_likelihood_optimization(
    multi_res_submaps, 
    multi_res_global_maps, 
    init_pose,
    true_pose,
    visualize=False,
    debug=True,
    n_candidates=3
)
```

### 3. 命令行使用

```bash
# 单分辨率似然优化
python optimize_submap.py data/mapping_0506/ --likelihood

# 多分辨率似然优化
python optimize_submap.py data/mapping_0506/ --likelihood --multi-res 5

# 自定义候选数量
python optimize_submap.py data/mapping_0506/ --likelihood --multi-res 3 --candidates 5

# 调试模式
python optimize_submap.py data/mapping_0506/ --likelihood --debug
```

## 优化逻辑结构

### 选项组合

| --likelihood | --multi-res | 优化策略 |
|--------------|-------------|----------|
| ❌ | ❌ | 单分辨率粒子滤波 |
| ✅ | ❌ | 单分辨率似然优化 |
| ❌ | ✅ | 多分辨率粒子滤波 |
| ✅ | ✅ | 多分辨率似然优化 |

### 逻辑流程

```
1. 检查 --multi-res 选项
   ├─ 有: 加载多分辨率地图
   └─ 无: 加载单分辨率地图

2. 根据 --likelihood 选择优化方法
   ├─ 有: 似然优化策略
   │   ├─ 多分辨率: multi_resolution_likelihood_optimization()
   │   └─ 单分辨率: match_submap_with_likelihood()
   └─ 无: 粒子滤波策略
       ├─ 多分辨率: multi_resolution_optimization()
       └─ 单分辨率: optimize_submap_pose()
```

## 参数说明

### 单分辨率似然优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `submap_res` | float | 0.05 | 子图分辨率(m) |
| `global_res` | float | 0.1 | 全局地图分辨率(m) |
| `max_iterations` | int | 100 | 最大迭代次数 |
| `tolerance` | float | 1e-6 | 收敛容差 |
| `method` | str | 'Nelder-Mead' | 优化算法 |
| `debug` | bool | False | 调试模式 |
| `visualize` | bool | False | 可视化模式 |

### 多分辨率似然优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_candidates` | int | 3 | 候选位姿数量 |
| `visualize` | bool | False | 可视化模式 |
| `debug` | bool | False | 调试模式 |

### 优化算法选项

- **Nelder-Mead**: 单纯形法，鲁棒性好，适合非光滑函数
- **BFGS**: 拟牛顿法，收敛快，需要梯度信息
- **L-BFGS-B**: 有限内存BFGS，适合大规模问题
- **Powell**: 方向集方法，不需要梯度信息

## 性能特点

### 优势
1. **计算效率高**: 梯度优化比粒子滤波更高效
2. **精度高**: 基于概率地图，保留更多信息
3. **收敛性好**: 多种优化算法可选
4. **内存友好**: 不需要维护大量粒子

### 局限性
1. **局部最优**: 可能陷入局部最优解
2. **初始值敏感**: 对初始位姿要求较高
3. **鲁棒性**: 在复杂环境下可能不如粒子滤波稳定

## 测试

运行测试脚本验证功能：

```bash
# 测试似然优化器
python test_likelihood_optimizer.py

# 测试优化逻辑
python test_optimization_logic.py
```

## 注意事项

1. **地图格式**: 确保子图和全局地图使用相同的概率表示
2. **分辨率匹配**: 多分辨率优化需要对应分辨率的全局地图文件
3. **初始位姿**: 似然优化对初始位姿敏感，建议使用较好的初始估计
4. **参数调优**: 根据具体应用场景调整优化参数
5. **结果验证**: 建议结合可视化结果验证优化效果

## 扩展开发

### 添加新的优化算法

```python
def custom_optimizer(objective_func, initial_params, **kwargs):
    # 实现自定义优化算法
    pass

# 在 match_submap_with_likelihood 中添加支持
if method == 'custom':
    result = custom_optimizer(objective_func, initial_params, **kwargs)
```

### 添加新的似然函数

```python
def custom_likelihood_function(submap, global_map, pose, **kwargs):
    # 实现自定义似然计算
    pass
```

## 版本历史

- **v1.0**: 初始版本，支持单分辨率似然优化
- **v1.1**: 添加多分辨率优化支持
- **v1.2**: 重构为独立模块，与粒子滤波并列
- **v1.3**: 优化逻辑结构，支持单层和多层模式 