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

### 默认调试选项
为了快速开始调试和测试匹配功能，建议优先使用以下命令：

```bash
python .\tools\optimize_submap.py ..\data\submap_200_visual\ --use-gt --submap 18 --multi-res
```

这个命令使用真实轨迹（ground truth）数据，针对第18个子地图进行多分辨率优化匹配，是推荐的默认调试选项。

