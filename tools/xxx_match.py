import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import rotate
from typing import Tuple

from fuse_submaps import GridMap, downsample_map
from particle_filter_matcher import compute_matching_error


def _grid_to_binary(grid: np.ndarray) -> np.ndarray:
    """将概率栅格转换为二值占用栅格"""
    return (grid > 0.6).astype(float)


def match_submap_with_grid_correlation(
    submap: GridMap,
    global_map: GridMap,
    init_pose: np.ndarray,
    search_range: float = 1.0,
    theta_range_deg: float = 10.0,
    theta_step_deg: float = 1.0,
    submap_res: float = 0.05,
    global_res: float = 0.1,
    visualize: bool = False,
) -> Tuple[np.ndarray, float]:
    """使用二维相关性在栅格层面匹配子图

    参数:
        submap: 待匹配的子图
        global_map: 全局地图
        init_pose: 初始位姿估计
        search_range: 平移搜索范围 (米)
        theta_range_deg: 角度搜索范围 (度)
        theta_step_deg: 角度步长 (度)
    返回:
        最优位姿和匹配误差
    """
    if abs(submap_res - global_res) > 1e-6 and submap_res < global_res:
        submap = downsample_map(submap, submap_res, global_res)
        submap_res = global_res

    global_grid = _grid_to_binary(global_map.to_matrix())
    submap_grid = _grid_to_binary(submap.to_matrix())

    init_theta = np.arctan2(init_pose[1, 0], init_pose[0, 0])
    cx, cy = init_pose[0, 3], init_pose[1, 3]
    center_i = int(round(cx / global_res)) - global_map.min_i
    center_j = int(round(cy / global_res)) - global_map.min_j
    range_cells = int(search_range / global_res) + max(submap_grid.shape) // 2

    patch_min_i = max(0, center_i - range_cells)
    patch_max_i = min(global_grid.shape[0], center_i + range_cells)
    patch_min_j = max(0, center_j - range_cells)
    patch_max_j = min(global_grid.shape[1], center_j + range_cells)
    patch = global_grid[patch_min_i:patch_max_i, patch_min_j:patch_max_j]

    best_score = -np.inf
    best_pose = init_pose.copy()

    for ang in np.arange(-theta_range_deg, theta_range_deg + theta_step_deg, theta_step_deg):
        theta = init_theta + np.deg2rad(ang)
        rotated = rotate(submap_grid, ang, reshape=False, order=1, prefilter=False)
        corr = correlate2d(patch, rotated, mode="valid")
        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr[iy, ix]
        if score > best_score:
            best_score = score
            gi = patch_min_i + iy + submap_grid.shape[0] // 2
            gj = patch_min_j + ix + submap_grid.shape[1] // 2
            x = (gi + global_map.min_i) * global_res
            y = (gj + global_map.min_j) * global_res
            c, s = np.cos(theta), np.sin(theta)
            best_pose = np.array(
                [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

    final_error = compute_matching_error(
        submap, global_map, best_pose, submap_res=submap_res, global_res=global_res
    )
    return best_pose, final_error
