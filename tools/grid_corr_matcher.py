"""Grid correlation based matching utilities."""

import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import rotate
from typing import Dict, Tuple

from fuse_submaps import GridMap, downsample_map
from particle_filter_matcher import compute_matching_error


def _prob_to_signed(grid: np.ndarray) -> np.ndarray:
    """Convert probability grid to signed values.

    Occupied cells become +1, free cells -1, unknown 0. This emphasises
    mismatches during correlation while keeping unknown cells neutral.
    """
    signed = np.zeros_like(grid, dtype=float)
    signed[grid > 0.6] = 1.0
    signed[grid < 0.4] = -1.0
    return signed


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

    global_grid = _prob_to_signed(global_map.to_matrix())
    submap_grid = _prob_to_signed(submap.to_matrix())

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

    ones_kernel = np.ones_like(submap_grid)

    for ang in np.arange(-theta_range_deg, theta_range_deg + theta_step_deg, theta_step_deg):
        theta = init_theta + np.deg2rad(ang)
        rotated = rotate(submap_grid, ang, reshape=False, order=0, prefilter=False)
        rot_energy = np.sum(rotated ** 2)
        if rot_energy == 0:
            continue

        corr = correlate2d(patch, rotated, mode="valid")
        patch_energy = correlate2d(patch ** 2, ones_kernel, mode="valid")
        norm_corr = corr / np.sqrt(np.maximum(patch_energy * rot_energy, 1e-6))

        iy, ix = np.unravel_index(np.argmax(norm_corr), norm_corr.shape)
        score = norm_corr[iy, ix]
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


def coarse_to_fine_grid_match(
    multi_res_submaps: Dict[float, GridMap],
    multi_res_global_maps: Dict[float, GridMap],
    init_pose: np.ndarray,
    visualize: bool = False,
) -> Tuple[np.ndarray, float]:
    """Perform hierarchical grid correlation over multiple resolutions."""
    resolutions = [1.6, 0.8, 0.4, 0.2, 0.1]

    # Search parameters roughly mimic the particle filter settings
    config = {
        1.6: (4.0, 30.0),
        0.8: (2.0, 15.0),
        0.4: (1.0, 15.0),
        0.2: (0.5, 10.0),
        0.1: (0.3, 5.0),
    }

    current_pose = init_pose.copy()
    for res in resolutions:
        if res not in multi_res_submaps or res not in multi_res_global_maps:
            continue
        search, theta = config[res]
        current_pose, _ = match_submap_with_grid_correlation(
            multi_res_submaps[res],
            multi_res_global_maps[res],
            current_pose,
            search_range=search,
            theta_range_deg=theta,
            theta_step_deg=1.0,
            submap_res=res,
            global_res=res,
            visualize=visualize,
        )

    final_error = compute_matching_error(
        multi_res_submaps[0.1], multi_res_global_maps[0.1], current_pose,
        submap_res=0.1, global_res=0.1
    )
    return current_pose, final_error
