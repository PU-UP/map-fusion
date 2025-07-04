import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from fuse_submaps import GridMap, decode_key, encode_key
import matplotlib as mpl
import time

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP' , 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def extract_occupied_cells(submap: GridMap, res: float) -> np.ndarray:
    """Return (N,2) array of world coordinates for cells with prob > 0.6."""
    coords = []
    for key, p in submap.occ_map.items():
        if p > 0.6:
            i, j = decode_key(key)
            coords.append((i * res, j * res))
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)

class Particle:
    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        
    def to_matrix(self) -> np.ndarray:
        """Convert particle pose to 4x4 transformation matrix"""
        T = np.eye(4)
        c, s = np.cos(self.theta), np.sin(self.theta)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T[:3, 3] = [self.x, self.y, 0]
        return T

class ParticleFilter:
    def __init__(self, 
                 n_particles: int = 100,
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),
                 measurement_noise: float = 0.1):
        self.n_particles = n_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.particles: List[Particle] = []
        
    def initialize_particles(self, init_pose: np.ndarray, spread: Tuple[float, float, float]):
        """Initialize particles around initial pose with given spread"""
        self.particles = []
        
        # Extract initial position and orientation
        x = init_pose[0, 3]
        y = init_pose[1, 3]
        theta = np.arctan2(init_pose[1, 0], init_pose[0, 0])
        
        # Generate particles with Gaussian distribution
        for _ in range(self.n_particles):
            px = np.random.normal(x, spread[0])
            py = np.random.normal(y, spread[1])
            ptheta = np.random.normal(theta, spread[2])
            self.particles.append(Particle(px, py, ptheta, 1.0/self.n_particles))
            
    def predict(self, delta_pose: np.ndarray):
        """Move particles according to motion model with noise"""
        dx = delta_pose[0, 3]
        dy = delta_pose[1, 3]
        dtheta = np.arctan2(delta_pose[1, 0], delta_pose[0, 0])
        
        for p in self.particles:
            p.x += dx + np.random.normal(0, self.motion_noise[0])
            p.y += dy + np.random.normal(0, self.motion_noise[1])
            p.theta += dtheta + np.random.normal(0, self.motion_noise[2])
            p.theta = np.mod(p.theta + np.pi, 2*np.pi) - np.pi
            
    def update_weights(self,
                      occupied_cells: np.ndarray,
                      global_map: 'GridMap',
                      global_res: float = 0.1):
        """Update particle weights based on map matching score"""
        total_weight = 0.0

        if occupied_cells.size == 0:
            return

        occ_points = occupied_cells

        for particle in self.particles:
            c, s = np.cos(particle.theta), np.sin(particle.theta)
            rot = np.array([[c, -s], [s, c]])
            trans = np.array([particle.x, particle.y])

            world_pts = occ_points @ rot.T + trans

            gi = np.round(world_pts[:, 0] / global_res).astype(np.int64)
            gj = np.round(world_pts[:, 1] / global_res).astype(np.int64)

            keys = (gi.astype(np.int64) << 32) | (gj.astype(np.int64) & 0xFFFFFFFF)
            probs = np.array([global_map.occ_map.get(int(k), 0.0) for k in keys])

            p_binary = np.where(probs > 0.6, 1.0, 0.0)
            diff = np.abs(p_binary - 1.0)

            avg_score = diff.mean() if diff.size > 0 else 1.0
            weight = np.exp(-avg_score / self.measurement_noise)
            particle.weight = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
                
    def resample(self):
        """Resample particles based on their weights"""
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum([p.weight for p in self.particles])
        
        new_particles = []
        i = 0
        for position in positions:
            while cumsum[i] < position:
                i += 1
            old_particle = self.particles[i]
            new_particles.append(Particle(
                old_particle.x,
                old_particle.y,
                old_particle.theta,
                1.0/self.n_particles
            ))
        
        self.particles = new_particles
        
    def get_estimated_pose(self) -> np.ndarray:
        """Get weighted average pose from particles"""
        if not self.particles:
            return np.eye(4)
            
        # Weighted average of position
        avg_x = sum(p.x * p.weight for p in self.particles)
        avg_y = sum(p.y * p.weight for p in self.particles)
        
        # For orientation, handle circular mean
        cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles)
        sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles)
        avg_theta = np.arctan2(sin_sum, cos_sum)
        
        # Convert to transformation matrix
        T = np.eye(4)
        c, s = np.cos(avg_theta), np.sin(avg_theta)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T[:3, 3] = [avg_x, avg_y, 0]
        return T

def match_submap_with_particle_filter(submap: 'GridMap',
                                    global_map: 'GridMap',
                                    init_pose: np.ndarray,
                                    n_particles: int = 500,
                                    n_iterations: int = 200,
                                    visualize: bool = True,
                                    spread: Tuple[float, float, float] = (0.5, 0.5, np.pi/6),
                                    submap_res: float = 0.05,
                                    global_res: float = 0.1,
                                    debug: bool = False) -> Tuple[np.ndarray, float]:
    """Match submap to global map using particle filter"""
    
    # Initialize particle filter
    pf = ParticleFilter(
        n_particles=n_particles,
        motion_noise=(0.05, 0.05, 0.02),
        measurement_noise=0.08
    )
    
    pf.initialize_particles(init_pose, spread=spread)

    occupied_cells = extract_occupied_cells(submap, submap_res)
    
    # Setup visualization
    if visualize:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Particle Filter Map Matching')
        global_grid_vis = global_map.to_matrix()
    
    best_pose = init_pose.copy()
    min_error = float('inf')

    no_improvement_threshold = 20
    error_tolerance = 0.001
    no_improvement_count = 0
    
    # 耗时检测相关变量
    timing_start_iter = 5  # 从第5次迭代开始计时
    iteration_times = []
    total_timing_time = 0.0
    
    # 误差变化记录（用于debug模式下的绘图）
    error_history = []
    iteration_history = []
    
    for iter in range(n_iterations):
        # 开始计时（从第5次迭代开始）
        if iter >= timing_start_iter:
            iter_start_time = time.time()
        
        # Predict (random walk for exploration)
        pf.predict(np.eye(4))

        # Update weights based on map matching
        pf.update_weights(occupied_cells, global_map, global_res=global_res)
        
        # Get current estimate
        current_pose = pf.get_estimated_pose()
        
        # Calculate current error
        error = compute_matching_error(occupied_cells, global_map, current_pose, global_res=global_res)
        
        # 记录误差历史（用于debug模式）
        if debug:
            error_history.append(error)
            iteration_history.append(iter)
        
        # Update best pose if needed
        if error < min_error:
            if min_error - error > error_tolerance:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            min_error = error
            best_pose = current_pose.copy()
            print(f"Iteration {iter}: New best pose found, error = {error:.3f}")
        else:
            no_improvement_count += 1
        
        # 计算并记录耗时（从第5次迭代开始）
        if iter >= timing_start_iter:
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            iteration_times.append(iter_time)
            total_timing_time += iter_time
            
            # 显示耗时信息（仅在debug模式下）
            if debug and (iter % 10 == 0 or iter == n_iterations - 1):  # 每10次迭代显示一次，或最后一次
                avg_time = total_timing_time / len(iteration_times)
                print(f"迭代 {iter}: 耗时 {iter_time:.4f}s, 平均耗时 {avg_time:.4f}s")
        
        if no_improvement_count >= no_improvement_threshold:
            print(
                f"Early stopping due to no significant improvement for {no_improvement_threshold} iterations."
            )
            break

        # Visualize if requested
        if visualize and iter % 5 == 0:
            ax1.clear()
            ax2.clear()

            # Plot global map
            ax1.imshow(global_grid_vis, cmap='gray', origin='upper')
            
            # Plot particles
            particle_matrix_cols = [(p.y / global_res) - global_map.min_j for p in pf.particles]
            particle_matrix_rows = [(p.x / global_res) - global_map.min_i for p in pf.particles]
            weights = [p.weight * 1000 for p in pf.particles]
            
            ax1.scatter(particle_matrix_cols, particle_matrix_rows, c='red', s=weights, alpha=0.5)
            
            # Plot transformed submap
            temp_map = transform_submap(submap, current_pose, submap_res, global_res)
            match_grid = temp_map.to_matrix()
            ax2.imshow(match_grid, cmap='gray', origin='upper')
            ax2.set_title(f'Matching Result (Iter {iter})')
            
            plt.pause(0.1)
        
        # Resample particles
        pf.resample()
    
    # 显示总体耗时统计（仅在debug模式下）
    if debug and iteration_times:
        total_iterations = len(iteration_times)
        total_time = sum(iteration_times)
        avg_time = total_time / total_iterations
        min_time = min(iteration_times)
        max_time = max(iteration_times)
        
        print(f"\n========== 耗时统计 (从第{timing_start_iter}次迭代开始) ==========")
        print(f"总迭代次数: {total_iterations}")
        print(f"总耗时: {total_time:.4f}s")
        print(f"平均耗时: {avg_time:.4f}s")
        print(f"最快耗时: {min_time:.4f}s")
        print(f"最慢耗时: {max_time:.4f}s")
    
    if visualize:
        plt.ioff()
        plt.close(fig)

    result = (best_pose, min_error)

    # 绘制误差变化图（仅在debug模式下，且不计入耗时）
    if debug and error_history:
        plt.figure(figsize=(12, 8))
        plt.plot(iteration_history, error_history, 'b-', linewidth=2, label='匹配误差')
        plt.xlabel('迭代次数')
        plt.ylabel('匹配误差')
        plt.title('粒子滤波优化过程中的误差变化')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 标记最佳误差点
        min_error_idx = np.argmin(error_history)
        plt.plot(iteration_history[min_error_idx], error_history[min_error_idx], 'ro', 
                markersize=10, label=f'最佳误差: {error_history[min_error_idx]:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return result

def compute_matching_error(occupied_cells: np.ndarray,
                         global_map: 'GridMap',
                         pose: np.ndarray,
                         global_res: float = 0.1) -> float:
    """Compute matching error between submap and global map"""
    if occupied_cells.size == 0:
        return 0.0

    c, s = pose[0, 0], pose[1, 0]
    rot = np.array([[c, -s], [s, c]])
    trans = pose[:2, 3]

    world_pts = occupied_cells @ rot.T + trans

    gi = np.round(world_pts[:, 0] / global_res).astype(np.int64)
    gj = np.round(world_pts[:, 1] / global_res).astype(np.int64)

    keys = (gi.astype(np.int64) << 32) | (gj.astype(np.int64) & 0xFFFFFFFF)
    probs = np.array([global_map.occ_map.get(int(k), 0.0) for k in keys])

    valid = probs > -1  # all entries valid, keep for consistency
    if np.any(valid):
        errors = np.abs(probs[valid] - 1.0)
        return errors.mean()
    return 0.0

def transform_submap(submap: 'GridMap', pose: np.ndarray,
                     submap_res: float = 0.05,
                     global_res: float = 0.1) -> 'GridMap':
    """Transform submap using given pose"""
    transformed = GridMap()

    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        p_s = np.array([sub_i * submap_res, sub_j * submap_res, 0.0])
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]

        gi_glob = int(np.round(p_w[0] / global_res))
        gj_glob = int(np.round(p_w[1] / global_res))

        transformed.update_occ(gi_glob, gj_glob, p_sub)

    return transformed
