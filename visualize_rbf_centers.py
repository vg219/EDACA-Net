#!/usr/bin/env python3
"""
可视化RBF核心位置分布和训练过程中的变化
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_rbf_centers_distribution(n_kernel=16):
    """可视化RBF核心的初始分布"""
    
    # 模拟初始化过程
    grid_size = int(np.sqrt(n_kernel))
    if grid_size * grid_size < n_kernel:
        grid_size += 1
    
    # 标准方法：[-1, 1]
    x_std = torch.linspace(-1, 1, grid_size)
    y_std = torch.linspace(-1, 1, grid_size)
    xx_std, yy_std = torch.meshgrid(x_std, y_std, indexing='ij')
    centers_std = torch.stack([xx_std.flatten(), yy_std.flatten()], dim=1)[:n_kernel]
    centers_std += 0.1 * torch.randn_like(centers_std)
    
    # 自适应方法：扩大范围
    rbf_range = 1.0 + 1.0/8 + 0.3  # 1.425
    x_ada = torch.linspace(-rbf_range, rbf_range, grid_size)
    y_ada = torch.linspace(-rbf_range, rbf_range, grid_size)
    xx_ada, yy_ada = torch.meshgrid(x_ada, y_ada, indexing='ij')
    centers_ada = torch.stack([xx_ada.flatten(), yy_ada.flatten()], dim=1)[:n_kernel]
    centers_ada += 0.1 * torch.randn_like(centers_ada)
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 标准方法
    ax1.scatter(centers_std[:, 0], centers_std[:, 1], c='blue', s=100, alpha=0.7)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('标准初始化：[-1, 1] x [-1, 1]')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    
    # 查询范围框
    ax1.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='red', linewidth=2, label='查询范围'))
    ax1.legend()
    
    # 自适应方法
    ax2.scatter(centers_ada[:, 0], centers_ada[:, 1], c='green', s=100, alpha=0.7)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('自适应初始化：[-1.425, 1.425] x [-1.425, 1.425]')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    
    # 查询范围框
    ax2.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='red', linewidth=2, label='查询范围'))
    ax2.add_patch(plt.Rectangle((-rbf_range, -rbf_range), 2*rbf_range, 2*rbf_range, 
                               fill=False, edgecolor='green', linewidth=2, linestyle='--', label='RBF覆盖范围'))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rbf_centers_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return centers_std, centers_ada

def simulate_training_movement(n_kernel=16, epochs=100):
    """模拟训练过程中核心位置的移动"""
    
    # 初始位置
    grid_size = int(np.sqrt(n_kernel))
    if grid_size * grid_size < n_kernel:
        grid_size += 1
    
    rbf_range = 1.425
    x = torch.linspace(-rbf_range, rbf_range, grid_size)
    y = torch.linspace(-rbf_range, rbf_range, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    centers_init = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_kernel]
    centers_init += 0.1 * torch.randn_like(centers_init)
    
    # 模拟训练过程的移动
    centers_history = [centers_init.clone()]
    centers_current = centers_init.clone()
    
    for epoch in range(epochs):
        # 模拟梯度更新：向特定区域聚集
        # 假设右上角区域需要更密集的核心
        target_region = torch.tensor([0.5, 0.5])  # 目标区域中心
        
        for i in range(n_kernel):
            # 计算当前核心到目标区域的距离
            dist_to_target = torch.norm(centers_current[i] - target_region)
            
            # 模拟梯度：距离越远，移动越快
            if dist_to_target > 0.5:  # 只有距离较远的核心才移动
                direction = (target_region - centers_current[i]) / (dist_to_target + 1e-8)
                move_strength = min(0.02, dist_to_target * 0.01)  # 移动强度
                centers_current[i] += direction * move_strength
        
        # 添加随机扰动模拟训练噪声
        centers_current += 0.001 * torch.randn_like(centers_current)
        
        # 每10个epoch记录一次
        if epoch % 10 == 0:
            centers_history.append(centers_current.clone())
    
    # 可视化移动轨迹
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制初始位置
    init_centers = centers_history[0]
    ax.scatter(init_centers[:, 0], init_centers[:, 1], 
              c='blue', s=100, alpha=0.7, label='初始位置', marker='o')
    
    # 绘制最终位置
    final_centers = centers_history[-1]
    ax.scatter(final_centers[:, 0], final_centers[:, 1], 
              c='red', s=100, alpha=0.7, label='训练后位置', marker='s')
    
    # 绘制移动轨迹
    for i in range(n_kernel):
        trajectory_x = [centers[i, 0] for centers in centers_history]
        trajectory_y = [centers[i, 1] for centers in centers_history]
        ax.plot(trajectory_x, trajectory_y, 'gray', alpha=0.5, linewidth=1)
    
    # 标记目标区域
    target_circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color='green', 
                              linewidth=2, linestyle='--', label='目标密集区域')
    ax.add_patch(target_circle)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)
    ax.set_title('训练过程中RBF核心位置的学习移动')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('rbf_centers_training_movement.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return centers_history

def analyze_center_density(centers, grid_resolution=50):
    """分析核心密度分布"""
    centers_np = centers.numpy()
    
    # 创建网格
    x_range = np.linspace(-2, 2, grid_resolution)
    y_range = np.linspace(-2, 2, grid_resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # 计算每个网格点的核心密度
    density = np.zeros((grid_resolution, grid_resolution))
    bandwidth = 0.2  # 密度估计带宽
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            point = np.array([xx[i, j], yy[i, j]])
            # 计算到所有核心的距离
            distances = np.linalg.norm(centers_np - point, axis=1)
            # 高斯核密度估计
            density[i, j] = np.sum(np.exp(-distances**2 / (2 * bandwidth**2)))
    
    # 绘制密度热图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, density, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label='核心密度')
    plt.scatter(centers_np[:, 0], centers_np[:, 1], c='red', s=50, alpha=0.8)
    plt.title('RBF核心密度分布')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rbf_centers_density.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== RBF核心位置分析 ===")
    
    # 1. 可视化初始分布
    print("1. 可视化初始分布对比...")
    centers_std, centers_ada = visualize_rbf_centers_distribution(n_kernel=16)
    
    # 2. 模拟训练移动
    print("2. 模拟训练过程中的位置学习...")
    training_history = simulate_training_movement(n_kernel=16, epochs=100)
    
    # 3. 分析密度分布
    print("3. 分析最终的核心密度分布...")
    analyze_center_density(training_history[-1])
    
    # 4. 统计信息
    print("\n=== 统计信息 ===")
    print(f"初始核心范围: [{centers_ada.min():.3f}, {centers_ada.max():.3f}]")
    print(f"训练后核心范围: [{training_history[-1].min():.3f}, {training_history[-1].max():.3f}]")
    
    # 计算核心移动距离
    init_centers = training_history[0]
    final_centers = training_history[-1]
    movement_distances = torch.norm(final_centers - init_centers, dim=1)
    print(f"平均移动距离: {movement_distances.mean():.3f}")
    print(f"最大移动距离: {movement_distances.max():.3f}")
    print(f"移动距离标准差: {movement_distances.std():.3f}")
