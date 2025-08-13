#!/usr/bin/env python3
"""
简化版本的Hermite RBF内存优化测试
避免复杂依赖，专注于验证内存优化算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class HermiteMemoryTest(nn.Module):
    """
    专门用于测试Hermite RBF内存优化的简化版本
    """
    def __init__(self, n_kernel=32, hermite_order=2, hsi_dim=31):
        super().__init__()
        self.n_kernel = n_kernel
        self.hermite_order = hermite_order
        self.hsi_dim = hsi_dim
        
        # RBF参数
        self.rbf_centers = nn.Parameter(self._init_rbf_centers(n_kernel))
        self.rbf_sigmas = nn.Parameter(self._init_rbf_sigmas(n_kernel))
        
        # Hermite权重
        hermite_dim = self._get_hermite_dim(2, hermite_order)
        self.rbf_weights = nn.Parameter(torch.randn(n_kernel, hermite_dim, hsi_dim))
        
        # 简化的MLP
        self.hermite_mlp = SimpleMLP(in_dim=66, out_dim=hsi_dim, hidden_list=[128, 64])
    
    def _get_hermite_dim(self, in_dim: int, order: int) -> int:
        """计算Hermite基函数的维度"""
        dim = 1  # 0阶
        if order >= 1:
            dim += in_dim  # 1阶
        if order >= 2:
            dim += in_dim * (in_dim + 1) // 2  # 2阶
        return dim
    
    def _init_rbf_centers(self, n_kernel):
        """智能初始化RBF中心"""
        grid_size = int(math.sqrt(n_kernel))
        if grid_size * grid_size < n_kernel:
            grid_size += 1
            
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_kernel]
        centers += 0.1 * torch.randn_like(centers)
        return centers
    
    def _init_rbf_sigmas(self, n_kernel):
        """智能初始化RBF带宽"""
        avg_distance = 2.0 / math.sqrt(n_kernel)
        base_sigma = avg_distance * 0.8
        sigmas = base_sigma * (0.5 + torch.rand(n_kernel))
        return sigmas
    
    def compute_hermite_memory_efficient(self, mlp_output, coord_diff, sigmas, gaussian_weights):
        """内存高效版本：逐核心计算"""
        B, N, out_dim = mlp_output.shape
        K = coord_diff.shape[2]
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)
        
        final_pred = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
        
        # 逐个核心处理
        for k in range(K):
            dx = coord_diff[:, :, k, 0]  # [B, N]
            dy = coord_diff[:, :, k, 1]  # [B, N]
            sigma_k = sigmas[k]
            weight_k = gaussian_weights[:, :, k]  # [B, N]
            
            epsilon = 1e-6
            idx = 0
            kernel_contribution = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
            
            for order in range(self.hermite_order + 1):
                for alpha in range(order + 1):
                    beta = order - alpha
                    
                    # 计算Hermite系数
                    if alpha == 0 and beta == 0:
                        hermite_coeff = torch.ones_like(dx)
                    elif alpha == 1 and beta == 0:
                        hermite_coeff = -dx / (sigma_k**2 + epsilon)
                    elif alpha == 0 and beta == 1:
                        hermite_coeff = -dy / (sigma_k**2 + epsilon)
                    elif alpha == 2 and beta == 0:
                        hermite_coeff = (dx**2) / (sigma_k**4 + epsilon) - 1.0 / (sigma_k**2 + epsilon)
                    elif alpha == 1 and beta == 1:
                        hermite_coeff = (dx * dy) / (sigma_k**4 + epsilon)
                    elif alpha == 0 and beta == 2:
                        hermite_coeff = (dy**2) / (sigma_k**4 + epsilon) - 1.0 / (sigma_k**2 + epsilon)
                    else:
                        hermite_coeff = torch.zeros_like(dx)
                    
                    # 应用系数和权重
                    weighted_hermite_term = (hermite_coeff * weight_k).unsqueeze(-1) * mlp_output
                    kernel_contribution += weighted_hermite_term * self.rbf_weights[k, idx, :].unsqueeze(0).unsqueeze(0)
                    idx += 1
            
            final_pred += kernel_contribution
        
        return final_pred
    
    def compute_hermite_original(self, mlp_output, coord_diff, sigmas, gaussian_weights):
        """原始版本：创建大型5D张量"""
        B, N, out_dim = mlp_output.shape
        K = coord_diff.shape[2]
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)
        
        # 创建大型张量 - 这里容易OOM
        hermite_features = torch.zeros(B, N, K, hermite_dim, out_dim, 
                                     device=mlp_output.device, dtype=mlp_output.dtype)
        
        mlp_expanded = mlp_output.unsqueeze(2).expand(B, N, K, out_dim)
        
        dx = coord_diff[..., 0]  # [B, N, K]
        dy = coord_diff[..., 1]  # [B, N, K]
        sigma_expanded = sigmas.view(1, 1, K)
        
        epsilon = 1e-6
        idx = 0
        
        for order in range(self.hermite_order + 1):
            for alpha in range(order + 1):
                beta = order - alpha
                
                if alpha == 0 and beta == 0:
                    hermite_coeff = torch.ones_like(dx)
                elif alpha == 1 and beta == 0:
                    hermite_coeff = -dx / (sigma_expanded**2 + epsilon)
                elif alpha == 0 and beta == 1:
                    hermite_coeff = -dy / (sigma_expanded**2 + epsilon)
                elif alpha == 2 and beta == 0:
                    hermite_coeff = (dx**2) / (sigma_expanded**4 + epsilon) - 1.0 / (sigma_expanded**2 + epsilon)
                elif alpha == 1 and beta == 1:
                    hermite_coeff = (dx * dy) / (sigma_expanded**4 + epsilon)
                elif alpha == 0 and beta == 2:
                    hermite_coeff = (dy**2) / (sigma_expanded**4 + epsilon) - 1.0 / (sigma_expanded**2 + epsilon)
                else:
                    hermite_coeff = torch.zeros_like(dx)
                
                hermite_features[:, :, :, idx, :] = hermite_coeff.unsqueeze(-1) * mlp_expanded
                idx += 1
        
        # 应用高斯权重 - 另一个大型张量
        weighted_hermite = hermite_features * gaussian_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 最终计算
        final_pred = torch.einsum('bnkho,kho->bno', weighted_hermite, self.rbf_weights)
        return final_pred

def test_memory_comparison():
    """比较不同方法的内存使用"""
    print("=== Hermite RBF 内存优化测试 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试配置
    configs = [
        {"n_kernel": 16, "size": (32, 32), "name": "小配置"},
        {"n_kernel": 32, "size": (64, 64), "name": "中配置"},
        {"n_kernel": 64, "size": (128, 128), "name": "大配置"},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"内核数量: {config['n_kernel']}, 图像大小: {config['size']}")
        
        try:
            model = HermiteMemoryTest(n_kernel=config['n_kernel']).to(device)
            
            H, W = config['size']
            B, N = 1, H * W
            
            # 创建测试数据
            mlp_output = torch.randn(B, N, 31, device=device)
            coord_diff = torch.randn(B, N, config['n_kernel'], 2, device=device)
            gaussian_weights = torch.randn(B, N, config['n_kernel'], device=device)
            
            # 估算内存
            hermite_dim = model._get_hermite_dim(2, 2)
            tensor_size = B * N * config['n_kernel'] * hermite_dim * 31
            memory_mb = tensor_size * 4 / (1024 ** 2)
            print(f"估算5D张量内存: {memory_mb:.1f} MB")
            
            # 测试内存高效版本
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()
            
            try:
                result_efficient = model.compute_hermite_memory_efficient(
                    mlp_output, coord_diff, model.rbf_sigmas, gaussian_weights
                )
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                    efficient_mem = torch.cuda.memory_allocated() - start_mem
                    print(f"内存高效版本: 成功, GPU内存: +{efficient_mem / 1024 / 1024:.1f}MB")
                else:
                    print(f"内存高效版本: 成功")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"内存高效版本: 内存不足")
                else:
                    print(f"内存高效版本: 错误 - {str(e)}")
            
            # 测试原始版本（可能会OOM）
            if memory_mb < 500:  # 只在预计内存较小时测试
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    start_mem = torch.cuda.memory_allocated()
                
                try:
                    result_original = model.compute_hermite_original(
                        mlp_output, coord_diff, model.rbf_sigmas, gaussian_weights
                    )
                    
                    if device == 'cuda':
                        torch.cuda.synchronize()
                        original_mem = torch.cuda.memory_allocated() - start_mem
                        print(f"原始版本: 成功, GPU内存: +{original_mem / 1024 / 1024:.1f}MB")
                    else:
                        print(f"原始版本: 成功")
                    
                    # 验证结果一致性
                    if 'result_efficient' in locals():
                        diff = torch.max(torch.abs(result_efficient - result_original))
                        print(f"结果差异: {diff.item():.6f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"原始版本: 内存不足 (预期)")
                    else:
                        print(f"原始版本: 错误 - {str(e)}")
            else:
                print(f"原始版本: 跳过 (预计内存过大: {memory_mb:.1f}MB)")
                
        except Exception as e:
            print(f"配置失败: {str(e)}")

if __name__ == "__main__":
    test_memory_comparison()
