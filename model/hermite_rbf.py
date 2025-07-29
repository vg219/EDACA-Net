import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional


class HermiteRBF(nn.Module):
    """
    Hermite Radial Basis Function Network
    结合传统RBF和Hermite插值的优势，支持高阶导数信息
    """
    
    def __init__(self, 
                 cmin: float = -1.0, 
                 cmax: float = 1.0, 
                 s_dims: Tuple[int, int] = (64, 64),
                 in_dim: int = 2, 
                 out_dim: int = 31,
                 n_kernel: int = 256,
                 hermite_order: int = 2,
                 rbf_type: str = 'hermite_gauss',
                 adaptive_sigma: bool = True,
                 use_bias: bool = True,
                 **kwargs):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_kernel = n_kernel
        self.hermite_order = hermite_order
        self.rbf_type = rbf_type
        self.adaptive_sigma = adaptive_sigma
        self.use_bias = use_bias
        
        # 核心参数
        self.centers = nn.Parameter(torch.randn(n_kernel, in_dim))
        
        if adaptive_sigma:
            # 每个核心独立的sigma
            self.sigmas = nn.Parameter(torch.ones(n_kernel))
        else:
            # 全局sigma
            self.register_parameter('sigma_global', nn.Parameter(torch.tensor(1.0)))
        
        # Hermite权重 - 包含函数值和各阶导数的权重
        hermite_dim = self._get_hermite_dim(in_dim, hermite_order)
        self.hermite_weights = nn.Parameter(torch.randn(n_kernel, hermite_dim, out_dim))
        
        # Hermite形状参数
        self.alpha = nn.Parameter(torch.ones(n_kernel))  # 主要形状参数
        self.beta = nn.Parameter(torch.ones(n_kernel))   # 二次形状参数
        
        # 偏置项
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # 初始化参数
        self._init_parameters(cmin, cmax)
        
    def _get_hermite_dim(self, in_dim: int, order: int) -> int:
        """计算Hermite基函数的维度"""
        dim = 1  # 0阶: 函数值
        if order >= 1:
            dim += in_dim  # 1阶: 一阶导数
        if order >= 2:
            dim += in_dim * (in_dim + 1) // 2  # 2阶: 二阶导数
        return dim
    
    def _init_parameters(self, cmin: float, cmax: float):
        """初始化参数"""
        with torch.no_grad():
            # 初始化中心点为规则网格
            n_per_dim = int(math.ceil(self.n_kernel**(1/self.in_dim)))
            
            if self.in_dim == 2:
                # 2D网格
                x = torch.linspace(cmin, cmax, n_per_dim)
                y = torch.linspace(cmin, cmax, n_per_dim)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            else:
                # 通用多维网格
                coords = [torch.linspace(cmin, cmax, n_per_dim) for _ in range(self.in_dim)]
                grid = torch.meshgrid(*coords, indexing='ij')
                centers = torch.stack([g.flatten() for g in grid], dim=-1)
            
            # 只取前n_kernel个点
            self.centers.data = centers[:self.n_kernel]
            
            # 初始化sigma
            avg_spacing = (cmax - cmin) / n_per_dim
            if self.adaptive_sigma:
                self.sigmas.data.fill_(avg_spacing)
            else:
                self.sigma_global.data.fill_(avg_spacing)
            
            # 初始化Hermite参数
            self.alpha.data.uniform_(0.5, 1.5)
            self.beta.data.uniform_(-0.5, 0.5)
            
            # 初始化权重
            hermite_dim = self.hermite_weights.shape[1]
            nn.init.xavier_uniform_(self.hermite_weights.data)
            
            # 对0阶权重给更大的初始值（函数值比导数更重要）
            self.hermite_weights.data[:, 0, :] *= 2.0
    
    def compute_hermite_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算Hermite基函数及其导数
        Args:
            x: [p, d] 查询点坐标
        Returns:
            hermite_features: [p, k, hermite_dim] Hermite基函数特征
        """
        p, d = x.shape
        k = self.centers.shape[0]
        
        # 计算距离向量
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # [p, k, d]
        r2 = torch.sum(diff**2, dim=-1)  # [p, k]
        
        # 获取sigma
        if self.adaptive_sigma:
            sigma = self.sigmas.unsqueeze(0)  # [1, k]
        else:
            sigma = self.sigma_global.expand(1, k)  # [1, k]
        
        # 标准化距离
        normalized_r2 = r2 / (sigma**2 + 1e-8)
        
        # 基础高斯函数
        phi_0 = torch.exp(-0.5 * normalized_r2)  # [p, k]
        
        hermite_features = []
        
        # 0阶: 函数值 - 带Hermite修正
        if self.hermite_order >= 0:
            alpha = self.alpha.unsqueeze(0)  # [1, k]
            beta = self.beta.unsqueeze(0)   # [1, k]
            h0 = phi_0 * (alpha + beta * normalized_r2)
            hermite_features.append(h0)
        
        # 1阶: 一阶导数
        if self.hermite_order >= 1:
            inv_sigma2 = 1.0 / (sigma**2 + 1e-8)  # [1, k]
            for i in range(d):
                # ∂φ/∂xi with Hermite correction
                grad_term = diff[:, :, i] * inv_sigma2  # [p, k]
                h1_i = phi_0 * grad_term * (alpha + beta * (normalized_r2 - 1))
                hermite_features.append(h1_i)
        
        # 2阶: 二阶导数  
        if self.hermite_order >= 2:
            inv_sigma2 = 1.0 / (sigma**2 + 1e-8)  # [1, k]
            
            # 对角二阶导数 ∂²φ/∂xi²
            for i in range(d):
                laplacian_term = inv_sigma2 * (
                    normalized_r2 - 1 - 
                    diff[:, :, i]**2 * inv_sigma2
                )
                h2_ii = phi_0 * (alpha + beta * (normalized_r2 - 2)) * laplacian_term
                hermite_features.append(h2_ii)
            
            # 交叉二阶导数 ∂²φ/∂xi∂xj (i < j)
            for i in range(d):
                for j in range(i+1, d):
                    cross_term = diff[:, :, i] * diff[:, :, j] * (inv_sigma2**2)
                    h2_ij = phi_0 * (alpha + beta * (normalized_r2 - 2)) * cross_term
                    hermite_features.append(h2_ij)
        
        return torch.stack(hermite_features, dim=-1)  # [p, k, hermite_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [p, d] 查询点坐标 (归一化到[-1, 1])
        Returns:
            output: [p, out_dim] 输出特征
        """
        # 计算Hermite基函数
        hermite_basis = self.compute_hermite_basis(x)  # [p, k, hermite_dim]
        
        # 与权重进行张量乘积: [p, k, h] × [k, h, o] -> [p, o]
        output = torch.einsum('pkh,kho->po', hermite_basis, self.hermite_weights)
        
        # 添加偏置
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def get_kernel_importance(self) -> torch.Tensor:
        """计算每个核心的重要性分数"""
        # 基于权重的L2范数计算重要性
        importance = torch.norm(self.hermite_weights, dim=(1, 2))
        return importance
    
    def prune_kernels(self, threshold: float = 1e-6) -> int:
        """
        修剪不重要的核心
        Args:
            threshold: 重要性阈值
        Returns:
            pruned_count: 被修剪的核心数量
        """
        with torch.no_grad():
            importance = self.get_kernel_importance()
            keep_mask = importance > threshold
            
            if keep_mask.sum() == 0:
                return 0  # 避免删除所有核心
            
            old_count = self.n_kernel
            new_count = keep_mask.sum().item()
            
            # 更新参数
            self.centers.data = self.centers.data[keep_mask]
            if self.adaptive_sigma:
                self.sigmas.data = self.sigmas.data[keep_mask]
            self.hermite_weights.data = self.hermite_weights.data[keep_mask]
            self.alpha.data = self.alpha.data[keep_mask]
            self.beta.data = self.beta.data[keep_mask]
            
            self.n_kernel = new_count
            
            return old_count - new_count


class HermiteLoss(nn.Module):
    """
    专门为Hermite RBF设计的损失函数
    包含重建损失、梯度一致性损失和平滑性损失
    """
    
    def __init__(self, 
                 lambda_grad: float = 0.1,
                 lambda_smooth: float = 0.01,
                 lambda_sparsity: float = 0.001):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_smooth = lambda_smooth
        self.lambda_sparsity = lambda_sparsity
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                model: Optional[HermiteRBF] = None) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失
        Args:
            pred: [B, C, H, W] 预测结果
            target: [B, C, H, W] 目标图像
            model: HermiteRBF模型（用于计算稀疏性损失）
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        
        # 1. 基础重建损失
        recon_loss = self.l1_loss(pred, target)
        loss_dict['recon'] = recon_loss
        
        # 2. 梯度一致性损失
        grad_loss = self._gradient_consistency_loss(pred, target)
        loss_dict['grad'] = grad_loss
        
        # 3. 平滑性损失
        smooth_loss = self._smoothness_loss(pred)
        loss_dict['smooth'] = smooth_loss
        
        # 4. 稀疏性损失（鼓励使用较少的核心）
        sparsity_loss = 0.0
        if model is not None:
            sparsity_loss = self._sparsity_loss(model)
            loss_dict['sparsity'] = sparsity_loss
        
        # 总损失
        total_loss = (recon_loss + 
                     self.lambda_grad * grad_loss + 
                     self.lambda_smooth * smooth_loss +
                     self.lambda_sparsity * sparsity_loss)
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _gradient_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算梯度一致性损失"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), 
                              sobel_x, padding=1).view_as(pred)
        pred_grad_y = F.conv2d(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), 
                              sobel_y, padding=1).view_as(pred)
        
        target_grad_x = F.conv2d(target.view(-1, 1, target.shape[-2], target.shape[-1]), 
                                sobel_x, padding=1).view_as(target)
        target_grad_y = F.conv2d(target.view(-1, 1, target.shape[-2], target.shape[-1]), 
                                sobel_y, padding=1).view_as(target)
        
        # L1损失
        grad_loss = (self.l1_loss(pred_grad_x, target_grad_x) + 
                    self.l1_loss(pred_grad_y, target_grad_y))
        
        return grad_loss
    
    def _smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """计算平滑性损失（总变分损失）"""
        # 计算相邻像素差异
        tv_h = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        
        return tv_h + tv_w
    
    def _sparsity_loss(self, model: HermiteRBF) -> torch.Tensor:
        """计算稀疏性损失"""
        # L1正则化鼓励权重稀疏
        l1_reg = torch.sum(torch.abs(model.hermite_weights))
        
        # 核心重要性的熵损失（鼓励少数核心占主导）
        importance = model.get_kernel_importance()
        importance_prob = F.softmax(importance, dim=0)
        entropy_loss = -torch.sum(importance_prob * torch.log(importance_prob + 1e-8))
        
        return l1_reg + 0.1 * entropy_loss


def create_hermite_network(hsi_dim: int = 31, 
                          msi_dim: int = 3,
                          feat_dim: int = 128,
                          hermite_order: int = 2,
                          n_kernel: int = 256) -> dict:
    """
    创建Hermite RBF网络的工厂函数
    Returns:
        包含网络和损失函数的字典
    """
    
    # Hermite RBF网络
    hermite_rbf = HermiteRBF(
        cmin=-1.0, cmax=1.0,
        in_dim=2, out_dim=hsi_dim,
        n_kernel=n_kernel,
        hermite_order=hermite_order,
        adaptive_sigma=True,
        use_bias=True
    )
    
    # 损失函数
    criterion = HermiteLoss(
        lambda_grad=0.1,
        lambda_smooth=0.01,
        lambda_sparsity=0.001
    )
    
    return {
        'hermite_rbf': hermite_rbf,
        'criterion': criterion
    }


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建网络
    components = create_hermite_network(hsi_dim=31, hermite_order=2, n_kernel=64)
    hermite_rbf = components['hermite_rbf'].to(device)
    criterion = components['criterion']
    
    # 测试数据
    batch_size = 8
    num_points = 256
    coord = torch.rand(batch_size, num_points, 2, device=device) * 2 - 1  # [-1, 1]
    target = torch.rand(batch_size, 31, 64, 64, device=device)
    
    print(f"Hermite RBF网络参数量: {sum(p.numel() for p in hermite_rbf.parameters()):,}")
    print(f"Hermite维度: {hermite_rbf._get_hermite_dim(2, 2)}")
    
    # 前向传播测试
    with torch.no_grad():
        for b in range(batch_size):
            output = hermite_rbf(coord[b])  # [num_points, 31]
            print(f"Batch {b}: 输入形状 {coord[b].shape}, 输出形状 {output.shape}")
    
    print("Hermite RBF网络测试完成！")
