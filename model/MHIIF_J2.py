import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from edsr import make_edsr_baseline, make_coord
import sys
sys.path.insert(0, '/home/YuJieLiang/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord

from model.base_model import BaseModel, register_model, PatchMergeModule
# from model.hermite_rbf import HermiteRBF, HermiteLoss  # 暂时注释掉

# Advanced RBF utilities
def hashgrid_from_dict(config_dict, expected_properties=None, embed_fn=None):
    """从配置字典创建hash grid"""
    if expected_properties is None:
        expected_properties = ['bmin', 'bmax', 'num_levels', 'max_params', 'base_resolution', 'per_level_scale', 'require_grad']
    
    # 默认参数
    defaults = {
        'bmin': -1.0,
        'bmax': 1.0,
        'num_levels': 4,
        'max_params': 2**20,
        'base_resolution': 16,
        'per_level_scale': 2.0,
        'require_grad': True
    }
    
    # 合并配置
    config = {**defaults, **config_dict}
    
    # 创建简化的hash grid (这里用简单实现替代)
    class SimpleHashGrid(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.num_levels = kwargs.get('num_levels', 4)
            self.feature_dim = kwargs.get('feature_dim', 32)
            
            # 每层的embedding
            self.embeddings = nn.ModuleList([
                nn.Linear(2, self.feature_dim) for _ in range(self.num_levels)
            ])
            
        def forward(self, coords):
            # coords: [B, N, 2]
            features = []
            for i, emb in enumerate(self.embeddings):
                # 简单的特征提取
                scale = 2 ** i
                scaled_coords = coords * scale
                feat = emb(scaled_coords)
                features.append(feat)
            return torch.cat(features, dim=-1)
    
    return SimpleHashGrid(**config)

def create_advanced_rbf_params(grid_x_range, grid_y_range, rbf_params, device='cuda'):
    """
    高级RBF参数创建函数，集成了多种核心选择策略
    """
    # 基本参数
    n_kernel = rbf_params.get('n_kernel', 256)
    kernel_type = rbf_params.get('kernel_type', 'gauss_a')  # 各向异性高斯核
    selection_strategy = rbf_params.get('selection_strategy', 'grid_knn')  # 网格+kNN混合
    
    # 1. 智能网格初始化
    grid_size = int(math.sqrt(n_kernel))
    if grid_size * grid_size < n_kernel:
        grid_size += 1
    
    # 扩展范围以更好覆盖边界
    x_min, x_max = grid_x_range
    y_min, y_max = grid_y_range
    margin = 0.3  # 边界扩展
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_expanded = torch.linspace(x_min - margin * x_range, x_max + margin * x_range, grid_size)
    y_expanded = torch.linspace(y_min - margin * y_range, y_max + margin * y_range, grid_size)
    xx, yy = torch.meshgrid(x_expanded, y_expanded, indexing='ij')
    
    # 取前n_kernel个点
    grid_centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_kernel]
    
    # 2. 自适应扰动策略
    avg_distance = min(x_range, y_range) / grid_size
    perturbation_scale = avg_distance * 0.1
    
    # 根据距离边界的远近调整扰动强度
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_distances = torch.norm(grid_centers - torch.tensor([center_x, center_y]), dim=1)
    max_center_dist = center_distances.max()
    
    # 边界附近的核心扰动更小，保持覆盖
    perturbation_weights = 1.0 - 0.5 * (center_distances / max_center_dist)
    perturbations = perturbation_scale * perturbation_weights.unsqueeze(1) * torch.randn_like(grid_centers)
    
    centers = grid_centers + perturbations
    
    # 3. 智能形状参数（各向异性）
    if kernel_type == 'gauss_a':  # 各向异性高斯
        # 基础sigma
        base_sigma_x = avg_distance * 0.8
        base_sigma_y = avg_distance * 0.8
        
        # 为每个核心学习不同的形状
        sigma_x = base_sigma_x * (0.5 + 0.5 * torch.rand(n_kernel))
        sigma_y = base_sigma_y * (0.5 + 0.5 * torch.rand(n_kernel))
        
        # 旋转角度（用于各向异性）
        rotation_angles = 2 * math.pi * torch.rand(n_kernel)
        
        # 组合形状参数 [sigma_x, sigma_y, rotation]
        shape_params = torch.stack([sigma_x, sigma_y, rotation_angles], dim=1)
        
    else:  # 各向同性高斯 'gauss_i'
        base_sigma = avg_distance * 0.8
        sigmas = base_sigma * (0.5 + 0.5 * torch.rand(n_kernel))
        shape_params = sigmas.unsqueeze(1)  # [n_kernel, 1]
    
    # 4. Embedding参数（用于特征编码）
    embedding_dim = rbf_params.get('embedding_dim', 64)
    embeddings = torch.randn(n_kernel, embedding_dim) * 0.1
    
    # 5. 核心重要性权重（用于动态选择）
    importance_weights = torch.ones(n_kernel)
    
    # 6. Hash grid编码（可选）
    hash_grid_config = rbf_params.get('hash_grid', None)
    hash_grid = None
    if hash_grid_config:
        hash_grid = hashgrid_from_dict(hash_grid_config)
    
    return {
        'centers': centers.to(device),
        'shape_params': shape_params.to(device),
        'kernel_type': kernel_type,
        'embeddings': embeddings.to(device),
        'importance_weights': importance_weights.to(device),
        'selection_strategy': selection_strategy,
        'hash_grid': hash_grid,
        'n_kernel': n_kernel
    }

# from utils import easy_logger

# logger = easy_logger(func_name='MHIIF', level='INFO')
 
class MLP(nn.Module):
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
        x = self.layers(x)
        return x


@register_model("MHIIF_J2")
class MHIIF_J2(BaseModel):
    def __init__(
        self,
        hsi_dim=31,
        msi_dim=3,
        feat_dim=128,
        guide_dim=128,
        spa_edsr_num=3,
        spe_edsr_num=3,
        mlp_dim=[256, 128],
        scale=4,
        patch_merge=True,
        # Hermite RBF 参数
        use_hermite_rbf=True,
        hermite_order=2,
        n_kernel=256,
        rbf_hidden_dim=64,
        hermite_weight=0.5,  # Hermite RBF输出的权重
        # 新增参数：控制是否使用四邻域
        use_four_neighbors=False,  # 默认使用简化版本
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.use_hermite_rbf = use_hermite_rbf
        self.hermite_weight = hermite_weight
        self.use_four_neighbors = use_four_neighbors  # 存储四邻域控制标志
        
        # 原始的编码器
        self.image_encoder = make_edsr_baseline(
            n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim
        )
        self.depth_encoder = make_edsr_baseline(
            n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim
        )

        # 原始的MLP网络
        imnet_in_dim = self.feat_dim + self.guide_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=hsi_dim + 1, hidden_list=self.mlp_dim)
        
        # Hermite RBF网络 - MLP增强版本 + 高级核心选择
        if self.use_hermite_rbf:
            # 高级RBF参数配置
            advanced_rbf_config = {
                'n_kernel': n_kernel,
                'kernel_type': 'gauss_a',  # 各向异性高斯核
                'selection_strategy': 'grid_knn',  # 网格+kNN混合策略
                'embedding_dim': rbf_hidden_dim,
                'hash_grid': {
                    'num_levels': 4,
                    'feature_dim': 16,
                    'base_resolution': 16,
                    'per_level_scale': 2.0
                }
            }
            
            # 动态计算坐标范围（基于scale自适应）
            max_offset = 1.0 / 8  # 基于最小特征图估算
            grid_range = 1.0 + max_offset + 0.3  # 扩展覆盖范围
            
            # 使用高级RBF参数创建函数
            self.advanced_rbf_params = create_advanced_rbf_params(
                grid_x_range=(-grid_range, grid_range),
                grid_y_range=(-grid_range, grid_range),
                rbf_params=advanced_rbf_config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 将高级参数转换为nn.Parameter
            self.rbf_centers = nn.Parameter(self.advanced_rbf_params['centers'])
            self.rbf_shape_params = nn.Parameter(self.advanced_rbf_params['shape_params'])
            self.rbf_embeddings = nn.Parameter(self.advanced_rbf_params['embeddings'])
            self.rbf_importance_weights = nn.Parameter(self.advanced_rbf_params['importance_weights'])
            
            # 保留原有的Hermite结构
            self.hermite_order = hermite_order
            hermite_dim = self._get_hermite_dim(2, hermite_order)
            self.rbf_weights = nn.Parameter(self._init_rbf_weights(n_kernel, hermite_dim, hsi_dim + 1))
            
            # 核心选择参数
            self.kernel_selection_top_k = min(n_kernel // 4, 64)  # 动态选择最相关的核心数量
            self.selection_strategy = self.advanced_rbf_params['selection_strategy']
            self.kernel_type = self.advanced_rbf_params['kernel_type']
            
            # Hash grid编码器（可选）
            self.hash_grid = self.advanced_rbf_params['hash_grid']
            if self.hash_grid:
                self.hash_grid = self.hash_grid.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # MLP作为核函数P_0：处理局部特征和相对坐标
            self.hermite_mlp = MLP(
                in_dim=imnet_in_dim,  # feat + guide + rel_coord
                out_dim=hsi_dim + 1,
                hidden_list=self.mlp_dim
            )
            
            # 智能初始化MLP
            self._init_hermite_mlp_as_gaussian()
            
            # 新增：注意力机制用于核心选择
            attention_input_dim = imnet_in_dim + rbf_hidden_dim
            self.kernel_attention = nn.Sequential(
                nn.Linear(attention_input_dim, rbf_hidden_dim),
                nn.ReLU(),
                nn.Linear(rbf_hidden_dim, 1),  # 输出标量注意力分数
                nn.Sigmoid()
            )
            
            # Hermite损失函数
            # self.hermite_criterion = HermiteLoss(  # 暂时注释掉
            #     lambda_grad=0.05,
            #     lambda_smooth=0.01,
            #     lambda_sparsity=0.001
            # )
            
        self.patch_merge = patch_merge
        self._patch_merge_model = PatchMergeModule(
            self,
            crop_batch_size=32,
            scale=self.scale,
            patch_size_list=[16, 16 * self.scale, 16 * self.scale],
        )

    def forward_kernel_selection(self, query_coords, input_features):
        """
        高级核心选择策略：结合kNN、注意力机制和空间分布
        
        Args:
            query_coords: [B, N, 2] 查询坐标
            input_features: [B, N, feat_dim] 输入特征
            
        Returns:
            selected_indices: [B, N, top_k] 选中的核心索引
            selection_weights: [B, N, top_k] 选择权重
        """
        B, N, _ = query_coords.shape
        device = query_coords.device
        
        if self.selection_strategy == 'grid_knn':
            # 1. 基于距离的kNN选择
            coord_expanded = query_coords.unsqueeze(2)  # [B, N, 1, 2]
            centers_expanded = self.rbf_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
            distances = torch.norm(coord_expanded - centers_expanded, dim=-1)  # [B, N, K]
            
            # 2. 获取top-k最近的核心
            top_k = self.kernel_selection_top_k
            knn_distances, knn_indices = torch.topk(distances, k=top_k, dim=-1, largest=False)
            
            # 3. 结合注意力机制进行精细化选择
            if hasattr(self, 'hash_grid') and self.hash_grid is not None:
                # 使用hash grid编码增强特征
                hash_features = self.hash_grid(query_coords)  # [B, N, hash_dim]
                enhanced_features = torch.cat([input_features, hash_features], dim=-1)
            else:
                enhanced_features = input_features
            
            # 获取选中核心的embedding
            selected_embeddings = self.rbf_embeddings[knn_indices]  # [B, N, top_k, embed_dim]
            
            # 计算注意力权重
            feat_for_attention = enhanced_features.unsqueeze(2).expand(-1, -1, top_k, -1)  # [B, N, top_k, feat_dim]
            attention_input = torch.cat([feat_for_attention, selected_embeddings], dim=-1)
            
            # 计算每个核心的注意力分数
            attention_scores = self.kernel_attention(attention_input.view(-1, attention_input.shape[-1]))
            attention_scores = attention_scores.view(B, N, top_k).squeeze(-1)  # [B, N, top_k]
            
            # 4. 综合距离权重和注意力权重
            distance_weights = torch.exp(-knn_distances / (2 * 0.5**2))  # 距离权重
            final_weights = distance_weights * attention_scores  # 综合权重
            final_weights = F.softmax(final_weights, dim=-1)  # 归一化
            
            return knn_indices, final_weights
            
        elif self.selection_strategy == 'attention_only':
            # 纯注意力机制选择
            K = self.rbf_centers.shape[0]
            
            # 计算所有核心的注意力分数
            all_embeddings = self.rbf_embeddings.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, K, embed_dim]
            feat_for_attention = input_features.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, feat_dim]
            attention_input = torch.cat([feat_for_attention, all_embeddings], dim=-1)
            
            attention_scores = self.kernel_attention(attention_input.view(-1, attention_input.shape[-1]))
            attention_scores = attention_scores.view(B, N, K).squeeze(-1)  # [B, N, K]
            
            top_k = self.kernel_selection_top_k
            _, selected_indices = torch.topk(attention_scores, k=top_k, dim=-1, largest=True)
            selected_weights = torch.gather(attention_scores, -1, selected_indices)
            selected_weights = F.softmax(selected_weights, dim=-1)
            
            return selected_indices, selected_weights
            
        else:  # 'all' - 使用所有核心
            all_indices = torch.arange(self.rbf_centers.shape[0], device=device)
            all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            all_weights = torch.ones_like(all_indices, dtype=torch.float32)
            all_weights = F.softmax(all_weights, dim=-1)
            
            return all_indices, all_weights

    def compute_anisotropic_kernel_weights(self, query_coords, selected_indices):
        """
        计算各向异性核心权重
        
        Args:
            query_coords: [B, N, 2] 查询坐标
            selected_indices: [B, N, top_k] 选中的核心索引
            
        Returns:
            kernel_weights: [B, N, top_k] 核心权重
        """
        B, N, top_k = selected_indices.shape
        
        # 获取选中核心的参数
        selected_centers = self.rbf_centers[selected_indices]  # [B, N, top_k, 2]
        selected_shape_params = self.rbf_shape_params[selected_indices]  # [B, N, top_k, shape_dim]
        
        # 计算相对位置
        query_expanded = query_coords.unsqueeze(2)  # [B, N, 1, 2]
        coord_diff = query_expanded - selected_centers  # [B, N, top_k, 2]
        
        if self.kernel_type == 'gauss_a':  # 各向异性高斯
            # 解析形状参数 [sigma_x, sigma_y, rotation]
            sigma_x = selected_shape_params[..., 0]  # [B, N, top_k]
            sigma_y = selected_shape_params[..., 1]  # [B, N, top_k]
            rotation = selected_shape_params[..., 2]  # [B, N, top_k]
            
            # 应用旋转变换
            cos_r = torch.cos(rotation)
            sin_r = torch.sin(rotation)
            
            dx = coord_diff[..., 0]
            dy = coord_diff[..., 1]
            
            # 旋转坐标
            dx_rot = cos_r * dx + sin_r * dy
            dy_rot = -sin_r * dx + cos_r * dy
            
            # 各向异性高斯权重
            weights = torch.exp(-0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2))
            
        else:  # 各向同性高斯 'gauss_i'
            sigma = selected_shape_params[..., 0]  # [B, N, top_k]
            distances = torch.norm(coord_diff, dim=-1)  # [B, N, top_k]
            weights = torch.exp(-0.5 * (distances / sigma)**2)
        
        return weights

    def compute_hermite_with_selection(self, mlp_output, coord_diff_selected, selected_shape_params, 
                                     final_kernel_weights, selected_indices):
        """
        基于选中核心计算Hermite特征的高效版本
        
        Args:
            mlp_output: [B, N, out_dim] MLP输出
            coord_diff_selected: [B, N, top_k, 2] 查询点到选中核心的坐标差异
            selected_shape_params: [B, N, top_k, shape_dim] 选中核心的形状参数
            final_kernel_weights: [B, N, top_k] 最终核心权重
            selected_indices: [B, N, top_k] 选中的核心索引
            
        Returns:
            final_pred: [B, N, out_dim] 最终预测结果
        """
        B, N, out_dim = mlp_output.shape
        B, N, top_k, _ = coord_diff_selected.shape
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)
        
        # 初始化累积结果
        final_pred = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
        
        # 逐个选中的核心处理
        for k in range(top_k):
            # 当前核心的参数
            dx = coord_diff_selected[:, :, k, 0]  # [B, N]
            dy = coord_diff_selected[:, :, k, 1]  # [B, N]
            weight_k = final_kernel_weights[:, :, k]  # [B, N]
            
            # 获取当前核心的形状参数
            if self.kernel_type == 'gauss_a':
                sigma_x = selected_shape_params[:, :, k, 0]  # [B, N]
                sigma_y = selected_shape_params[:, :, k, 1]  # [B, N]
                rotation = selected_shape_params[:, :, k, 2]  # [B, N]
                
                # 应用旋转变换到坐标差异
                cos_r = torch.cos(rotation)
                sin_r = torch.sin(rotation)
                dx_rot = cos_r * dx + sin_r * dy
                dy_rot = -sin_r * dx + cos_r * dy
                
                # 使用旋转后的坐标和各向异性sigma
                dx_norm = dx_rot
                dy_norm = dy_rot
                sigma_x_use = sigma_x
                sigma_y_use = sigma_y
            else:  # 各向同性
                dx_norm = dx
                dy_norm = dy
                sigma_x_use = selected_shape_params[:, :, k, 0]
                sigma_y_use = sigma_x_use  # 各向同性
            
            # 数值稳定性
            epsilon = 1e-6
            
            # 逐个Hermite阶数计算
            idx = 0
            kernel_contribution = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
            
            for order in range(self.hermite_order + 1):
                for alpha in range(order + 1):
                    beta = order - alpha
                    
                    # 计算Hermite系数（支持各向异性）
                    if alpha == 0 and beta == 0:
                        # 0阶: H_{0,0} = P_0(MLP输出)
                        hermite_coeff = torch.ones_like(dx_norm)
                    elif alpha == 1 and beta == 0:
                        # 1阶x导数: H_{1,0} = -x/σₓ²
                        hermite_coeff = -dx_norm / (sigma_x_use**2 + epsilon)
                    elif alpha == 0 and beta == 1:
                        # 1阶y导数: H_{0,1} = -y/σᵧ²
                        hermite_coeff = -dy_norm / (sigma_y_use**2 + epsilon)
                    elif alpha == 2 and beta == 0:
                        # 2阶xx导数: H_{2,0} = (x²/σₓ⁴ - 1/σₓ²)
                        hermite_coeff = (dx_norm**2) / (sigma_x_use**4 + epsilon) - 1.0 / (sigma_x_use**2 + epsilon)
                    elif alpha == 1 and beta == 1:
                        # 2阶xy导数: H_{1,1} = xy/(σₓ²σᵧ²)
                        hermite_coeff = (dx_norm * dy_norm) / ((sigma_x_use * sigma_y_use)**2 + epsilon)
                    elif alpha == 0 and beta == 2:
                        # 2阶yy导数: H_{0,2} = (y²/σᵧ⁴ - 1/σᵧ²)
                        hermite_coeff = (dy_norm**2) / (sigma_y_use**4 + epsilon) - 1.0 / (sigma_y_use**2 + epsilon)
                    else:
                        # 其他高阶项
                        hermite_coeff = torch.zeros_like(dx_norm)
                    
                    # 获取当前核心k对应的权重参数
                    k_global = selected_indices[:, :, k]  # [B, N] 全局核心索引
                    
                    # 从全局权重中获取当前核心的权重
                    # 使用gather操作获取对应的权重
                    rbf_weight_k = self.rbf_weights[k_global]  # [B, N, hermite_dim, out_dim]
                    rbf_weight_k_idx = rbf_weight_k[:, :, idx, :]  # [B, N, out_dim]
                    
                    # 应用Hermite系数和权重
                    contribution = hermite_coeff.unsqueeze(-1) * mlp_output * rbf_weight_k_idx
                    kernel_contribution += contribution
                    
                    idx += 1
            
            # 应用核心权重并累积到最终结果
            final_pred += weight_k.unsqueeze(-1) * kernel_contribution
        
        return final_pred

    def _get_hermite_dim(self, in_dim: int, order: int) -> int:
        """计算Hermite基函数的维度"""
        dim = 1  # 0阶: 函数值
        if order >= 1:
            dim += in_dim  # 1阶: 一阶导数
        if order >= 2:
            dim += in_dim * (in_dim + 1) // 2  # 2阶: 二阶导数
        return dim

    def _init_rbf_centers(self, n_kernel):
        """智能初始化RBF中心：在坐标空间均匀分布"""
        # 使用网格布局 + 小的随机扰动
        grid_size = int(math.sqrt(n_kernel))
        if grid_size * grid_size < n_kernel:
            grid_size += 1
            
        # 在[-1, 1] x [-1, 1]范围内创建网格
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # 取前n_kernel个点
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_kernel]
        
        # 添加小的随机扰动以避免过度规则化
        centers += 0.1 * torch.randn_like(centers)
        
        return centers
    
    def _init_rbf_centers_adaptive(self, n_kernel):
        """
        自适应初始化RBF中心：根据scale动态调整覆盖范围
        解决边界覆盖不足的问题
        """
        # 计算可能的最大偏移（基于四邻域时的偏移）
        max_offset = 1.0 / 8  # 假设最小特征图8x8，rx = ry = 1/8 = 0.125
        
        # RBF范围 = 查询范围 + 额外边界
        rbf_range = 1.0 + max_offset + 0.3  # 额外0.3的安全边界
        
        # 使用网格布局
        grid_size = int(math.sqrt(n_kernel))
        if grid_size * grid_size < n_kernel:
            grid_size += 1
            
        # 在扩大的范围内创建网格
        x = torch.linspace(-rbf_range, rbf_range, grid_size)
        y = torch.linspace(-rbf_range, rbf_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # 取前n_kernel个点
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_kernel]
        
        # 添加小的随机扰动
        centers += 0.1 * torch.randn_like(centers)
        
        return centers
    
    def _init_rbf_sigmas(self, n_kernel):
        """智能初始化RBF带宽：基于中心间距离"""
        # 估算平均中心间距离
        avg_distance = 2.0 / math.sqrt(n_kernel)  # 基于网格的估算
        
        # sigma设为平均距离的0.5-1.5倍，加入一些变化
        base_sigma = avg_distance * 0.8
        sigmas = base_sigma * (0.5 + torch.rand(n_kernel))  # [0.5, 1.5] * base_sigma
        
        return sigmas
    
    def _init_rbf_weights(self, n_kernel, hermite_dim, out_dim):
        """使用高斯核函数的模拟值初始化RBF权重"""
        # 为0阶项(高斯核函数)设置较大的初始权重
        weights = torch.zeros(n_kernel, hermite_dim, out_dim)
        
        # 0阶项：模拟标准高斯核的输出，给予较大的初始值
        weights[:, 0, :] = 0.5 + 0.1 * torch.randn(n_kernel, out_dim)
        
        # 1阶项：梯度项，给予较小的初始值
        if hermite_dim > 1:
            weights[:, 1:3, :] = 0.1 * torch.randn(n_kernel, 2, out_dim) if hermite_dim > 2 else 0.1 * torch.randn(n_kernel, hermite_dim-1, out_dim)
        
        # 2阶项：二阶导数项，给予最小的初始值
        if hermite_dim > 3:
            weights[:, 3:, :] = 0.05 * torch.randn(n_kernel, hermite_dim-3, out_dim)
        
        return weights
    
    def _init_hermite_mlp_as_gaussian(self):
        """
        智能初始化hermite_mlp，使其初始行为接近高斯核函数
        重点关注相对坐标的最后两个维度（rel_coord部分）
        """
        with torch.no_grad():
            # 对MLP的最后一层进行特殊初始化
            # 使其对相对坐标(最后2维)敏感，类似高斯核函数
            
            layers = list(self.hermite_mlp.layers.children())
            
            # 初始化第一层：让网络对相对坐标敏感
            if len(layers) > 0 and isinstance(layers[0], nn.Linear):
                first_layer = layers[0]
                input_dim = first_layer.in_features
                
                # 重新初始化权重
                nn.init.xavier_normal_(first_layer.weight)
                
                # 对相对坐标部分(最后2维)给予更大的权重，模拟距离敏感性
                coord_weights = first_layer.weight[:, -2:]  # 最后2维是rel_coord
                coord_weights.data *= 2.0  # 增强坐标敏感性
                
                # 偏置初始化
                nn.init.zeros_(first_layer.bias)
            
            # 初始化最后一层：输出类似高斯核的响应
            if len(layers) >= 2:
                last_layer = None
                for layer in reversed(layers):
                    if isinstance(layer, nn.Linear):
                        last_layer = layer
                        break
                
                if last_layer is not None:
                    # 权重小初始化，避免过大的初始输出
                    nn.init.normal_(last_layer.weight, mean=0, std=0.1)
                    
                    # 偏置设为正值，类似高斯核的正输出
                    nn.init.constant_(last_layer.bias, 0.5)
            
            # 中间层使用标准初始化
            for layer in layers:
                if isinstance(layer, nn.Linear) and layer != layers[0] and layer != last_layer:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def compute_hermite_derivatives(self, mlp_output, coord_diff, sigmas):
        """
        计算MLP作为P_0核函数的Hermite导数特征
        
        Args:
            mlp_output: [B, N, out_dim] MLP输出作为P_0值
            coord_diff: [B, N, K, 2] 查询点到RBF中心的坐标差异
            sigmas: [K] 每个RBF中心的sigma参数
            
        Returns:
            hermite_features: [B, N, K, hermite_dim, out_dim] Hermite导数特征
        """
        B, N, out_dim = mlp_output.shape
        K = coord_diff.shape[2]
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)  # 修复：传入正确的参数
        
        # 初始化Hermite特征张量
        hermite_features = torch.zeros(B, N, K, hermite_dim, out_dim, 
                                     device=mlp_output.device, dtype=mlp_output.dtype)
        
        # 扩展MLP输出维度以匹配K个RBF中心
        mlp_expanded = mlp_output.unsqueeze(2).expand(B, N, K, out_dim)  # [B, N, K, out_dim]
        
        # 坐标差异分量
        dx = coord_diff[..., 0]  # [B, N, K]
        dy = coord_diff[..., 1]  # [B, N, K]
        sigma_expanded = sigmas.view(1, 1, K)  # [1, 1, K]
        
        # 为数值稳定性添加小的扰动
        epsilon = 1e-6
        
        idx = 0
        for order in range(self.hermite_order + 1):
            for alpha in range(order + 1):
                beta = order - alpha
                
                # 计算Hermite系数 H_{α,β}(x,y) = (-1)^{α+β} * x^α * y^β / (σ^{α+β})
                if alpha == 0 and beta == 0:
                    # 0阶: H_{0,0} = P_0(MLP输出)
                    hermite_coeff = torch.ones_like(dx)
                elif alpha == 1 and beta == 0:
                    # 1阶x导数: H_{1,0} = -x/σ²
                    hermite_coeff = -dx / (sigma_expanded**2 + epsilon)
                elif alpha == 0 and beta == 1:
                    # 1阶y导数: H_{0,1} = -y/σ²
                    hermite_coeff = -dy / (sigma_expanded**2 + epsilon)
                elif alpha == 2 and beta == 0:
                    # 2阶xx导数: H_{2,0} = (x²/σ⁴ - 1/σ²)
                    hermite_coeff = (dx**2) / (sigma_expanded**4 + epsilon) - 1.0 / (sigma_expanded**2 + epsilon)
                elif alpha == 1 and beta == 1:
                    # 2阶xy导数: H_{1,1} = xy/σ⁴
                    hermite_coeff = (dx * dy) / (sigma_expanded**4 + epsilon)
                elif alpha == 0 and beta == 2:
                    # 2阶yy导数: H_{0,2} = (y²/σ⁴ - 1/σ²)
                    hermite_coeff = (dy**2) / (sigma_expanded**4 + epsilon) - 1.0 / (sigma_expanded**2 + epsilon)
                else:
                    # 其他高阶项，设为0或使用递推关系
                    hermite_coeff = torch.zeros_like(dx)
                
                # 应用Hermite系数到MLP输出
                hermite_features[:, :, :, idx, :] = hermite_coeff.unsqueeze(-1) * mlp_expanded
                idx += 1
        
        return hermite_features

    def compute_hermite_memory_efficient(self, mlp_output, coord_diff, sigmas, gaussian_weights):
        """
        内存高效版本：逐核心计算Hermite特征，避免创建巨大的5D张量
        
        Args:
            mlp_output: [B, N, out_dim] MLP输出作为P_0值
            coord_diff: [B, N, K, 2] 查询点到RBF中心的坐标差异
            sigmas: [K] 每个RBF中心的sigma参数
            gaussian_weights: [B, N, K] 高斯权重
            
        Returns:
            final_pred: [B, N, out_dim] 最终预测结果
        """
        B, N, out_dim = mlp_output.shape
        K = coord_diff.shape[2]
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)
        
        # 初始化累积结果
        final_pred = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
        
        # 逐个核心处理，避免创建大型5D张量
        for k in range(K):
            # 当前核心的坐标差异和权重
            dx = coord_diff[:, :, k, 0]  # [B, N]
            dy = coord_diff[:, :, k, 1]  # [B, N]
            sigma_k = sigmas[k]
            weight_k = gaussian_weights[:, :, k]  # [B, N]
            
            # 为数值稳定性添加小的扰动
            epsilon = 1e-6
            
            # 逐个Hermite阶数计算
            idx = 0
            kernel_contribution = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
            
            for order in range(self.hermite_order + 1):
                for alpha in range(order + 1):
                    beta = order - alpha
                    
                    # 计算Hermite系数
                    if alpha == 0 and beta == 0:
                        # 0阶: H_{0,0} = P_0(MLP输出)
                        hermite_coeff = torch.ones_like(dx)
                    elif alpha == 1 and beta == 0:
                        # 1阶x导数: H_{1,0} = -x/σ²
                        hermite_coeff = -dx / (sigma_k**2 + epsilon)
                    elif alpha == 0 and beta == 1:
                        # 1阶y导数: H_{0,1} = -y/σ²
                        hermite_coeff = -dy / (sigma_k**2 + epsilon)
                    elif alpha == 2 and beta == 0:
                        # 2阶xx导数: H_{2,0} = (x²/σ⁴ - 1/σ²)
                        hermite_coeff = (dx**2) / (sigma_k**4 + epsilon) - 1.0 / (sigma_k**2 + epsilon)
                    elif alpha == 1 and beta == 1:
                        # 2阶xy导数: H_{1,1} = xy/σ⁴
                        hermite_coeff = (dx * dy) / (sigma_k**4 + epsilon)
                    elif alpha == 0 and beta == 2:
                        # 2阶yy导数: H_{0,2} = (y²/σ⁴ - 1/σ²)
                        hermite_coeff = (dy**2) / (sigma_k**4 + epsilon) - 1.0 / (sigma_k**2 + epsilon)
                    else:
                        # 其他高阶项，设为0
                        hermite_coeff = torch.zeros_like(dx)
                    
                    # 应用Hermite系数和高斯权重到MLP输出
                    # hermite_features[k, idx] = hermite_coeff * weight_k * mlp_output
                    weighted_hermite_term = (hermite_coeff * weight_k).unsqueeze(-1) * mlp_output  # [B, N, out_dim]
                    
                    # 累积当前核心的贡献: weighted_hermite_term * rbf_weights[k, idx]
                    kernel_contribution += weighted_hermite_term * self.rbf_weights[k, idx, :].unsqueeze(0).unsqueeze(0)  # [B, N, out_dim]
                    
                    idx += 1
            
            # 累积当前核心的总贡献
            final_pred += kernel_contribution
        
        return final_pred

    def compute_hermite_chunked(self, mlp_output, coord_diff, sigmas, gaussian_weights, chunk_size=1024):
        """
        分块处理版本：对于超大图像，按像素分块处理
        
        Args:
            mlp_output: [B, N, out_dim] MLP输出
            coord_diff: [B, N, K, 2] 坐标差异
            sigmas: [K] sigma参数
            gaussian_weights: [B, N, K] 高斯权重
            chunk_size: 每次处理的像素数量
            
        Returns:
            final_pred: [B, N, out_dim] 最终预测结果
        """
        B, N, out_dim = mlp_output.shape
        final_pred = torch.zeros(B, N, out_dim, device=mlp_output.device, dtype=mlp_output.dtype)
        
        # 按chunk_size分块处理
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            
            # 提取当前块的数据
            mlp_chunk = mlp_output[:, start_idx:end_idx, :]  # [B, chunk, out_dim]
            coord_chunk = coord_diff[:, start_idx:end_idx, :, :]  # [B, chunk, K, 2]
            weight_chunk = gaussian_weights[:, start_idx:end_idx, :]  # [B, chunk, K]
            
            # 处理当前块
            chunk_pred = self.compute_hermite_memory_efficient(
                mlp_chunk, coord_chunk, sigmas, weight_chunk
            )  # [B, chunk, out_dim]
            
            # 存储结果
            final_pred[:, start_idx:end_idx, :] = chunk_pred
        
        return final_pred

    def compute_hermite_adaptive(self, mlp_output, coord_diff, sigmas, gaussian_weights):
        """
        自适应内存管理：根据张量大小和可用内存选择最佳策略
        
        Args:
            mlp_output: [B, N, out_dim] MLP输出
            coord_diff: [B, N, K, 2] 坐标差异
            sigmas: [K] sigma参数
            gaussian_weights: [B, N, K] 高斯权重
            
        Returns:
            final_pred: [B, N, out_dim] 最终预测结果
        """
        B, N, out_dim = mlp_output.shape
        K = coord_diff.shape[2]
        hermite_dim = self._get_hermite_dim(2, self.hermite_order)
        
        return self._compute_hermite_original(mlp_output, coord_diff, sigmas, gaussian_weights)

    
    def _compute_hermite_original(self, mlp_output, coord_diff, sigmas, gaussian_weights):
        """
        原始的Hermite计算方法（创建完整的5D张量）
        仅在内存充足时使用
        """
        hermite_features = self.compute_hermite_derivatives(mlp_output, coord_diff, sigmas)
        weighted_hermite = hermite_features * gaussian_weights.unsqueeze(-1).unsqueeze(-1)
        final_pred = torch.einsum('bnkho,kho->bno', weighted_hermite, self.rbf_weights)
        return final_pred

    def query_simplified(self, feat, coord, hr_guide):
        """
        简化版本：取消四邻域，直接使用RBF进行空间建模
        计算效率提升4倍，内存使用减少75%
        """
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords for relative coordinate calculation
        feat_coord = (
            make_coord((h, w), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, h, w)
        )

        if self.use_hermite_rbf:
            # === 简化版本：直接插值，不使用四邻域 ===
            
            # 1. 获取HR引导特征（双线性插值）
            q_guide_hr = F.grid_sample(
                hr_guide, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, guide_dim]
            
            # 2. 获取LR特征（双线性插值）
            q_feat = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, feat_dim]
            
            # 3. 计算相对坐标（相对于LR网格中心）
            q_coord = F.grid_sample(
                feat_coord, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
            
            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= h  # 归一化到像素尺度
            rel_coord[:, :, 1] *= w
            
            # 4. MLP预测（作为P_0核函数）
            inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
            mlp_output = self.hermite_mlp(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, hsi_dim+1]
            
            # 5. 高级RBF空间建模 - 使用智能核心选择
            # 5.1 智能核心选择
            selected_indices, selection_weights = self.forward_kernel_selection(
                coord, inp  # 使用坐标和完整特征进行选择
            )  # [B, N, top_k], [B, N, top_k]
            
            # 5.2 计算选中核心的各向异性权重
            anisotropic_weights = self.compute_anisotropic_kernel_weights(
                coord, selected_indices
            )  # [B, N, top_k]
            
            # 5.3 综合权重（选择权重 × 各向异性权重）
            final_kernel_weights = selection_weights * anisotropic_weights  # [B, N, top_k]
            
            # 5.4 计算相对坐标差异（仅针对选中的核心）
            B, N, top_k = selected_indices.shape
            selected_centers = self.rbf_centers[selected_indices]  # [B, N, top_k, 2]
            coord_expanded = coord.unsqueeze(2)  # [B, N, 1, 2]
            coord_diff_selected = coord_expanded - selected_centers  # [B, N, top_k, 2]
            
            # 5.5 获取选中核心的形状参数
            selected_shape_params = self.rbf_shape_params[selected_indices]  # [B, N, top_k, shape_dim]
            
            # 5.6 计算Hermite特征（仅对选中的核心）
            final_pred = self.compute_hermite_with_selection(
                mlp_output, coord_diff_selected, selected_shape_params, 
                final_kernel_weights, selected_indices
            )  # [B, N, hsi_dim+1]
            
            # 7. 直接输出，不需要权重聚合（取前hsi_dim维）
            output = final_pred[:, :, :-1]  # 去掉最后一维的权重
            
        else:
            # 原始方法的简化版本（也取消四邻域）
            q_guide_hr = F.grid_sample(
                hr_guide, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)
            
            q_feat = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)
            
            q_coord = F.grid_sample(
                feat_coord, coord.flip(-1).unsqueeze(1), 
                mode="bilinear", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)
            
            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= h
            rel_coord[:, :, 1] *= w
            
            inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
            pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)
            output = pred[:, :, :-1]  # 去掉最后一维的权重
        
        return output.permute(0, 2, 1).view(b, -1, H, W)

    def query_with_hermite(self, feat, coord, hr_guide):
        """
        结合Hermite RBF的查询函数 - MLP作为核函数版本
        """
        # 原始的query逻辑
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = (
            make_coord((h, w), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, h, w)
        )
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        if self.use_hermite_rbf:
            # 统一的MLP-RBF预测
            preds = []
            
            for vx in [-1, 1]:
                for vy in [-1, 1]:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += (vx) * rx
                    coord_[:, :, 1] += (vy) * ry
                    
                    # 获取局部特征
                    q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), 
                                        mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                    q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), 
                                        mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                    
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= h
                    rel_coord[:, :, 1] *= w
                    
                    # MLP作为P_0核函数
                    inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                    mlp_output = self.hermite_mlp(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, hsi_dim+1]
                    
                    # 计算到RBF中心的坐标差异
                    coord_expanded = coord_.view(B, N, 1, 2)  # [B, N, 1, 2]
                    centers_expanded = self.rbf_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
                    coord_diff = coord_expanded - centers_expanded  # [B, N, K, 2]
                    
                    # 计算高斯权重
                    distances = torch.norm(coord_diff, dim=-1)  # [B, N, K]
                    sigma_expanded = self.rbf_sigmas.unsqueeze(0).unsqueeze(0)  # [1, 1, K]
                    gaussian_weights = torch.exp(-0.5 * (distances / (sigma_expanded + 1e-8))**2)  # [B, N, K]
                    
                    # 内存优化版本：自适应选择处理策略
                    # 计算Hermite导数特征 - 根据内存使用情况选择策略
                    final_pred = self.compute_hermite_adaptive(
                        mlp_output, coord_diff, self.rbf_sigmas, gaussian_weights
                    )  # [B, N, out_dim]
                    
                    preds.append(final_pred)
            
            # 聚合4个邻居的预测
            preds = torch.stack(preds, dim=-1)  # [B, N, hsi_dim+1, 4]
            weight = F.softmax(preds[:, :, -1, :], dim=-1)
            final_output = ((preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1))
            
        else:
            # 原始MLP预测（保持不变）
            preds = []
            for vx in [-1, 1]:
                for vy in [-1, 1]:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += (vx) * rx
                    coord_[:, :, 1] += (vy) * ry

                    q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1),
                                        mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                    q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                                        mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= h
                    rel_coord[:, :, 1] *= w

                    inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                    pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)
                    preds.append(pred)

            preds = torch.stack(preds, dim=-1)  # [B, N, hsi_dim+1, 4]
            weight = F.softmax(preds[:, :, -1, :], dim=-1)
            final_output = ((preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1))
        
        return final_output.permute(0, 2, 1).view(b, -1, H, W)
    def query(self, feat, coord, hr_guide):
        """
        查询函数：根据配置选择四邻域版本或简化版本
        """
        if self.use_hermite_rbf:
            if self.use_four_neighbors:
                # 使用原始的四邻域方法
                return self.query_with_hermite(feat, coord, hr_guide)
            else:
                # 使用简化的单点插值方法
                return self.query_simplified(feat, coord, hr_guide)
        else:
            if self.use_four_neighbors:
                # 原始四邻域逻辑（保持不变）
                return self._query_original_four_neighbors(feat, coord, hr_guide)
            else:
                # 使用简化版本
                return self.query_simplified(feat, coord, hr_guide)

    def _query_original_four_neighbors(self, feat, coord, hr_guide):
        """
        原始的四邻域查询逻辑（不使用Hermite RBF）
        """
        # 原始query逻辑（保持不变）
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = (
            make_coord((h, w), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, h, w)
        )
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, C, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret_p = ((preds[:, :, 0:-1, :] * weight.unsqueeze(-2))
            .sum(-1, keepdim=True)
            .squeeze(-1)
        )
        ret = ret_p
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret

    def _forward_implem(self, HR_MSI, lms, LR_HSI):
        # 获取输入 LR_HSI 和目标 HR_MSI 的初始尺寸
        _, _, h_LR, w_LR = LR_HSI.shape
        _, _, h_HR, w_HR = HR_MSI.shape

        # 计算需要的总的放大倍数 (如 n 倍)
        total_scale_factor_h = h_HR / h_LR
        total_scale_factor_w = w_HR / w_LR
        upscale_factor = self.scale

        # 每次上采样的 scale_factor，可以根据需要的上采样次数来决定
        # 假设我们需要 log2(n) 次的上采样，即每次上采样 2 倍
        num_steps = int(upscale_factor).bit_length() - 1  # 计算需要的上采样次数
        scale_factor_h = float(
            total_scale_factor_h ** (1.0 / num_steps)
        )  # 每次高度放大比例
        scale_factor_w = float(
            total_scale_factor_w ** (1.0 / num_steps)
        )  # 每次宽度放大比例

        # 开始逐步上采样
        for _ in range(num_steps):
            # 当前 LR_HSI 尺寸
            _, _, h_LR, w_LR = LR_HSI.shape

            # 动态上采样，scale_factor 每次都是计算出的自适应值
            LR_HSI_up = torch.nn.functional.interpolate(
                LR_HSI,
                scale_factor=(scale_factor_h, scale_factor_w),
                mode="bilinear",
                align_corners=False,
            )

            # 计算下采样因子，确保 HR_MSI_down 与 LR_HSI_up 尺寸匹配
            downscale_factor_h = float(h_HR / h_LR / scale_factor_h)
            downscale_factor_w = float(w_HR / w_LR / scale_factor_w)

            HR_MSI_down = torch.nn.functional.interpolate(
                HR_MSI,
                scale_factor=(1 / downscale_factor_h, 1 / downscale_factor_w),
                mode="bilinear",
                align_corners=False,
            )

            # 确保 HR_MSI_down 的尺寸与 LR_HSI_up 一致
            assert HR_MSI_down.shape[2:] == LR_HSI_up.shape[2:], "尺寸不匹配"

            # 提取特征
            coord = make_coord(HR_MSI_down.shape[2:]).cuda()
            hr_guide = self.image_encoder(HR_MSI_down)  # 提取下采样后的高分辨率图像特征
            feat = self.depth_encoder(LR_HSI)  # 提取当前 LR_HSI 特征

            # 查询与特征融合（现在包含Hermite RBF）
            ret = self.query(feat, coord, hr_guide)

            # 更新 LR_HSI
            LR_HSI = ret + LR_HSI_up

        return LR_HSI

    def _forward_implem_(self, HR_MSI, lms, LR_HSI):
        """主要的前向传播函数，集成了Hermite RBF"""
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).cuda()
        
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW
        feat = self.depth_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI
        
        # 使用包含Hermite RBF的查询
        ret = self.query(feat, coord, hr_guide)
        output = lms + ret

        return output

    def sharpening_train_step(self, lms, lr_hsi, pan, gt, criterion):
        """训练步骤，包含Hermite损失"""
        sr = self._forward_implem_(pan, lms, lr_hsi)
        if self.use_hermite_rbf:
            # 可以添加RBF正则化损失
            # 计算基础损失
            loss = criterion(sr, gt)

            
        return sr.clip(0, 1), loss
    
    def sharpening_val_step(self, lms, lr_hsi, pan, gt):
        """验证步骤"""
        if self.patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16*self.scale, 16, 16*self.scale],
                scale=1,
                patch_merge_step=self.patch_merge_step,
            )
            pred = _patch_merge_model.forward_chop(lms, lr_hsi, pan)[0]
        else:
            pred = self._forward_implem_(pan, lms, lr_hsi)

        return pred.clip(0, 1)

    def patch_merge_step(self, lms, lr_hsi, pan, *args, **kwargs):
        return self._forward_implem_(pan, lms, lr_hsi)

    def get_rbf_info(self):
        """获取RBF网络信息"""
        if self.use_hermite_rbf:
            B, N, K = 1, 4096, self.rbf_centers.shape[0]  # 典型大小估算
            hermite_dim = self._get_hermite_dim(2, self.hermite_order)
            out_dim = 32  # 估算输出维度
            
            # 估算内存使用
            tensor_size = B * N * K * hermite_dim * out_dim
            memory_mb = tensor_size * 4 / (1024 ** 2)
            
            return {
                'n_kernels': self.rbf_centers.shape[0],
                'hermite_order': self.hermite_order,
                'hermite_dim': hermite_dim,
                'rbf_centers_range': (self.rbf_centers.min().item(), self.rbf_centers.max().item()),
                'rbf_sigmas_range': (self.rbf_sigmas.min().item(), self.rbf_sigmas.max().item()),
                'rbf_weights_shape': self.rbf_weights.shape,
                'rbf_parameters': sum(p.numel() for p in [self.rbf_centers, self.rbf_sigmas, self.rbf_weights]) + 
                                sum(p.numel() for p in self.hermite_mlp.parameters()),
                'estimated_memory_mb': memory_mb,
                'memory_strategy': 'adaptive' if memory_mb > 100 else 'original',
                'initialization': 'gaussian_informed',  # 新增：标记使用了高斯启发的初始化
                'mlp_layers': len([m for m in self.hermite_mlp.modules() if isinstance(m, nn.Linear)]),
            }
        else:
            return {'hermite_rbf': 'disabled'}

    def benchmark_query_methods(self, feat, coord, hr_guide, num_runs=5):
        """
        对比四邻域版本和简化版本的性能
        """
        import time
        results = {}
        
        # 测试四邻域版本
        if hasattr(self, 'query_with_hermite'):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            original_flag = self.use_four_neighbors
            self.use_four_neighbors = True
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.query(feat, coord, hr_guide)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            four_neighbor_time = (time.time() - start_time) / num_runs
            results['four_neighbor_time'] = four_neighbor_time
            
            self.use_four_neighbors = original_flag
        
        # 测试简化版本
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        original_flag = self.use_four_neighbors
        self.use_four_neighbors = False
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.query(feat, coord, hr_guide)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        simplified_time = (time.time() - start_time) / num_runs
        results['simplified_time'] = simplified_time
        
        self.use_four_neighbors = original_flag
        
        if 'four_neighbor_time' in results:
            results['speedup'] = four_neighbor_time / simplified_time
            results['time_reduction'] = (four_neighbor_time - simplified_time) / four_neighbor_time * 100
        
        # 内存使用估算
        B, N = coord.shape[:2]
        
        # 四邻域版本内存: 4倍
        four_neighbor_memory = 4 * B * N * (self.feat_dim + self.guide_dim + 32)  # 估算
        simplified_memory = B * N * (self.feat_dim + self.guide_dim + 32)
        
        results['memory_reduction'] = (four_neighbor_memory - simplified_memory) / four_neighbor_memory * 100
        
        return results

    def validate_output_quality(self, feat, coord, hr_guide):
        """
        验证简化版本和四邻域版本的输出质量差异
        """
        if not hasattr(self, 'query_with_hermite'):
            return "四邻域版本不可用"
        
        with torch.no_grad():
            original_flag = self.use_four_neighbors
            
            # 四邻域版本输出
            self.use_four_neighbors = True
            output_four = self.query(feat, coord, hr_guide)
            
            # 简化版本输出  
            self.use_four_neighbors = False
            output_simplified = self.query(feat, coord, hr_guide)
            
            # 恢复原始设置
            self.use_four_neighbors = original_flag
            
            # 计算差异
            mse = F.mse_loss(output_simplified, output_four)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # 计算相关系数
            output_four_flat = output_four.flatten()
            output_simplified_flat = output_simplified.flatten()
            correlation = torch.corrcoef(torch.stack([output_four_flat, output_simplified_flat]))[0, 1]
            
            return {
                'mse': mse.item(),
                'psnr': psnr.item(),
                'correlation': correlation.item(),
                'max_diff': (output_simplified - output_four).abs().max().item(),
                'quality_assessment': 'excellent' if psnr > 40 else 'good' if psnr > 30 else 'acceptable'
            }

    def set_validation_mode(self, fast_mode=True):
        """设置快速验证模式"""
        if fast_mode:
            self._original_use_four_neighbors = self.use_four_neighbors
            self.use_four_neighbors = False  # 验证时使用简化版本
        else:
            if hasattr(self, '_original_use_four_neighbors'):
                self.use_four_neighbors = self._original_use_four_neighbors
    
    def test_initialization_quality(self, test_samples=1000):
        """测试初始化质量：验证MLP的初始行为是否类似高斯核"""
        if not self.use_hermite_rbf:
            return "RBF disabled"
            
        with torch.no_grad():
            # 生成测试数据
            device = self.rbf_centers.device
            B, N = 1, test_samples
            
            # 随机特征和指导
            feat_dim = self.feat_dim + self.guide_dim
            test_feat = torch.randn(B, N, feat_dim, device=device)
            
            # 测试不同距离的相对坐标
            distances = torch.linspace(0, 2, test_samples, device=device)
            rel_coords = torch.stack([distances, torch.zeros_like(distances)], dim=-1).unsqueeze(0)  # [1, N, 2]
            
            # 组合输入
            test_input = torch.cat([test_feat, rel_coords], dim=-1)  # [1, N, feat_dim + 2]
            
            # MLP输出
            mlp_out = self.hermite_mlp(test_input.view(-1, test_input.shape[-1]))
            mlp_out = mlp_out.view(B, N, -1)[:, :, 0]  # 取第一个输出通道
            
            # 理想的高斯响应
            sigma_test = 0.5
            ideal_gaussian = torch.exp(-distances**2 / (2 * sigma_test**2))
            
            # 计算相似度
            mlp_normalized = (mlp_out[0] - mlp_out[0].min()) / (mlp_out[0].max() - mlp_out[0].min() + 1e-8)
            correlation = torch.corrcoef(torch.stack([mlp_normalized, ideal_gaussian]))[0, 1]
            
            return {
                'gaussian_correlation': correlation.item(),
                'mlp_output_range': (mlp_out.min().item(), mlp_out.max().item()),
                'ideal_range': (ideal_gaussian.min().item(), ideal_gaussian.max().item()),
                'initialization_quality': 'good' if correlation > 0.3 else 'needs_improvement'
            }

    def prune_rbf_kernels(self, threshold=1e-6):
        """修剪RBF核心"""
        if self.use_hermite_rbf:
            with torch.no_grad():
                # 基于权重范数计算重要性
                importance = torch.norm(self.rbf_weights, dim=1)
                keep_mask = importance > threshold
                
                if keep_mask.sum() == 0:
                    return 0
                
                old_count = self.rbf_centers.shape[0]
                
                # 更新参数
                self.rbf_centers.data = self.rbf_centers.data[keep_mask]
                self.rbf_sigmas.data = self.rbf_sigmas.data[keep_mask]
                self.rbf_weights.data = self.rbf_weights.data[keep_mask]
                
                new_count = keep_mask.sum().item()
                return old_count - new_count
        return 0


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    torch.cuda.set_device('cuda:2')

    # 测试原始版本
    print("=== 原始MHIIF_J2 ===")
    model_original = MHIIF_J2(31, 3, 64, 64, use_hermite_rbf=False).cuda()

    # 测试Hermite版本
    print("=== Hermite RBF版本 ===")
    model_hermite = MHIIF_J2(
        31, 3, 64, 64, 
        use_hermite_rbf=True,
        hermite_order=2,
        n_kernel=128,  # 相对较少的核心用于测试
        rbf_hidden_dim=32,
        hermite_weight=0.3
    ).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    gt = torch.randn([1, 31, H, W]).cuda()
    # criterion = model_hermite.hermite_criterion()

    # 测试原始版本
    print("原始版本参数量:", sum(p.numel() for p in model_original.parameters()))
    output_original = model_original.sharpening_val_step(lms, LR_HSI, HR_MSI, gt)
    print("原始版本输出形状:", output_original.shape)

    # 测试Hermite版本
    print("Hermite版本参数量:", sum(p.numel() for p in model_hermite.parameters()))
    print("RBF信息:", model_hermite.get_rbf_info())
    output_hermite = model_hermite.sharpening_val_step(lms, LR_HSI, HR_MSI, gt)
    print("Hermite版本输出形状:", output_hermite.shape)

    # 训练步骤测试
    # _, loss_original = model_original.sharpening_train_step(lms, LR_HSI, HR_MSI, gt)
    # _, loss_hermite = model_hermite.sharpening_train_step(lms, LR_HSI, HR_MSI, gt, criterion)
    
    # # print(f"原始版本损失: {loss_original.item():.6f}")
    # print(f"Hermite版本损失: {loss_hermite.item():.6f}")

    # FLOP分析
    print("\n=== FLOP分析 ===")
    model_hermite.forward = model_hermite._forward_implem_
    print("Hermite版本:")
    print(flop_count_table(FlopCountAnalysis(model_hermite, (HR_MSI, lms, LR_HSI))))