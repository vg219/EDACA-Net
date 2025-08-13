import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from edsr import make_edsr_baseline, make_coord
import sys
sys.path.insert(0, '/home/YuJieLiang/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord

from model.base_model import BaseModel, register_model, PatchMergeModule
from pykdtree.kdtree import KDTree
# from model.hermite_rbf import HermiteRBF, HermiteLoss

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



def query_chunked_torch(kd_tree, x, k, chunk_size=int(2e8), device='cuda'):
    """
    与RBF网络保持一致的分块KNN查询
    Args:
        kd_tree: KDTree对象
        x: 查询点 [N, 2] numpy array
        k: 近邻数量
        chunk_size: 分块大小
        device: 目标设备
    Returns:
        kernel_idx: [N, k] torch tensor
    """
    if chunk_size >= x.shape[0]: 
        _, idx = kd_tree.query(x, k=k, sqr_dists=True)
        return torch.tensor(idx.astype(np.int32), device=device)
    
    # 分块处理
    idx = np.zeros([x.shape[0], k], dtype=np.uint32)
    for i in range(0, x.shape[0], chunk_size):
        _, idx[i:i+chunk_size] = kd_tree.query(x[i:i+chunk_size], k=k, sqr_dists=True)
    
    if k == 1:
        idx = idx[:, 0]
    return torch.tensor(idx.astype(np.int32), device=device)

@register_model("MHIIF_J2_Hermite")
class MHIIF_J2_Hermite(BaseModel):
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
        # === RBF风格参数 ===
        n_kernels=2048,
        k_neighbors=4,
        rbf_type='nlin_s',
        ks_alpha=1.0,
        sparse_kernel_grad=False,
        # === KNN相关参数 ===
        knn_chunk_size=int(2e8),    # 与RBF网络一致的分块大小
        use_precomputed_knn=False,  # 是否预计算KNN (训练时使用)
    ):
        super().__init__()
        self.guide_dim = guide_dim
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.hsi_dim = hsi_dim
        self.k_neighbors = k_neighbors
        self.rbf_type = rbf_type
        self.knn_chunk_size = knn_chunk_size
        self.use_precomputed_knn = use_precomputed_knn

        # 编码器
        self.image_encoder = make_edsr_baseline(
            n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim
        )
        self.depth_encoder = make_edsr_baseline(
            n_resblocks=spe_edsr_num, n_feats=feat_dim, n_colors=hsi_dim
        )

        # === RBF风格的可学习核参数 ===
        self.kernel_centers = nn.Embedding(n_kernels, 2, sparse=sparse_kernel_grad)
        self.kernel_features = nn.Embedding(n_kernels, feat_dim, sparse=sparse_kernel_grad)
        
        # 核形状参数
        if rbf_type.endswith('_s'):
            ks_dim = 1
        elif rbf_type.endswith('_d'):
            ks_dim = 2
        elif rbf_type.endswith('_a'):
            ks_dim = 4
        else:
            raise ValueError(f"Unknown rbf_type: {rbf_type}")
            
        self.kernel_scales = nn.Embedding(n_kernels, ks_dim, sparse=sparse_kernel_grad)
        self.ks_dim = ks_dim

        # MLP核函数
        imnet_in_dim = feat_dim + feat_dim + self.guide_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=hsi_dim + 1, hidden_list=self.mlp_dim)
        
        # 初始化
        self._init_rbf_kernels(n_kernels, ks_alpha)
        
        # KDTree缓存
        self._kdtree_cache = None
        self._kernel_centers_cache = None
        
        self.patch_merge = patch_merge
        if self.patch_merge:
            self._patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=32,
                scale=self.scale,
                patch_size_list=[16, 16 * self.scale, 16 * self.scale],
            )

    def _init_rbf_kernels(self, n_kernels, ks_alpha):
        """RBF风格的核参数初始化"""
        with torch.no_grad():
            # 规则网格初始化核心位置
            n_per_dim = int(np.ceil(n_kernels**(1/2)))
            x = torch.linspace(-1, 1, n_per_dim)
            y = torch.linspace(-1, 1, n_per_dim)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            
            init_points = points[:n_kernels] if len(points) >= n_kernels else points
            self.kernel_centers.weight[:len(init_points)] = init_points
            if len(init_points) < n_kernels:
                self.kernel_centers.weight[len(init_points):].uniform_(-1, 1)

            # 核形状初始化
            side_interval = 2.0 / (n_per_dim - 1) if n_per_dim > 1 else 2.0
            init_scale = ks_alpha / side_interval
            
            if self.rbf_type.endswith('_s'):
                self.kernel_scales.weight.fill_(init_scale)
            elif self.rbf_type.endswith('_d'):
                self.kernel_scales.weight.fill_(init_scale)
            elif self.rbf_type.endswith('_a'):
                eye_flat = torch.eye(2).flatten() * init_scale
                self.kernel_scales.weight[:] = eye_flat

            # 核特征初始化
            nn.init.uniform_(self.kernel_features.weight, -0.1, 0.1)

    def _build_kdtree(self):
        """构建KDTree，与RBF网络保持一致"""
        kernel_centers_np = self.kernel_centers.weight.detach().cpu().numpy().astype(np.float32)
        
        # 检查是否需要重建KDTree
        if (self._kdtree_cache is None or 
            self._kernel_centers_cache is None or
            not np.array_equal(self._kernel_centers_cache, kernel_centers_np)):
            
            self._kdtree_cache = KDTree(kernel_centers_np)
            self._kernel_centers_cache = kernel_centers_np.copy()
            
        return self._kdtree_cache

    def _find_k_nearest_rbf_style(self, coord):
        """
        RBF风格的KNN搜索
        Args:
            coord: [B, N, 2] 查询坐标
        Returns:
            topk_indices: [B, N, k] 最近邻索引
        """
        B, N, _ = coord.shape
        
        # 构建KDTree
        kdtree = self._build_kdtree()
        
        # 转换为numpy并reshape为[B*N, 2]
        coord_np = coord.detach().cpu().numpy().astype(np.float32).reshape(B * N, 2)
        
        # === 与RBF网络完全一致的KNN查询 ===
        topk_indices_flat = query_chunked_torch(
            kdtree, coord_np, k=self.k_neighbors, 
            chunk_size=self.knn_chunk_size, device=coord.device
        )
        
        # Reshape回[B, N, k]
        if self.k_neighbors == 1:
            topk_indices = topk_indices_flat.view(B, N, 1)
        else:
            topk_indices = topk_indices_flat.view(B, N, self.k_neighbors)
            
        return topk_indices

    def clip_kernel_params(self):
        """RBF风格的参数约束"""
        with torch.no_grad():
            self.kernel_centers.weight.clamp_(-1.5, 1.5)
            
            if self.rbf_type.endswith('_s') or self.rbf_type.endswith('_d'):
                self.kernel_scales.weight.clamp_(0.01, 10.0)
            elif self.rbf_type.endswith('_a'):
                self.kernel_scales.weight.clamp_(-10.0, 10.0)

    def _apply_kernel_scaling(self, rel_coord, kernel_scales):
        """根据核形状参数标准化相对坐标"""
        if self.rbf_type.endswith('_s'):
            scaled_rel_coord = rel_coord / kernel_scales.unsqueeze(-1)
        elif self.rbf_type.endswith('_d'):
            scaled_rel_coord = rel_coord / kernel_scales
        elif self.rbf_type.endswith('_a'):
            B, N, k, _ = rel_coord.shape
            scale_matrices = kernel_scales.view(B, N, k, 2, 2)
            scaled_rel_coord = torch.einsum('bnkij,bnkj->bnki', scale_matrices, rel_coord)
        else:
            scaled_rel_coord = rel_coord
            
        return scaled_rel_coord

    def update_precomputed_knn(self, coord_list, device='cuda'):
        """
        预计算KNN索引，适用于训练时的固定坐标
        与RBF网络的update_point_kernel_idx保持一致
        Args:
            coord_list: 训练中使用的坐标列表
            device: 目标设备
        """
        if not self.use_precomputed_knn:
            return
            
        print("Precomputing KNN indices...")
        kdtree = self._build_kdtree()
        
        self._precomputed_knn = {}
        for i, coord in enumerate(coord_list):
            B, N, _ = coord.shape
            coord_np = coord.detach().cpu().numpy().astype(np.float32).reshape(B * N, 2)
            
            topk_indices = query_chunked_torch(
                kdtree, coord_np, k=self.k_neighbors,
                chunk_size=self.knn_chunk_size, device=device
            )
            
            self._precomputed_knn[i] = topk_indices.view(B, N, self.k_neighbors)
        print(f"Precomputed KNN for {len(coord_list)} coordinate sets")

    def query_with_hybrid_kernels(self, feat, coord, hr_guide):
        """混合RBF查询，使用RBF风格的KNN"""
        B, N, _ = coord.shape

        # === 步骤1: RBF风格的K近邻搜索 ===
        topk_indices = self._find_k_nearest_rbf_style(coord)

        # === 步骤2-6: 其余步骤保持不变 ===
        q_kernel_feat = self.kernel_features(topk_indices)
        q_kernel_coord = self.kernel_centers(topk_indices)
        q_kernel_scales = self.kernel_scales(topk_indices)

        rel_coord = coord.unsqueeze(-2) - q_kernel_coord
        scaled_rel_coord = self._apply_kernel_scaling(rel_coord, q_kernel_scales)

        q_instance_feat = F.grid_sample(
            feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        q_instance_feat = q_instance_feat.unsqueeze(-2).expand(-1, -1, self.k_neighbors, -1)
        
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        q_guide_hr = q_guide_hr.unsqueeze(-2).expand(-1, -1, self.k_neighbors, -1)

        inp = torch.cat([q_kernel_feat, q_instance_feat, q_guide_hr, scaled_rel_coord], dim=-1)
        preds = self.imnet(inp.view(B * N * self.k_neighbors, -1))
        preds = preds.view(B, N, self.k_neighbors, -1)

        weight = F.softmax(preds[..., -1:], dim=-2)
        ret_values = (preds[..., :-1] * weight).sum(dim=-2)
        
        b, _, H, W = hr_guide.shape
        ret = ret_values.permute(0, 2, 1).view(b, self.hsi_dim, H, W)
        
        return ret
    
    def _forward_implem_(self, HR_MSI, lms, LR_HSI):
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W], flatten=True).to(HR_MSI.device)
        coord = coord.unsqueeze(0).expand(HR_MSI.shape[0], -1, -1)

        hr_guide = self.image_encoder(HR_MSI)
        feat = self.depth_encoder(LR_HSI)
        
        ret = self.query_with_hybrid_kernels(feat, coord, hr_guide)
        
        output = lms + ret
        return output
    
    def sharpening_train_step(self,lms, lr_hsi, pan, gt, criterion):
        # ms = self._construct_ms(lms)
        sr = self._forward_implem_(pan,lms,lr_hsi)
        loss = criterion(sr, gt)
        self.clip_kernel_params()
        # 使KDTree缓存失效（因为核心位置可能已改变）
        self._kdtree_cache = None
        self._kernel_centers_cache = None
        return sr.clip(0, 1), loss
    
    def sharpening_val_step(self,lms, lr_hsi, pan, gt):
        
        if self.patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16*self.scale, 16, 16*self.scale],
                scale=1,
                patch_merge_step=self.patch_merge_step,
            )
            pred = _patch_merge_model.forward_chop(lms,lr_hsi,pan)[0]
        else:
            pred = self._forward_implem_(pan,lms,lr_hsi)

        return pred.clip(0, 1)
    

    def patch_merge_step(self,lms, lr_hsi, pan, *args, **kwargs):
        return self._forward_implem_(pan,lms,lr_hsi)



if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    torch.cuda.set_device('cuda:1')

    # 测试RBF风格的MHIIF_J2_Hermite模型
    print("=== MHIIF_J2_Hermite RBF风格测试 ===")
    
    # 创建模型实例
    model = MHIIF_J2_Hermite(
        hsi_dim=31,
        msi_dim=3,
        feat_dim=128,
        guide_dim=128,
        spa_edsr_num=3,
        spe_edsr_num=3,
        mlp_dim=[256, 128],
        scale=4,
        patch_merge=True,
        # RBF风格参数
        n_kernels=2048,          # 核的数量
        k_neighbors=8,          # 固定的K值
        rbf_type='nlin_a',      # 各向异性核
        ks_alpha=1.0,           # 核形状初始化因子
        sparse_kernel_grad=True # 稀疏梯度优化
    ).cuda()

    # 测试数据
    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    gt = torch.randn([1, 31, H, W]).cuda()
    criterion = torch.nn.L1Loss()

    # 模型信息
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"RBF类型: {model.rbf_type}")
    print(f"核心数量: {model.kernel_centers.num_embeddings}")
    print(f"K邻居数: {model.k_neighbors}")
    print(f"核形状维度: {model.ks_dim}")

    # 前向传播测试
    print("\n=== 前向传播测试 ===")
    with torch.no_grad():
        output = model._forward_implem_(HR_MSI, lms, LR_HSI)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 训练步骤测试
    # print("\n=== 训练步骤测试 ===")
    # model.train()
    
    # # 模拟训练步骤
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # for epoch in range(3):
    #     optimizer.zero_grad()
        
    #     # 前向传播
    #     output = model._forward_implem_(HR_MSI, lms, LR_HSI)
    #     loss = criterion(output, gt)
        
    #     # 反向传播
    #     loss.backward()
        
    #     # 参数约束（RBF风格）
    #     model.clip_kernel_params()
        
    #     optimizer.step()
        
    #     print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    # 核心选择测试
    # print("\n=== 核心选择分析 ===")
    # model.eval()
    # with torch.no_grad():
    #     # 创建查询坐标
    #     coord = make_coord([H, W], flatten=True).cuda()
    #     coord = coord.unsqueeze(0).expand(B, -1, -1)
        
    #     # 计算到所有核心的距离
    #     dists = torch.cdist(coord, model.kernel_centers.weight.unsqueeze(0).expand(B, -1, -1))
    #     min_dists, _ = torch.min(dists, dim=-1)
    #     avg_min_dist = min_dists.mean().item()
        
    #     print(f"平均最小距离: {avg_min_dist:.4f}")
    #     print(f"核心位置范围: [{model.kernel_centers.weight.min().item():.4f}, {model.kernel_centers.weight.max().item():.4f}]")
    #     print(f"核形状参数范围: [{model.kernel_scales.weight.min().item():.4f}, {model.kernel_scales.weight.max().item():.4f}]")

    # FLOP分析
    print("\n=== FLOP分析 ===")
    model.eval()
    try:
        model.forward = model._forward_implem_
        flops = FlopCountAnalysis(model, (HR_MSI, lms, LR_HSI))
        print(f"总FLOPs: {flops.total():,}")
        print("\n详细FLOP表:")
        print(flop_count_table(flops, max_depth=2))
    except Exception as e:
        print(f"FLOP分析出错: {e}")

    print("\n=== 测试完成 ===")
