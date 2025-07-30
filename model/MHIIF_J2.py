import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from edsr import make_edsr_baseline, make_coord
import sys
sys.path.insert(0, '/home/YuJieLiang/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord

from model.base_model import BaseModel, register_model, PatchMergeModule
from model.hermite_rbf import HermiteRBF, HermiteLoss

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
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.use_hermite_rbf = use_hermite_rbf
        self.hermite_weight = hermite_weight
        
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
        
        # Hermite RBF网络 - MLP增强版本
        if self.use_hermite_rbf:
            # RBF核心：智能初始化策略
            self.rbf_centers = nn.Parameter(self._init_rbf_centers(n_kernel))  # 均匀分布的核心位置
            self.rbf_sigmas = nn.Parameter(self._init_rbf_sigmas(n_kernel))    # 自适应的核心带宽
            
            # Hermite阶数和权重
            self.hermite_order = hermite_order
            hermite_dim = self._get_hermite_dim(2, hermite_order)  # 2D坐标的Hermite维度
            self.rbf_weights = nn.Parameter(self._init_rbf_weights(n_kernel, hermite_dim, hsi_dim + 1))
            
            # MLP作为核函数P_0：处理局部特征和相对坐标
            # 这个MLP现在被解释为Hermite基函数的0阶项
            self.hermite_mlp = MLP(
                in_dim=imnet_in_dim,  # feat + guide + rel_coord
                out_dim=hsi_dim + 1,
                hidden_list=self.mlp_dim
            )
            
            # 智能初始化MLP，使其初始行为类似高斯核函数
            self._init_hermite_mlp_as_gaussian()
            
            # Hermite损失函数
            self.hermite_criterion = HermiteLoss(
                lambda_grad=0.05,
                lambda_smooth=0.01,
                lambda_sparsity=0.001
            )
            
        self.patch_merge = patch_merge
        self._patch_merge_model = PatchMergeModule(
            self,
            crop_batch_size=32,
            scale=self.scale,
            patch_size_list=[16, 16 * self.scale, 16 * self.scale],
        )

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
                    
                    # 计算Hermite导数特征
                    hermite_features = self.compute_hermite_derivatives(
                        mlp_output, coord_diff, self.rbf_sigmas
                    )  # [B, N, K, hermite_dim, out_dim]
                    
                    # 应用高斯权重到Hermite特征
                    weighted_hermite = hermite_features * gaussian_weights.unsqueeze(-1).unsqueeze(-1)  # [B, N, K, hermite_dim, out_dim]
                    
                    # 与RBF权重结合: [B, N, K, hermite_dim, out_dim] × [K, hermite_dim, out_dim] -> [B, N, out_dim]
                    final_pred = torch.einsum('bnkho,kho->bno', weighted_hermite, self.rbf_weights)
                    
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
        """保持原始接口，内部调用带Hermite的版本"""
        if self.use_hermite_rbf:
            return self.query_with_hermite(feat, coord, hr_guide)
        else:
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
            
            # RBF正则化：鼓励核心分布均匀，权重稀疏
            if hasattr(self, 'rbf_centers'):
                # 核心分散性损失
                centers_dist = torch.pdist(self.rbf_centers)
                diversity_loss = torch.exp(-centers_dist).mean()
                
                # 权重稀疏性损失
                sparsity_loss = torch.norm(self.rbf_weights, p=1)
                
                # 添加正则化
                loss = loss + 0.01 * diversity_loss + 0.001 * sparsity_loss
        else:
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
            return {
                'n_kernels': self.rbf_centers.shape[0],
                'hermite_order': self.hermite_order,
                'rbf_centers_range': (self.rbf_centers.min().item(), self.rbf_centers.max().item()),
                'rbf_sigmas_range': (self.rbf_sigmas.min().item(), self.rbf_sigmas.max().item()),
                'rbf_weights_shape': self.rbf_weights.shape,
                'rbf_parameters': sum(p.numel() for p in [self.rbf_centers, self.rbf_sigmas, self.rbf_weights]) + 
                                sum(p.numel() for p in self.hermite_mlp.parameters()),
                'initialization': 'gaussian_informed',  # 新增：标记使用了高斯启发的初始化
                'mlp_layers': len([m for m in self.hermite_mlp.modules() if isinstance(m, nn.Linear)]),
            }
        else:
            return {'hermite_rbf': 'disabled'}
    
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