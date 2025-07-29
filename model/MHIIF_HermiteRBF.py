"""
Hermite RBF集成到MHIIF框架的适配器
Integration of Hermite RBF into MHIIF framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .hermite_rbf import HermiteRBF, HermiteLoss
from model.base_model import BaseModel, register_model, PatchMergeModule

@register_model("MHIIF_HermiteRBF")
class MHIIF_HermiteRBF(BaseModel):
    """
    MHIIF with Hermite RBF Network
    将Hermite RBF集成到MHIIF架构中用于高光谱图像超分辨率
    """
    
    def __init__(self, 
                 hsi_channels: int = 31,
                 msi_channels: int = 3,
                 scale_factor: int = 4,
                 hermite_order: int = 2,
                 n_kernel: int = 512,
                 use_hash_encoding: bool = True,
                 hash_levels: int = 16,
                 hash_features: int = 2,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 patch_merge=True,
                 **kwargs):
        super().__init__()
        
        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels
        self.scale_factor = scale_factor
        self.hermite_order = hermite_order
        self.patch_merge = patch_merge
        
        # 特征提取网络
        self.hsi_encoder = self._build_encoder(hsi_channels, hidden_dim // 2)
        self.msi_encoder = self._build_encoder(msi_channels, hidden_dim // 2)
        
        # 坐标编码（简化版本 - 使用位置编码）
        if use_hash_encoding:
            # 简单的位置编码替代Hash编码
            self.coord_encoder = self._build_positional_encoding(2, hash_levels * hash_features)
            coord_dim = hash_levels * hash_features
        else:
            self.coord_encoder = None
            coord_dim = 2
        
        # Hermite RBF网络
        self.hermite_rbf = HermiteRBF(
            cmin=-1.0, cmax=1.0,
            in_dim=coord_dim,
            out_dim=hidden_dim,
            n_kernel=n_kernel,
            hermite_order=hermite_order,
            adaptive_sigma=True,
            use_bias=True
        )
        
        # 融合网络
        self.fusion_net = self._build_fusion_net(hidden_dim, hsi_channels)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hsi_channels)
        
    def _build_encoder(self, in_channels: int, out_channels: int) -> nn.Module:
        """构建编码器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
    
    def _build_fusion_net(self, in_dim: int, out_dim: int) -> nn.Module:
        """构建融合网络"""
        return nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def _build_positional_encoding(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建位置编码网络（替代Hash编码）"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def get_coordinates(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """生成归一化坐标"""
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        return coords.view(-1, 2)  # [H*W, 2]
    
    def _forward_implem(self, 
                hsi_lr: torch.Tensor, 
                msi_hr: torch.Tensor,
                target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            hsi_lr: [B, C_hsi, H, W] 低分辨率HSI
            msi_hr: [B, C_msi, H*scale, W*scale] 高分辨率MSI
            target_size: 目标输出尺寸
        Returns:
            hsi_hr: [B, C_hsi, H*scale, W*scale] 高分辨率HSI
        """
        B, C_hsi, H, W = hsi_lr.shape
        _, C_msi, H_hr, W_hr = msi_hr.shape
        
        if target_size is None:
            target_size = (H_hr, W_hr)
        
        # 1. 特征提取
        hsi_feat = self.hsi_encoder(hsi_lr)  # [B, hidden_dim//2, H, W]
        msi_feat = self.msi_encoder(msi_hr)  # [B, hidden_dim//2, H_hr, W_hr]
        
        # 2. 上采样HSI特征到目标分辨率
        hsi_feat_up = F.interpolate(hsi_feat, size=target_size, 
                                   mode='bilinear', align_corners=False)
        
        # 3. 生成坐标
        coords = self.get_coordinates(target_size[0], target_size[1], 
                                    hsi_lr.device)  # [H_hr*W_hr, 2]
        
        # 4. 坐标编码
        if self.coord_encoder is not None:
            coord_feat = self.coord_encoder(coords)  # [H_hr*W_hr, encoded_dim]
        else:
            coord_feat = coords  # [H_hr*W_hr, 2]
        
        # 5. Hermite RBF推理
        rbf_feat = self.hermite_rbf(coord_feat)  # [H_hr*W_hr, hidden_dim]
        rbf_feat = rbf_feat.view(target_size[0], target_size[1], -1)  # [H_hr, W_hr, hidden_dim]
        rbf_feat = rbf_feat.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, hidden_dim, H_hr, W_hr]
        
        # 6. 特征融合
        fused_feat = torch.cat([hsi_feat_up, msi_feat], dim=1)  # [B, hidden_dim, H_hr, W_hr]
        
        # 7. 与RBF特征结合
        combined_feat = fused_feat + rbf_feat
        
        # 8. 最终输出
        output = self.fusion_net(combined_feat.permute(0, 2, 3, 1))  # [B, H_hr, W_hr, C_hsi]
        output = output.permute(0, 3, 1, 2)  # [B, C_hsi, H_hr, W_hr]
        
        return output
    
    # def get_rbf_info(self) -> Dict[str, Any]:
    #     """获取RBF网络信息"""
    #     return {
    #         'n_kernels': self.hermite_rbf.n_kernel,
    #         'hermite_order': self.hermite_rbf.hermite_order,
    #         'kernel_importance': self.hermite_rbf.get_kernel_importance(),
    #         'parameters': sum(p.numel() for p in self.hermite_rbf.parameters())
    #     }

    def sharpening_train_step(self,lms, lr_hsi, pan, gt, criterion):
        # ms = self._construct_ms(lms)
        sr = self._forward_implem(pan,lms,lr_hsi)
        loss = criterion(sr, gt)
        return sr.clip(0, 1), loss
    
    
    def sharpening_val_step(self,lms, lr_hsi, pan, gt):

        # gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
        #                         data['ms'].cuda(), data['pan'].cuda()
        # sr1 = self(ms, lms, pan)
        # with torch.no_grad():
        #     metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
        #     metrics.update(metrics)
        
        if self.patch_merge:
            # logger.debug(f"using patch merge module")
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16*self.scale, 16, 16*self.scale],
                scale=1,
                patch_merge_step=self.patch_merge_step,
            )
            pred = _patch_merge_model.forward_chop(lms,lr_hsi,pan)[0]
        else:
            pred = self._forward_implem(pan,lms,lr_hsi)
            # pred = self._forward_implem(ms, pan)


        return pred.clip(0, 1)
    
    # def set_metrics(self, criterion, rgb_range=1.0):
    #     self.rgb_range = rgb_range
    #     self.criterion = criterion

    def patch_merge_step(self,lms, lr_hsi, pan, *args, **kwargs):
        return self._forward_implem(pan,lms,lr_hsi)
        # return self._forward_implem(ms, pan)

class HermiteTrainer:
    """
    Hermite RBF训练器
    支持渐进式训练和自适应核心调整
    """
    
    def __init__(self, 
                 model: MHIIF_HermiteRBF,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 max_hermite_order: int = 2,
                 progressive_training: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_hermite_order = max_hermite_order
        self.progressive_training = progressive_training
        
        # 损失函数
        self.criterion = HermiteLoss(
            lambda_grad=0.1,
            lambda_smooth=0.01,
            lambda_sparsity=0.001
        )
        
        # 训练状态
        self.current_order = 0 if progressive_training else max_hermite_order
        self.order_switch_epochs = [50, 100]  # 切换Hermite阶数的epoch
        
    def set_hermite_order(self, order: int):
        """设置当前Hermite阶数"""
        self.current_order = min(order, self.max_hermite_order)
        self.model.hermite_rbf.hermite_order = self.current_order
        print(f"切换到Hermite阶数: {self.current_order}")
    
    def train_step(self, 
                   hsi_lr: torch.Tensor, 
                   msi_hr: torch.Tensor, 
                   hsi_hr_gt: torch.Tensor,
                   epoch: int) -> Dict[str, float]:
        """
        训练步骤
        """
        self.model.train()
        
        # 渐进式训练：动态调整Hermite阶数
        if self.progressive_training:
            if epoch in self.order_switch_epochs:
                new_order = self.order_switch_epochs.index(epoch) + 1
                self.set_hermite_order(new_order)
        
        # 前向传播
        pred = self.model(hsi_lr, msi_hr)
        
        # 计算损失
        total_loss, loss_dict = self.criterion(pred, hsi_hr_gt, self.model.hermite_rbf)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 转换为标量
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in loss_dict.items()}
        
        return loss_dict
    
    def evaluate_step(self, 
                     hsi_lr: torch.Tensor, 
                     msi_hr: torch.Tensor, 
                     hsi_hr_gt: torch.Tensor) -> Dict[str, float]:
        """
        评估步骤
        """
        self.model.eval()
        
        with torch.no_grad():
            pred = self.model(hsi_lr, msi_hr)
            
            # 计算评估指标
            psnr = self._calculate_psnr(pred, hsi_hr_gt)
            ssim = self._calculate_ssim(pred, hsi_hr_gt)
            sam = self._calculate_sam(pred, hsi_hr_gt)
            
            # 计算损失
            _, loss_dict = self.criterion(pred, hsi_hr_gt, self.model.hermite_rbf)
            loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in loss_dict.items()}
            
            metrics = {
                'PSNR': psnr,
                'SSIM': ssim,
                'SAM': sam,
                **loss_dict
            }
            
            return metrics
    
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算SSIM（简化版本）"""
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim.item()
    
    def _calculate_sam(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算SAM（光谱角度映射）"""
        pred_norm = F.normalize(pred.view(pred.shape[0], pred.shape[1], -1), dim=1)
        target_norm = F.normalize(target.view(target.shape[0], target.shape[1], -1), dim=1)
        
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        sam = torch.acos(cos_sim).mean()
        
        return torch.rad2deg(sam).item()
    
    def prune_kernels(self, threshold: float = 1e-6) -> int:
        """修剪不重要的RBF核心"""
        return self.model.hermite_rbf.prune_kernels(threshold)


def create_hermite_mhiif_system(hsi_channels: int = 31,
                               scale_factor: int = 4,
                               hermite_order: int = 2,
                               n_kernel: int = 512,
                               learning_rate: float = 2e-4) -> Dict[str, Any]:
    """
    创建完整的Hermite MHIIF系统
    """
    # 模型
    model = MHIIF_HermiteRBF(
        hsi_channels=hsi_channels,
        scale_factor=scale_factor,
        hermite_order=hermite_order,
        n_kernel=n_kernel,
        use_hash_encoding=False  # 默认不使用编码
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500, eta_min=1e-6
    )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'trainer_class': HermiteTrainer
    }


if __name__ == "__main__":
    # 测试集成系统
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建系统
    system = create_hermite_mhiif_system(hsi_channels=31, hermite_order=2)
    model = system['model'].to(device)
    optimizer = system['optimizer']
    
    # 创建训练器
    trainer = HermiteTrainer(model, optimizer, device, max_hermite_order=2)
    
    # 测试数据
    B, H, W = 4, 16, 16
    hsi_lr = torch.rand(B, 31, H, W, device=device)
    msi_hr = torch.rand(B, 3, H*4, W*4, device=device)
    hsi_hr_gt = torch.rand(B, 31, H*4, W*4, device=device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"RBF信息: {model.get_rbf_info()}")
    
    # 训练步骤测试
    loss_dict = trainer.train_step(hsi_lr, msi_hr, hsi_hr_gt, epoch=0)
    print(f"训练损失: {loss_dict}")
    
    # 评估步骤测试
    metrics = trainer.evaluate_step(hsi_lr, msi_hr, hsi_hr_gt)
    print(f"评估指标: {metrics}")
    
    print("Hermite MHIIF集成系统测试完成！")
