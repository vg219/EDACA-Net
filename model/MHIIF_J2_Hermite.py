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
        
        # Hermite RBF网络
        if self.use_hermite_rbf:
            self.hermite_rbf = HermiteRBF(
                cmin=-1.0, cmax=1.0,
                in_dim=2,  # 2D坐标
                out_dim=rbf_hidden_dim,
                n_kernel=n_kernel,
                hermite_order=hermite_order,
                adaptive_sigma=True,
                use_bias=True
            )
            
            # RBF特征融合网络
            self.rbf_fusion = nn.Sequential(
                nn.Linear(rbf_hidden_dim + self.feat_dim + self.guide_dim, 
                         (rbf_hidden_dim + self.feat_dim + self.guide_dim) // 2),
                nn.ReLU(inplace=True),
                nn.Linear((rbf_hidden_dim + self.feat_dim + self.guide_dim) // 2, hsi_dim),
            )
            
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

    def query_with_hermite(self, feat, coord, hr_guide):
        """
        结合Hermite RBF的查询函数
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

        # 原始MLP预测
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

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
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, hsi_dim+1]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, hsi_dim+1, 4]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        mlp_output = ((preds[:, :, 0:-1, :] * weight.unsqueeze(-2))
            .sum(-1, keepdim=True)
            .squeeze(-1)
        )  # [B, N, hsi_dim]
        
        # Hermite RBF预测
        if self.use_hermite_rbf:
            # 归一化坐标到[-1,1]
            norm_coord = coord.clone()  # [B, N, 2]
            norm_coord = norm_coord.view(-1, 2)  # [B*N, 2]
            
            # Hermite RBF推理
            rbf_feat = self.hermite_rbf(norm_coord)  # [B*N, rbf_hidden_dim]
            rbf_feat = rbf_feat.view(B, N, -1)  # [B, N, rbf_hidden_dim]
            
            # 特征融合：RBF特征 + 图像特征 + 引导特征
            # 平均池化获取全局特征作为上下文
            global_feat = F.adaptive_avg_pool2d(feat, 1).view(b, -1, 1).expand(b, -1, N).permute(0, 2, 1)  # [B, N, feat_dim]
            global_guide = F.adaptive_avg_pool2d(hr_guide, 1).view(b, -1, 1).expand(b, -1, N).permute(0, 2, 1)  # [B, N, guide_dim]
            
            # 融合所有特征
            combined_feat = torch.cat([rbf_feat, global_feat, global_guide], dim=-1)  # [B, N, rbf_hidden_dim + feat_dim + guide_dim]
            rbf_output = self.rbf_fusion(combined_feat)  # [B, N, hsi_dim]
            
            # 加权融合MLP和RBF输出
            final_output = (1 - self.hermite_weight) * mlp_output + self.hermite_weight * rbf_output
        else:
            final_output = mlp_output
        
        # 重塑输出
        ret = final_output.permute(0, 2, 1).view(b, -1, H, W)
        return ret

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
        
        # 基础损失
        base_loss = criterion(sr, gt)
        
        # 如果使用Hermite RBF，添加Hermite损失
        if self.use_hermite_rbf and hasattr(self, 'hermite_criterion'):
            hermite_loss, _ = self.hermite_criterion(sr, gt, self.hermite_rbf)
            total_loss = base_loss + 0.1 * hermite_loss  # 权重可调
        else:
            total_loss = base_loss
            
        return sr.clip(0, 1), total_loss
    
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
                'n_kernels': self.hermite_rbf.n_kernel,
                'hermite_order': self.hermite_rbf.hermite_order,
                'kernel_importance': self.hermite_rbf.get_kernel_importance(),
                'rbf_parameters': sum(p.numel() for p in self.hermite_rbf.parameters()),
                'hermite_weight': self.hermite_weight
            }
        else:
            return {'hermite_rbf': 'disabled'}

    def prune_rbf_kernels(self, threshold=1e-6):
        """修剪RBF核心"""
        if self.use_hermite_rbf:
            return self.hermite_rbf.prune_kernels(threshold)
        return 0


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    torch.cuda.set_device('cuda:1')

    # 测试原始版本
    print("=== 原始MHIIF_J2 ===")
    model_original = MHIIF_J2_Hermite(31, 3, 64, 64, use_hermite_rbf=False).cuda()

    # 测试Hermite版本
    print("=== Hermite RBF版本 ===")
    model_hermite = MHIIF_J2_Hermite(
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
    criterion = torch.nn.L1Loss()

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
    _, loss_original = model_original.sharpening_train_step(lms, LR_HSI, HR_MSI, gt, criterion)
    _, loss_hermite = model_hermite.sharpening_train_step(lms, LR_HSI, HR_MSI, gt, criterion)
    
    print(f"原始版本损失: {loss_original.item():.6f}")
    print(f"Hermite版本损失: {loss_hermite.item():.6f}")

    # FLOP分析
    print("\n=== FLOP分析 ===")
    model_hermite.forward = model_hermite._forward_implem_
    print("Hermite版本:")
    print(flop_count_table(FlopCountAnalysis(model_hermite, (HR_MSI, lms, LR_HSI))))
