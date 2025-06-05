import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from edsr import make_edsr_baseline, make_coord
import sys
sys.path.insert(0, '/home/YuJieLiang/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord

from model.base_model import BaseModel, register_model, PatchMergeModule

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
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.image_encoder = make_edsr_baseline(
            n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim
        )
        self.depth_encoder = make_edsr_baseline(
            n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim
        )

        imnet_in_dim = self.feat_dim + self.guide_dim + 2
        imnet_grad_in_dim = 2 * hsi_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=hsi_dim + 1, hidden_list=self.mlp_dim)
        self.imnet_grad = MLP(
            imnet_grad_in_dim, out_dim=hsi_dim + 1, hidden_list=self.mlp_dim
        )
        # I know that. One of the values is depth, and another is the weight.
        self.patch_merge = patch_merge
        self._patch_merge_model = PatchMergeModule(
            self,
            crop_batch_size=32,
            scale=self.scale,
            patch_size_list=[16, 16 * self.scale, 16 * self.scale],
        )


    def query_v2(self, feat, coord, hr_guide):
        # ... (前面的代码不变) ...
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord_expanded = coord.expand(b, H * W, 2) # 使用新名称以避免与循环中的 coord_ 混淆
        B, N, _ = coord_expanded.shape

        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord_expanded.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        rx = 1 / h
        ry = 1 / w

        preds_val_list = [] # 存储 self.imnet 的原始输出
        preds_deriv_list = [] # 存储 self.imnet_grad 处理导数后的输出

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord_expanded.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry

                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord_sampled = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                rel_coord = coord_expanded - q_coord_sampled # 注意：这里是 coord_expanded (中心查询点) - q_coord_sampled (采样点坐标)
                                                        # 通常 rel_coord 是相对于采样点邻域中心的坐标
                                                        # 您之前的 rel_coord = coord - q_coord 可能是对的，取决于 q_coord 的定义
                                                        # 假设这里的 coord_expanded 是我们关心的“查询点”，而 q_coord_sampled 是“支持点”
                                                        # 我们需要一个“输入坐标”来对 pred 求导。
                                                        # 如果 rel_coord 是用于 imnet 的输入坐标，那它应该是求导的目标。
                                                        # 让我们假设 rel_coord 就是那个应该求导的相对坐标输入。
                
                # 为了清晰，我们重构 rel_coord，使其是 MLP 关心的那个坐标输入
                # 如果 MLP 输入的是绝对坐标，那么 coord_ 就是求导目标
                # 如果 MLP 输入的是相对坐标，那么这个 rel_coord (经过处理的) 是求导目标
                # 假设您的 `rel_coord` (原 inp 的一部分) 是要求导的坐标部分
                # 我们需要确保这个 rel_coord 被正确设置并传递

                current_rel_coord = coord_expanded - q_coord_sampled # 重新计算，作为示例
                current_rel_coord[:, :, 0] *= h # h from LR space
                current_rel_coord[:, :, 1] *= w # w from LR space
                with torch.enable_grad():  # must enable grad to compute gradients
                    # ---- 开始 Hermite 计算 ----
                    q_feat_flat = q_feat.reshape(B * N, -1)
                    q_guide_hr_flat_loop = q_guide_hr.reshape(B * N, -1) # q_guide_hr 在循环外计算，对所有点相同
                    rel_coord_flat_loop = current_rel_coord.reshape(B * N, -1)

                    if not rel_coord_flat_loop.requires_grad:
                        rel_coord_flat_loop.requires_grad_(True)
                    elif rel_coord_flat_loop.grad is not None:
                        rel_coord_flat_loop.grad.zero_()

                    inp_for_imnet = torch.cat([
                        q_feat_flat.detach(), 
                        q_guide_hr_flat_loop.detach(), # q_guide_hr 在循环外计算，这里应该是每个点的对应值
                        rel_coord_flat_loop
                    ], dim=-1)

                    pred_raw = self.imnet(inp_for_imnet) # [B*N, hsi_dim+1]

                    # --- 一阶导数 ---
                    first_deriv_parts = []
                    for i in range(pred_raw.shape[1]):
                        grad_outputs_i = torch.zeros_like(pred_raw)
                        grad_outputs_i[:, i] = 1.0
                        grad_i_wrt_coords = torch.autograd.grad(
                            outputs=pred_raw, inputs=rel_coord_flat_loop,
                            grad_outputs=grad_outputs_i, retain_graph=True, create_graph=True
                        )[0]
                        first_deriv_parts.append(grad_i_wrt_coords.unsqueeze(1))
                    first_deriv_tensor = torch.cat(first_deriv_parts, dim=1) # [B*N, hsi_dim+1, 2]

                    # --- 二阶导数 (可选) ---
                    # (如果需要，按照上面的方法计算 second_deriv_tensor)
                    # second_deriv_tensor = ... # [B*N, hsi_dim+1, 2, 2]

                    # --- 准备 imnet_grad 输入 ---
                    # 方案1: 只用一阶导数，匹配原维度
                    inp_for_imnet_grad = torch.cat([
                        first_deriv_tensor[:, :, 0],
                        first_deriv_tensor[:, :, 1]
                    ], dim=-1) # [B*N, 2 * (hsi_dim+1)]
                    
                    # (如果使用了二阶导数，并修改了 imnet_grad_in_dim，则相应调整这里的拼接)

                    pred_from_deriv_terms = self.imnet_grad(inp_for_imnet_grad).view(B, N, -1)
                    # ---- 结束 Hermite 计算 ----

                preds_val_list.append(pred_raw.view(B,N,-1))
                preds_deriv_list.append(pred_from_deriv_terms)
        
        # --- 后续处理 preds_val_list 和 preds_deriv_list ---
        # 您原来的 preds 和 preds_grad 现在对应 preds_val_list 和 preds_deriv_list
        preds_val_stacked = torch.stack(preds_val_list, dim=-1)  # [B, N, C, kk]
        preds_deriv_stacked = torch.stack(preds_deriv_list, dim=-1)  # [B, N, C, kk]
        
        # 您原来的权重计算逻辑是基于 pred 的最后一个通道
        # pred_raw 的最后一个通道是权重通道
        weight_from_val = F.softmax(preds_val_stacked[:, :, -1, :], dim=-1) 
        # pred_from_deriv_terms 的最后一个通道也应该是权重 (如果 imnet_grad 输出一致)
        weight_from_deriv = F.softmax(preds_deriv_stacked[:, :, -1, :], dim=-1)

        ret_p = (preds_val_stacked[:, :, 0:-1, :] * weight_from_val.unsqueeze(-2)).sum(-1) # No keepdim, sum reduces last dim
        ret_g = (preds_deriv_stacked[:, :, 0:-1, :] * weight_from_deriv.unsqueeze(-2)).sum(-1)
        
        ret = ret_p + ret_g # 或者其他组合方式
        ret = ret.permute(0, 2, 1).view(b, -1, H, W) # out_dim-1 because weight channel is removed

        return ret

    def query_v1(self, feat, coord, hr_guide):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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
        preds_grad = []
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

                with torch.enable_grad():  # must enable grad to compute gradients
                    inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                    inp.requires_grad_(True)
                    pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                    # 分离光谱预测和权重
                    pred_spectral = pred[:, :, :-1]  # [B, N, hsi_dim]
                    # pred_weight = pred[:, :, -1:]    # [B, N, 1]
                    # pred_spectral_ = self.compute_pred_grad_sobel(pred_spectral.permute(0, 2, 1).view(b, -1, H, W), rel_coord)

                    # --- hermite gradient --- #
                    pred_spectral_grad = self.compute_pred_grad(pred_spectral, inp, rel_coord)

                # --- mlp inject gradient --- #
                pred_grad = self.imnet_grad(pred_spectral_grad.view(B * N, -1)).view(B, N, -1)

                # pred_ = torch.cat([pred_spectral_, pred_weight], dim=-1)

                preds.append(pred)
                preds_grad.append(pred_grad)

        preds = torch.stack(preds, dim=-1)  # [B, N, C, kk]
        preds_grad = torch.stack(preds_grad, dim=-1)  # [B, N, C, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        weight_grad = F.softmax(preds_grad[:, :, -1, :], dim=-1)
        ret_p = ((preds[:, :, 0:-1, :] * weight.unsqueeze(-2))
            .sum(-1, keepdim=True)
            .squeeze(-1)
        )
        ret_g = (preds_grad[:, :, 0:-1, :] * weight_grad.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret_p  + ret_g
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

            # 查询与特征融合
            ret = self.query(feat, coord, hr_guide)

            # 更新 LR_HSI
            LR_HSI = ret + LR_HSI_up

        return LR_HSI

    def _forward_implem_v3(self, HR_MSI, lms, LR_HSI):
        # 获取输入 LR_HSI 和目标 HR_MSI 的初始尺寸
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW# Bx128xhxw
        feat = self.depth_encoder(LR_HSI)  # Bx128xhxw
        _, _, h_LR, w_LR = feat.shape
        _, _, h_HR, w_HR = hr_guide.shape
        coord = make_coord(hr_guide.shape[2:]).cuda()
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
        LR_HSI_up = feat.clone()
        for _ in range(num_steps):
            # 当前 LR_HSI 尺寸

            ret = self.query(LR_HSI_up, coord, hr_guide)

            # 动态上采样，scale_factor 每次都是计算出的自适应值
            LR_HSI_up = torch.nn.functional.interpolate(
                LR_HSI_up,
                scale_factor=(scale_factor_h, scale_factor_w),
                mode="bilinear",
                align_corners=False,
            )

            # 计算下采样因子，确保 HR_MSI_down 与 LR_HSI_up 尺寸匹配
            # downscale_factor_h = float(h_HR / h / scale_factor_h)
            # downscale_factor_w = float(w_HR / w / scale_factor_w)

            # HR_MSI_down = torch.nn.functional.interpolate(hr_guide, scale_factor=(1/downscale_factor_h, 1/downscale_factor_w), mode='bilinear', align_corners=False)

            # 确保 HR_MSI_down 的尺寸与 LR_HSI_up 一致
            # assert HR_MSI_down.shape[2:] == LR_HSI_up.shape[2:], "尺寸不匹配"

            # 提取特征

            # hr_guide = self.image_encoder(HR_MSI_down)  # 提取下采样后的高分辨率图像特征
            # feat = self.depth_encoder(LR_HSI)  # 提取当前 LR_HSI 特征

            # 查询与特征融合
            # ret = self.query(LR_HSI_up, coord, hr_guide)

            # LR_HSI_up = self.conv_1x1(LR_HSI_up)

            # 更新 LR_HSI
            output = ret + lms
        # output = self.conv_layer(feat)

        return output

    def _forward_implem_(self, HR_MSI, lms, LR_HSI):
        # image, depth, coord, res, lr_image = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data['lr_image']
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).cuda()
        # feat = torch.cat([HR_MSI, lms], dim=1)
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW
        # lr_guide = self.image_encoder(lr_image)  # Bx128xhxw
        feat = self.depth_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI
        ret = self.query_v2(feat, coord, hr_guide)
        output = lms + ret

        # else:
        #     N = coord.shape[1] # coord ~ [B, N, 2]
        #     n = 30720
        #     tmp = []
        #     for start in range(0, N, n):
        #         end = min(N, start + n)
        #         ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide, data['hr_depth'].repeat(1,3,1,1)) # [B, N, 1]
        #         tmp.append(ans)
        #     res = res + torch.cat(tmp, dim=1)

        return output

    def _forward_implem_flop_analysis(self, HR_MSI, lms, LR_HSI):
        """用于FLOP分析的简化版本, 不包含autograd.grad操作"""
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).cuda()
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW
        feat = self.depth_encoder(LR_HSI)  # Bx128xhxw

        # 简化的query操作，不包含梯度计算
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        feat_coord = (
            make_coord((h, w), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, h, w)
        )
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        rx = 1 / h
        ry = 1 / w

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
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                # 只使用基本的MLP预测，不包含梯度计算
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (
            (preds[:, :, 0:-1, :] * weight.unsqueeze(-2))
            .sum(-1, keepdim=True)
            .squeeze(-1)
        )
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        output = lms + ret
        return output


    
    def sharpening_train_step(self,lms, lr_hsi, pan, gt, criterion):
        # ms = self._construct_ms(lms)
        sr = self._forward_implem(pan,lms,lr_hsi)
        loss = criterion(sr, gt)
        return sr.clip(0, 1), loss
    
    # def sharpening_train_step(self, ms, lms, pan, gt, criterion):
    #     sr = self._forward_implem(pan, lms, ms)
    #     # sr = self._forward_implem(ms,pan)

    #     loss = criterion(sr, gt)
    #     # loss_2 = criterion(sr_g, self.grad_gt(gt-lms))
    #     # loss = loss_1[0] + 0.1 * loss_2[0]
        
    #     # log_vars = {}
    #     # with torch.no_grad():
    #     #     metrics = analysis_accu(gt, sr, 4, choices=4)
    #     #     log_vars.update(metrics)

    #     # return {'loss': loss, 'log_vars': log_vars}
    #     return sr.clip(0, 1), loss
    
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
            pred = self._forward_implem_(pan,lms,lr_hsi)
            # pred = self._forward_implem(ms, pan)


        return pred.clip(0, 1)
    
    # def set_metrics(self, criterion, rgb_range=1.0):
    #     self.rgb_range = rgb_range
    #     self.criterion = criterion

    def patch_merge_step(self,lms, lr_hsi, pan, *args, **kwargs):
        return self._forward_implem_(pan,lms,lr_hsi)
        # return self._forward_implem(ms, pan)

if __name__ == '__main__':
    # from fvcore.nn import FlopCountAnalysis, flop_count_table

    # torch.cuda.set_device('cuda:1')

    # model = JIIF_(31, 3, 128, 128).cuda()

    # B, C, H, W = 1, 31, 512, 512
    # scale = 4

    # HR_MSI = torch.randn([B, 3, H, W]).cuda()
    # HSI_up = torch.randn([B, C, H, W]).cuda()
    # LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()

    # output = model.val_step(LR_HSI, HSI_up, HR_MSI)
    # print(output.shape)




    from fvcore.nn import FlopCountAnalysis, flop_count_table

    torch.cuda.set_device('cuda:1')

    model = MHIIF_J2(31 ,3 ,64, 64).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    criterion = torch.nn.L1Loss()
    gt = torch.randn([1, 31, H, W]).cuda()
    
    # output, loss= model.train_step(LR_HSI, lms, HR_MSI,gt,criterion)
    # print(output.shape)
    model.forward = model._forward_implem_
    # output = model._forward_implem_v3(HR_MSI,lms,LR_HSI)
    # print(output.shape)
    output = model.sharpening_val_step(lms, LR_HSI, HR_MSI, gt)
    # output = model.val_step(LR_HSI, lms, HR_MSI)
    print(output.shape)


    # output,grad_gt = model._forward_implem(HR_MSI,lms,LR_HSI)
    # print(output.shape)
    # gt = torch.randn(1, 31, H, W).cuda()
    # gr = torch.randn(1, 31, H, W).cuda()
    # criterion = torch.nn.L1Loss()
    # loss_1 = criterion(output, gt)
    # loss_2 = criterion(grad_gt, gr)
    # loss = loss_1+0.1*loss_2

    print(flop_count_table(FlopCountAnalysis(model, (HR_MSI,lms,LR_HSI))))
    # ### 3.551M                 | 10.533G ###
    # ### 2.832M                 | 8.352G            
    # ### 0.771M                 | 3.194G            
    # ### 0.771M                 | 2.883G    baseline
    # ### 0.899M                 | 6.175G    grad
    # ###v2 0.843504M              | 4.354982G     |
    ####v3 0.739345M              | 4.958314G     |
    ####v3 0.706321M              | 4.287226G  
    ####   0.660976M              | 3.171402G  MHIIF
