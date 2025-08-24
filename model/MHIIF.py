import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from edsr import make_edsr_baseline, make_coord
import sys
sys.path.insert(0, '/home/YuJieLiang/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord, ComplexGaborLayer, PositionalEmbedding, MLP_P, MLP, hightfre, ImplicitDecoder

from model.base_model import BaseModel, register_model, PatchMergeModule

from utils import easy_logger

logger = easy_logger(func_name='MHIIF', level='INFO')

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

class Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_g = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim-1)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pred, grad):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = pred.shape
        
        qkv_c = self.qkv_c(pred).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_g = self.qkv_g(grad).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]  # make torchscript happy (cannot use tensor as tuple)
        q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2]
        
        q_c = q_c * self.scale
        attn = (q_c @ k_g.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v_g).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class FixedSymmetricConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, direction):
        super(FixedSymmetricConvolution, self).__init__()
        self.direction = direction
        # 创建卷积核参数，并将其初始化为随机值
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            
        self.symmetricize_weight()

    def forward(self, x):
        # 应用卷积操作
        return nn.functional.conv2d(x, self.weight, padding=1)

    def symmetricize_weight(self):
        # 对称化卷积核：中间一列为0，第一列和第三列的值互为相反数
        kernel_size = self.weight.size(-1)
        if self.direction == 0:
            middle_col = kernel_size // 2
            self.weight.data[:, :, :, middle_col] = 0
            self.weight.data[:, :, :, 0] = -self.weight.data[:, :, :, -1].clone().detach()
            # 固定中间一列为0
            self.weight.data[:, :, :, middle_col].requires_grad = False
        elif self.direction == 1:
            middle_row = kernel_size // 2
            self.weight.data[:, :,middle_row,:] = 0
            # 第一行和第三行的值互为相反数
            self.weight.data[:, :, 0, :] = -self.weight.data[:, :, -1, :].clone().detach()
            self.weight.data[:, :, middle_row, :].requires_grad = False

        

@register_model('MHIIF')
class MHIIF_(BaseModel):

    def __init__(self, hsi_dim=31, msi_dim=3, feat_dim=128, guide_dim=128, spa_edsr_num=3, spe_edsr_num=3, mlp_dim=[256,128], scale = 4, patch_merge=True,):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.image_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim)
        self.depth_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim)

        imnet_in_dim = self.feat_dim + self.guide_dim + 2
        # imnet_grad_in_dim = self.feat_dim + self.guide_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=hsi_dim+1 ,hidden_list=self.mlp_dim)
        # self.fixed_symmetric_conv_x = FixedSymmetricConvolution(32, 32, kernel_size=3,direction=0)
        # self.fixed_symmetric_conv_y = FixedSymmetricConvolution(32, 32, kernel_size=3,direction=1)
        # self.imnet_grad = MLP(64, out_dim=hsi_dim+1, hidden_list=self.mlp_dim)
        imnet_grad_in_dim = 2*(hsi_dim+1) + 2*self.guide_dim + 2
        self.imnet_grad = MLP(imnet_grad_in_dim, out_dim=hsi_dim+1, hidden_list=self.mlp_dim)
        # self.conv_layer = nn.Conv2d(in_channels=feat_dim, out_channels=hsi_dim, kernel_size=3, padding=1)
        # self.conv_1x1 = nn.Conv2d(in_channels=hsi_dim, out_channels=feat_dim, kernel_size=1, padding=0)


        # I know that. One of the values is depth, and another is the weight.
        self.patch_merge = patch_merge
        # self._patch_merge_model = PatchMergeModule(self, crop_batch_size=32,
        #                                            scale=self.scale, 
        #                                            patch_size_list=[16, 16*self.scale, 16*self.scale],
        #                                            )

    def grad_gt(self, feat):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        # 使用卷积操作计算梯度
        b, c, h, w = feat.shape
        gradient_x = F.conv2d(feat, sobel_x.expand(c, 1, 3, 3), padding=1, groups=c)
        gradient_y = F.conv2d(feat, sobel_y.expand(c, 1, 3, 3), padding=1, groups=c)

        # 计算梯度大小和方向
        # gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        # gradient_direction = torch.atan2(gradient_y, gradient_x)

        return torch.cat([gradient_x, gradient_y],dim=1)
    
    # def grad(self, feat):
        
    #     gradient_x = self.fixed_symmetric_conv_x(feat)  #Bx128x64x64
    #     gradient_y = self.fixed_symmetric_conv_y(feat)
    #     gradient = torch.cat([gradient_x, gradient_y], dim=1)
    #     # gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-6)
    #     # gradient_direction = torch.atan2(gradient_y, gradient_x)
    #     # gradient = torch.stack([gradient_x, gradient_y], dim=1)#Bx2x32x64x64
    #     return gradient
    
    def query(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]
        # mean = nn.AdaptiveMaxPool2d((128, 128))
        # lr_guide = mean(hr_guide)
        # q_guide_hr_grad = self.grad(q_guide_hr.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)

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
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
                # feat_grad = self.grad(q_feat.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)
                # q_feat_grad = F.grid_sample(feat_grad, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                #          :].permute(0, 2, 1)  # [B, N, c]
                # q_guide_hr_grad = self.grad(q_guide_hr.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[ :, :, 0, :].permute(0, 2, 1) # [B, N, C]
                # q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)
                # inp_grad = torch.cat([q_feat,q_guide_hr,feat_grad,q_guide_hr_grad,rel_coord], dim=-1)
                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                # q_feat_t = self.grad(q_feat.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)
                # q_guide_t = self.grad(q_guide_hr.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)
                # inp_grad = torch.cat([q_feat_t, q_guide_t, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                inp_grad = self.grad_gt(pred.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)
                q_guide_grad = self.grad_gt(q_guide_hr.permute(0, 2, 1).view(b, -1, H, W)).permute(0, 2, 3, 1).reshape(b, H*W, -1)
                # pred_exp = torch.cat([pred, pred], dim=-1)  # [B, N, 2*C]
                # 计算梯度差异
                # rel_grad = inp_grad - pred_exp  # [B, N, 2*C] - [B, N, 2*C]
                inp_ = torch.cat([inp_grad, q_guide_grad, rel_coord], dim=-1)
                pred_grad = self.imnet_grad(inp_.view(B * N, -1)).view(B, N, -1)

                preds.append(pred)
                preds_grad.append(pred_grad)

        preds = torch.stack(preds, dim=-1)  # [B, N, C, kk]
        preds_grad = torch.stack(preds_grad, dim=-1)  # [B, N, C, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        weight_grad = F.softmax(preds_grad[:, :, -1, :], dim=-1)
        ret_p = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret_g = (preds_grad[:, :, 0:-1, :] * weight_grad.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret_p + ret_g
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
        scale_factor_h = float(total_scale_factor_h ** (1.0 / num_steps))  # 每次高度放大比例
        scale_factor_w = float(total_scale_factor_w ** (1.0 / num_steps))  # 每次宽度放大比例
        
        # 开始逐步上采样
        for _ in range(num_steps):
            # 当前 LR_HSI 尺寸
            _, _, h_LR, w_LR = LR_HSI.shape
            
            # 动态上采样，scale_factor 每次都是计算出的自适应值
            LR_HSI_up = torch.nn.functional.interpolate(LR_HSI, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)
            
            # 计算下采样因子，确保 HR_MSI_down 与 LR_HSI_up 尺寸匹配
            downscale_factor_h = float(h_HR / h_LR / scale_factor_h)
            downscale_factor_w = float(w_HR / w_LR / scale_factor_w)
            
            HR_MSI_down = torch.nn.functional.interpolate(HR_MSI, scale_factor=(1/downscale_factor_h, 1/downscale_factor_w), mode='bilinear', align_corners=False)
            
            # 确保 HR_MSI_down 的尺寸与 LR_HSI_up 一致
            assert HR_MSI_down.shape[2:] == LR_HSI_up.shape[2:], "尺寸不匹配"
            
            # 提取特征
            coord = make_coord(HR_MSI_down.shape[2:]).cuda()
            hr_guide = self.image_encoder(HR_MSI_down)  # 提取下采modelmodel/MIMO_SST.py/MIMO_SST.py样后的高分辨率图像特征
            feat = self.depth_encoder(LR_HSI)  # 提取当前 LR_HSI 特征
            
            # 查询与特征融合
            ret = self.query(feat, coord, hr_guide)
            
            # 更新 LR_HSI
            LR_HSI = ret + LR_HSI_up
        
        return LR_HSI
    
    def _forward_implem_v3(self, HR_MSI, lms, LR_HSI):
        # 获取输入 LR_HSI 和目标 HR_MSI 的初始尺寸
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW# Bx128xhxw
        feat = self.depth_encoder(LR_HSI)      # Bx128xhxw 
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
        scale_factor_h = float(total_scale_factor_h ** (1.0 / num_steps))  # 每次高度放大比例
        scale_factor_w = float(total_scale_factor_w ** (1.0 / num_steps))  # 每次宽度放大比例
        
        # 开始逐步上采样
        LR_HSI_up = feat.clone()
        for _ in range(num_steps):
            # 当前 LR_HSI 尺寸            
            
            ret = self.query(LR_HSI_up, coord, hr_guide)
            
            # 动态上采样，scale_factor 每次都是计算出的自适应值
            LR_HSI_up = torch.nn.functional.interpolate(LR_HSI_up, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)
            
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
        ret = self.query(feat, coord, hr_guide)
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
    
    def sharpening_train_step(self,lms, lr_hsi, pan, gt, criterion):
        # ms = self._construct_ms(lms)
        sr = self._forward_implem_(pan,lms,lr_hsi)
        loss = criterion(sr, gt)
        return sr.clip(0, 1), loss
    

    
    def sharpening_val_step(self,
                            lms: torch.Tensor,
                            lr_hsi: torch.Tensor,
                            pan: torch.Tensor,
                            txt: "torch.Tensor | None"=None,
                            patch_merge: "callable | bool | None"=True,
                            *,
                            inference_wo_txt: bool=False,
                            **_kwargs):  
        if patch_merge is None:
            patch_merge = self.patch_merge
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

        return pred.clip(0, 1)
    
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

    torch.cuda.set_device('cuda:0')

    model = MHIIF_(31 ,3 ,64, 64).cuda()

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
