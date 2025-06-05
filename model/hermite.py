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

class GradientFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 水平梯度卷积核
        self.conv_x = nn.Conv2d(in_channels, in_channels, 
                               kernel_size=3, padding=1, groups=in_channels, bias=False)
        # 垂直梯度卷积核
        self.conv_y = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        # 初始化Sobel算子
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 扩展为多通道卷积核
        self.conv_x.weight.data = sobel_kernel_x.repeat(in_channels,1,1,1) / 8.0
        self.conv_y.weight.data = sobel_kernel_y.repeat(in_channels,1,1,1) / 8.0
        self.conv_x.weight.requires_grad_(False)  # 固定梯度计算参数
        self.conv_y.weight.requires_grad_(False)

    def forward(self, x):
        grad_x = self.conv_x(x)  # [B,C,h,w]
        grad_y = self.conv_y(x)  # [B,C,h,w]
        return torch.cat([grad_x, grad_y], dim=1)  # [B,2C,h,w]


class PositionalEncoder(nn.Module):
    """ 改进的位置编码器，包含导数信息 """
    def __init__(self, in_dim=2, out_dim=64):
        super().__init__()
        self.freq_bands = 2 ** torch.linspace(0, 5, out_dim//4)
        self.proj = nn.Sequential(
            nn.Linear(8*len(self.freq_bands), out_dim),
            nn.GELU()
        )
        
    def forward(self, coord):
        # coord: [..., 2]
        scaled_coord = coord * torch.pi  # 扩展到[-π, π]
        
        # 计算基础编码
        sins = torch.sin(scaled_coord[..., None] * self.freq_bands.to(coord.device))  # [B, N, 2, 16]
        coss = torch.cos(scaled_coord[..., None] * self.freq_bands.to(coord.device))  # [B, N, 2, 16]
        
        # 展平最后两个维度
        sins = sins.flatten(start_dim=-2)  # [B, N, 32]
        coss = coss.flatten(start_dim=-2)  # [B, N, 32]
            
        # 计算导数编码
        dx = scaled_coord[..., 0:1] * self.freq_bands.to(coord.device)
        dy = scaled_coord[..., 1:2] * self.freq_bands.to(coord.device)
        der_sin = torch.cat([torch.cos(dx), torch.cos(dy)], dim=-1)
        der_cos = torch.cat([-torch.sin(dx), -torch.sin(dy)], dim=-1)
        
        return self.proj(torch.cat([sins, coss, der_sin, der_cos], dim=-1))



@register_model('hermite')
class hermite(BaseModel):

    def __init__(self, hsi_dim=31, msi_dim=3, feat_dim=128, guide_dim=128, spa_edsr_num=3, spe_edsr_num=3, mlp_dim=[256,128], scale = 4, patch_merge=True,):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.hsi_dim = hsi_dim
        self.image_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim)
        self.depth_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim)

        imnet_in_dim = 3*self.feat_dim + 2*self.guide_dim + 2
        # imnet_grad_in_dim = self.feat_dim + self.guide_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=2*hsi_dim+3 ,hidden_list=self.mlp_dim)
        # self.imnet = MLP(imnet_in_dim, out_dim=hsi_dim+2 ,hidden_list=self.mlp_dim)
        self.position_encoder = PositionalEncoder(in_dim=2, out_dim=self.feat_dim)
        self.grad_extractor = GradientFeatureExtractor(self.feat_dim)
        self.output_proj = nn.Sequential(
                                            nn.Linear(self.feat_dim*2, 256),   # 输入: 3C=768 (256*3)
                                            nn.ReLU(inplace=True),
                                            nn.Linear(256, hsi_dim)          # 输出: RGB三通道
                                        )
        # self.channel_align = nn.Linear(self.feat_dim, self.feat_dim)

        # I know that. One of the values is depth, and another is the weight.
        self.patch_merge = patch_merge


    
    def hermite_query(self, feat, coord, hr_guide):
        """
        feat: 低分辨率特征 [B, C, h, w]
        coord: 高分辨率坐标 [B, H*W, 2] (归一化到[-1,1])
        hr_guide: 高分辨率引导特征 [B, C_guide, H, W]
        """
        B, C, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        N = H * W
        coord = coord.expand(B, H * W, 2)
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(B, 2, h, w)

        # ================== 第一步：提取梯度特征 ==================
        grad_feat = self.grad_extractor(feat)  # [B, 2C, h, w]
        
        # ================== 第二步：四邻域采样循环 ==================
        all_feats = []
        all_grads = []
        all_coords = []
        
        rx = 1 / h  # 单个LR像素在归一化坐标系中的宽度
        ry = 1 / w
        
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                # 生成偏移坐标
                offset_coord = coord.clone()
                offset_coord[..., 0] += vx * rx  # x方向偏移
                offset_coord[..., 1] += vy * ry  # y方向偏移
                
                # 采样原始特征
                q_feat = F.grid_sample(
                    feat, 
                    offset_coord.flip(-1).unsqueeze(1),  # grid_sample需要[x,y]顺序
                    mode='nearest',
                    align_corners=False
                ).squeeze(2).permute(0,2,1)  # [B, N, C]
                
                # 采样梯度特征 (包含x和y方向梯度)
                q_grad = F.grid_sample(
                    grad_feat,
                    offset_coord.flip(-1).unsqueeze(1),
                    mode='nearest',
                    align_corners=False
                ).squeeze(2).permute(0,2,1)  # [B, N, 2C]
                
                # # 计算相对坐标
                # feat_coord = make_coord((h,w)).to(feat.device)  # [h*w, 2]
                # feat_coord = feat_coord.permute(2,0,1).unsqueeze(0).expand(B,2,h,w)
                
                q_coord = F.grid_sample(
                    feat_coord,
                    offset_coord.flip(-1).unsqueeze(1),
                    mode='nearest',
                    align_corners=False
                ).squeeze(2).permute(0,2,1)  # [B, N, 2]
                
                rel_coord = coord - q_coord  # 相对坐标差
                rel_coord[..., 0] *= h  # 转换为实际像素距离
                rel_coord[..., 1] *= w
                
                # 保存所有采样结果
                all_feats.append(q_feat)
                all_grads.append(q_grad)
                all_coords.append(rel_coord)
        
        # ================== 第三步：构建Hermite输入特征 ==================
        # 拼接四个邻域的特征
        stacked_feats = torch.stack(all_feats, dim=2)  # [B, N, 4, C]
        stacked_grads = torch.stack(all_grads, dim=2)  # [B, N, 4, 2C]
        stacked_coords = torch.stack(all_coords, dim=2)  # [B, N, 4, 2]
        
        # 高分辨率引导特征采样
        hr_guide_feat = F.grid_sample(
            hr_guide,
            coord.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False
        ).squeeze(2).permute(0,2,1)  # [B, N, C_guide]
        
        # 位置编码
        pos_embed = self.position_encoder(stacked_coords.view(B*N*4,2))  # [B*N*4, pe_dim]
        pos_embed = pos_embed.view(B, N, 4, -1)  # [B, N, 4, pe_dim]
        
        # 构建最终输入特征
        hermite_input = torch.cat([
            stacked_feats,                  # 原始特征 [B,N,4,C]
            stacked_grads,                  # 梯度特征 [B,N,4,2C]
            hr_guide_feat.unsqueeze(2).expand(-1,-1,4,-1),  # 高分辨率引导 [B,N,4,C_guide]
            pos_embed,                      # 位置编码 [B,N,4,pe_dim]
            stacked_coords                  # 相对坐标差 [B, N, 4, 2]
        ], dim=-1)  # 最终维度: C + 2C + pe_dim + C_guide + 2
        
        # ================== 第四步：动态权重预测 ==================
        # 通过MLP预测每个邻域的Hermite系数
        hermite_coeff = self.imnet(
            hermite_input.view(B*N*4, -1)
        ).view(B, N, 4, -1)  # 3 = 值权重 + 梯度权重 + 混合系数
        

        # ================== 第五步：Hermite插值计算 ==================
        # 分解预测系数
        value_weights = torch.softmax(hermite_coeff[..., 0], dim=-1)  # [B,N,4]
        grad_weights = torch.sigmoid(hermite_coeff[..., 1])           # [B,N,4]
        blend_factors = hermite_coeff[..., 2]                         # [B,N,4]
        preds = hermite_coeff[...,3:3+self.hsi_dim]
        preds_grad = hermite_coeff[...,3+self.hsi_dim:]
        # 计算各邻域贡献
        value_terms = (preds * value_weights.unsqueeze(-1)).sum(dim=2)  # [B,N,C]
        grad_terms = (preds_grad * grad_weights.unsqueeze(-1)).sum(dim=2)    # [B,N,2C]



        # # 修改后的混合计算
        # aligned_value = self.channel_align(value_terms)  # [B,N,2C]
        # 混合值项和梯度项
        final_output = (
            blend_factors.mean(dim=-1, keepdim=True) * value_terms +
            (1 - blend_factors.mean(dim=-1, keepdim=True)) * grad_terms
        )  # [B,N,2C]
        
        # ================== 第六步：输出重构 ==================
        # 通道调整（示例：假设输出为3通道RGB）
        # output = self.output_proj(final_output)  # [B,N,3]
        return final_output.permute(0,2,1).view(B, -1, H, W)


    
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
            hr_guide = self.image_encoder(HR_MSI_down)  # 提取下采样后的高分辨率图像特征
            feat = self.depth_encoder(LR_HSI)  # 提取当前 LR_HSI 特征
            
            # 查询与特征融合
            ret = self.hermite_query(feat, coord, hr_guide)
            
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
            
            ret = self.hermite_query(LR_HSI_up, coord, hr_guide)
            
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
        ret = self.hermite_query(feat, coord, hr_guide)
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
            logger.debug(f"using patch merge module")
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

    model = hermite(31 ,3 ,64, 64).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    criterion = torch.nn.L1Loss()
    gt = torch.randn([1, 31, H, W]).cuda()

    model.forward = model._forward_implem
    output= model.sharpening_val_step(lms,LR_HSI, HR_MSI,gt)
    # print(output.shape)

    # output = model._forward_implem_v3(HR_MSI,lms,LR_HSI)
    # print(output.shape)
    # output = model.sharpening_val_step(lms, LR_HSI, HR_MSI, gt)
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
    #####  0.711M                 | 3.369G     |  48...
    #####  0.711M                 | 3.307G     51.7737
   ######  0.711M                 | 4.4G      m