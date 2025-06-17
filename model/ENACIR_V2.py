import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat')
from model.module.fe_block import make_edsr_baseline, make_coord

from model.base_model import BaseModel, register_model, PatchMergeModule

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

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


        

@register_model('ENACIR_V2')
class ENACIR_V2(BaseModel):

    def __init__(self, hsi_dim=31, msi_dim=3, feat_dim=128, guide_dim=128, spa_edsr_num=3, spe_edsr_num=3, mlp_dim=[256,128], scale = 4, n_heads=8, patch_merge=True, ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.scale = scale
        self.n_heads = n_heads
        self.image_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=msi_dim)
        self.depth_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim)

        imnet_in_dim = 2 * self.feat_dim + self.guide_dim + 2
        self.w_q = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)
        self.w_k = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)
        self.w_v = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)
        self.W_prob = nn.Linear(self.feat_dim, 1)
        self.W_o = nn.Linear(2*(self.feat_dim), hsi_dim)
        self.gaussian_kernel = (1./(3*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*3-1),3*3), 2)/(2*torch.pow(torch.Tensor([3]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])

        self.imnet = MLP(imnet_in_dim, hsi_dim, hidden_list=self.mlp_dim)

        # I know that. One of the values is depth, and another is the weight.
        self.patch_merge = patch_merge
        # self._patch_merge_model = PatchMergeModule(self, crop_batch_size=32,
        #                                            scale=self.scale, 
        #                                            patch_size_list=[16, 16*self.scale, 16*self.scale])
        # self.device = device


    def gen_qkv(self, feat):
        feat_q = self.w_q(feat)
        feat_k = self.w_k(feat)
        feat_v = self.w_v(feat)
        return feat_q, feat_k, feat_v
        
    def query(self, feat, coord, hr_guide):
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        device = feat.device
        
        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]
        
        coord_ = coord.clone()
        
        q_feat_q, q_feat_k, q_feat_v = self.gen_qkv(feat)
        q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)
        
        # 2D空间采样而不是1D序列采样
        q_feat_q = F.grid_sample(q_feat_q, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        q_feat_k = F.grid_sample(q_feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        q_feat_v = F.grid_sample(q_feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
        
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= h
        rel_coord[:, :, 1] *= w
        
        bs, spat, feats = q_feat_k.shape
        
        # ===== 改进的2D空间聚类部分 =====
        
        # 1. 计算2D空间熵（而不是1D序列熵）
        prob_k = F.softmax(self.W_prob(q_feat_k).squeeze(-1), -1) + 1e-8  # [bs, spat]
        
        # 将1D序列重塑为2D空间
        spatial_h = int(np.sqrt(spat))
        spatial_w = spat // spatial_h
        if spatial_h * spatial_w != spat:
            # 如果不是完全平方数，调整为最接近的矩形
            spatial_h = int(np.sqrt(spat))
            spatial_w = spat // spatial_h
            remaining = spat - spatial_h * spatial_w
            if remaining > 0:
                # 补齐到矩形
                prob_k = prob_k[:, :spatial_h * spatial_w]
                q_feat_k = q_feat_k[:, :spatial_h * spatial_w, :]
                q_feat_v = q_feat_v[:, :spatial_h * spatial_w, :]
                spat = spatial_h * spatial_w
        
        # 重塑为2D
        prob_k_2d = prob_k.view(bs, spatial_h, spatial_w)  # [bs, spatial_h, spatial_w]
        entropy_2d = -prob_k_2d * torch.log(prob_k_2d) / torch.log(self.base.to(device))
        
        # 2. 2D高斯平滑
        gaussian_2d = self.gaussian_kernel.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size]
        # 分别在两个维度进行1D卷积来近似2D高斯
        entropy_smooth = F.conv1d(entropy_2d.view(bs * spatial_h, 1, spatial_w), 
                                gaussian_2d, padding='same').view(bs, spatial_h, spatial_w)
        entropy_smooth = F.conv1d(entropy_smooth.permute(0, 2, 1).contiguous().view(bs * spatial_w, 1, spatial_h), 
                                gaussian_2d, padding='same').view(bs, spatial_w, spatial_h).permute(0, 2, 1)
        
        # 3. 2D边缘检测（Sobel算子）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(entropy_smooth.unsqueeze(1), sobel_x, padding=1)
        grad_y = F.conv2d(entropy_smooth.unsqueeze(1), sobel_y, padding=1)
        entropy_step_2d = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)  # [bs, spatial_h, spatial_w]
        
        # 4. 自适应聚类大小计算
        means = []
        for b_ in range(bs):
            # 基于2D梯度幅值计算聚类大小
            gradient_map = entropy_step_2d[b_]
            threshold = gradient_map.mean() + 0.5 * gradient_map.std()
            high_gradient_regions = (gradient_map > threshold).float()
            
            # 计算连通区域的平均大小
            region_count = high_gradient_regions.sum().item()
            if region_count > 0:
                avg_region_size = (spatial_h * spatial_w) / max(region_count, 1)
            else:
                avg_region_size = 4  # 默认聚类大小
            means.append(avg_region_size)
        
        clst_sh = max(2, min(16, round(np.mean(means))))  # 限制聚类大小在合理范围内
        
        # 5. 重新采样和聚类
        # 将2D熵图重新flatten，但保持空间邻近性
        entropy_flat = entropy_smooth.reshape(bs, -1)  # [bs, spatial_h * spatial_w]
        
        # 确保可以整除
        valid_length = (spat // clst_sh) * clst_sh
        start_idx = (spat - valid_length) // 2
        
        q_feat_k_valid = q_feat_k[:, start_idx:start_idx + valid_length, :]
        q_feat_v_valid = q_feat_v[:, start_idx:start_idx + valid_length, :]
        entropy_valid = entropy_flat[:, start_idx:start_idx + valid_length]
        
        # 重塑并聚类
        q_feat_k_clustered = q_feat_k_valid.view(bs, valid_length // clst_sh, clst_sh, feats)
        q_feat_v_clustered = q_feat_v_valid.view(bs, valid_length // clst_sh, clst_sh, feats)
        entropy_weights = F.softmax(entropy_valid.view(bs, valid_length // clst_sh, clst_sh), -1).unsqueeze(-1)
        
        # 加权聚合
        q_feat_k = (entropy_weights * q_feat_k_clustered).sum(-2)
        q_feat_v = (entropy_weights * q_feat_v_clustered).sum(-2)
        
        # 6. 多头注意力计算（保持原有逻辑）
        q_feat_q = q_feat_q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        q_feat_k = q_feat_k.view(bs, q_feat_k.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        q_feat_v = q_feat_v.view(bs, q_feat_v.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)

        weight = F.softmax(torch.matmul(q_feat_q, q_feat_k.transpose(2,3)), -1)/(feats//self.n_heads)
        ret_ = torch.matmul(weight,q_feat_v).permute(1,2,0,3).contiguous().view(B, N, -1)  # [B, N, 2]
        ret = self.imnet(torch.cat([q_feat, ret_, q_guide_hr, rel_coord], dim=-1))
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret
    

    def _forward_implem(self, HR_MSI, lms, LR_HSI):
        # image, depth, coord, res, lr_image = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data['lr_image']
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).cuda()
        # feat = torch.cat([HR_MSI, lms], dim=1) 
        hr_guide = self.image_encoder(HR_MSI)  # Bx128xHxW
        feat = self.depth_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI
        ret = self.query(feat, coord, hr_guide)
        output = lms + ret

        return output

    def sharpening_train_step(self,lms, lr_hsi, pan, gt, criterion):
        sr = self._forward_implem(pan, lms, lr_hsi)

        loss = criterion(sr, gt)

        return sr.clip(0, 1), loss
    
    def sharpening_val_step(self,lms, lr_hsi, pan, gt):  
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

        return pred.clip(0, 1)

    def patch_merge_step(self,lms, lr_hsi, pan, *args, **kwargs):
        return self._forward_implem(pan, lms, lr_hsi)


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

    model = ENACIR_V2(31 ,3 ,128, 128).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    criterion = torch.nn.L1Loss()
    gt = torch.randn([1, 31, H, W]).cuda()
    
    # output, loss= model.train_step(LR_HSI, lms, HR_MSI,gt,criterion)
    # print(output.shape)
    model.forward = model._forward_implem
    output = model._forward_implem(HR_MSI,lms,LR_HSI)
    print(output.shape)
    # output = model.val_step(LR_HSI, lms, HR_MSI)
    # print(output.shape)


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
    ####   0.660976M              | 3.171402G 
    
    ####MHIIF 0.67336M               | 2.407678G  
    # #####   0.677328M              | 2.423931G  
    # #####   0.722031M              | 1.811825G     
