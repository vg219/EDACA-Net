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


        

@register_model('ENACIR')
class ENACIR(BaseModel):

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

        self.imnet = MLP(imnet_in_dim, hsi_dim +1, hidden_list=self.mlp_dim)

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

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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
        q_feat_q = F.grid_sample(q_feat_q, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        q_feat_k = F.grid_sample(q_feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        q_feat_v = F.grid_sample(q_feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
        bs, spat, feats = q_feat_k.shape
        prob_k = F.softmax(self.W_prob(q_feat_k).squeeze(-1), -1) + 1e-8

        entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = STEFunction.apply(entropy_step)
        means = []
        # stds = []
        for b_ in range(bs):
            boundaries = torch.diff(entropy_step[b_].type(torch.int64), prepend=~entropy_step[b_][:1].type(torch.int64), append=~entropy_step[b_][-1:].type(torch.int64))
            region_lengths = torch.diff(torch.nonzero(boundaries).squeeze())
            mean_region_length = region_lengths.float().mean()  # 直接调用，不检查长度
            # std_region_length = region_lengths.float().std()
            means.append(mean_region_length.item())
            # stds.append(std_region_length.item())
        
        clst_sh = round(np.mean(means))
        q_feat_k = q_feat_k[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        q_feat_v = q_feat_v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        q_feat_k = q_feat_k.view(bs, q_feat_k.shape[1]//clst_sh, clst_sh, feats)
        q_feat_v = q_feat_v.view(bs, q_feat_v.shape[1]//clst_sh, clst_sh, feats)
        entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
        q_feat_k = (entropy*q_feat_k).sum(-2)
        q_feat_v = (entropy*q_feat_v).sum(-2)
        
        q_feat_q = q_feat_q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        q_feat_k = q_feat_k.view(bs, q_feat_k.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        q_feat_v = q_feat_v.view(bs, q_feat_v.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)

        weight = F.softmax(torch.matmul(q_feat_q, q_feat_k.transpose(2,3)), -1)/(feats//self.n_heads)
        pred_ = torch.matmul(weight,q_feat_v).permute(1,2,0,3).contiguous().view(B, N, -1)  # [B, N, 2]
        # q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
        rx = 1 / h
        ry = 1 / w

        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)
                
                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                # q_feat_q = F.grid_sample(q_feat_q, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
                # q_feat_k = F.grid_sample(q_feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
                # q_feat_v = F.grid_sample(q_feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,:].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
        
                # q_feat_v = self.imnet(torch.cat([q_feat_v, q_guide_hr, rel_coord], dim=-1))
                # q_feat_ = q_feat_v.clone()
                # bs, spat, feats = q_feat_k.shape
                # prob_k = F.softmax(self.W_prob(q_feat_k).squeeze(-1), -1) + 1e-8

                # entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(device))
                # entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
                
                # entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
                # entropy_step = STEFunction.apply(entropy_step)
                # means = []
                # # stds = []
                # for b_ in range(bs):
                #     boundaries = torch.diff(entropy_step[b_].type(torch.int64), prepend=~entropy_step[b_][:1].type(torch.int64), append=~entropy_step[b_][-1:].type(torch.int64))
                #     region_lengths = torch.diff(torch.nonzero(boundaries).squeeze())
                #     mean_region_length = region_lengths.float().mean()  # 直接调用，不检查长度
                #     # std_region_length = region_lengths.float().std()
                #     means.append(mean_region_length.item())
                #     # stds.append(std_region_length.item())
                
                # clst_sh = round(np.mean(means))
                # q_feat_k = q_feat_k[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
                # q_feat_v = q_feat_v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
                # q_feat_k = q_feat_k.view(bs, q_feat_k.shape[1]//clst_sh, clst_sh, feats)
                # q_feat_v = q_feat_v.view(bs, q_feat_v.shape[1]//clst_sh, clst_sh, feats)
                # entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
                # entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
                # q_feat_k = (entropy*q_feat_k).sum(-2)
                # q_feat_v = (entropy*q_feat_v).sum(-2)
                
                # q_feat_q = q_feat_q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2,0,1,3)
                # q_feat_k = q_feat_k.view(bs, q_feat_k.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)
                # q_feat_v = q_feat_v.view(bs, q_feat_v.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)

                # weight = F.softmax(torch.matmul(q_feat_q, q_feat_k.transpose(2,3)), -1)/(feats//self.n_heads)
                # pred_ = torch.matmul(weight,q_feat_v).permute(1,2,0,3).contiguous().view(B, N, -1)  # [B, N, 2]
                pred = self.imnet(torch.cat([q_feat, pred_, q_guide_hr, rel_coord], dim=-1))
                preds.append(pred)
        preds = torch.stack(preds, dim=-1)  # [B, N, C, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        # ret = self.W_o(torch.cat([ret, q_feat_], dim=-1))
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret
    
    def query_with_entropy_clustering_for_inr(self, feat, coord, hr_guide):
        """
        用熵驱动聚类替代传统角元素集成，直接将聚类增强特征输入INR，不再输入多头注意力特征
        """
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        device = feat.device

        # 采样特征
        feat_coord = make_coord((h, w), flatten=False).to(device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= h
        rel_coord[:, :, 1] *= w

        # QKV
        q_feat_q, q_feat_k, q_feat_v = self.gen_qkv(feat)
        q_feat_k = F.grid_sample(q_feat_k, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_feat_v = F.grid_sample(q_feat_v, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        bs, spat, feats = q_feat_k.shape

        # 熵驱动聚类
        prob_k = F.softmax(self.W_prob(q_feat_k).squeeze(-1), -1) + 1e-8
        entropy = -prob_k * torch.log(prob_k) / torch.log(self.base.to(device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = STEFunction.apply(entropy_step)

        means = []
        for b_ in range(bs):
            boundaries = torch.diff(entropy_step[b_].type(torch.int64), prepend=~entropy_step[b_][:1].type(torch.int64), append=~entropy_step[b_][-1:].type(torch.int64))
            region_lengths = torch.diff(torch.nonzero(boundaries).squeeze())
            mean_region_length = region_lengths.float().mean()
            means.append(mean_region_length.item())
        clst_sh = round(np.mean(means))

        # 聚类压缩
        q_feat_k_trimmed = q_feat_k[:, (spat % clst_sh)//2 : spat - (spat % clst_sh - (spat % clst_sh)//2), :]
        q_feat_v_trimmed = q_feat_v[:, (spat % clst_sh)//2 : spat - (spat % clst_sh - (spat % clst_sh)//2), :]
        q_feat_k_clustered = q_feat_k_trimmed.view(bs, q_feat_k_trimmed.shape[1] // clst_sh, clst_sh, feats)
        q_feat_v_clustered = q_feat_v_trimmed.view(bs, q_feat_v_trimmed.shape[1] // clst_sh, clst_sh, feats)
        entropy_trimmed = entropy[:, (spat % clst_sh)//2 : spat - (spat % clst_sh - (spat % clst_sh)//2)]
        entropy_weights = F.softmax(entropy_trimmed.view(bs, entropy_trimmed.shape[1] // clst_sh, clst_sh), -1).unsqueeze(-1)
        q_feat_v_compressed = (entropy_weights * q_feat_v_clustered).sum(-2)  # [bs, num_clusters, feats]
        # 将聚类特征插值回原始空间分辨率
        q_feat_v_expanded = F.interpolate(
            q_feat_v_compressed.transpose(1, 2).unsqueeze(-1), 
            size=spat, mode='linear', align_corners=False
        ).squeeze(-1).transpose(1, 2)  # [bs, spat, feats]

        ret = self.imnet(torch.cat([
            q_feat,                # 原始特征 [bs, spat, feats]
            q_feat_v_expanded,     # 插值后的聚类特征 [bs, spat, feats]
            q_guide_hr,            # HR引导
            rel_coord              # 相对坐标
        ], dim=-1))
        
        # 信息增强特征（每个查询点与聚类中心的相似性加权聚合）
        similarity_scores = torch.matmul(ret, q_feat_v_compressed.transpose(1, 2))  # [bs, spat, num_clusters]
        info_weights = F.softmax(similarity_scores / math.sqrt(feats), dim=-1)
        ret_ = torch.matmul(info_weights, ret)  # [bs, spat, feats]

        # 直接输入INR，不再输入多头注意力特征
        # ret = self.imnet(torch.cat([
        #     q_feat,                # 原始特征
        #     cluster_enhanced_feat, # 聚类增强特征
        #     q_guide_hr,            # HR引导
        #     rel_coord              # 相对坐标
        # ], dim=-1))
        ret_ = ret_.permute(0, 2, 1).view(b, -1, H, W)
        return ret_
    
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

    torch.cuda.set_device('cuda:6')

    model = ENACIR(31 ,3 ,128, 128).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    criterion = torch.nn.L1Loss()
    gt = torch.randn([1, 31, H, W]).cuda()
    
    # output, loss= model.sharpening_train_step(lms, LR_HSI, HR_MSI,gt,criterion)
    # print(output.shape)
    model.forward = model._forward_implem
    # output = model._forward_implem(HR_MSI,lms,LR_HSI)
    # print(output.shape)
    output = model.sharpening_val_step(lms, LR_HSI, HR_MSI,gt)
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
    ####   0.660976M              | 3.171402G 
    
    ####MHIIF 0.67336M               | 2.407678G  
    # #####   0.677328M              | 2.423931G  
    # #####   0.722031M              | 1.811825G
