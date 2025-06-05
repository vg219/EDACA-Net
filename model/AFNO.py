import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Utils import make_coord
from ADCI import ADCI


class AAFNO(nn.Module):
    def __init__(self):
        super().__init__()

        self.shallow_encoder1 = nn.Conv2d(31, 32, 1)
        self.shallow_encoder2 = nn.Conv2d(34, 32, 1)

        self.conv0 = simple_attn(32, 8)
        self.conv1 = simple_attn(32, 8)

        self.convw = nn.Conv2d(98, 32, 1)

        self.conv00 = nn.Conv2d(128, 64, 1)
        self.conv01 = nn.Conv2d(64, 32, 1)
        self.act = nn.ReLU()

        self.fc1 = nn.Conv2d(32, 32, 1)
        self.fc2 = nn.Conv2d(32, 31, 1)

        self.ADCI1_1 = ADCI(32, 32)
        self.ADCI1_2 = ADCI(32, 32)
        self.ADCI1_3 = ADCI(32, 32)

        self.ADCI2_1 = ADCI(32, 32)
        self.ADCI2_2 = ADCI(32, 32)
        self.ADCI2_3 = ADCI(32, 32)

    def INFI(self, feat, hsmi, coord):

        # feat: [B, 128, h, w]
        # hs: [B, 64, H, W]
        B = feat.shape[0]

        h, w = feat.shape[-2:]

        # coord(B,H,W,2)

        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(B, 2, h, w)
        # q_hsmi = F.grid_sample(hsmi, coord.flip(-1), mode='nearest', align_corners=False)

        rx = 1 / h
        ry = 1 / w
        inps = []
        areas = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                q_feat = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)

                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= h
                rel_coord[:, 1, :, :] *= w
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                inp = torch.cat([q_feat, hsmi, rel_coord], dim=1)
                inp = self.convw(inp)
                inps.append(inp)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        for index, area in enumerate(areas):
            inps[index] = inps[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat(inps, dim=1)

        x = self.conv00(grid)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.act(x)
        x = nn.functional.interpolate(x, scale_factor=1/2, mode='bicubic', align_corners=False)

        x = self.conv01(x)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)
        feat = x
        # f1 = self.fc1(feat)
        # f1 = F.interpolate(f1, scale_factor=2, mode='bicubic', align_corners=False)
        # f1 = F.gelu(f1)
        # f1 = F.interpolate(f1, scale_factor=1/2, mode='bicubic', align_corners=False)
        # ret = self.fc2(f1)
        ret = self.fc2(F.gelu(self.fc1(feat)))

        return ret

    def forward(self, lr_hsi, hr_msi, coord, sf):

        lr_hsi_up = F.interpolate(lr_hsi, scale_factor=sf, mode='bicubic', align_corners=False)
        hm_si = torch.cat([hr_msi, lr_hsi_up], dim=1)

        hm_si = self.shallow_encoder2(hm_si)
        lr_hsi = self.shallow_encoder1(lr_hsi)
        hm_si = self.ADCI1_3(self.ADCI1_2(self.ADCI1_1(hm_si)))
        lr_hsi = self.ADCI2_3(self.ADCI2_2(self.ADCI2_1(lr_hsi)))

        hm_si_down = F.interpolate(hm_si, scale_factor=1/sf, mode='bicubic', align_corners=False)

        feat = torch.cat([hm_si_down, lr_hsi], dim=1)

        # 进行query
        res = self.INFI(feat, hm_si, coord) + lr_hsi_up
        return res


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        x = self.o_proj1(ret)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic',align_corners=False)
        x = self.act(x)
        x = nn.functional.interpolate(x, scale_factor=1/2, mode='bicubic',align_corners=False)
        x = self.o_proj2(x)

        bias = x + bias

        return bias

