from functools import partial
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
import torch.utils.checkpoint as cp
from timm.layers import DropPath

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from dataclasses import dataclass

import sys
sys.path.insert(1, "./")

from model.module.rwkv_v2 import (
    CrossScanTriton,
    CrossMergeTriton,
    CrossScan,
    CrossMerge,
    CrossScanTritonSelect,
    CrossMergeTritonSelect,
)
from model.module.rwkv_v2 import VRWKV_ChannelMix as RWKV_CMix
from model.module.rwkv_v2 import VRWKV_SpatialMix_V6 as RWKV_TMix
from model.module.rwkv_v2 import HEAD_SIZE, TIME_DECAY_DIM
from model.module.layer_norm import LayerNorm
from model.base_model import BaseModel, register_model, PatchMergeModule

from utils import easy_logger

logger = easy_logger(func_name='RWKVFusion_v9')


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    windows = rearrange(x, 'b c (nh p1) (nw p2) -> (b nh nw) c p1 p2', p1=window_size, p2=window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = rearrange(windows, '(b nh nw) c p1 p2 -> b c (nh p1) (nw p2)', p1=window_size, p2=window_size, b=B, 
                  c=windows.shape[1], nh=H // window_size, nw=W // window_size)
    return x

    
class Modulator(nn.Module):
    def __init__(self, dim, double=True):
        super().__init__()
        self.double = double
        
        self.multip = 6 if double else 3
        self.modulated = nn.Conv1d(dim, dim * self.multip, 1)
        
    def forward(self, x):
        return self.modulated(F.silu(x)).chunk(self.multip, dim=1)
    
        
class SingleStreamRWKVBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        n_layer=8,
        n_embd=64,
        cond_chan=1,
        drop_path=0.0,
        dim_att=64,
        dim_ffn=64,
        scan_mode="K2",
        attn_bias=True,
        ffn_bias=True,
        cond_inj_mode='add',
        attn_groups=4,
        ffn_hidden_rate=1,
        ffn_groups=4,
        *,
        no_out_norm=True,
        shift_type="q_shift",
        scan_id=0,
        window_size=8,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.cond_inj_mode = cond_inj_mode
        self.window_size = window_size
        self.roll_size = window_size // 2 if (scan_id % 2 == 0 and window_size > 0) else 0
        
        # block config
        logger.info(
            f"layer_id: {layer_id}, scan_mode: {scan_mode}, cond_inj_mode: {cond_inj_mode}, window_size: {window_size}, ",
            f"scan_id: {scan_id}, roll_size: {self.roll_size}, attn_ffn_groups: [{attn_groups}, {ffn_groups}], out_norm: {not no_out_norm}"
        )

        if not no_out_norm:
            self.out_norm = LayerNorm(n_embd, 'BiasFree')  # nn.GroupNorm(1, n_embd)
                        
        # MIFM
        self.x_convs = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1,
                      groups=n_embd, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, 1, bias=False),
        )
        self.cond_convs = nn.Conv2d(cond_chan, n_embd, kernel_size=3, padding=1)

        self.MIFM_norm = LayerNorm(n_embd, "BiasFree")
        self.adap_pool = nn.AdaptiveMaxPool2d(1)
        # bottleneck
        self.act_conv = nn.Sequential(
            nn.Conv2d(n_embd, n_embd // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(n_embd // 4, n_embd, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.act = nn.Sequential(nn.Conv2d(n_embd, n_embd, 1),
                                 nn.SiLU())
        
        if cond_inj_mode == 'cat':
            self.cat_cond_conv = nn.Conv2d(n_embd * 2, n_embd, 1, 1)

        ############# BRWKV #############
        assert scan_mode in [
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K2, K4, K8]"
        K = int(scan_mode[1])
        self.K = K
        n_embd = n_embd * K
        dim_att = dim_att * K
        dim_ffn = dim_ffn * K
        
        self.modulator = Modulator(n_embd)

        N_HEAD = n_embd // HEAD_SIZE
        self.att = RWKV_TMix(
            dim_att,
            N_HEAD,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            n_groups=attn_groups,
            attn_bias=attn_bias,
        )
        self.ffn = RWKV_CMix(
            dim_ffn,
            N_HEAD,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            hidden_rate=ffn_hidden_rate,
            n_groups=ffn_groups,
            ffn_bias=ffn_bias,
        )

        self.x_ln1 = LayerNorm(n_embd, 'WithBias')
        self.x_ln2 = LayerNorm(n_embd, 'WithBias')

        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        if scan_mode == "K2":
            self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
            self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
        elif scan_mode == "K4":
            self.scan = CrossScanTriton.apply
            self.merge = CrossMergeTriton.apply
        else:
            self.scan = CrossScan.apply
            self.merge = CrossMerge.apply
            
    def MIFM_forward(self, x, cond):
        ## MIFM
        x = self.MIFM_norm(x)
        x_pool = self.adap_pool(x)
        x_mul = self.act(self.act_conv(x_pool) * x)
        x_conv = self.x_convs(x)
        x_act = x_conv * x_mul
        cond_act = cond * x_mul

        ## conditional concanation or addition
        if self.cond_inj_mode == 'cat':
            cat_x = torch.cat([x_act, cond_act], dim=1)
            mifm_x = self.cat_cond_conv(cat_x)
        else:
            mifm_x = x_act + cond_act
            
        return mifm_x

    def forward(self, x: torch.Tensor, cond: torch.Tensor, patch_resolution=None):
        B, C, H, W = x.shape
        inp = x

        cond = self.cond_convs(cond)
        
        ## MIFM
        x = self.MIFM_forward(x, cond)
        
        ## BRWKV
        # window partition
        if self.window_size > 0:
            if self.roll_size > 0:
                x = x.roll(shifts=(-self.roll_size, -self.roll_size), dims=(2, 3))
            x = window_partition(x, self.window_size)
            h, w = self.window_size, self.window_size
            patch_resolution = (self.window_size, self.window_size)
        else:
            h, w = H, W
            
        # ESS
        x = self.scan(x).view(x.size(0), -1, h * w)
        sc1, sh1, ga1, sc2, sh2, ga2 = self.modulator(x)
           
        # Spatial mixing
        prenorm_x = self.x_ln1(x)
        mod_x_attn = (1 + sc1) * prenorm_x + sh1
        x = ga1 * self.drop_path(self.att(mod_x_attn, patch_resolution)) + x
        
        # Channel mixing
        prenorm_x = self.x_ln2(x)
        mod_x_ffn = (1 + sc2) * prenorm_x + sh2
        x = ga2 * self.drop_path(self.ffn(mod_x_ffn, patch_resolution)) + x

        # merge ESS
        x = rearrange(x, "b (k d) (h w) -> b k d h w", k=self.K, h=h, w=w)
        x = self.merge(x).view(x.size(0), C, h, w)
        
        # window reverse
        if self.window_size > 0:
            x = window_reverse(x, self.window_size, H, W)
            if self.roll_size > 0:
                x = x.roll(shifts=(self.roll_size, self.roll_size), dims=(2, 3))
        
        # norm output
        if not self.no_out_norm:
            out = self.out_norm(x) + inp
        else:
            out = (x / self.K) + inp

        return out


class FusionSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.mods = nn.ModuleList(args)
 
    def __getitem__(self, idx):
        return self.mods[idx]

    def enc_forward(self, feat, cond, patch_resolution):
        outp = feat
        for mod in self.mods:
            outp = mod(outp, cond, patch_resolution)
        return outp.contiguous() + feat

    def dec_forward(self, feat, cond, patch_resolution):
        outp = self.mods[0](feat)
        feat = outp
        for mod in self.mods[1:]:
            outp = mod(outp, cond, patch_resolution)
        return outp.contiguous() + feat


############# RWKVFusion Model ################
def down(chan, down_type="conv", r=2, chan_r=2):
    if down_type == "conv":
        return nn.Sequential(
            nn.Conv2d(chan, chan * chan_r, r, r, bias=False),
            # LayerNorm(chan * chan_r),
        )
    else:
        raise NotImplementedError(f"down type {down_type} not implemented")


def up(chan, r=2, chan_r=2):
    return nn.Sequential(
        nn.Conv2d(chan, chan // chan_r, 1, bias=False),
        # nn.PixelShuffle(2),
        nn.Upsample(scale_factor=r, mode="bilinear"),
        # LayerNorm(chan // chan_r),
    )


@register_model("panRWKV_v9_local")
class RWKVFusion(BaseModel):
    def __init__(
        self,
        img_channel: int=3,
        condition_channel: int=1,
        out_channel: int=3,
        width: int=16,
        middle_blk_num: int=1,
        mid_window_size: int=0,
        enc_blk_nums: list=[],
        dec_blk_nums: list=[],
        chan_upscales: list=[],
        window_sizes: list=[],
        attn_groups: list=[],
        ffn_groups: list=[],
        ffn_hidden_rates: list=[],
        upscale: int=1,
        if_abs_pos: bool=False,
        if_rope: bool=False,
        pt_img_size: int=64,
        drop_path_rate: float=0.1,
        fusion_prior: Literal['max', 'mean', 'none'] = "max",
        patch_merge: bool=True,
        use_mask_as_cond: bool=False,
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        self.if_rope = if_rope
        self.pt_img_size = pt_img_size
        self.fusion_prior = fusion_prior
        self.patch_merge = patch_merge
        self.use_mask_as_cond = use_mask_as_cond

        assert fusion_prior in [
            "max",
            "mean",
            "none",
        ], "`fusion_prior` should be one of [max, mean, none]"
        logger.info(f"{__class__.__name__}: fusion_prior: {fusion_prior}")

        if if_abs_pos:
            self.abs_pos = nn.Parameter(
                torch.randn(1, width, pt_img_size, pt_img_size), requires_grad=True
            )

        logger.print(f"{__class__.__name__}: {img_channel=}, {condition_channel=}\n")
        self.patch_embd = nn.Conv2d(
            in_channels=img_channel + condition_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=True,
        )
        # assume we condition on intro (patch embedded patches)
        condition_channel = img_channel + condition_channel
        if use_mask_as_cond:
            condition_channel += 1

        self.proj_out = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        ## main body
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        depth = sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)
        n_layers = len(enc_blk_nums) + 1 + len(dec_blk_nums)
        inter_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        chan = width
        n_prev_blks = 0
        n_prev_layers = 0
        # encoder
        for layer_id, num in enumerate(enc_blk_nums):
            self.encoders.append(
                FusionSequential(
                    *[
                        SingleStreamRWKVBlock(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            dim_att=chan,
                            dim_ffn=chan,
                            window_size=window_sizes[layer_id],
                            cond_chan=condition_channel,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + layer_id,
                            attn_groups=attn_groups[layer_id],
                            ffn_groups=ffn_groups[layer_id],
                            ffn_hidden_rate=ffn_hidden_rates[layer_id],
                        )
                        for i in range(num)
                    ]
                )
            )
            self.downs.append(down(chan, r=2, chan_r=chan_upscales[layer_id]))
            chan = chan * chan_upscales[layer_id]
            n_prev_blks += num
            pt_img_size //= 2
        n_prev_layers += len(enc_blk_nums)

        # middle layer
        self.middle_blks = FusionSequential(
            *[
                SingleStreamRWKVBlock(
                    layer_id=n_prev_blks + i,
                    n_layer=depth,
                    n_embd=chan,
                    dim_att=chan,
                    dim_ffn=chan,
                    window_size=mid_window_size,
                    cond_chan=condition_channel,
                    drop_path=inter_dpr[n_prev_blks + i],
                    scan_id=n_prev_blks + i,  # n_prev_layers
                    attn_groups=attn_groups[-1],
                    ffn_groups=ffn_groups[-1],
                    ffn_hidden_rate=ffn_hidden_rates[-1],
                )
                for i in range(middle_blk_num)
            ]
        )
        n_prev_blks += middle_blk_num
        n_prev_layers += 1

        self.skip_scales = nn.ParameterList([])
        # decoder
        for dec_layer_id, num in enumerate(reversed(dec_blk_nums)):
            self.ups.append(up(chan, chan_r=chan_upscales[::-1][dec_layer_id]))
            chan = chan // chan_upscales[::-1][dec_layer_id]
            window_size = window_sizes[::-1][dec_layer_id]
            attn_group = attn_groups[::-1][dec_layer_id]
            ffn_group = ffn_groups[::-1][dec_layer_id]
            ffn_hidden_rate = ffn_hidden_rates[::-1][dec_layer_id]
            pt_img_size *= 2
            last_layer = dec_layer_id == len(dec_blk_nums) - 1
            self.skip_scales.append(
                nn.Parameter(torch.ones(1, chan, 1, 1), requires_grad=True)
            )

            self.decoders.append(
                FusionSequential(
                    nn.Conv2d(chan * 2, chan, 1, bias=True),
                    *[
                        SingleStreamRWKVBlock(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            dim_att=chan,
                            dim_ffn=chan,
                            window_size=window_size,
                            cond_chan=condition_channel,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + dec_layer_id,
                            attn_groups=attn_group,
                            ffn_groups=ffn_group,
                            ffn_hidden_rate=ffn_hidden_rate,
                            # no_out_norm=last_layer and i == num - 1,
                        )
                        for i in range(num)
                    ],
                )
            )
            n_prev_blks += num
        n_prev_layers += len(dec_blk_nums)

        # self.padder_size = 2 ** len(self.encoders)

        # segmentation plugger
        # if seg_plug:
        #     assert seg_plug_dataset == 'joint'
        #     chan_prod = np.cumprod([width * chan_upscales[0]] + chan_upscales[1:])
        #     chan_prod = np.insert(chan_prod, 0, out_channel)
        #     logger.info(f'Segmentaion plgger: feature chans {chan_prod} ')
        #     self.seg_plugger = FusionSegmentationPlugger(**seg_plugger_cfg_joint(chan_prod.tolist(), output_size=128))
        #     self.out_feat = True
        # else:
        self.out_feat = False

        # init
        logger.print(
            f"============= {__class__.__name__}: init network ================="
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        rng = torch.Generator().manual_seed(2025)
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, generator=rng)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.body.weight, 1.0)
            if hasattr(m.body, "bias"):
                nn.init.constant_(m.body.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu", generator=rng)
            trunc_normal_(m.weight, std=0.02, generator=rng)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def resize_pos_embd(self,
                        pos_embd: torch.Tensor,
                        inp_size: tuple):
        if pos_embd.size(-1) != inp_size[-1] or pos_embd.size(-2) != inp_size[-2]:
            pos_embd = F.interpolate(pos_embd, inp_size[-2:], mode='bilinear', align_corners=False)
        
        return pos_embd

    def _forward_implem(self, inp, cond, mask=None):
        x = inp
        bs, _, H, W = x.shape

        cat_x_cond = torch.cat([x, cond], dim=1)
        x = self.patch_embd(cat_x_cond)

        # U-Net conditioning
        if mask is not None and self.use_mask_as_cond:
            u_cond = torch.cat([cat_x_cond, mask], dim=1)
        elif mask is None and self.use_mask_as_cond:
            u_cond = torch.cat([cat_x_cond, torch.zeros(bs, 1, H, W).to(x)], dim=1)
        else:
            u_cond = cat_x_cond

        if self.if_abs_pos:
            x = x + self.resize_pos_embd(self.abs_pos, inp_size=(H, W))
            
        if self.if_rope:
            x = self.rope(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            cond1 = F.interpolate(
                u_cond, (H, W), mode="bilinear", align_corners=False
            )
            x = encoder.enc_forward(x, cond1, (H, W))
            encs.append(x)
            x = down(x)
            H = H // 2
            W = W // 2

        cond2 = F.interpolate(
            u_cond, (H, W), mode="bilinear", align_corners=False,
        )
        x = self.middle_blks.enc_forward(x, cond2, (H, W))

        # if self.out_feat:
        #     feat = []
        for decoder, up, enc_skip, skip_scale in zip(
            self.decoders, self.ups, encs[::-1], self.skip_scales
        ):
            x = up(x)
            H = H * 2
            W = W * 2
            cond3 = F.interpolate(
                u_cond, (H, W), mode="bilinear", align_corners=False,
            )
            x = torch.cat([x, enc_skip * skip_scale], dim=1)
            x = decoder.dec_forward(x, cond3, (H, W))

            # if self.out_feat:
            #     feat.append(x)

        x = self.proj_out(x)

        # if self.out_feat:
        #     # feat.append(x)
        #     return x, feat
        return x

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(lms, pan) + lms
        loss = criterion(sr, gt)

        return sr, loss

    @torch.no_grad()
    def val_step(self, ms, lms, pan, patch_merge=None):
        if patch_merge is None:
            patch_merge = self.patch_merge

        if patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16, 16 * self.upscale, 16 * self.upscale],
                scale=self.upscale,
                patch_merge_step=self.patch_merge_step,
            )
            sr = _patch_merge_model.forward_chop(ms, lms, pan)[0] + lms
        else:
            sr = self._forward_implem(lms, pan) + lms

        return sr

    def only_fusion_step(self, vis, ir, mask=None):
        outp = self._forward_implem(vis, ir, mask)
        if self.fusion_prior == "max":
            prior = torch.max(vis, ir)
        elif self.fusion_prior == "mean":
            prior = (vis + ir) / 2
        elif self.fusion_prior == "none":
            prior = 0.0
        else:
            raise ValueError(f"Invalid fusion_prior: {self.fusion_prior}")

        if self.out_feat:
            fused, feats = outp
            fused = fused + prior
            feats.append(fused)

            return fused.clip(0, 1), feats
        else:
            return (outp + prior).clip(0, 1)

    def fusion_train_step(
        self,
        vis: "torch.Tensor",
        ir: "torch.Tensor",
        mask: "torch.Tensor"=None,
        gt: "torch.Tensor"=None,
        fusion_criterion: callable=None,
        to_rgb_fn: callable = None,
        has_gt: bool = False,
    ):
        if not self.use_mask_as_cond:
            mask = None
        elif mask is not None:
            mask = mask.to(vis)
        fused_outp = self.only_fusion_step(vis, ir, mask)
        fused = fused_outp

        if mask is not None:
            mask_in = mask.clone().detach()
        else:
            mask_in = None

        fused_for_loss = to_rgb_fn(fused) if to_rgb_fn is not None else fused
        if has_gt:
            fusion_gt = gt
            boundary_gt = (vis, ir)
        else:
            fusion_gt = None
            boundary_gt = gt
        loss = list(
            fusion_criterion(
                fused_for_loss,
                boundary_gt=boundary_gt,
                fusion_gt=fusion_gt,
                mask=mask_in,
            )
        )

        return fused.clip(0, 1), loss

    @torch.no_grad()
    def fusion_val_step(
        self,
        vis,
        ir,
        mask,
        *,
        patch_merge=False,
        ret_seg_map: bool = False,
        to_rgb_fn: callable = None,
    ):
        if not self.use_mask_as_cond:
            mask = None
        elif mask is not None:
            mask = mask.to(vis)
        fused_outp = self.only_fusion_step(vis, ir, mask)

        if self.out_feat:
            fused, feats = fused_outp
        else:
            fused = fused_outp

        fused = to_rgb_fn(fused) if to_rgb_fn is not None else fused

        if self.out_feat:
            if ret_seg_map:
                seg_map = self.seg_plugger(feats)
                seg_map = seg_map.argmax(dim=1)[:, None]
                if seg_map.shape[-2:] != fused.shape[-2:]:
                    seg_map = F.interpolate(
                        seg_map.float(),
                        fused.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

        if self.out_feat and ret_seg_map:
            return fused.clip(0, 1), seg_map[:, 0].long()
        else:
            return fused.clip(0, 1)

    def patch_merge_step(self, ms, lms, pan, **kwargs):
        sr = self._forward_implem(lms, pan)
        return sr
    


if __name__ == "__main__":
    from torch.cuda import memory_summary
    
    class Tester:
        def __init__(self,
                     bs,
                     img_size,
                     task,
                     with_mask=False,
                     device='cuda:0'
                     ):
            # cfg 1 for pansharpening and HMIF
            cfg_pan = dict(
                    img_channel=8,
                    condition_channel=1,
                    out_channel=8,
                    width=32,
                    middle_blk_num=1,
                    enc_blk_nums=[2, 1, 1],
                    dec_blk_nums=[2, 1, 1],
                    chan_upscales=[2, 1, 1],
                    mid_window_size=8,
                    window_sizes=[16, 16, 16],
                    attn_groups=[1, 2, 2],
                    ffn_groups=[1, 2, 2],
                    ffn_hidden_rates=[2,2,1],
                    pt_img_size=64,
                    drop_path_rate=0.1,
                    if_abs_pos=False,
                    if_rope=False,
                    upscale=1,
                    use_mask_as_cond=False
                )
            
            # cfg 2 for VIF
            cfg_vif = dict(
                    img_channel=3,
                    condition_channel=1,
                    out_channel=3,
                    width=32,
                    middle_blk_num=1,
                    enc_blk_nums=[1, 1],
                    dec_blk_nums=[1, 1],
                    chan_upscales=[1, 1],
                    mid_window_size=8,
                    window_sizes=[16, 16],
                    attn_groups=[2, 2],
                    ffn_groups=[2, 2],
                    ffn_hidden_rates=[2,2,1],
                    pt_img_size=64,
                    if_abs_pos=False,
                    if_rope=False,
                    upscale=1,
                    use_mask_as_cond=False
                )
            cfgs = {'pan': cfg_pan, 'vif': cfg_vif}
            
            self.cfg = cfgs[task]
            self.bs = bs
            self.img_size = img_size
            self.task = task
            self.device = device
            self.with_mask = with_mask
            
            self.scale = self.cfg['upscale']
            self.channel_img = {'pan': 8, 'vif': 3}[task]
            self.channel_cond = {'pan': 1, 'vif': 1}[task]
            
            self._logger = easy_logger(func_name='model_test')
            self.print = self._logger.info
            self.info = self._logger.info
            self.debug = self._logger.debug
            self.warning = self._logger.warning
            self.error = self._logger.error
            
            ## init model
            self.net = RWKVFusion(**self.cfg).to(device)
        
        def prepare_data(self):
            ## test forward and backward
            img = torch.randn(self.bs, self.channel_img, self.img_size, self.img_size).to(device)
            cond = torch.randn(self.bs, self.channel_cond, self.img_size, self.img_size).to(device)
            
            if self.with_mask:
                mask = torch.randint(0, 9, (self.bs, 1, self.img_size, self.img_size)).to(img)
            else:
                mask = None
                
            return img, cond, mask
        
        def summary_mem(self):
            ## memory usage
            self.print("Memory usage:")
            self.print(memory_summary(device=device, abbreviated=True))
        
        def params_and_flops(self):
            ## Count params, flops
            from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table
            from model.module.rwkv_v2 import WKV_6, vrwkv6_flops

            self.net.forward = self.net._forward_implem
            img, cond, mask = self.prepare_data()
            flops_count = FlopCountAnalysis(self.net, (img, cond, mask))

            custom_ops = {"prim::PythonOp.WKV_6": vrwkv6_flops}
            flops_count.set_op_handle(**custom_ops)

            self.print(flop_count_table(flops_count))
            
        @torch.no_grad()
        def measure_throughput(self):
            assert not self.with_mask, '`with_mask` is not allowed in throughput measurement'
            self.info(f'measure throughput for task: {self.task}')
            
            ## measure throughput
            from utils import measure_throughput

            self.net.forward = self.net._forward_implem
            throghtput = measure_throughput(self.net, [(self.channel_img, self.img_size, self.img_size),
                                                    (self.channel_cond, self.img_size, self.img_size,)],
                                            16, num_warmup=10, num_iterations=50)

            self.print(f'throughput: {throghtput} imgs/s')
            
        def test_forward(self):
            img, cond, mask = self.prepare_data()
            out = self.net._forward_implem(img, cond, mask)
            sr = torch.randn(1, self.channel_img, self.img_size, self.img_size).to(device)
            loss = F.mse_loss(out, sr)
            logger.print(loss)
            loss.backward()
            
            # find unused params and big-normed gradient
            d_grads = {}
            n_params = 0
            for n, p in self.net.named_parameters():
                n_params += p.numel()
                if p.grad is None:
                    print(n, "has no grad")
                else:
                    p_sum = torch.abs(p.grad).sum().item()
                    d_grads[n] = p_sum

            ## topk
            d_grads = dict(sorted(d_grads.items(), key=lambda item: item[1], reverse=True))
            for k, v in list(d_grads.items())[:20]:
                self.print(k, v)

            ## params
            self.print("total params:", n_params / 1e6, "M")
        
        
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)   
    tester = Tester(1, 64, 'pan', with_mask=False, device=device)
    # tester.params_and_flops()
    tester.test_forward()
    # tester.measure_throughput()

