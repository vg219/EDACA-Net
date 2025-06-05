from functools import partial
from typing import Literal
import numpy as np
from pyparsing import C
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
from torch.nn.init import trunc_normal_
import math
from timm.layers import DropPath, to_2tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys
sys.path.insert(1, "./")

from model.module import pos_embedding
from model.module.pos_embedding import interpolate_pos_embed_2d, pos_emb_sincos_2d_register
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
from model.module.layer_norm import LayerNorm
from model.base_model import BaseModel, register_model, PatchMergeModule

from utils import easy_logger

logger = easy_logger(func_name='RWKVFusion')

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "b (d r) ... -> b d r ...", r=2)
    x1, x2 = x.unbind(dim=2)
    x = torch.stack((-x2, x1), dim=2)
    return rearrange(x, "b d r ... -> b (d r) ...")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        logger.print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="pixel",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.freqs = freqs

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        self.seq_len = ft_seq_len

        self.alter_seq_len(ft_seq_len, pt_seq_len)

    def alter_seq_len(self, ft_seq_len, pt_seq_len):
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, self.freqs)
        # freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = (
            freqs.cos().cuda().permute(-1, 0, 1)
        )  # .view(-1, freqs.shape[-1]).cuda()
        freqs_sin = (
            freqs.sin().cuda().permute(-1, 0, 1)
        )  # .view(-1, freqs.shape[-1]).cuda()

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t):
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = (
                t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            )
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin

    def __repr__(self):
        return (
            f"VisionRotaryEmbeddingFast(seq_len={self.seq_len}, freqs_cos={tuple(self.freqs_cos.shape)}, "
            + f"freqs_sin={tuple(self.freqs_sin.shape)})"
        )


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Modulator(nn.Module):
    def __init__(self, dim, double=True):
        super().__init__()
        self.double = double
        
        self.multip = 6 if double else 3
        self.modulated = nn.Conv1d(dim, dim * self.multip, 1)
        
    def forward(self, x):
        return self.modulated(F.silu(x)).chunk(self.multip, dim=1)
    

class RWKVBlock_v4(nn.Module):
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
        ffn_groups=4,
        *,
        no_out_norm=False,
        shift_type="q_shift",
        scan_id=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.cond_inj_mode = cond_inj_mode
        logger.info(
            f"layer_id: {layer_id}, scan_mode: {scan_mode}, cond_inj_mode: {cond_inj_mode}, " + \
            f"scan_id: {scan_id}, attn_ffn_groups: [{attn_groups}, {ffn_groups}], out_norm: {not no_out_norm}"
        )

        if not no_out_norm:
            self.out_norm = LayerNorm(n_embd, 'BiasFree')  # nn.GroupNorm(1, n_embd)
            

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

        HEAD_SIZE = 32
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
            hidden_rate=1,
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

    def forward(self, inp: list[torch.Tensor, tuple]):
        x, patch_resolution = inp
        
        b, c, h, w = x.shape
        inp = x

        ## BRWKV
        # scan
        x = self.scan(x).view(b, -1, h * w)  # .permute(0, -1, 1)  # [b, 8, d, h, w]
        sc1, sh1, ga1, sc2, sh2, ga2 = self.modulator(x)
           
        # spatial mixing
        prenorm_x = self.x_ln1(x)
        mod_x_attn = (1 + sc1) * prenorm_x + sh1
        x = ga1 * self.drop_path(self.att(mod_x_attn, patch_resolution)) + x
        # Channel mixing
        prenorm_x = self.x_ln2(x)
        mod_x_ffn = (1 + sc2) * prenorm_x + sh2
        x = ga2 * self.drop_path(self.ffn(mod_x_ffn, patch_resolution)) + x

        x = rearrange(x, "b (k d) (h w) -> b k d h w", k=self.K, h=h, w=w)
        x = self.merge(x).view(b, c, h, w)
        
        if not self.no_out_norm:
            out = self.out_norm(x) + inp
        else:
            out = (x / self.K) + inp

        return [out, patch_resolution]



class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    
    
class Sequential(nn.Module):
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


############# PanRWKV Model ################

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    
class PatchUnEmbd(nn.Module):
    def __init__(self,
                 patch_size: int,
                 in_chan: int,
                 out_chan: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_unembd = nn.Sequential(
            LayerNorm(in_chan, 'BiasFree'),
            nn.Conv1d(in_chan, out_chan * patch_size ** 2, 1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)
        
        x = self.patch_unembd(x)
        return rearrange(x, 'b (c p1 p2) (h w) -> b c (h p1) (w p2)', 
                         p1=self.patch_size, p2=self.patch_size, h=H, w=W)
        

class PatchUnEmbdByUpsample(nn.Module):
    def __init__(self,
                 patch_size: int, 
                 in_chan: int,
                 out_chan: int,
                 mode: Literal['upsample', 'trans_conv']='upsample'):
        super().__init__()
        self.patch_size = patch_size
        if mode == 'upsample':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
                                          nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, bias=False))
        elif mode == 'trans_conv':
            self.upsample = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=patch_size, stride=patch_size)
        else:
            raise NotImplementedError(f"mode {mode} not implemented")
        
    def forward(self, x):
        B, L, C = x.size()
        h = w = int(math.sqrt(L))
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.upsample(x)
        
        return x
        
    
@register_model("panRWKV_plain_v6")
class RWKVFusion(BaseModel):
    def __init__(
        self,
        img_channel=3,
        condition_channel=9,
        out_channel=3,
        width=16,
        n_layers=6,
        blks_by_layers=[2,2,2,2,2,2],
        upscale=4,
        patch_size=4,
        by_pass_high_res=False,
        if_sin_cos_pos=False,
        if_learnable_abs_pos=False,
        pt_img_size=64,
        drop_path_rate=0.1,
        patch_merge=True,
        fusion_prior: str = "max",
        # seg_plug=False,
        # seg_plug_dataset="joint",
    ):
        super().__init__()
        self.upscale = upscale
        self.if_sin_cos_pos = if_sin_cos_pos
        self.if_learnable_abs_pos = if_learnable_abs_pos
        self.pt_img_size = pt_img_size
        self.fusion_prior = fusion_prior
        self.patch_size = patch_size
        self.patch_merge = patch_merge
        self.by_pass_high_res = by_pass_high_res

        assert fusion_prior in [
            "max",
            "mean",
            "none",
        ], "`fusion_prior` should be one of [max, mean, none]"
        logger.info(f"{__class__.__name__}: fusion_prior: {fusion_prior}")

        embd_h = pt_img_size // patch_size
        embd_w = pt_img_size // patch_size
        embd_len = embd_h * embd_w
        if if_sin_cos_pos:
            pos_embedding = pos_emb_sincos_2d_register(embd_h, embd_w, width)
            self.register_buffer('pos_embedding', pos_embedding)
        if if_learnable_abs_pos:
            self.learned_abs_pos = nn.Parameter(
                torch.zeros(1, embd_len, width), requires_grad=True
            )

        logger.print(f"{__class__.__name__}: {img_channel=}, {condition_channel=}\n")
        
        # assume we condition on intro (patch embedded patches)
        condition_channel = img_channel + condition_channel
        
        ## patch embedding first
        self.patch_embd = PatchEmbed(patch_size, condition_channel, width)

        ## plain backbone
        self.plain_backbone = nn.ModuleDict()
        
        prev_n_blks = 0
        n_blocks = sum(blks_by_layers)
        for layer_id in range(n_layers):
            self.plain_backbone[f"layer_{layer_id}"] = nn.Sequential(
                *[
                    RWKVBlock_v4(
                        layer_id=i + prev_n_blks,
                        n_layer=n_blocks,
                        n_embd=width,
                        cond_chan=condition_channel,
                        drop_path=drop_path_rate,
                        dim_att=width,
                        dim_ffn=width,
                        scan_mode="K2",
                        attn_bias=False,
                        ffn_bias=False,
                        no_out_norm=False,
                        shift_type="q_shift",
                        scan_id=i + prev_n_blks
                    )
                    for i in range(blks_by_layers[layer_id])
                ]
            )
            prev_n_blks += blks_by_layers[layer_id]
            
        self.patch_unembd = PatchUnEmbd(patch_size, width, out_channel)
            
        ## bypass with high-resolution conv modules
        
        
        # init
        logger.print(
            f"============= {__class__.__name__}: init network ================="
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        rng = torch.Generator().manual_seed(2024)

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
            trunc_normal_(m.weight, std=0.02, generator=rng)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _forward_implem(self, inp, cond):
        x = inp
        B, _, H, W = x.shape
        hp = H // self.patch_size
        wp = W // self.patch_size

        cat_x_cond = torch.cat([x, cond], dim=1)
        x = self.patch_embd(cat_x_cond)
        
        if self.if_sin_cos_pos:
            pos_embedding = interpolate_pos_embed_2d(self.pos_embedding, x.shape[1])
            x = x + pos_embedding

        if self.if_learnable_abs_pos:
            learned_abs_pos = interpolate_pos_embed_2d(self.learned_abs_pos, x.shape[1])
            x = x + learned_abs_pos

        x = x.permute(0, 2, 1).view(B, -1, hp, wp)
        
        for _, m in self.plain_backbone.items():
            x, _ = m([x, (hp, wp)])
            
        x = self.patch_unembd(x)

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

    def only_fusion_step(self, vis, ir):
        outp = self._forward_implem(vis, ir)
        if self.fusion_prior == "max":
            prior = torch.max(vis, ir)
        elif self.fusion_prior == "mean":
            prior = (vis + ir) / 2
        elif self.fusion_prior == "none":
            prior = 0.0

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
        mask: "torch.Tensor",
        gt: "torch.Tensor",
        fusion_criterion: callable,
        to_rgb_fn: callable = None,
        has_gt: bool = False,
    ):
        fused_outp = self.only_fusion_step(vis, ir)
        
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
        fused_outp = self.only_fusion_step(vis, ir)

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
        sr = self._forward_implem(lms, pan)  # sr[:,[29,19,9]]
        return sr


if __name__ == "__main__":
    from torch.cuda import memory_summary

    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    net = RWKVFusion(
        img_channel=8,
        condition_channel=1,
        out_channel=8,
        width=32,
        patch_size=4,
        pt_img_size=64,
        if_sin_cos_pos=True,
        upscale=1,
    ).to(device)

    scale = 4
    img_size = 64 // scale
    bs = 1

    img = torch.randn(bs, 8, img_size * scale, img_size * scale).to(device)
    cond = torch.randn(bs, 1, img_size * scale, img_size * scale).to(device)

    ### Count params, flops
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table

    # net.forward = net._forward_implem
    # flops_count = FlopCountAnalysis(net, (img, cond))

    # from model.module.rwkv import WKV_6, vrwkv6_flops

    # custom_ops = {"prim::PythonOp.WKV_6": vrwkv6_flops}
    # flops_count.set_op_handle(**custom_ops)

    # print(flop_count_table(flops_count))

    ### measure throughput
    # from utils import measure_throughput

    # throghtput = measure_throughput(net, [(3, img_size*scale, img_size*scale),
    #                                       (1, img_size*scale, img_size*scale)],
    #                                 32, num_warmup=20, num_iterations=200)

    # print(f'throughput: {throghtput} imgs/s')

    for _ in range(1):
        ms = torch.randn(bs, 8, img_size, img_size).to(device)
        img = torch.randn(bs, 8, img_size * scale, img_size * scale).to(device)
        cond = torch.randn(bs, 1, img_size * scale, img_size * scale).to(device)
        fusion_gt = torch.cat([img, cond], dim=1)
        mask = (
            torch.randint(0, 9, (bs, 1, img_size * scale, img_size * scale))
            .to(device)
            .float()
        )

        # sr = net.val_step(ms, img, cond)
        # print(sr.shape)

        # net = torch.compile(net)

        out = net._forward_implem(img, cond)
        # out.mean().backward()
        # print(out.shape)
        sr = torch.randn(1, 8, img_size * scale, img_size * scale).to(device)
        loss = F.mse_loss(out, sr)
        print(loss)
        loss.backward()

        # test segmentation heads and seg loss
        # from utils import get_loss

        # fusion_loss = get_loss('drmffusion').to(device)
        # fused, loss = net.fusion_train_step(img, cond, mask, fusion_gt, fusion_loss)
        # loss[0].backward()
        # print(loss)

        # test patch merge
        # sr = net.val_step(ms, img, cond)
        # print(sr.shape)

        # find unused params and big-normed gradient
        d_grads = {}
        n_params = 0
        for n, p in net.named_parameters():
            n_params += p.numel()
            if p.grad is None:
                print(n, "has no grad")
            else:
                p_sum = torch.abs(p.grad).sum().item()
                d_grads[n] = p_sum

        ## topk
        d_grads = dict(sorted(d_grads.items(), key=lambda item: item[1], reverse=True))
        for k, v in list(d_grads.items())[:20]:
            print(k, v)

        ## params
        print("total params:", n_params / 1e6, "M")
