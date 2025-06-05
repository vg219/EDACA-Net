from functools import partial
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

import sys

sys.path.insert(1, "./")

from model.module.rwkv_v2 import (
    CrossScanTriton,
    CrossMergeTriton,
    CrossScan,
    CrossMerge,
    CrossScanTritonSelect,
    CrossMergeTritonSelect,
    CrossScanTritonSelectK1,
    CrossMergeTritonSelectK1,
    CrossScanK1Torch,
)
from model.module.rwkv_v2 import VRWKV_ChannelMix as RWKV_CMix
from model.module.rwkv_v2 import VRWKV_SpatialMix_V6 as RWKV_TMix
from model.module.layer_norm import LayerNorm
from model.base_model import BaseModel, register_model, PatchMergeModule

# from model.downstream_module import seg_plugger_cfg_joint, FusionSegmentationPlugger
from utils import easy_logger

logger = easy_logger()

# from utils.network_utils import get_local
# get_local.activate()


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
        scan_mode="K1",
        attn_bias=False,
        ffn_bias=False,
        cond_inj_mode='add',
        attn_groups=1,
        ffn_groups=1,
        *,
        swap_channel=False,
        no_out_norm=False,
        shift_type="q_shift",
        init_values=None,
        scan_id=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.swap_channel = swap_channel
        self.no_out_norm = no_out_norm
        self.cond_inj_mode = cond_inj_mode
        logger.info(
            f"layer_id: {layer_id}, scan_mode: {scan_mode}, cond_inj_mode: {cond_inj_mode}, " + \
            f"scan_id: {scan_id}, attn_ffn_groups: [{attn_groups}, {ffn_groups}], out_norm: {not no_out_norm}"
        )

        # if self.layer_id == 0:
        #     self.ln0 = nn.GroupNorm(1, n_embd)

        if not no_out_norm:
            self.out_norm = nn.GroupNorm(1, n_embd)  # LayerNorm(n_embd, "BiasFree")  # nn.GroupNorm(1, n_embd)

        # MIFM
        self.x_convs = nn.Sequential(
            # nn.GroupNorm(1, n_embd),
            # LayerNorm(n_embd, "BiasFree"),
            nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1,
                      groups=n_embd, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, 1, bias=False),
        )
        self.cond_convs = nn.Sequential(
            # LayerNorm(n_embd, "BiasFree"),
            nn.Conv2d(cond_chan, n_embd, kernel_size=1, bias=False),
            nn.Conv2d(
                n_embd, n_embd, kernel_size=3, padding=1, bias=False, groups=n_embd
            ),
        )

        self.MIFM_norm = nn.GroupNorm(1, n_embd)  # LayerNorm(n_embd, "BiasFree")
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        # bottleneck
        self.act_conv = nn.Sequential(
            nn.Conv2d(n_embd, n_embd // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(n_embd // 4, n_embd, 1, 1, 0),
            # nn.Sigmoid(),
        )
        self.act = nn.Sigmoid()
        
        if cond_inj_mode == 'cat':
            self.cat_cond_conv = nn.Conv2d(n_embd * 2, n_embd, 1, 1)

        if self.swap_channel and layer_id != 0:
            # self.swap_conv_feat = nn.Conv2d(n_embd, n_embd, 1, 1)
            self.swap_conv_cond = nn.Conv2d(n_embd, n_embd, 1, 1)

        assert scan_mode in [
            "K1",
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K1, K2, K4, K8]"
        K = int(scan_mode[1])
        self.K = K
        n_embd = n_embd * K
        dim_att = dim_att * K
        dim_ffn = dim_ffn * K

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

        self.x_ln1 = nn.GroupNorm(1, n_embd)  # LayerNorm(n_embd, 'BiasFree')
        self.x_ln2 = nn.GroupNorm(1, n_embd)  # LayerNorm(n_embd, 'BiasFree')

        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        if scan_mode == "K1":
            self.scan = lambda x: CrossScanK1Torch.scan(x, scan_id % 4)
            self.merge = lambda x: CrossScanK1Torch.merge(x, scan_id % 4)
        elif scan_mode == "K2":
            self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
            self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
        elif scan_mode == "K4":
            self.scan = CrossScanTriton.apply
            self.merge = CrossMergeTriton.apply
        else:
            self.scan = CrossScan.apply
            self.merge = CrossMerge.apply

        self.layer_scale = init_values is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((1, n_embd, 1)), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((1, n_embd, 1)), requires_grad=True
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, patch_resolution=None):
        b, c, h, w = x.shape
        inp = x
        T = h * w


        # first layer norm
        # if self.layer_id == 0:
        #     x = self.ln0(x)

        
        # first layer norm
        # if self.layer_id == 0:
        #     x = self.ln0(x)

        ## MIFM
        x = self.MIFM_norm(x)
        cond = self.cond_convs(cond)
        # swap channel
        if self.swap_channel and self.layer_id != 0:
            swap_c = self.n_embd // 2
            x = torch.cat([x[:, :swap_c], cond[:, swap_c:]], dim=1)
            cond = torch.cat([cond[:, :swap_c], x[:, swap_c:]], dim=1)
            # x = self.swap_conv_feat(x)
            cond = self.swap_conv_cond(cond)
        x_pool = self.adap_pool(x)
        x_act = self.act(self.act_conv(x_pool) * x)
        x = self.x_convs(x)
        x = x * x_act
        cond = cond * x_act

        ## scan to BRWKV
        if self.cond_inj_mode == 'cat':
            cat_x = torch.cat([x, cond], dim=1)
            scan_in = self.cat_cond_conv(cat_x)
        else:
            scan_in = x + cond
        x = self.scan(scan_in).view(b, -1, h * w)  # .permute(0, -1, 1)  # [b, 8, d, h, w]
        if self.layer_scale:
            # spatial mixing
            x = (
                self.drop_path(self.gamma1 * self.att(self.x_ln1(x), patch_resolution))
                + x
            )
            # Channel mixing
            x = (
                self.drop_path(self.gamma2 * self.ffn(self.x_ln2(x), patch_resolution))
                + x
            )
        else:
            # spatial mixing
            x = self.drop_path(self.att(self.x_ln1(x), patch_resolution)) + x
            # Channel mixing
            x = self.drop_path(self.ffn(self.x_ln2(x), patch_resolution)) + x

        # x = rearrange(x, "b (h w) (k d) -> b k d h w", k=self.K, h=h, w=w)
        x = rearrange(x, "b (k d) (h w) -> b k d h w", k=self.K, h=h, w=w)
        x = self.merge(x).view(b, c, h, w)
        if not self.no_out_norm:
            out = self.out_norm(x) + inp
        else:
            out = (x / self.K) + inp

        return out


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


class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


############# PanRWKV Model ################


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False
        )
        self.norm = norm_layer(4 * dim)
        
    def forward(self):
        pass


def down(chan, down_type="conv", r=2, chan_r=2):
    if down_type == "conv":
        return nn.Sequential(
            nn.Conv2d(chan, chan * chan_r, r, r, bias=False),
            # LayerNorm(chan * chan_r),
        )
    elif down_type == "patch_merge":
        assert False
        return PatchMerging2D(chan, chan * 2)
    else:
        raise NotImplementedError(f"down type {down_type} not implemented")


def up(chan, r=2, chan_r=2):
    return nn.Sequential(
        nn.Conv2d(chan, chan // chan_r, 1, bias=False),
        # nn.PixelShuffle(2),
        nn.Upsample(scale_factor=r, mode="bilinear"),
        # LayerNorm(chan // chan_r),
    )


@register_model("panRWKV_k1_v7")
class RWKVFusion(BaseModel):
    def __init__(
        self,
        img_channel=3,
        condition_channel=9,
        out_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        chan_upscales=[],
        upscale=1,
        if_abs_pos=False,
        if_rope=False,
        pt_img_size=64,
        drop_path_rate=0.1,
        patch_merge=True,
        fusion_prior: str = "max",
        # seg_plug=False,
        # seg_plug_dataset="joint",
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        self.if_rope = if_rope
        self.pt_img_size = pt_img_size
        self.fusion_prior = fusion_prior
        self.patch_merge = patch_merge

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

        if if_rope:
            self.rope = VisionRotaryEmbeddingFast(
                width, pt_seq_len=pt_img_size, ft_seq_len=None
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
                Sequential(
                    *[
                        RWKVBlock_v4(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            dim_att=chan,
                            dim_ffn=chan,
                            cond_chan=condition_channel,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + layer_id,
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
        self.middle_blks = Sequential(
            *[
                RWKVBlock_v4(
                    layer_id=n_prev_blks + i,
                    n_layer=depth,
                    n_embd=chan,
                    dim_att=chan,
                    dim_ffn=chan,
                    cond_chan=condition_channel,
                    drop_path=inter_dpr[n_prev_blks + i],
                    scan_id=n_prev_blks + i,  # n_prev_layers
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
            pt_img_size *= 2
            last_layer = dec_layer_id == len(dec_blk_nums) - 1
            self.skip_scales.append(
                nn.Parameter(torch.ones(1, chan, 1, 1), requires_grad=True)
            )

            self.decoders.append(
                Sequential(
                    nn.Conv2d(chan * 2, chan, 1, bias=True),
                    *[
                        RWKVBlock_v4(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            dim_att=chan,
                            dim_ffn=chan,
                            cond_chan=condition_channel,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + dec_layer_id,
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
        #     self.out_feat = False

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

    def alter_ropes(self, ft_img_size):
        if ft_img_size != self.pt_img_size and hasattr(self, "rope"):
            self.rope.alter_seq_len(ft_img_size, self.pt_seq_len)
            
    def resize_pos_embd(self,
                        pos_embd: torch.Tensor,
                        inp_size: tuple):
        if pos_embd.size(-1) != inp_size[-1] or pos_embd.size(-2) != inp_size[-2]:
            pos_embd = F.interpolate(pos_embd, inp_size[-2:], mode='bilinear', align_corners=False)
        
        return pos_embd

    def _forward_implem(self, inp, cond):
        x = inp
        *_, H, W = x.shape

        cat_x_cond = torch.cat([x, cond], dim=1)
        x = self.patch_embd(cat_x_cond)

        # U-Net conditioning
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
            self.alter_ropes(pan.shape[-1])
            sr = self._forward_implem(lms, pan) + lms
            self.alter_ropes(self.pt_img_size)

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
        # if self.out_feat:
        #     fused, feats = fused_outp
        # else:
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

        # if self.out_feat and mask is not None:
        #     # feats.append(fused)
        #     seg_mask = self.seg_plugger(feats)
        #     if seg_mask.shape[-2:] != mask.shape[-2:]:
        #         seg_mask = F.interpolate(seg_mask, mask.shape[-2:], mode='bilinear', align_corners=False)
        #     mask_in = mask.clone().detach()
        #     seg_loss = self.seg_plugger.loss(seg_mask, mask_in.long())
        #     loss[0] += seg_loss
        #     loss[1].update({'seg_loss': seg_loss})

        #     with torch.no_grad():
        #         # now we used MSRS, M3FD, LLVIP datasets all treat background as class 0

        #         # we need to reduce the mask to 0-8, for example, MSRS dataset
        #         # and the backgound to 255 (for train to ignore) or 9 (for visualization)

        #         ## reduce mask with backgroud class 0 to 9
        #         # bg_mask = mask == 0
        #         # mask[mask >= 1] = mask[mask >= 1] - 1
        #         # mask[bg_mask] = seg_mask.shape[1]  # to max class   # seg_mask shape as [bs, class(wo bg), h, w]

        #         ## add one dim for background
        #         # class channels: [bg, *obj_classes]

        #         # ver1: kornia metrics
        #         # seg_mask = torch.cat([torch.ones_like(seg_mask[:, :1]) * (- torch.inf), seg_mask], dim=1)
        #         # iou = mean_iou(seg_mask.argmax(1, keepdim=True), mask.long(), num_classes=seg_mask.shape[1])
        #         # iou_without_bg = iou[:, 1:]
        #         # # mask iou == 1. (maybe just no mask of this class)
        #         # iou_not_all_1_mask = iou_without_bg != 1.
        #         # iou_not_1 = iou_without_bg[iou_not_all_1_mask]
        #         # mean_iou_wo_bg = iou_not_1.mean()

        #         # bg_mask = mask == 0
        #         # mask[mask >= 1] = mask[mask >= 1] - 1
        #         # mask[bg_mask] = seg_mask.shape[1] - 1

        #         gt_mask_wo_bg = F.one_hot(mask.squeeze(1).long(), num_classes=seg_mask.shape[1])
        #         gt_mask_wo_bg = gt_mask_wo_bg.permute(0, 3, 1, 2)

        #         seg_mask = F.one_hot(seg_mask.argmax(1), num_classes=seg_mask.shape[1])
        #         seg_mask = seg_mask.permute(0, 3, 1, 2)

        #         mean_iou = MeanIoU(seg_mask.shape[1], per_class=True).to(seg_mask.device)
        #         mean_iou_wo_bg = mean_iou(seg_mask, gt_mask_wo_bg)[1:].mean()  # remove bg

        #         loss[1].update({'mean_iou': mean_iou_wo_bg})

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
        sr = self._forward_implem(lms, pan)
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
        middle_blk_num=1,
        enc_blk_nums=[3, 2, 1],
        dec_blk_nums=[3, 2, 1],
        chan_upscales=[2, 1, 1],
        pt_img_size=64,
        if_abs_pos=False,
        if_rope=False,
        upscale=4,
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
