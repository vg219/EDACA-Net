from functools import partial
from re import T
from typing import Literal, Callable
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
from typeguard import typechecked

import sys
sys.path.insert(1, "./")

from model.module.rwkv_v4_multi_modal import (
    CrossScanTriton,
    CrossMergeTriton,
    CrossScan,
    CrossMerge,
    CrossScanTritonSelect,
    CrossMergeTritonSelect,
)
from model.module.rwkv_v4_multi_modal import VRWKV_ChannelMix as RWKV_CMix
from model.module.rwkv_v4_multi_modal import VRWKV_SpatialMix_V6 as RWKV_TMix
from model.module.rwkv_v4_multi_modal import HEAD_SIZE, TIME_DECAY_DIM
from model.module.layer_norm import LayerNorm
from model.module.pos_embedding import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from model.base_model import BaseModel, register_model
from model.module import PatchMergeModule

from utils import easy_logger

logger = easy_logger(func_name='RWKVFusion_v10', level='INFO')


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
    

# TODO: add rope positional encoding
    

@dataclass
class ConditionInput:
    modalities: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None     # cat two modalities
    llm_feature: torch.Tensor = None                                        # llm feature
    mask_input: torch.Tensor = None                                         # mask input

    def __post_init__(self):
        assert self.modalities is not None, "two modalities must be involved"
        if isinstance(self.modalities, tuple):
            assert len(self.modalities) == 2, "two modalities must be involved"
            self.modalities = torch.cat(self.modalities, dim=1)
            
    def __repr__(self):
        return f"modalities: {self.modalities.shape}, " + \
               f"llm_feature: {self.llm_feature.shape if self.llm_feature is not None else None}, " + \
               f"mask_input: {self.mask_input.shape if self.mask_input is not None else None}, "
            
            
class MultiModalityFusion(nn.Module):
    """
    MIFM for U-Net conditions pre-fusion
    """
    def __init__(self,
                 n_embd: int,
                 modal_chan: int,
                 mask_chan: int=None,
                 llm_chan: int=None,
                 feat_drop: float = 0.0,
                 llm_drop: float = 0.0,
                 llm_pe_type: Literal['none', 'sinusoidal']='none',
                 img_pe_type: Literal['none', 'sinusoidal']='none',
                 scan_mode: str='K2',
                 *,
                 # multi-modal RWKV configure
                 layer_id: int=0,
                 n_layer: int=8,
                 scan_id: int=0,
                 shift_type: str='q_shift',
                 attn_groups: int=4,
                 attn_bias: bool=False,
                 img_ds_r_by_llm: int=4,
                 multi_modal_mlp_hidden_ratio: int=2,
                 add_mm_tokens: bool=False,
                 checkpoint: bool=False,
                 ):
        super().__init__()
        
        self.n_embd = n_embd
        self.modal_chan = modal_chan
        self.mask_chan = mask_chan
        self.llm_chan = llm_chan
        self.img_ds_r_by_llm = img_ds_r_by_llm
        self.add_mm_tokens = add_mm_tokens
        self.with_mask = False
        self.with_llm_feat = False
        
        assert self.img_ds_r_by_llm >= 1, "img_ds_r_by_llm should be greater than or equal to 1"
        
        # previous feature encoder
        self.feat_norm = LayerNorm(n_embd, "BiasFree")
        self.feat_convs = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1,
                      groups=n_embd, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, 1, bias=False),
        )
        
        # modalities encoder
        self.modal_convs = nn.Conv2d(modal_chan, n_embd, kernel_size=3, padding=1)
        self.modal_norm = LayerNorm(n_embd, "BiasFree")
        self.adap_pool = nn.AdaptiveMaxPool2d(1)
        
        # modals bottleneck
        self.act_conv = nn.Sequential(
            nn.Conv2d(n_embd, n_embd // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(n_embd // 4, n_embd, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.act = nn.Sequential(nn.Conv2d(n_embd, n_embd, 1),
                                 nn.SiLU())
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0.0 else nn.Identity()
        
        # mask prompt encoder
        if mask_chan is not None:
            self.with_mask = True
            # progessive encoding
            self.mask_convs = nn.Sequential(
                nn.Conv2d(mask_chan, n_embd // 2, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(n_embd // 2, n_embd, kernel_size=3, padding=1, groups=n_embd // 2, bias=False),
                LayerNorm(n_embd, "BiasFree"),
                nn.SiLU(),
            )
            
        # llm feature encoder
        if llm_chan is not None:
            self.with_llm_feat = True
            
            K = int(scan_mode[1])
            
            # use <img> and <txt> token
            if self.add_mm_tokens:
                assert not add_mm_tokens, 'not support.'
                self.img_token = nn.Parameter(torch.zeros(2, 1, n_embd * K))
                self.txt_token = nn.Parameter(torch.zeros(2, 1, n_embd * K))
            
            # linear head for llm feature
            self.llm_dense = nn.Sequential(
                LayerNorm(llm_chan, 'BiasFree'),  # pre-norm
                nn.Conv1d(llm_chan, n_embd * K, 1)
            )
            self.llm_drop = nn.Dropout(llm_drop) if llm_drop > 0.0 else nn.Identity()
            assert scan_mode in [
                "K2",
                "K4",
                "K8",
            ], "scan_mode should be one of [K2, K4, K8]"
            self.K = K    
            n_embd = n_embd * K
            dim_att = n_embd
            
            # multi-modal RWKV attention
            N_HEAD = n_embd // HEAD_SIZE
            self.multi_modal_attn = RWKV_TMix(
                dim_att,
                N_HEAD,
                n_layer,
                layer_id,
                shift_mode=shift_type,
                n_groups=attn_groups,
                attn_bias=attn_bias,
                with_cp=checkpoint,
            )
            self.multi_modal_img_ln = LayerNorm(n_embd, 'BiasFree')
            self.multi_modal_txt_ln = LayerNorm(n_embd, 'BiasFree')
            
            # image and text mlps
            n_embd = n_embd // K
            hidden_size = n_embd * multi_modal_mlp_hidden_ratio
            self.img_mlp = nn.Sequential(
                nn.Conv2d(n_embd, hidden_size, 1, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Conv2d(hidden_size, n_embd, 1, bias=True),
            )
            self.txt_mlp = nn.Sequential(
                nn.Conv1d(n_embd * K, hidden_size * K, 1, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Conv1d(hidden_size * K, n_embd * K, 1, bias=True),   
            )
            
            # ESS
            if scan_mode == "K2":
                self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
                self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
            elif scan_mode == "K4":
                self.scan = CrossScanTriton.apply
                self.merge = CrossMergeTriton.apply
            else:
                self.scan = CrossScan.apply
                self.merge = CrossMerge.apply
                
        # postional encoding
        self.img_pe_type = img_pe_type
        self.llm_pe_type = llm_pe_type
        
    def cat_special_tokens_to_seqs(self, img: torch.Tensor, txt: torch.Tensor):
        # add special tokens to sequences
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        
        assert C_t == C_i, "modalities should have the same channels"
        
        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"
        
        soi = self.img_token[0].repeat(bs_img, 1, 1)
        eoi = self.img_token[1].repeat(bs_img, 1, 1)
        
        sot = self.txt_token[0].repeat(bs_txt, 1, 1)
        soi = self.txt_token[1].repeat(bs_txt, 1, 1)
        
        return torch.cat([soi, img, eoi,
                          sot, txt, soi], dim=-1)
            
    def add_img_pe(self, img):
        bs, _, h, w = img.shape
        
        if self.img_pe_type == 'sinusoidal':
            img_pe = get_2d_sincos_pos_embed(self.n_embd, (h, w)).unsqueeze(0).repeat(bs, 1, 1, 1)
            # add pe
            img = img + img_pe.to(img.device)
        
        return img
    
    def add_llm_pe(self, llm_txt):
        bs, l, d = llm_txt.shape if llm_txt is not None else None, None, None
        
        if self.llm_pe_type == 'sinusoidal' and llm_txt is not None:
            llm_ids = torch.arange(l, device=llm_txt.device).float()
            llm_pe = get_1d_sincos_pos_embed_from_grid(self.n_embd, llm_ids)
            # add pe
            llm_txt = llm_txt + llm_pe.unsqueeze(0).repeat(bs, 1, 1)
            
        return llm_txt
    
    def extract_img_and_llm_feat(self, h: int, w: int, att_or_ffn_out: torch.Tensor, has_llm: bool):
        # att_or_ffn_out: [bs, K, C, L]
        if has_llm:
            img_feat, llm_feat = att_or_ffn_out[..., :h*w], att_or_ffn_out[...,h*w:]
        else:
            img_feat = att_or_ffn_out
            llm_feat = None
            
        return img_feat, llm_feat
    
    def forward(self, img_feat: torch.Tensor, cond_input: ConditionInput):
        modals = cond_input.modalities
        mask = cond_input.mask_input
        llm_feat = cond_input.llm_feature
        
        # feature encoder
        img_feat = self.add_img_pe(img_feat)
        img_feat = self.feat_norm(img_feat)
        feat_pool = self.adap_pool(img_feat)
        feat_mul = self.act(self.act_conv(feat_pool) * img_feat)
        
        img_feat = self.feat_convs(img_feat)
        img_feat = self.feat_drop(img_feat)
        img_feat = img_feat * feat_mul
        
        # modalities encoder
        modals = self.modal_convs(modals)
        modals = self.modal_norm(modals)
        modals = modals * feat_mul
        
        # fusion mask in feat and modals
        if mask is not None and self.with_mask:
            mask = self.mask_convs(mask)
            img_feat = img_feat * mask + img_feat
            modals = modals * mask + modals
        
        # fusion feat and modals
        img_feat = img_feat + modals
        
        ################################# LLM text and image attention ###########################
        # fusion image and llm features
        if llm_feat is not None and self.with_llm_feat:
            llm_feat = self.llm_drop(self.llm_dense(llm_feat))
            llm_feat = self.add_llm_pe(llm_feat)
            
            # downsample image
            img_feat, full_size = self.downsample_img_by_llm(img_feat)
            
            # ESS
            bs, C, H, W = img_feat.shape
            img_feat = self.scan(img_feat).view(bs, -1, H * W)
            
            # multi-modal RWKV attention
            img_feat = self.multi_modal_img_ln(img_feat)
            llm_feat = self.multi_modal_txt_ln(llm_feat)
            multi_modal_attn_out = self.multi_modal_attn(img_feat, llm_feat, (H, W))
            
            # extract img and llm feat
            img_feat_out, llm_feat_out = self.extract_img_and_llm_feat(H, W, multi_modal_attn_out, self.with_llm_feat)
            
            # residual
            img_feat = img_feat + img_feat_out
            llm_feat = llm_feat + llm_feat_out
            
            # merge ESS
            img_feat = rearrange(img_feat, "b (k d) (h w) -> b k d h w", k=self.K, h=H, w=W)
            img_feat = self.merge(img_feat).view(bs, C, H, W) / self.K
            
            # upsample image
            img_feat = self.upsample_img_by_llm(img_feat, full_size)
            
            # mlps
            img_feat = self.img_mlp(img_feat)
            llm_feat = self.txt_mlp(llm_feat)
            
        ###########################################################################################
            
        return img_feat, modals, llm_feat
    
    def downsample_img_by_llm(self, img):
        full_size = img.shape[-2:]
        img_ds_size = (img.shape[-2] // self.img_ds_r_by_llm, img.shape[-1] // self.img_ds_r_by_llm)
        return F.interpolate(
            img, 
            img_ds_size, 
            mode='bilinear',
            align_corners=True
        ), full_size
            
    def upsample_img_by_llm(self, img, img_upsample_size: tuple[int, int]):
        return F.interpolate(
            img, 
            img_upsample_size,
            mode='bilinear',
            align_corners=True,
            antialias=True
        )
            
    def __repr__(self):
        return f"{self.__class__.__name__}(modal_chan={self.modal_chan}, \
            n_embd={self.n_embd}, mask_chan={self.mask_chan}, llm_chan={self.llm_chan})"


class DoubleStreamRWKVBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        n_layer=8,
        n_embd=64,
        drop_path=0.0,
        scan_mode="K2",
        attn_bias=True,
        ffn_bias=True,
        cond_inj_mode='add',
        attn_groups=4,
        ffn_hidden_rate=1,
        ffn_groups=4,
        *,
        no_out_norm=False,
        shift_type="q_shift",
        scan_id=0,
        window_size=0,
        MIFM_modal_chan=1,
        MIFM_mask_chan=1,
        MIFM_llm_chan=None,
        MIFM_feat_drop=0.0,
        MIFM_llm_drop=0.0,
        MIFM_llm_pe_type='none',
        MIFM_img_pe_type='none',
        img_downsample_ratio_by_llm=4,
        checkpoint=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.cond_inj_mode = cond_inj_mode
        self.window_size = window_size
        self.roll_size = window_size // 2 if (scan_id % 2 == 0 and window_size > 0) else 0
        self.img_downsample_ratio_by_llm = img_downsample_ratio_by_llm
        
        # flags
        self.has_llm = MIFM_llm_chan is not None
        self.has_wind_part = window_size > 0
        self.has_roll = self.roll_size > 0
        
        # block config
        logger.info(
            f"DoubleStreamRWKVBlock: layer_id: {layer_id}, scan_mode: {scan_mode}, cond_inj_mode: {cond_inj_mode}, window_size: {window_size}, ",
            f"scan_id: {scan_id}, roll_size: {self.roll_size}, attn_ffn_groups: [{attn_groups}, {ffn_groups}], out_norm: {not no_out_norm}"
        )

        if not no_out_norm:
            self.out_norm = LayerNorm(n_embd, 'BiasFree')  # nn.GroupNorm(1, n_embd)
                        
        K = int(scan_mode[1])
        
        ############# MIFM ##############
        self.MIFM = MultiModalityFusion(n_embd, MIFM_modal_chan, MIFM_mask_chan, MIFM_llm_chan, 
                                        MIFM_feat_drop, MIFM_llm_drop, MIFM_llm_pe_type,
                                        MIFM_img_pe_type, scan_id=scan_id, layer_id=layer_id,
                                        img_ds_r_by_llm=img_downsample_ratio_by_llm,
                                        attn_groups=attn_groups, attn_bias=attn_bias,
                                        shift_type=shift_type, n_layer=n_layer,)
            
        ############# BRWKV #############
        assert scan_mode in [
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K2, K4, K8]"
        self.K = K
        n_embd = n_embd * K
        dim_att = n_embd
        dim_ffn = n_embd
        
        self.modulator1 = Modulator(n_embd, double=False)
        self.modulator2 = Modulator(n_embd, double=False)

        N_HEAD = n_embd // HEAD_SIZE
        
        # Attention
        self.att_img = RWKV_TMix(
            dim_att,
            N_HEAD,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            n_groups=attn_groups,
            attn_bias=attn_bias,
            with_cp=checkpoint,
        )
        
        # FFN
        self.ffn_fusion = RWKV_CMix(
            dim_ffn,
            N_HEAD,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            hidden_rate=ffn_hidden_rate,
            n_groups=ffn_groups,
            ffn_bias=ffn_bias,
            with_cp=checkpoint,
        )
        
        # if self.has_llm:
        #     # if clip windows, we should add a global vision/language BRWKV block
        #     # an img/txt multi-modal attention and ffn
        #     self.att_multi_modal = RWKV_TMix(
        #         dim_att,
        #         N_HEAD,
        #         n_layer,
        #         layer_id,
        #         shift_mode=shift_type,
        #         n_groups=attn_groups,
        #         attn_bias=attn_bias,
        #     )
        #     self.multi_modal_txt_ln = LayerNorm(n_embd, 'WithBias')
        #     self.multi_modal_img_ln = LayerNorm(n_embd, 'WithBias')

        self.only_img_attn_ln = LayerNorm(n_embd, 'BiasFree')
        self.fusion_img_llm_ln = LayerNorm(n_embd, 'BiasFree')

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
    
    def partition_img_by_window(self, H, W, x):
        if self.has_wind_part:
            if self.has_roll:
                x = x.roll(shifts=(-self.roll_size, -self.roll_size), dims=(2, 3))
            x = window_partition(x, self.window_size)
            h, w = self.window_size, self.window_size
            # patch_resolution = (self.window_size, self.window_size)
        else:
            h, w = H, W
            
        return h, w, x
    
    def reverse_img_by_window(self, H, W, x):
        if self.has_wind_part:
            x = window_reverse(x, self.window_size, H, W)
            if self.has_roll:
                x = x.roll(shifts=(self.roll_size, self.roll_size), dims=(2, 3))
                
        return x
        
    def extract_img_and_llm_feat(self, h: int, w: int, att_or_ffn_out: torch.Tensor, has_llm: bool):
        # att_or_ffn_out: [bs, K, C, L]
        if has_llm:
            img_feat, llm_feat = att_or_ffn_out[..., :h*w], att_or_ffn_out[...,h*w:]
        else:
            img_feat = att_or_ffn_out
            llm_feat = None
            
        return img_feat, llm_feat
    
    def BRWKV_img_forward(self, 
                          x: torch.Tensor, 
                          MIFM_img: torch.Tensor=None, 
                          llm_feat: torch.Tensor=None,
                          patch_resolution=None):
        """
        ##! 1. Image-only attention with clipping windows
        w_img = window_partition(img, window_size)
        w_img = scan(img)
        w_img = only_img_attention(img)
        
        ##! 2. Multi-modal attention with txt feature
        # solution 1 (downsample image)
        img = window_reverse(w_img, window_size, H, W)
        ds_img = downsample_scanned_img(img)
        ds_img, txt = multi_modal_attention(ds_img, txt)
        upsample_img = upsample_scanned_img(ds_img)
        img = (upsample_img + img) / 2
        
        # solution 2 (repeat txt feature on bs-dim)
        rep_bs_txt = repeat(txt, 'bs d l -> (bs n_winds) d l')
        w_img, rep_bs_txt = multi_modal_attention(w_img, rep_bs_txt)
        img = window_reverse(w_img, window_size, H, W)
        txt = rearrange(rep_bs_txt, '(bs n_winds) d l -> bs n_winds d l', bs=B)
        txt = txt.mean(dim=1)
        
        ##! 3. Multi-modal FFN
        #! Flux.1 says we should have two FFNs, one for image and one for txt
        img = FFN(img, txt)
        
        """
        
        B, C, H, W = x.shape
        has_llm = (llm_feat is not None) and self.has_llm
        
        # pre-fusion previous (non-downsampled) image and MIFM image
        x = x + MIFM_img
        
        # window partition
        # (bs, c, h, w) -> (bs x n_winds, c, wind, wind)
        h, w, x = self.partition_img_by_window(H, W, x)
            
        # ESS
        # (bs x n_winds, c, wind, wind) -> (bs x n_winds, c x K, wind * wind)
        x = self.scan(x).view(x.size(0), -1, h * w)
        
        ######################## Image-only Attention ##########################
           
        # Image spatial mixing
        prenorm_x = self.only_img_attn_ln(x)
        sc1, sh1, ga1 = self.modulator1(x)
        x = (1 + sc1) * prenorm_x + sh1  # modulated img
        x = self.drop_path(self.att_img(x, None, (h, w)))
        x = ga1 * x + x

        ########################################################################
        
        ######################## Prepare for multi-modal FFN ######################

        # multi-modal spatial mixing
        #------------------------------------
        #! Question: if there needs to scan the image? or just flatten the image? 
        # if partition windows before, we should reverse the image first
        # because the caption is for the whole image
        if self.has_wind_part:
            # merge ESS
            # (bs x n_winds, c x K, wind * wind) -> (bs, c, wind, wind)
            x = rearrange(x, "b (k d) (h w) -> b k d h w", k=self.K, h=h, w=w)
            x = self.merge(x).view(x.size(0), C, h, w) / self.K
            
            # window reverse
            # (bs x n_winds, c, wind, wind) -> (bs, c, h, w)
            x = self.reverse_img_by_window(H, W, x)
            
            # scan the whole image that captioned without window partition
            x = self.scan(x).view(x.size(0), -1, H * W)

        ########################################################################
            
        ######################## Multi-modality FFN ############################
        
        # Channel mixing
        x = self.fusion_img_llm_ln(x)
        sc2, sh2, ga2 = self.modulator2(x)
        mod_x_ffn = (1 + sc2) * x + sh2
        # 1. multi-modal FFN
        x = self.drop_path(self.ffn_fusion(mod_x_ffn, llm_feat, patch_resolution))
        x, llm_feat = self.extract_img_and_llm_feat(H, W, x, has_llm)
        # 2. pure image FFN
        # x = self.drop_path(self.ffn_fusion(mod_x_ffn, None, (h, w)))
        
        x = ga2 * x + x
        
        #########################################################################
        
        # merge ESS
        # (bs x n_winds, c x K, wind * wind) -> (bs, c, wind, wind)
        x = rearrange(x, "b (k d) (h w) -> b k d h w", k=self.K, h=H, w=W)
        x = self.merge(x).view(x.size(0), C, H, W) / self.K
        
        return x, llm_feat
        
    def forward(self, x: torch.Tensor, cond: ConditionInput, patch_resolution=None):
        inp = x
        ## MIFM
        MIFM_img_feat, modals, llm_feat = self.MIFM(x, cond)
        
        ## BRWKV
        x, llm_feat = self.BRWKV_img_forward(x, MIFM_img_feat, llm_feat, patch_resolution)

        # norm output
        if not self.no_out_norm:
            out = self.out_norm(x) + inp
        else:
            out = (x / self.K) + inp

        return out, ConditionInput(cond.modalities, llm_feat, cond.mask_input)


class SingleStreamRWKVBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        n_layer=8,
        n_embd=64,
        drop_path=0.0,
        scan_mode="K2",
        attn_bias=True,
        ffn_bias=True,
        cond_inj_mode='add',
        attn_groups=4,
        ffn_hidden_rate=1,
        ffn_groups=4,
        *,
        no_out_norm=False,
        shift_type="q_shift",
        scan_id=0,
        window_size=0,
        MIFM_modal_chan=1,
        MIFM_feat_drop=0.0,
        checkpoint=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.window_size = window_size
        self.roll_size = window_size // 2 if (scan_id % 2 == 0 and window_size > 0) else 0
        
        # block config
        logger.info(
            f"SingleStreamRWKVBlock: layer_id: {layer_id}, scan_mode: {scan_mode}, window_size: {window_size}, ",
            f"scan_id: {scan_id}, roll_size: {self.roll_size}, attn_ffn_groups: [{attn_groups}, {ffn_groups}], out_norm: {not no_out_norm}"
        )

        if not no_out_norm:
            self.out_norm = LayerNorm(n_embd, 'BiasFree')  # nn.GroupNorm(1, n_embd)

        ############# MIFM ##############
        # only use the modal as condition
        self.MIFM = MultiModalityFusion(n_embd,
                                        MIFM_modal_chan, 
                                        feat_drop=MIFM_feat_drop)

        ############# BRWKV #############
        assert scan_mode in [
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K2, K4, K8]"
        K = int(scan_mode[1])
        self.K = K
        n_embd = n_embd * K
        dim_att = n_embd
        dim_ffn = n_embd
        
        self.img_modulator = Modulator(n_embd)

        N_HEAD = n_embd // HEAD_SIZE
        self.att = RWKV_TMix(
            dim_att,
            N_HEAD,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            n_groups=attn_groups,
            attn_bias=attn_bias,
            with_cp=checkpoint,
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
            with_cp=checkpoint,
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

    def forward(self, 
                x: torch.Tensor, 
                cond: ConditionInput,
                patch_resolution: tuple | None = None):
        B, C, H, W = x.shape
        inp = x
        
        ## MIFM
        x, *_ = self.MIFM(x, cond)
        
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
        sc1, sh1, ga1, sc2, sh2, ga2 = self.img_modulator(x)
           
        # Spatial mixing
        prenorm_x = self.x_ln1(x)
        mod_x_attn = (1 + sc1) * prenorm_x + sh1
        x = ga1 * self.drop_path(self.att(mod_x_attn, None, patch_resolution)) + x
        
        # Channel mixing
        prenorm_x = self.x_ln2(x)
        mod_x_ffn = (1 + sc2) * prenorm_x + sh2
        x = ga2 * self.drop_path(self.ffn(mod_x_ffn, None, patch_resolution)) + x

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

    def enc_forward(self, feat, cond, patch_resolution) -> tuple[torch.Tensor, ConditionInput]:
        outp = feat
        for mod in self.mods:
            outp, cond = mod(outp, cond, patch_resolution)
        return outp.contiguous() + feat, cond

    def dec_forward(self, feat, cond, patch_resolution) -> torch.Tensor:
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



@register_model("panRWKV_v10_local")
class RWKVFusion(BaseModel):
    @typechecked
    def __init__(
        self,
        img_channel: int=3,
        modal_channel: int=1,
        mask_channel: int | None=None,
        llm_channel: int | None=None,
        out_channel: int=3,
        width: int=16,
        middle_blk_num: int=1,
        mid_window_size: int=0,
        enc_blk_nums: list=[],
        dec_blk_nums: list=[],
        chan_upscales: list=[],
        enc_window_sizes: list=[],
        dec_window_sizes: list=[],
        attn_groups: list=[],
        ffn_groups: list=[],
        ffn_hidden_rates: list=[],
        MIFM_img_downsample_ratios: list | None=None,
        with_checkpoint: bool=False,
        upscale: int=1,
        if_abs_pos: bool=False,
        pt_img_size: int=64,
        drop_path_rate: float=0.1,
        fusion_prior: Literal['max', 'mean', 'none'] = "max",
        patch_merge: bool=True,
        drop_txt_ratio: float=0.0,
        txt_embd_drop_ratio: float=0.0,
        multi_value_mask_max_classes: int | None=None,
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        self.pt_img_size = pt_img_size
        self.fusion_prior = fusion_prior
        self.patch_merge = patch_merge
        self.llm_channel = llm_channel
        self.mask_channel = mask_channel
        self.drop_txt_ratio = drop_txt_ratio
        self.multi_value_mask_max_classes = multi_value_mask_max_classes
        
        # flags
        self.has_mask = mask_channel is not None
        self.has_txt = llm_channel is not None
        
        # type check
        assert len(enc_blk_nums) == len(dec_blk_nums) == len(chan_upscales) == len(enc_window_sizes) == \
               len(dec_window_sizes) == len(attn_groups) == len(ffn_groups) == len(ffn_hidden_rates) == \
               len(MIFM_img_downsample_ratios), "config length not match"
        assert fusion_prior in [
            "max",
            "mean",
            "none",
        ], "`fusion_prior` should be one of [max, mean, none]"
        
        # cfgs
        net_name = __class__.__name__
        logger.info(f"{net_name}: multi-modal settings, input mask: {self.has_mask}, input txt: {self.has_txt}")
        logger.info(f"{net_name}: enc_blk_nums: {enc_blk_nums}, dec_blk_nums: {dec_blk_nums}")
        logger.info(f"{net_name}: chan_upscales: {chan_upscales}, enc_window_sizes: {enc_window_sizes}, dec_window_sizes: {dec_window_sizes}")
        logger.info(f"{net_name}: checkpoint: {with_checkpoint} {'the training will be slow for large batch size' if with_checkpoint else ''}")
        logger.info(f"{net_name}: fusion_prior: {fusion_prior}, use patch merge: {patch_merge}")
        logger.info(f"{net_name}: txt input directly drop ratio: {drop_txt_ratio}, txt embd drop ratio: {txt_embd_drop_ratio}")

        if if_abs_pos:
            self.abs_pos = nn.Parameter(
                torch.randn(1, width, pt_img_size, pt_img_size), requires_grad=True
            )

        logger.print(f"{__class__.__name__}: {img_channel=}, {modal_channel=}\n")
        self.patch_embd = nn.Conv2d(
            in_channels=img_channel + modal_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=True,
        )
        modal_channel = img_channel + modal_channel
        
        if self.llm_channel is not None:
            self.llm_embd = nn.Sequential(
                nn.LayerNorm(self.llm_channel),
                nn.Linear(self.llm_channel, width * np.prod(chan_upscales), bias=True)
            )
            self.embd_drop = nn.Dropout(txt_embd_drop_ratio)
            llm_channel = width * np.prod(chan_upscales)
            
        # mask embedding
        if self.multi_value_mask_max_classes is not None:
            self.mask_embd = nn.Sequential(
                nn.LayerNorm(self.multi_value_mask_max_classes),
                nn.Conv2d(self.multi_value_mask_max_classes, width, bias=True),
            )
            mask_channel = default(mask_channel, width)
        
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
        # stochastic depth decay rule
        inter_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth // 2)
        ]
        inter_dpr = inter_dpr + inter_dpr[::-1]
        if len(inter_dpr) < depth:
            inter_dpr.append(inter_dpr[-1])
        logger.info(f"{net_name}: inter_dpr: {np.array(inter_dpr).round(4)} for depth: {depth}")

        chan = width
        n_prev_blks = 0
        n_prev_layers = 0
        # encoder
        for layer_id, num in enumerate(enc_blk_nums):
            self.encoders.append(
                FusionSequential(
                    *[
                        DoubleStreamRWKVBlock(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            window_size=enc_window_sizes[layer_id],
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + layer_id,
                            attn_groups=attn_groups[layer_id],
                            ffn_groups=ffn_groups[layer_id],
                            ffn_hidden_rate=ffn_hidden_rates[layer_id],
                            MIFM_modal_chan=modal_channel,
                            MIFM_mask_chan=mask_channel,
                            MIFM_llm_chan=(llm_channel if n_prev_blks == 0 else prev_chan * 2) if self.has_txt else None,
                            MIFM_feat_drop=inter_dpr[n_prev_blks + i],
                            MIFM_llm_drop=inter_dpr[n_prev_blks + i],
                            img_downsample_ratio_by_llm=MIFM_img_downsample_ratios[layer_id],
                            checkpoint=with_checkpoint,
                        )
                        for i in range(num)
                    ]
                )
            )
            # self.downs.append(
            #     nn.Sequential(
            #         OrderedDict([
            #                 ('x_down', down(chan, r=2, chan_r=chan_upscales[layer_id])),
            #                 ('modal_down', down(chan, r=2, chan_r=chan_upscales[layer_id]))
            #             ])
            #     )
            # )
            self.downs.append(down(chan, r=2, chan_r=chan_upscales[layer_id]))
            prev_chan = chan
            chan = chan * chan_upscales[layer_id]
            n_prev_blks += num
            pt_img_size //= 2
        n_prev_layers += len(enc_blk_nums)

        # middle layer
        self.middle_blks = FusionSequential(
            *[
                DoubleStreamRWKVBlock(
                    layer_id=n_prev_blks + i,
                    n_layer=depth,
                    n_embd=chan,
                    window_size=mid_window_size,
                    drop_path=inter_dpr[n_prev_blks + i],
                    scan_id=n_prev_blks + i,  # n_prev_layers + layer_id,
                    attn_groups=attn_groups[layer_id],
                    ffn_groups=ffn_groups[layer_id],
                    ffn_hidden_rate=ffn_hidden_rates[layer_id],
                    MIFM_modal_chan=modal_channel,
                    MIFM_mask_chan=mask_channel,
                    MIFM_llm_chan=prev_chan * 2 if self.has_txt else None,
                    MIFM_feat_drop=inter_dpr[n_prev_blks + i],
                    MIFM_llm_drop=inter_dpr[n_prev_blks + i],
                    img_downsample_ratio_by_llm=MIFM_img_downsample_ratios[-1],
                    checkpoint=with_checkpoint,
                )
                for i in range(middle_blk_num)
            ]
        )
        n_prev_blks += middle_blk_num
        n_prev_layers += 1

        # decoder
        self.skip_scales = nn.ParameterList([])
        for dec_layer_id, num in enumerate(reversed(dec_blk_nums)):
            # self.ups.append(
            #     nn.Sequential(
            #         OrderedDict([
            #                 ('x_up', up(chan, r=2, chan_r=chan_upscales[::-1][dec_layer_id])),
            #                 ('modal_up', up(chan, r=2, chan_r=chan_upscales[::-1][dec_layer_id]))
            #             ])
            #     )
            # )
            self.ups.append(up(chan, r=2, chan_r=chan_upscales[::-1][dec_layer_id]))
            chan = chan // chan_upscales[::-1][dec_layer_id]
            window_size = dec_window_sizes[::-1][dec_layer_id]
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
                            window_size=window_size,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,  # n_prev_layers + dec_layer_id,
                            attn_groups=attn_group,
                            ffn_groups=ffn_group,
                            ffn_hidden_rate=ffn_hidden_rate,
                            MIFM_modal_chan=modal_channel,
                            MIFM_feat_drop=inter_dpr[n_prev_blks + i],
                            checkpoint=with_checkpoint,
                        )
                        for i in range(num)
                    ],
                )
            )
            n_prev_blks += num
        n_prev_layers += len(dec_blk_nums)

        self.out_feat = False
        
        # interpoalte
        self.interpolate = lambda x, H, W: F.interpolate(x, (H, W), mode="bilinear",
                                                         align_corners=True, antialias=True)

        # init
        logger.print(
            f"============= {__class__.__name__}: init network ================="
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        rng_linear = torch.Generator().manual_seed(2025)
        rng_conv = torch.Generator().manual_seed(2026)
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, generator=rng_linear)
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
            trunc_normal_(m.weight, std=0.02, generator=rng_conv)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def resize_pos_embd(self,
                        pos_embd: torch.Tensor,
                        inp_size: tuple):
        if pos_embd.size(-1) != inp_size[-1] or pos_embd.size(-2) != inp_size[-2]:
            pos_embd = F.interpolate(pos_embd, inp_size[-2:], mode='bilinear', align_corners=False)
        
        return pos_embd
    
    def resize_conds(self, H, W, x: torch.Tensor=None):
        if exists(x):
            x = self.interpolate(x, H, W)
        
        return x

    def _forward_implem(self, inp, modal, mask=None, llm_feature=None):
        x = inp
        bs, _, H, W = x.shape

        cat_x_modal = torch.cat([x, modal], dim=1)
        x = self.patch_embd(cat_x_modal)
        modal = cat_x_modal
        # x, modal = x.chunk(2, dim=1)
        
        if self.if_abs_pos:
            x = x + self.resize_pos_embd(self.abs_pos, inp_size=(H, W))
            
        if llm_feature is not None and self.llm_channel is not None:
            llm_feature = self.llm_embd(llm_feature).permute(0, 2, 1)
            llm_feature = self.embd_drop(llm_feature)
            
        if mask is not None and hasattr(self, 'mask_embd'):
            mask = self.mask_embd(mask)
            
        # init condition
        u_cond = ConditionInput(modal, llm_feature, mask)
                        
        ## encoder
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            u_cond.modalities = self.resize_conds(H, W, u_cond.modalities)
            u_cond.mask_input = self.resize_conds(H, W, u_cond.mask_input)
            x, u_cond = encoder.enc_forward(x, u_cond, (H, W))
            encs.append(x)
            # x = down.x_down(x)
            # modal = down.modal_down(u_cond.modalities)
            x = down(x)
            H = H // 2
            W = W // 2

        ## middle layer
        u_cond.modalities = self.resize_conds(H, W, u_cond.modalities)
        u_cond.mask_input = self.resize_conds(H, W, u_cond.mask_input)
        x, _ = self.middle_blks.enc_forward(x, u_cond, (H, W))
         
        ## decoder
        for decoder, up, enc_skip, skip_scale in zip(
            self.decoders, self.ups, encs[::-1], self.skip_scales
        ):
            # x = up.x_up(x)
            # modal = up.modal_up(modal)
            x = up(x)
            H = H * 2
            W = W * 2
            u_cond.modalities = self.resize_conds(H, W, u_cond.modalities)
            u_cond.mask_input = self.resize_conds(H, W, u_cond.mask_input)
            x = torch.cat([x, enc_skip * skip_scale], dim=1)
            x = decoder.dec_forward(x, u_cond, (H, W))

        x = self.proj_out(x)

        return x

    def drop_txt(self, txt):
        if self.has_txt and txt is not None and self.drop_txt_ratio > 0:
            if np.random.rand() < self.drop_txt_ratio:
                txt = torch.zeros_like(txt)
        
        return txt

    def sharpening_train_step(self, 
                              ms: torch.Tensor,
                              lms: torch.Tensor, 
                              pan: torch.Tensor,
                              gt: torch.Tensor,
                              txt: torch.Tensor | None=None,
                              criterion: "Callable | None"=None,
                              **_kwargs):
        assert criterion is not None, "criterion should be provided"
        
        txt = self.drop_txt(txt)
        sr = self._forward_implem(lms, pan, None, txt) + lms
        sr = sr.clip(0, 1)
        loss = criterion(sr, gt)

        return sr, loss

    @torch.no_grad()
    def sharpening_val_step(self,
                            ms: torch.Tensor,
                            lms: torch.Tensor,
                            pan: torch.Tensor,
                            txt: "torch.Tensor | None"=None,
                            patch_merge: "callable | bool | None"=True,
                            *,
                            inference_wo_txt: bool=False,
                            **_kwargs):
        if patch_merge is None:
            patch_merge = self.patch_merge
            
        if inference_wo_txt and txt is not None and self.has_txt:
            txt = torch.zeros_like(txt)

        if patch_merge:
            logger.debug(f"using patch merge module")
            # for any patch merge, we drop the txt feature
            if txt is not None and self.has_txt:
                pm_step = partial(self.patch_merge_step, llm_feature=txt)
            else:
                pm_step = self.patch_merge_step
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16 * self.upscale, 16 * self.upscale],
                scale=1,
                patch_merge_step=pm_step,
            )
            sr = _patch_merge_model.forward_chop(lms, pan)[0] + lms
        else:
            sr = self._forward_implem(lms, pan, None, txt) + lms

        return sr.clip(0, 1)

    def only_fusion_step(self, vi, ir, mask=None, txt=None, **_kwargs):
        outp = self._forward_implem(vi, ir, mask, txt)
        if self.fusion_prior == "max":
            prior = torch.max(vi, ir)
        elif self.fusion_prior == "mean":
            prior = (vi + ir) / 2
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
        
    def check_multi_modal_inputs(self, 
                                 vis: torch.Tensor,
                                 ir: torch.Tensor,
                                 mask: torch.Tensor | None,
                                 txt: torch.Tensor | None,
                                 device: "torch.device | str",
                                 dtype: torch.dtype,):
        if not self.has_mask:
            mask = None
        elif mask is not None:
            mask = mask.to(device).type(dtype)
        else: # has mask in model but input mask is None
            mask = torch.zeros(1, self.mask_channel, *vis.shape[-2:], device=device, dtype=dtype)
            
        if not self.has_txt:
            txt = None
        elif txt is not None:
            txt = txt.to(device).type(dtype)
            
        return vis, ir, mask, txt
    
    def check_multi_value_mask_to_one_hot(self, mask: torch.Tensor | None):
        if mask is not None and self.multi_value_mask_max_classes is not None and self.has_mask:
            if mask.ndim == 3:
                mask[mask > self.multi_value_mask_max_classes] = 0.  # larger than max classes are set to background
                new_mask = torch.zeros(1, self.multi_value_mask_max_classes, *mask.shape[-2:], device=mask.device, dtype=torch.float32)
                new_mask.scatter_(1, mask.long()[:, None], 1.)                
                # mask = F.one_hot(mask, num_classes=self.multi_value_mask_max_classes)
        else:
            new_mask = None
            
        return new_mask

    def fusion_train_step(
        self,
        vi: "torch.Tensor",
        ir: "torch.Tensor",
        mask: "torch.Tensor | None"=None,
        gt: "torch.Tensor | None"=None,
        txt: "torch.Tensor | None"=None,
        fusion_criterion: "Callable | None"=None,
        to_rgb_fn: "Callable | None" = None,
        has_gt: bool = False,
        **_kwargs,
    ):
        vi, ir, mask, txt = self.check_multi_modal_inputs(vi, ir, mask, txt, vi.device, vi.dtype)
        txt = self.drop_txt(txt)
        
        mask = self.check_multi_value_mask_to_one_hot(mask)
        
        fused_outp = self.only_fusion_step(vi, ir, mask, txt)

        # mask for loss, should be detached
        if mask is not None:
            mask_for_loss = mask.clone().detach()
        else:
            mask_for_loss = None

        fused_for_loss = to_rgb_fn(fused_outp) if to_rgb_fn is not None else fused_outp
        if has_gt or gt.size(1) == 3: # TODO: find more robust way to check if gt is available
            # if we have supervised GT, we use it to compute the supervised loss
            # sometimes, for MEF fusion task, we can access the GTs
            fusion_gt = gt
            # two different modalities for GT to compute the unsupervised loss
            boundary_gt = (vi, ir)
        else:
            fusion_gt = None
            boundary_gt = (vi, ir)
            
        # compute supervised and unsupervised losses
        assert fusion_criterion is not None, "fusion_criterion should be provided"
        loss = list(
            fusion_criterion(
                fused_for_loss,
                boundary_gt=boundary_gt,
                fusion_gt=fusion_gt,
                mask=mask_for_loss,
            )
        )

        return fused_for_loss.clip(0, 1), loss

    @torch.no_grad()
    def fusion_val_step(
        self,
        vi: "torch.Tensor",
        ir: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
        txt: "torch.Tensor | None" = None,
        *,
        patch_merge: bool = False,
        ret_seg_map: bool = False,
        to_rgb_fn: callable = None,
        **_kwargs,
    ):
        vi, ir, mask, txt = self.check_multi_modal_inputs(vi, ir, mask, txt, vi.device, vi.dtype)
            
        fused_outp = self.only_fusion_step(vi, ir, mask, txt)
        if to_rgb_fn is not None:
            fused = to_rgb_fn(fused_outp)
        else:
            fused = fused_outp

        return fused.clip(0, 1)

    def patch_merge_step(self, lms, pan, **kwargs):
        if self.has_txt and 'llm_feature' in kwargs:
            llm_feature = kwargs['llm_feature']
            if llm_feature is not None:
                assert llm_feature.size(0) == 1, 'when using PatchMergeModule, input batch size should be 1'
                llm_feature = llm_feature.repeat(lms.size(0), 1, 1)
            kwargs['llm_feature'] = llm_feature
        
        fused = self._forward_implem(lms, pan, **kwargs)
        return fused
    
    def flops(self, inputs: tuple[torch.Tensor, ...]):
        img, cond, mask, llm_feature = inputs
        
        from fvcore.nn import flop_count_table, FlopCountAnalysis
        from model.module.rwkv_v2 import WKV_6, vrwkv6_flops

        self.forward = self._forward_implem
        flops_count = FlopCountAnalysis(self, (img, cond, mask, llm_feature))

        custom_ops = {"prim::PythonOp.WKV_6": vrwkv6_flops}
        flops_count.set_op_handle(**custom_ops)

        logger.info(flop_count_table(flops_count))
        


if __name__ == "__main__":
    from torch.cuda import memory_summary
    from utils import catch_any_error
    
    class Tester:
        def __init__(self,
                     bs,
                     img_size,
                     task,
                     model_size,
                     with_mask=True,
                     with_llm_feature=True,
                     device='cuda:0',
                     dtype=torch.float32,
                     ):
            # cfg 1 for pansharpening and HMIF
            cfg_pan_small = dict(
                    img_channel=8,
                    modal_channel=1,
                    mask_channel=1 if with_mask else None,
                    llm_channel=512 if with_llm_feature else None,
                    out_channel=8,
                    width=32,
                    middle_blk_num=1,
                    enc_blk_nums=[2, 1, 1],
                    dec_blk_nums=[2, 1, 1],
                    chan_upscales=[2, 1, 1],
                    mid_window_size=0,
                    enc_window_sizes=[16, 16, 16],
                    dec_window_sizes=[16, 16, 16],
                    attn_groups=[4,4,4],
                    ffn_groups=[4,4,4],
                    MIFM_img_downsample_ratios=[4,2,1],
                    ffn_hidden_rates=[1,1,1],
                    with_checkpoint=False,
                    pt_img_size=64,
                    drop_path_rate=0.1,
                    if_abs_pos=False,
                    upscale=4,
                )
            
            # cfg 2 for VIF
            cfg_vif_tiny = dict(
                    img_channel=3,
                    modal_channel=1,
                    mask_channel=1 if with_mask else None,
                    llm_channel=512 if with_llm_feature else None,
                    out_channel=3,
                    width=32,
                    middle_blk_num=1,
                    enc_blk_nums=[1, 1],
                    dec_blk_nums=[1, 1],
                    chan_upscales=[1, 1],
                    mid_window_size=14,
                    enc_window_sizes=[14,14],
                    dec_window_sizes=[14,14],
                    attn_groups=[1, 1],
                    ffn_groups=[1, 1],
                    ffn_hidden_rates=[2,2],
                    MIFM_img_downsample_ratios=[4,4],
                    pt_img_size=64,
                    drop_path_rate=0.1,
                    if_abs_pos=False,
                    upscale=1,
                    with_checkpoint=True,
                )
            cfg_vif_small = dict(
                    img_channel=3,
                    modal_channel=1,
                    mask_channel=1 if with_mask else None,
                    llm_channel=512 if with_llm_feature else None,
                    out_channel=3,
                    width=32,
                    middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1],
                    dec_blk_nums=[1, 1, 1],
                    chan_upscales=[2, 1, 1],
                    mid_window_size=16,
                    enc_window_sizes=[16,16,16],#[14, 14, 14],
                    dec_window_sizes=[16,16,16],#[14, 14, 14],
                    attn_groups=[1, 1, 2],
                    ffn_groups=[1, 1, 2],
                    ffn_hidden_rates=[2, 2, 2],
                    MIFM_img_downsample_ratios=[4,2,1],
                    with_checkpoint=True,
                    pt_img_size=64,
                    drop_path_rate=0.1,
                    if_abs_pos=False,
                    upscale=1,
                )
            cfgs = {'pan_small': cfg_pan_small, 'vif_tiny': cfg_vif_tiny, 'vif_small': cfg_vif_small}
            logger.info(f'using dtype: {dtype}')
            
            self.cfg = cfgs[task + "_" + model_size]
            self.bs = bs
            self.dtype = dtype
            self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
            self.task = task
            self.device = device
            self.with_mask = with_mask
            self.with_llm_feature = with_llm_feature
            
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
            # for pan task: img is lms, cond is pan
            # for vif task: img is vis, cond is ir
            img = torch.randn(self.bs, self.channel_img, self.img_size[0], self.img_size[1], dtype=self.dtype).to(device)
            cond = torch.randn(self.bs, self.channel_cond, self.img_size[0], self.img_size[1], dtype=self.dtype).to(device)
            
            if self.with_mask:
                mask = torch.randint(0, 9, (self.bs, 1, self.img_size[0], self.img_size[1]), dtype=self.dtype).to(img)
            else:
                mask = None
            
            if self.with_llm_feature:
                llm_feature = torch.randn(self.bs, 512, self.cfg['llm_channel'], dtype=self.dtype).to(device)
            else:
                llm_feature = None
                
            return img, cond, mask, llm_feature
        
        def summary_mem(self):
            ## memory usage
            self.print("Memory usage:")
            self.print(memory_summary(device=device, abbreviated=True))
        
        def params_and_flops(self):
            ## Count params, flops
            from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table
            from model.module.rwkv_v2 import WKV_6, vrwkv6_flops

            self.net.forward = self.net._forward_implem
            img, cond, mask, llm_feature = self.prepare_data()
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
            throghtput = measure_throughput(self.net, [(self.channel_img, self.img_size[0], self.img_size[1]),
                                                       (self.channel_cond, self.img_size[0], self.img_size[1],)],
                                            16, num_warmup=10, num_iterations=50)

            self.print(f'throughput: {throghtput} imgs/s')

        # @torch.no_grad()
        def test_forward(self):
            img, cond, mask, llm_feature = self.prepare_data()
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                out = self.net._forward_implem(img, cond, mask, llm_feature)
            self.info(f'output shape: {out.shape} with dtype: {out.dtype}')
            # self.summary_mem()
            
        def test_forward_backward(self):
            img, cond, mask, llm_feature = self.prepare_data()
            out = self.net._forward_implem(img, cond, mask, llm_feature)
            sr = torch.randn(self.bs, self.channel_img, self.img_size[0], self.img_size[1]).to(device)
            print(f'output shape: {out.shape} with dtype: {out.dtype}')
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
            
            ## memory usage
            self.summary_mem()
        
        @catch_any_error
        def test_val_step(self):
            if self.task == 'pan':
                img, cond, mask, llm_feature = self.prepare_data()
                fused = self.net.sharpening_val_step(None, img, cond, llm_feature, patch_merge=True)
                self.info(f'fused shape: {fused.shape}')
                self.summary_mem()
            elif self.task == 'vif':
                # self.warning('use patch_merge_step only for sharpening task')
                img, cond, mask, llm_feature = self.prepare_data()
                fused = self.net.fusion_val_step(img, cond, mask, llm_feature)
                self.info(f'fused shape: {fused.shape}')
                self.summary_mem()
                
        def test_params(self,):
            for n, p in self.net.named_parameters():
                if p.dtype != torch.float32:
                    logger.warning(f'{n} has dtype: {p.dtype}')
                if not p.is_floating_point():
                    logger.warning(f'{n} is not floating point, has dtype: {p.dtype}')
                    
            logger.info(f'total params: {sum(p.numel() for p in self.net.parameters())}')
            
            
        def test_fused_optimizer(self,):
            import torch.optim as optim
            from utils.optim_utils import get_optimizer
            
            optimizer = optim.AdamW(self.net.parameters(), lr=1e-3, fused=True)
            # optimizer = get_optimizer(self.net, self.net.parameters(), lr=1e-3,
            #                           name='shampoo_ddp', use_pytorch_compile=False)
            img, cond, mask, llm_feature = self.prepare_data()
            sr = torch.randn(self.bs, self.channel_img, self.img_size[0], self.img_size[1]).to(device)
            out = self.net._forward_implem(img, cond, mask, llm_feature)
            loss = F.mse_loss(out, sr)
            loss.backward()
            optimizer.step()
            
            logger.info(f'optimizer: {optimizer} step done.')
            
                    
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    
    # img size: (280, 224) for VIF, and (64, 64) for PAN - training
    # img size: (672, 504) for VIF (padder with base 56), and (256, 256) for PAN - testing
    tester = Tester(1, (224, 280), 'vif', 'tiny', 
                    with_mask=True, with_llm_feature=True,
                    device=device, dtype=torch.float32)
    

    # tester.params_and_flops()
    # tester.test_forward()
    tester.test_forward_backward()
    # tester.test_val_step()
    # tester.measure_throughput()
    # tester.test_params()
    # tester.test_fused_optimizer()

