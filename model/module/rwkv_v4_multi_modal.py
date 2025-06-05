# Copyright (c) Shanghai AI Lab. All rights reserved.
from functools import partial
from typing import Sequence
import os

import torch
import torch.cuda.amp
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch.utils.checkpoint as cp
import triton
from timm.layers import DropPath
from einops import rearrange
from einops.layers.torch import Rearrange

from model.module.csm_triton import (triton_cross_scan,
                                     triton_cross_merge,
                                     
                                     triton_cross_scan_same_and_trans,
                                     triton_cross_merge_trans_and_flips,
                                     triton_cross_merge_same_and_trans,
                                     triton_cross_scan_trans_and_flips,
                                     
                                     triton_cross_scan_same,
                                     triton_cross_merge_same,
                                     triton_cross_scan_trans,
                                     triton_cross_merge_trans,
                                     triton_cross_scan_trans_same,
                                     triton_cross_merge_trans_same,
                                     triton_cross_scan_flips,
                                     triton_cross_merge_flips,
                                     )
from model.module.layer_norm import RMSNorm

from utils import easy_logger
from utils.utils_modules.shampoo_optimizers.distributed_shampoo.examples.trainer_utils import DType
logger = easy_logger(func_name='rwkv_v4_multi_modal')

############### K = 8 ################
    
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,C,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,C,H,W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,D,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,D,H,W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B,C,H,W))
        xs[:, 5] = antidiagonal_gather(x.view(B,C,H,W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)

############### K = 4 ################

class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y#.view(B, 4, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x

class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x#.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y

############## K = 2 ##################

class CrossScanTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 1 or 2')
        
        return y#.view(B, 2, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l) 
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 1 or 2')
        
        return x, None

class CrossMergeTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return x#.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return y, None


############ K = 1 ################

class CrossScanTritonSelectK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 1, C, H, W))
        if scan_mode == 0:
            triton_cross_scan_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_scan_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 2:
            triton_cross_scan_trans_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 3:
            triton_cross_scan_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 0, 1, 2 or 3')
        
        return y#.view(B, 1, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l) 
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 1, C, H, W)
        x = y.new_empty((B, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_merge_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 2:
            triton_cross_merge_trans_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 3:
            triton_cross_merge_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 0, 1, 2 or 3')
        
        return x, None

class CrossMergeTritonSelectK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 1, C, H, W)
        x = y.new_empty((B, C, H, W))
        if scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_merge_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 2:
            triton_cross_merge_trans_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 3:
            triton_cross_merge_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return x#.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 1, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_scan_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 2:
            triton_cross_scan_trans_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 3:
            triton_cross_scan_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return y, None


class CrossScanK1Torch():
    @staticmethod
    def scan(x, scan_id):
        B, C, H, W = x.size()
        if scan_id == 0:
            return x.view(B, 1, C, -1)
        elif scan_id == 1:
            return x.transpose(dim0=2, dim1=3).flatten(2, 3)
        elif scan_id == 2:
            return torch.flip(x.view(B, 1, C, -1), dims=[-1])
        elif scan_id == 3:
            return torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
    
    @staticmethod
    def merge(x, scan_id):
        B, _, C, H, W = x.size()
        if scan_id == 0:
            return x.view(B, C, H, W)
        elif scan_id == 1:
            return x.view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, H, W)
        elif scan_id == 2:
            x = x.view(B, C, -1)
            return torch.flip(x, dims=[-1]).view(B, C, H, W)
        elif scan_id == 3:
            x = x.view(B, C, H, W)
            return torch.flip(x.view(B, C, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]).view(B, C, H, W)
        
        
################# K=0 ################

# do not change x
# only flatten 

###! just to be aligned with K=1, 2, 4, 8
#! the rearrange is not necessary, just for alignment

def cross_scan_k0(x):
    B, C, H, W = x.size()
    
    return x.view(B, 1, C, H, W)
    
def cross_merge_k0(x):
    B, _, C, H, W = x.size()
    x = x.squeeze(1)  # k to be 1
    return x#.view(B, C, -1)


###################################### RWKV ############################################

wkv_version = int(os.getenv('WKV_VERSION', 6))
logger.info(f'WKV_VERSION is set to {wkv_version}')

logger.info('-'*80)
if wkv_version == 6:
    _TMAX_DEFAULT = 4096
    T_MAX = int(os.getenv("T_MAX", _TMAX_DEFAULT))
    if T_MAX > _TMAX_DEFAULT:
        logger.warning(f"T_MAX is set to {T_MAX}, which is greater than {_TMAX_DEFAULT} by default. This may consume a lot of memory")
    else:
        logger.info(f"T_MAX is set to {T_MAX}")
    HEAD_SIZE = int(os.getenv("HEAD_SIZE", 32))
    TIME_DECAY_DIM = int(os.getenv('TIME_DECAY_DIM', 32))  # previous value: 64
    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_cuda_vrwkv.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_op_vrwkv.cpp"
    BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv6_{T_MAX}/"
    os.makedirs(BUILD_PATH, exist_ok=True)
    
    logger.info('loading CUDA extension for [green]Vision RWKV6 Multi-Modal Operator[/green]🎉')
    logger.info(f'with configure: [g] HEAD_SIZE={HEAD_SIZE}, TIME_DECAY_DIM={TIME_DECAY_DIM}, T_MAX={T_MAX}[/g]')
    wkv6_cuda = load(
        name="wkv6",
        sources=[CPP_PATH, CUDA_PATH],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T_MAX}",
        ],
        build_directory=BUILD_PATH,
    )
elif wkv_version == 7:
    HEAD_SIZE = 32
    T_MAX = 256 * 256
    HEAD_SIZE = int(os.getenv("HEAD_SIZE", 32))
    TIME_DECAY_DIM = int(os.getenv('TIME_DECAY_DIM', 32))  # previous value: 64
    
    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv7.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv7_op.cpp"
    BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv7_{T_MAX}/"
    os.makedirs(BUILD_PATH, exist_ok=True)
    
    logger.info('loading CUDA extension for [green]Vision RWKV7 Multi-Modal Operator[/green]🎉')
    logger.info(f'with configure: [g] HEAD_SIZE={HEAD_SIZE}, TIME_DECAY_DIM={TIME_DECAY_DIM}, T_MAX={T_MAX}[/g]')
    load(name="wkv7",
         sources=[CPP_PATH, CUDA_PATH],
         is_python_module=False,
         verbose=True,
         extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T_MAX}"],
         build_directory=BUILD_PATH,
         )
        
logger.info(f'-'*80)

################################################## RWKV ###################################################

class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            DTYPE = torch.float32  # for LLM is bf16 is stable, when in image task, we recommend fp32
            assert HEAD_SIZE == C // H
            assert r.dtype == DTYPE
            assert w.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert a.dtype == DTYPE
            assert b.dtype == DTYPE
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y

def RUN_CUDA_RWKV7(r, w, k, v, a, b):
    return WKV_7.apply(r, w, k, v, a, b)


class WKV_6(torch.autograd.Function):
    @staticmethod
    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y
    
    @staticmethod
    # @torch.cuda.amp.custom_bwd
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            if not gy.is_contiguous():
                gy = gy.contiguous()
            # assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gw = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gu = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    # B = B.item()
    # T = T.item()
    # C = C.item()
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

################################################## Shift ###################################################

def q_shift_multihead(
    input,
    shift_pixel=1,
    head_dim=HEAD_SIZE,
    patch_resolution=None,
    with_cls_token=False,
):
    B, C, N = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1]
    )  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0 : int(head_dim * 1 / 4), :, shift_pixel:W] = input[
        :, :, 0 : int(head_dim * 1 / 4), :, 0 : W - shift_pixel
    ]
    output[:, :, int(head_dim / 4) : int(head_dim / 2), :, 0 : W - shift_pixel] = input[
        :, :, int(head_dim / 4) : int(head_dim / 2), :, shift_pixel:W
    ]
    output[:, :, int(head_dim / 2) : int(head_dim / 4 * 3), shift_pixel:H, :] = input[
        :, :, int(head_dim / 2) : int(head_dim / 4 * 3), 0 : H - shift_pixel, :
    ]
    output[:, :, int(head_dim * 3 / 4) : int(head_dim), 0 : H - shift_pixel, :] = input[
        :, :, int(head_dim * 3 / 4) : int(head_dim), shift_pixel:H, :
    ]
    if with_cls_token:
        output = output.reshape(B, C, N - 1)#.transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N)#.transpose(1, 2)
    return output


def groups_q_shift(input: torch.Tensor,
                   shift_pixel: int=1,
                   gamma: float=1/4,
                   H: int=64, 
                   W: int=64,
                   K: int=2):
    assert gamma <= 1/4
    B, KC, N = input.size()
    C = KC // K
    input = input.reshape(B, K, C, H, W)
    output = torch.zeros_like(input)
    output[..., 0:int(C*gamma), :, shift_pixel:W] = input[..., 0:int(C*gamma), :, 0:W-shift_pixel]
    output[..., int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[..., int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[..., int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[..., int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[..., int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[..., int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[..., int(C*gamma*4):, :, :] = input[..., int(C*gamma*4):, :, :]
    output = output.flatten(3)
    output = output.view(B, KC, H*W)
    return output


class ShiftByConv(nn.Module):
    def __init__(self, chan,):
        super().__init__()
        self.dwconv = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1, groups=chan, padding_mode='reflect')
        
    def forward(self, input, H, W):
        bs, c = input.shape[:2]
        input = input.view(bs, c, H, W)
        input = self.dwconv(input)
        input = input.view(bs, c, -1)
        
        return input
    
    
################################################# Vision RWKV v6 and v7 Module ###################################################

class VRWKV_SpatialMix_V7(nn.Module):
    def __init__(self,
                 n_embd,
                 n_head,
                 n_layer,
                 layer_id,
                 shift_mode="none",
                 shift_pixel=1,
                 n_groups=2,
                 attn_bias=False,
                 init_mode="fancy",
                 key_norm=False,
                 with_cls_token=False,
                 with_cp=False,
                 img_txt_cat_order=0,
                 scan_K=1,):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = n_embd // self.head_size
        self.n_embd = n_embd
        self.with_cp = with_cp
        self.img_txt_cat_order = img_txt_cat_order
        self.shift_mode = shift_mode
        self.shift_pixel = shift_pixel
        self.n_groups = n_groups
        self.attn_bias = attn_bias
        self.init_mode = init_mode
        self.key_norm = key_norm
        self.with_cls_token = with_cls_token
        
        assert n_embd % self.n_head == 0

        with torch.no_grad():
            ddd = torch.empty(1, n_embd, 1)
            self.time_maa_x = nn.Parameter(ddd)
            self.time_maa_r = nn.Parameter(ddd)
            self.time_maa_w = nn.Parameter(ddd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_v = nn.Parameter(ddd)
            self.time_maa_a = nn.Parameter(ddd)
            self.time_maa_g = nn.Parameter(ddd)

            decay_speed = torch.empty(n_embd)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, n_embd, 1))

            self.time_faaaa = nn.Parameter(torch.empty(self.n_head, self.head_size))
            self.time_aaaaa = nn.Parameter(torch.empty(1, n_embd, 1))

            D_MIX_LORA = HEAD_SIZE
            self.time_maa_w1 = nn.Parameter(torch.empty(D_MIX_LORA * 6, n_embd))
            self.time_maa_w2 = nn.Parameter(torch.empty(6, D_MIX_LORA, n_embd))

            # time decay
            D_DECAY_LORA = TIME_DECAY_DIM
            self.time_decay_w1 = nn.Parameter(torch.empty(D_DECAY_LORA, n_embd))
            self.time_decay_w2 = nn.Parameter(torch.empty(n_embd, D_DECAY_LORA))

            D_AAA_LORA = TIME_DECAY_DIM
            self.time_aaa_w1 = nn.Parameter(torch.empty(D_AAA_LORA, n_embd))
            self.time_aaa_w2 = nn.Parameter(torch.empty(n_embd, D_AAA_LORA))

            D_KKK_LORA = TIME_DECAY_DIM
            self.time_kkk_w1 = nn.Parameter(torch.empty(D_KKK_LORA, n_embd))
            self.time_kkk_w2 = nn.Parameter(torch.empty(n_embd, D_KKK_LORA))

            # gate
            D_GATE_LORA = n_embd // 4  # can be TIME_DECAY_DIM or just a dim smaller than n_embd
            self.gate_w1 = nn.Parameter(torch.empty(D_GATE_LORA, n_embd))
            self.gate_w2 = nn.Parameter(torch.empty(n_embd, D_GATE_LORA))
            # gate_divisor = 6  # 4
            # self.gate_w1 = nn.Conv1d(n_embd, n_embd // gate_divisor, 1, bias=attn_bias)
            # self.gate_w2 = nn.Conv1d(n_embd // gate_divisor, n_embd, 1, bias=attn_bias)

            if shift_mode == "q_shift_multihead":
                self.shift_func = q_shift_multihead
            elif shift_mode == 'conv':
                self.shift_func = ShiftByConv(self.n_embd)
            elif shift_mode == 'q_shift':
                self.shift_func = partial(groups_q_shift, K=scan_K)
            self.key = nn.Conv1d(n_embd, n_embd, 1, bias=attn_bias, groups=n_groups)
            self.value = nn.Conv1d(n_embd, n_embd, 1, bias=attn_bias, groups=n_groups)
            self.receptance = nn.Conv1d(n_embd, n_embd, 1, bias=attn_bias, groups=n_groups)
            self.output = nn.Conv1d(n_embd, n_embd, 1, bias=False, groups=n_groups)
            
            # head_size_divisor = 8
            # self.ln_x = nn.GroupNorm(self.n_head, n_embd, eps=(1e-5)*(head_size_divisor**2))
            self.ln_x = RMSNorm(-1, n_embd, 3)
            
    def cat_special_tokens_to_seqs(self, 
                            img: torch.Tensor, 
                            txt: torch.Tensor, 
                            img_token: torch.Tensor, 
                            txt_token: torch.Tensor):
        # add special tokens to sequences
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        
        assert C_t == C_i, "modalities should have the same channels"
        
        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"
        
        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)
        
        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)
        
        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi,
                              sot, txt, sot], dim=-1)
        else:
            return torch.cat([sot, txt, eot,
                              soi, img, eoi], dim=-1)

    def forward(self, x, txt=None, patch_resolution=None, mm_tokens=None):
        # B, T, C = x.size()
        H = self.n_head
        B, C, T = x.size()

        if self.shift_mode == 'q_shift_multihead':
            xx = (
                self.shift_func(
                    x,
                    self.shift_pixel,
                    patch_resolution=patch_resolution,
                    with_cls_token=self.with_cls_token,
                )
            )
        elif self.shift_mode == 'q_shift':
            xx = self.shift_func(x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1])
        elif self.shift_mode == 'conv':
            xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
        else:
            xx = x
            
        # fusion img_feature and llm_feature
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(-1) + 4
            else:
                x = torch.cat([x, txt], dim=-1)
                xx = torch.cat([xx, txt], dim=-1)
                T = T + txt.size(-1)
            
        xxx = x * self.time_maa_x + xx * (1 - self.time_maa_x)
        # [D_MIX_LORA * 6, C] @ [B, C, T] -> [B, D_MIX_LORA * 6, T]
        xxx = torch.tanh(self.time_maa_w1 @ xxx)
        xxx = rearrange(xxx, 'b (n d) t -> n (b t) d', b=B, t=T, n=6)
        
        xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, -1).transpose(-2, -1)
        mr, mw, mk, mv, ma, mg = xxx.unbind(dim=0)

        xr = x + xx * (self.time_maa_r + mr)
        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xa = x + xx * (self.time_maa_a + ma)
        xg = x + xx * (self.time_maa_g + mg)

        # conv(xr): [B, C, T] -> [B, C, T]
        # transpose(1, 2): [B, C, T] -> [B, T, C]
        r = self.receptance(xr).transpose(1, 2).contiguous()
        w = -F.softplus(
            -(self.time_decay + self.time_decay_w2 @ torch.tanh(self.time_decay_w1 @ xw))
        ) - 0.5 # soft-clamp to (-inf, -0.5)
        w = w.transpose(1, 2).contiguous()
        k = self.key(xk).transpose(1, 2).contiguous()
        v = self.value(xv).transpose(1, 2).contiguous()
        # gate: conv1d(xg) -> [B, C, T]
        # g = self.gate_w2(torch.tanh(self.gate_w1(xg)))
        
        # [D_GATE_LORA, C] @ [B, C, T] -> [B, D_GATE_LORA, T]
        # [C, D_GATE_LORA] @ [B, D_GATE_LORA, T] -> [B, C, T] -> [B, T, C]
        g = (self.gate_w2 @ torch.tanh(self.gate_w1 @ xg)).transpose(1, 2).contiguous()

        # [D_KKK_LORA, C] @ [B, C, T] -> [B, D_KKK_LORA, T]
        # [C, D_KKK_LORA] @ [B, D_KKK_LORA, T] -> [B, C, T] -> [B, T, C]
        kk = k + (self.time_kkk_w2 @ torch.tanh(self.time_kkk_w1 @ xk)).transpose(1, 2).contiguous()
        kk = F.normalize(kk, dim=-1, p=2.0)
        
        # [D_AAA_LORA, C] @ [B, C, T] -> [B, D_AAA_LORA, T]
        # [C, D_AAA_LORA] @ [B, D_AAA_LORA, T] -> [B, C, T] -> [B, T, C]
        a = torch.sigmoid(
            self.time_aaaaa + self.time_aaa_w2 @ (self.time_aaa_w1 @ xa)
        ) * 2.0 # a is "in-context learning rate"
        a = a.transpose(1, 2).contiguous()
        
        k = k * torch.clamp(w*0.5,max=0).exp()
        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk*a)  # [B, T, C]

        x = self.ln_x(x)
        
        # r, k, v: [B, T, C] -> [B, T, H, C//H]
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = x.transpose(1, 2)
        
        # import ipdb; ipdb.set_trace()
        x = self.output(x * g.transpose(1, 2))
        return x



class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        n_groups=2,
        attn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        img_txt_cat_order=0,
        scan_K=1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.img_txt_cat_order = img_txt_cat_order

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == "q_shift_multihead":
            self.shift_func = q_shift_multihead
        elif shift_mode == 'conv':
            self.shift_func = ShiftByConv(self.n_embd)
        elif shift_mode == 'q_shift':
            self.shift_func = partial(groups_q_shift, K=scan_K)

        # self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        
        self.key = nn.Conv1d(n_embd, self.attn_sz, 1, bias=attn_bias, groups=n_groups)
        self.value = nn.Conv1d(n_embd, n_embd, 1, bias=attn_bias, groups=n_groups)
        self.receptance = nn.Conv1d(n_embd, self.attn_sz, 1, bias=attn_bias, groups=n_groups)
        self.gate = nn.Conv1d(n_embd, self.attn_sz, 1, bias=attn_bias, groups=n_groups)
        
        if key_norm:
            self.key_norm = RMSNorm(-1, n_embd, 3) #nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        # self.output = nn.Linear(self.attn_sz, n_embd, bias=False)
        self.output = nn.Conv1d(self.attn_sz, n_embd, 1, bias=False, groups=n_groups)

        # self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
        self.ln_x = RMSNorm(-1, self.attn_sz, 3) # nn.LayerNorm(self.attn_sz, eps=1e-5)
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == "fancy":
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, self.n_embd, 1)
                for i in range(self.n_embd):
                    ddd[0, i, 0] = i / self.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = HEAD_SIZE  # generate TIME_MIX for w,k,v,r,g
                init_value = 0.01
                self.time_maa_w1 = nn.Parameter(
                    torch.zeros(TIME_MIX_EXTRA_DIM * 5, self.n_embd).uniform_(-init_value, init_value)
                )
                self.time_maa_w2 = nn.Parameter(
                    torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-init_value, init_value)
                )

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1, self.attn_sz, 1))

                TIME_DECAY_EXTRA_DIM = TIME_DECAY_DIM
                self.time_decay_w1 = nn.Parameter(
                    torch.zeros(TIME_DECAY_EXTRA_DIM, self.n_embd).uniform_(-init_value, init_value)
                )
                self.time_decay_w2 = nn.Parameter(
                    torch.zeros(self.attn_sz, TIME_DECAY_EXTRA_DIM).uniform_(-init_value, init_value)
                )

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError
        
    def cat_special_tokens_to_seqs(self, 
                                   img: torch.Tensor, 
                                   txt: torch.Tensor, 
                                   img_token: torch.Tensor, 
                                   txt_token: torch.Tensor):
        # add special tokens to sequences
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        
        assert C_t == C_i, "modalities should have the same channels"
        
        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"
        
        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)
        
        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)
        
        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi,
                              sot, txt, sot], dim=-1)
        else:
            return torch.cat([sot, txt, eot,
                              soi, img, eoi], dim=-1)
            
    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"
        
        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=-1)
        else:
            return torch.cat([txt, img], dim=-1)

    def jit_func(self, x, txt=None, patch_resolution=None, mm_tokens=None):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, C, T = x.size()

        #! shift will make the artifacts in corners
        if self.shift_mode == 'q_shift_multihead':
            xx = (
                self.shift_func(
                    x,
                    self.shift_pixel,
                    patch_resolution=patch_resolution,
                    with_cls_token=self.with_cls_token,
                )
            )
        elif self.shift_mode == 'q_shift':
            xx = self.shift_func(x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1])
            
        elif self.shift_mode == 'conv':
            xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
        else:
            xx = x
            
        # fusion img_feature and llm_feature
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(-1) + 4
            else:
                # print('spatial mix cat order:', self.img_txt_cat_order)
                # x = torch.cat([x, txt], dim=-1)
                # xx = torch.cat([xx, txt], dim=-1)
                x = self.cat_seqs_wo_special_tokens(x, txt)
                xx = self.cat_seqs_wo_special_tokens(xx, txt)
                T = T + txt.size(-1)
        
        xxx = x * self.time_maa_x + xx * (1 - self.time_maa_x)  # [B, C, T]
        # xxx = x + xx * self.time_maa_x
        
        # [5*max_dim, C] @ [B, C, T] -> [B, 5*max_dim, T]
        xxx = torch.tanh(self.time_maa_w1 @ xxx)
        xxx = rearrange(xxx, 'b (n d) t -> n (b t) d', b=B, t=T, n=5)
        
        # [5, B*T, max_dim] @ [5, max_dim, C] -> [5, B*T, C] -> [5, B, T, C] -> [5, B, C, T]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1).transpose(-2, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)  # [B, C, T]

        xw = x + xx * (self.time_maa_w + mw)  # [B, C, T] + [B, C, T] * ([1, C, 1] + [B, C, T]) -> [B, C, T]
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        # conv(xr): [B, C, T] -> [B, C, T]
        # transpose(1, 2): [B, C, T] -> [B, T, C]
        r = self.receptance(xr).transpose(1, 2).contiguous()
        k = self.key(xk).transpose(1, 2).contiguous()
        v = self.value(xv).transpose(1, 2).contiguous()
        g = F.silu(self.gate(xg))

        # ([B, T, C] @ [C, max_dim]) @ [max_dim, C] -> [B, T, C]
        # ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        
        # [C, max_dim] @ ([max_dim, C] @ [B, C, T]) -> [B, C, T]
        ww = self.time_decay_w2 @ torch.tanh(self.time_decay_w1 @ xw)
        # [1, C, 1] + [B, C, T] -> [B, C, T] -> [B, T, C]
        w = self.time_decay + ww
        w = w.transpose(1, 2).contiguous()

        return r, k, v, g, w, T

    def jit_func_2(self, x, g):
        B, T, C = x.size()

        x = self.ln_x(x).transpose(1, 2) # [B, C, T]
        x = self.output(x * g)  # [B, C, T]
        return x

    def forward(self, x, txt=None, patch_resolution=None, mm_tokens: "tuple[torch.Tensor] | None"=None):
        def _inner_forward(x, txt, patch_resolution):
            B, C, T= x.size()
            self.device = x.device

            r, k, v, g, w, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            if self.key_norm is not None:
                x = self.key_norm(x)
            return self.jit_func_2(x, g)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, txt, patch_resolution, use_reentrant=False)
        else:
            x = _inner_forward(x, txt, patch_resolution)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        hidden_rate=4,
        n_groups=2,
        ffn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        scan_K=1,
        img_txt_cat_order=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        self.img_txt_cat_order = img_txt_cat_order
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == "q_shift_multihead":
            self.shift_func = q_shift_multihead
        elif shift_mode == 'q_shift':
            self.shift_func = partial(groups_q_shift, K=scan_K)
        elif shift_mode == 'conv':
            self.shift_func = ShiftByConv(self.n_embd)

        hidden_sz = hidden_rate * n_embd
        # self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key = nn.Conv1d(n_embd, hidden_sz, 1, bias=ffn_bias, groups=n_groups)
        if key_norm:
            self.key_norm = RMSNorm(-1, hidden_sz, 3)  # nn.GroupNorm(1, hidden_sz)  # nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        # self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        # self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance = nn.Conv1d(n_embd, n_embd, 1, bias=ffn_bias, groups=n_groups)
        self.value = nn.Conv1d(hidden_sz, n_embd, 1, bias=ffn_bias, groups=n_groups)

    def _init_weights(self, init_mode):
        if init_mode == "fancy":
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                x = torch.ones(1, self.n_embd, 1)
                for i in range(self.n_embd):
                    x[0, i, 0] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError
    
    ########################################################## Channel mixing multi-modal version #########################################################
    # NOTE: channel mixing cat same as spatial mixing
    # but we do not use this multi-modal version, the mix concatenation in a simple MLP
    
    def cat_special_tokens_to_seqs(self, 
                                   img: torch.Tensor, 
                                   txt: torch.Tensor, 
                                   img_token: torch.Tensor, 
                                   txt_token: torch.Tensor):
        # add special tokens to sequences
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        
        assert C_t == C_i, "modalities should have the same channels"
        
        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"
        
        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)
        
        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)
        
        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi,
                              sot, txt, sot], dim=-1)
        else:
            return torch.cat([sot, txt, eot,
                              soi, img, eoi], dim=-1)
        
    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, C_t, L_t = txt.shape
        bs_img, C_i, L_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"
        
        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=-1)
        else:
            return torch.cat([txt, img], dim=-1)
    
    #########################################################################################################################################################
    
    def forward(self, x, txt=None, patch_resolution=None, mm_tokens: "tuple[torch.Tensor] | None"=None):
        def _inner_forward(x, txt, patch_resolution):
            B, C, T = x.size()
            
            if self.shift_mode == 'q_shift_multihead':
                xx = self.shift_func(
                            x,
                            self.shift_pixel,
                            patch_resolution=patch_resolution,
                            with_cls_token=self.with_cls_token,
                        )
            elif self.shift_mode == 'q_shift':
                xx = self.shift_func(x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1])
            elif self.shift_mode == 'conv':
                xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
            else:
                xx = x
                
            if txt is not None:
                if mm_tokens is not None:
                    x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                    xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                    T = T + txt.size(-1) + 4
                else:
                    # x = torch.cat([x, txt], dim=-1)
                    # xx = torch.cat([xx, txt], dim=-1)
                    # print('channel mix cat order:', self.img_txt_cat_order)
                    x = self.cat_seqs_wo_special_tokens(x, txt)
                    xx = self.cat_seqs_wo_special_tokens(xx, txt)
                    T = T + txt.size(-1)
                
            # [B, C, T] * [1, C, 1] + [B, C, T] * [1, C, 1] -> [B, C, T]
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            # x = F.silu(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, txt, patch_resolution, use_reentrant=False)
        else:
            x = _inner_forward(x, txt, patch_resolution)
        return x
    
#################################################################################################################


def vrwkv6_flops(inp, outp):
    sizes = inp[0].type().sizes()
    B, T, C = sizes
    head_size = C // 32  # nhead: 32
    dim = C
    n = T
    
    TIME_MIX_EXTRA_DIM = 32
    TIME_DECAY_EXTRA_DIM = TIME_DECAY_DIM
    
    addi_flops = 0
    addi_flops += n * dim * (TIME_MIX_EXTRA_DIM * 10 + TIME_DECAY_EXTRA_DIM * 2 + 7 * head_size + 17)
    addi_flops += n * (TIME_MIX_EXTRA_DIM * 5 + TIME_DECAY_EXTRA_DIM)
    return addi_flops * B


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                       shift_pixel, init_mode, key_norm=key_norm,
                                       with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cls_token=with_cls_token)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


if __name__ == '__main__':
    # x = torch.randn(2, 64, 256).cuda()
    # model = VRWKV_SpatialMix_V7(64, 64//32, 2, 1).cuda()
    # # model = VRWKV_SpatialMix_V6(64, 64//32, 2, 1).cuda()
    # # model = VRWKV_SpatialMix_V6(64, 2, 2, 1).cuda()
    # y = model(x, patch_resolution=(16, 16))
    # y.sum().backward()
    # print(y.shape)
    # print(model.gate.weight.grad)
    
    
    # x = torch.randn(1, 3, 64, 64).cuda()
    # x.requires_grad_()
    # scan = CrossScanTritonSelect
    # y = scan.apply(x)
    # print(y.shape)
    
    # y.sum().backward()
    # print(x.grad)
    
    #* test ESS
    
    from torchvision.io import read_image
    from einops import rearrange
    
    device = 'cuda'
    x = read_image('/Data3/cao/ZiHanCao/datasets/MEF-SICE/over/017.jpg')[None, ...].to(device) / 255.0
    x = torch.nn.functional.interpolate(x, (768, 1024), mode='bilinear', align_corners=False)
    H, W = x.shape[-2:]
    
    # window extraction
    def window_extract_2d(x: torch.Tensor, window_size: int):
        """
        Window extraction for 2D image and incorporate with ESS (scan) algorithm
        to reduce the computation complexity into window size.
        
        Difference from Swin's window partition.
        """
        
        B, C, H, W = x.shape
        
        assert H % window_size == 0 and W % window_size == 0, "image height and width must be divisible by window size"

        # NOTE: different from the window partition
        # nh = H // window_size
        # nw = W // window_size
        # x[..., ::nh, ::nw]
        x = rearrange(x, 'b c (p1 nh) (p2 nw) -> (b nh nw) c p1 p2', p1=window_size, p2=window_size)
    
        return x
    
    def window_reverse_2d(windows: torch.Tensor, window_size: int, H: int, W: int, format='bchw'):
        """
        Window reverse for 2D image and incorporate with ESS (merge) algorithm
        to reduce the computation complexity into window size.
        """
        if format == 'bchw':
            in_format = '(b nh nw) c p1 p2'
        elif format == 'bcl':
            in_format = '(b nh nw) c (p1 p2)'
        else:
            raise NotImplementedError
        
        # windows: [B x nh x nw, c, p1, p2]
        assert windows.shape[0] % (H * W / window_size / window_size) == 0, "windows shape must be divisible by (H * W / window_size / window_size)"
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        
        x = rearrange(windows, f'{in_format} -> b c (p1 nh) (p2 nw)', p1=window_size, p2=window_size, b=B, 
                    c=windows.shape[1], nh=H // window_size, nw=W // window_size)
        
        return x
    
    # window extraction
    x = window_extract_2d(x, 128)
    h = w = 128
    print(x.shape)
    
    # h = 768
    # w = 1024
    
    # ESS
    # [b, k, c, h * w]
    y1 = CrossScanTritonSelect.apply(x, 0)
    y2 = CrossScanTritonSelect.apply(x, 1)
    
    print(y1.shape, y2.shape)
    
    # ESS merge
    x1 = CrossMergeTritonSelect.apply(y1.view(x.size(0), 2, 3, h, w), 0)
    x2 = CrossMergeTritonSelect.apply(y2.view(x.size(0), 2, 3, h, w), 1)
    
    x1 = x1.view(x.size(0), 3, h, w) / 2
    x2 = x2.view(x.size(0), 3, h, w) / 2
    
    # window reverse
    y1 = y1.flatten(1, 2)
    y2 = y2.flatten(1, 2)
    
    y1 = window_reverse_2d(y1, 128, H, W, format='bcl')
    y2 = window_reverse_2d(y2, 128, H, W, format='bcl')
    
    
    print(x1.shape, x2.shape)