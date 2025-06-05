# Copyright (c) Shanghai AI Lab. All rights reserved.
from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import triton
from timm.layers import DropPath
from einops import rearrange


from model.module.csm_triton import (triton_cross_scan,
                                     triton_cross_merge,
                                     triton_cross_scan_same_and_trans,
                                     triton_cross_merge_trans_and_flips,
                                     triton_cross_merge_same_and_trans,
                                     triton_cross_scan_trans_and_flips)

logger = logging.getLogger(__name__)

    
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
        return y.view(B, 4, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x

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
        return x.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y

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


# Vmamba cross scan
    

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
        return y.view(B, 4, C, -1)
    
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
        return x.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y


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
        
        return y.view(B, 2, C, -1)
    
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
            
        return x.view(B, C, -1)
    
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

T_MAX = 8192
HEAD_SIZE = 32
CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_cuda_vrwkv.cu"
CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_op_vrwkv.cpp"
# BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv6/"

from torch.utils.cpp_extension import load

logger.info('loading CUDA extension for [green] Vision RWKV6 [/green]')
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
    # build_directory=BUILD_PATH,
)

class WKV_6(torch.autograd.Function):
    @staticmethod
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
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
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
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


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
                   W: int=64):
    assert gamma <= 1/4
    K = 2
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
        self.dwconv = nn.Conv2d(chan, chan, kernel_size=5, stride=1, padding=2, groups=chan)
        
    def forward(self, input, H, W):# -> Any:
        bs, c = input.size(0), input.size(1)
        input = input.view(bs, c, H, W)
        input = self.dwconv(input)
        input = input.view(bs, c, -1)
        
        return input


class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="q_shift",
        shift_pixel=1,
        n_groups=2,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

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
        elif shift_mode == 'q_shift':
            self.shift_func = groups_q_shift

        # self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        
        self.key = nn.Conv1d(n_embd, self.attn_sz, 1, bias=False, groups=n_groups)
        self.value = nn.Conv1d(n_embd, n_embd, 1, bias=False, groups=n_groups)
        self.receptance = nn.Conv1d(n_embd, self.attn_sz, 1, bias=False, groups=n_groups)
        self.gate = nn.Conv1d(n_embd, self.attn_sz, 1, bias=False, groups=n_groups)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        # self.output = nn.Linear(self.attn_sz, n_embd, bias=False)
        self.output = nn.Conv1d(self.attn_sz, n_embd, 1, bias=False, groups=n_groups)

        # self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
        self.ln_x = nn.LayerNorm(self.attn_sz, eps=1e-5)
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
                self.time_maa_x_cond = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(
                    torch.zeros(TIME_MIX_EXTRA_DIM * 4, self.n_embd).uniform_(-0.01, 0.01)
                )
                self.time_maa_w2 = nn.Parameter(
                    torch.zeros(4, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-0.01, 0.01)
                )
                
                self.time_maa_w1_cross = nn.Parameter(
                    torch.zeros(TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-0.01, 0.01)
                )
                self.time_maa_w2_cross = nn.Parameter(
                    torch.zeros(TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-0.01, 0.01)
                )

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -5 + 8 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1, self.attn_sz, 1))

                TIME_DECAY_EXTRA_DIM = 64
                self.time_decay_w1 = nn.Parameter(
                    torch.zeros(TIME_DECAY_EXTRA_DIM, self.n_embd).uniform_(-0.01, 0.01)
                )
                self.time_decay_w2 = nn.Parameter(
                    torch.zeros(self.attn_sz, TIME_DECAY_EXTRA_DIM).uniform_(-0.01, 0.01)
                )

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError

    # @torch.compile(fullgraph=True)
    def get_hiddens(self, x, x_cond, xx, xx_cond):
        B, C, T = x.size()
        
        xxx = x + xx * self.time_maa_x  # [B, C, T]
        xxx_cond = x_cond + xx_cond * self.time_maa_x_cond  # [B, C, T]
        
        ## mw, mk, mr, mg
        # [4*max_dim, C] @ [B, C, T] -> [B, 4*max_dim, T]
        xxx = torch.tanh(self.time_maa_w1 @ xxx)
        # make torch.compile happy
        # xxx = rearrange(xxx, 'b (n d) t -> n (b t) d', b=B, t=T, n=4)
        xxx = xxx.view(B, 4, -1, T).permute(1, 0, 2, 3).reshape(4, B * T, -1)
        
        # [4, B*T, max_dim] @ [4, max_dim, C] -> [4, B*T, C] -> [4, B, T, C] -> [4, B, C, T]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1).transpose(-2, -1)
        mw, mk, mr, mg = xxx.unbind(dim=0)  # [B, C, T]
        
        ## mv
        # [max_dim, C] @ [B, C, T] -> [B, max_dim, T] -> [B, T, max_dim]
        xxx_cond = torch.tanh(self.time_maa_w1_cross @ xxx_cond).transpose(1, 2)
        # [B, T, max_dim] @ [max_dim, C] -> [B, T, C]
        mv = (xxx_cond @ self.time_maa_w2_cross).transpose(1, 2)
        # fuse mv and mk
        mv = (mv + mk) / 2

        xw = x + xx * (self.time_maa_w + mw)  # [B, C, T] + [B, C, T] * ([1, C, 1] + [B, C, T]) -> [B, C, T]
        xk = x + xx * (self.time_maa_k + mk)
        xv = x_cond + xx_cond * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)
        
        return xw, xk, xv, xr, xg
        
    
    def jit_func(self, x, x_cond, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        if self.shift_mode == 'q_shift_multihead':
            xx = self.shift_func(
                    x,
                    self.shift_pixel,
                    patch_resolution=patch_resolution,
                    with_cls_token=self.with_cls_token,
                ) - x
            xx_cond = self.shift_func(
                    x_cond,
                    self.shift_pixel,
                    patch_resolution=patch_resolution,
                    with_cls_token=self.with_cls_token,
                ) - x_cond 
            
        elif self.shift_mode == 'q_shift':
            xx = self.shift_func(x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]) - x
            xx_cond = self.shift_func(x_cond, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]) - x_cond
        else:
            xx = x
            xx_cond = x_cond
            
        xw, xk, xv, xr, xg = self.get_hiddens(x, x_cond, xx, xx_cond)
        
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
        # [1, C, 1] + [B, C, T] -> [B, C, T]
        w = self.time_decay + ww
        w = w.transpose(1, 2).contiguous()

        return r, k, v, g, w

    # @torch.compile(dynamic=True)
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        # x = x.view(B * T, C)

        x = self.ln_x(x).transpose(1, 2) # [B, C, T]
        x = self.output(x * g)  # [B, C, T]
        return x

    def forward(self, x, x_cond, patch_resolution=None):
        def _inner_forward(x):
            B, C, T = x.size()
            self.device = x.device

            r, k, v, g, w = self.jit_func(x, x_cond, patch_resolution)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            if self.key_norm is not None:
                x = self.key_norm(x)
            return self.jit_func_2(x, g)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="q_shift",
        shift_pixel=1,
        hidden_rate=4,
        n_groups=2,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == "q_shift_multihead":
            self.shift_func = q_shift_multihead
        elif shift_mode == 'q_shift':
            self.shift_func = groups_q_shift
        elif shift_mode == 'conv':
            self.shift_func = ShiftByConv(self.n_embd)

        hidden_sz = hidden_rate * n_embd
        # self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key = nn.Conv1d(n_embd, hidden_sz, 1, bias=False, groups=n_groups)
        if key_norm:
            self.key_norm = nn.GroupNorm(1, hidden_sz)  # nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        # self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        # self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance = nn.Conv1d(n_embd, n_embd, 1, bias=False, groups=n_groups)
        self.value = nn.Conv1d(hidden_sz, n_embd, 1, bias=False, groups=n_groups)

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
    
    def forward(self, x, patch_resolution=None):
        
        # @torch.compile(dynamic=True)
        def _inner_forward(x):
            if self.shift_mode == 'q_shift_multihead':
                xx = self.shift_func(
                            x,
                            self.shift_pixel,
                            patch_resolution=patch_resolution,
                            with_cls_token=self.with_cls_token,
                        )
            elif self.shift_mode == 'q_shift':
                xx = self.shift_func(x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]) - x
            elif self.shift_mode == 'conv':
                xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
            else:
                xx = x
            
            # [B, C, T] * [1, C, 1] + [B, C, T] * [1, C, 1] -> [B, C, T]
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            # x = torch.sigmoid(self.receptance(xr)) * kv
            x = F.silu(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


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


# @BACKBONES.register_module()
# class VRWKV6(BaseBackbone):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_channels=3,
#                  out_indices=-1,
#                  drop_rate=0.,
#                  embed_dims=192,
#                  num_heads=3,
#                  depth=12,
#                  drop_path_rate=0.,
#                  shift_pixel=1,
#                  shift_mode='q_shift_multihead',
#                  init_mode='fancy',
#                  post_norm=False,
#                  key_norm=False,
#                  init_values=None,
#                  hidden_rate=4,
#                  final_norm=True,
#                  interpolate_mode='bicubic',
#                  output_cls_token=False,
#                  with_cls_token=False,
#                  with_cp=False,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         self.embed_dims = embed_dims
#         self.num_extra_tokens = 0
#         self.num_layers = depth
#         self.drop_path_rate = drop_path_rate

#         # Set cls token
#         if output_cls_token:
#             assert with_cls_token is True, f'with_cls_token must be True if' \
#                 f'set output_cls_token to True, but got {with_cls_token}'
#         self.with_cls_token = with_cls_token
#         self.output_cls_token = output_cls_token
#         if self.with_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

#         self.patch_embed = PatchEmbed(
#             in_channels=in_channels,
#             input_size=img_size,
#             embed_dims=self.embed_dims,
#             conv_type='Conv2d',
#             kernel_size=patch_size,
#             stride=patch_size,
#             bias=True)

#         self.patch_resolution = self.patch_embed.init_out_size
#         num_patches = self.patch_resolution[0] * self.patch_resolution[1]

#         # Set position embedding
#         self.interpolate_mode = interpolate_mode
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, self.embed_dims))

#         self.drop_after_pos = nn.Dropout(p=drop_rate)

#         if isinstance(out_indices, int):
#             out_indices = [out_indices]
#         assert isinstance(out_indices, Sequence), \
#             f'"out_indices" must by a sequence or int, ' \
#             f'get {type(out_indices)} instead.'
#         for i, index in enumerate(out_indices):
#             if index < 0:
#                 out_indices[i] = self.num_layers + index
#             assert 0 <= out_indices[i] <= self.num_layers, \
#                 f'Invalid out_indices {index}'
#         self.out_indices = out_indices
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.layers = ModuleList()
#         for i in range(self.num_layers):
#             self.layers.append(Block(
#                 n_embd=embed_dims,
#                 n_head=num_heads,
#                 n_layer=depth,
#                 layer_id=i,
#                 shift_pixel=shift_pixel,
#                 shift_mode=shift_mode,
#                 hidden_rate=hidden_rate,
#                 drop_path=dpr[i],
#                 init_mode=init_mode,
#                 post_norm=post_norm,
#                 key_norm=key_norm,
#                 init_values=init_values,
#                 with_cls_token=with_cls_token,
#                 with_cp=with_cp
#             ))

#         self.final_norm = final_norm
#         if final_norm:
#             self.ln1 = nn.LayerNorm(self.embed_dims)


#     def forward(self, x):
#         B = x.shape[0]
#         x, patch_resolution = self.patch_embed(x)

#         x = x + resize_pos_embed(
#             self.pos_embed,
#             self.patch_resolution,
#             patch_resolution,
#             mode=self.interpolate_mode,
#             num_extra_tokens=self.num_extra_tokens)
#         if self.with_cls_token:
#             cls_tokens = self.cls_token.expand(B, -1, -1)
#             x = torch.cat((x, cls_tokens), dim=1)  # post cls_token

#         x = self.drop_after_pos(x)

#         outs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x, patch_resolution)

#             if i == len(self.layers) - 1 and self.final_norm:
#                 x = self.ln1(x)

#             if i in self.out_indices:
#                 B, _, C = x.shape
#                 if self.with_cls_token:
#                     patch_token = x[:, :-1].reshape(B, *patch_resolution, C)
#                     patch_token = patch_token.permute(0, 3, 1, 2)
#                     cls_token = x[:, -1]
#                 else:
#                     patch_token = x.reshape(B, *patch_resolution, C)
#                     patch_token = patch_token.permute(0, 3, 1, 2)
#                 if self.output_cls_token:
#                     out = [patch_token, cls_token]
#                 else:
#                     out = patch_token
#                 outs.append(out)
#         return tuple(outs)
