#!/usr/bin/env python3
"""
Chikusei数据集生成脚本
基于CAVE数据集的处理流程，适配Chikusei数据集的特殊裁切策略

数据流程:
1. GT (Ground Truth) - 原始128通道高光谱数据
2. HRMSI - 3通道RGB数据  
3. LRHSI_X - GT通过Interp23Tap抗混叠下采样X倍得到的低分辨率高光谱数据
4. lms_X - LRHSI_X通过双线性插值上采样回原尺寸得到的低分辨率多光谱数据

裁切策略:
- 训练集: 按步长重叠裁切出若干个128x128的patch
- 测试集: 从左上角2048x2048区域裁出不重叠的1024x1024的四张

作者: Assistant  
日期: 2025-08-10
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
from pathlib import Path
import scipy.io as sio

# 配置参数
DEFAULT_CONFIG = {
    'chikusei_file': '/data2/users/yujieliang/dataset/Chikusei/Chikusei_ROSIS_RGB.h5',
    'output_dir': '/data2/users/yujieliang/dataset/Chikusei',
    'patch_size': 128,           # 训练patch尺寸
    'test_patch_size': 1024,     # 测试patch尺寸  
    'test_region_size': 2048,    # 测试区域尺寸
    'stride': 32,                # 训练patch步长
    'downsample_factors': [4, 8, 16, 32],
    'compression_level': 9,
    'rgb_bands': [29, 19, 9],    # Chikusei的RGB波段索引 (R, G, B)
}

class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.
    """

    def __init__(self, ratio: int, pad_mode: str = "replicate"):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))
        self.pad_mode = pad_mode

        # Define the 23-tap filter coefficients (CDF23 from MATLAB code)
        cdf23_coeffs = 2.0 * np.array(
            [
                0.5,
                0.305334091185,
                0.0,
                -0.072698593239,
                0.0,
                0.021809577942,
                0.0,
                -0.005192756653,
                0.0,
                0.000807762146,
                0.0,
                -0.000060081482,
            ]
        )
        # Make symmetric
        base_coeffs = np.concatenate([np.flip(cdf23_coeffs[1:]), cdf23_coeffs])
        base_coeffs_t = torch.tensor(base_coeffs, dtype=torch.float32)

        # Reshape kernel for 2D convolution (separable filter)
        kernel_h = base_coeffs_t.view(1, 1, -1, 1)  # Shape (1, 1, 23, 1)
        kernel_w = base_coeffs_t.view(1, 1, 1, -1)  # Shape (1, 1, 1, 23)

        # Register kernels as buffers
        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_w", kernel_w)

        # Calculate padding size (kernel_size=23)
        self.padding = (base_coeffs_t.shape[0] - 1) // 2  # Should be 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ratio == 1:
            return x

        current_img = x
        bs, c, h_curr, w_curr = current_img.shape

        for k in range(self.num_upsamples):
            h_curr *= 2
            w_curr *= 2

            # Upsample by inserting zeros
            upsampled = torch.zeros(
                bs, c, h_curr, w_curr, device=x.device, dtype=x.dtype
            )

            # Place original pixels according to MATLAB logic
            if k == 0:
                upsampled[..., 1::2, 1::2] = current_img
            else:
                upsampled[..., ::2, ::2] = current_img

            # Apply separable convolution with circular padding
            # Pad for horizontal filter (width)
            padded_w = F.pad(
                upsampled, (self.padding, self.padding, 0, 0), mode=self.pad_mode
            )
            kernel_w_grouped = self.kernel_w.repeat(c, 1, 1, 1)
            filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)

            # Pad for vertical filter (height)
            padded_h = F.pad(
                filtered_w, (0, 0, self.padding, self.padding), mode="circular"
            )
            kernel_h_grouped = self.kernel_h.repeat(c, 1, 1, 1)
            filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)

            current_img = filtered_h

        return current_img


def anti_aliasing_downsample(image_tensor, factor, device):
    """使用Interp23Tap进行抗混叠下采样"""
    if factor == 1:
        return image_tensor
    
    # 添加batch维度
    img_batch = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    # 使用Interp23Tap的滤波核进行抗混叠预处理
    cdf23_coeffs = 2.0 * np.array([
        0.5, 0.305334091185, 0.0, -0.072698593239, 0.0, 0.021809577942,
        0.0, -0.005192756653, 0.0, 0.000807762146, 0.0, -0.000060081482,
    ])
    base_coeffs = np.concatenate([np.flip(cdf23_coeffs[1:]), cdf23_coeffs])
    base_coeffs_t = torch.tensor(base_coeffs, dtype=torch.float32, device=device)
    
    # 创建分离滤波核
    kernel_h = base_coeffs_t.view(1, 1, -1, 1)
    kernel_w = base_coeffs_t.view(1, 1, 1, -1)
    padding = (len(base_coeffs) - 1) // 2
    
    # 应用抗混叠滤波
    bs, c, h, w = img_batch.shape
    
    # 水平滤波
    padded_w = F.pad(img_batch, (padding, padding, 0, 0), mode='replicate')
    kernel_w_grouped = kernel_w.repeat(c, 1, 1, 1)
    filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)
    
    # 垂直滤波
    padded_h = F.pad(filtered_w, (0, 0, padding, padding), mode='replicate')
    kernel_h_grouped = kernel_h.repeat(c, 1, 1, 1)
    filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)
    
    # 下采样（子采样）
    downsampled = filtered_h[:, :, ::factor, ::factor]
    
    return downsampled.squeeze(0)  # 移除batch维度


class ChikuseiDatasetGenerator:
    """Chikusei数据集生成器"""
    
    def __init__(self, config=None):
        """初始化生成器"""
        self.config = config or DEFAULT_CONFIG.copy()
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_chikusei_data(self):
        """加载原始Chikusei数据"""
        print("📂 加载原始Chikusei数据...")
        
        chikusei_file = self.config['chikusei_file']
        if not os.path.exists(chikusei_file):
            raise FileNotFoundError(f"Chikusei文件不存在: {chikusei_file}")
        
        # 加载H5文件
        print(f"   从 {chikusei_file} 加载数据...")
        
        with h5py.File(chikusei_file, 'r') as f:
            print(f"   H5文件键值: {list(f.keys())}")
            
            # 加载GT数据 (128通道高光谱)
            if 'gt' not in f:
                raise ValueError("文件中缺少'gt'数据集")
            
            self.gt_data = f['gt'][:].astype(np.float32)  # (128, 2335, 2517)
            print(f"   GT数据形状: {self.gt_data.shape}")
            print(f"   GT数值范围: [{self.gt_data.min():.2f}, {self.gt_data.max():.2f}]")
            
            # 加载HRMSI数据 (3通道RGB)
            if 'HRMSI' not in f:
                raise ValueError("文件中缺少'HRMSI'数据集")
            
            self.hrmsi_data = f['HRMSI'][:].astype(np.float32)  # (3, 2335, 2517)
            print(f"   HRMSI数据形状: {self.hrmsi_data.shape}")
            print(f"   HRMSI数值范围: [{self.hrmsi_data.min():.2f}, {self.hrmsi_data.max():.2f}]")
            
        
        # # 数据归一化到 [0, 1]
        # if self.gt_data.max() > 1.0:
        #     print(f"   GT数据归一化: [{self.gt_data.min()}, {self.gt_data.max()}] -> [0, 1]")
        #     self.gt_data = (self.gt_data - self.gt_data.min()) / (self.gt_data.max() - self.gt_data.min())
        
        # if self.hrmsi_data.max() > 1.0:
        #     print(f"   HRMSI数据归一化: [{self.hrmsi_data.min()}, {self.hrmsi_data.max()}] -> [0, 1]")
        #     self.hrmsi_data = (self.hrmsi_data - self.hrmsi_data.min()) / (self.hrmsi_data.max() - self.hrmsi_data.min())
        
        print(f"   ✅ 数据加载完成!")
        print(f"   最终GT数据: {self.gt_data.shape} (C×H×W)")
        print(f"   最终HRMSI数据: {self.hrmsi_data.shape} (C×H×W)")
        
    def crop_training_patches(self):
        """裁剪训练patches - 重叠裁切"""
        print("✂️ 裁剪训练patches (重叠裁切)...")
        
        patch_size = self.config['patch_size']
        stride = self.config['stride']
        
        c, h, w = self.gt_data.shape
        
        # 计算可裁剪的patch数量
        n_patches_h = (h - patch_size) // stride + 1
        n_patches_w = (w - patch_size) // stride + 1
        total_patches = n_patches_h * n_patches_w
        
        print(f"   原始图像尺寸: {h} × {w}")
        print(f"   Patch尺寸: {patch_size} × {patch_size}")
        print(f"   步长: {stride}")
        print(f"   预计patch数量: {total_patches}")
        
        # 裁剪patches
        gt_patches = []
        hrmsi_patches = []
        
        for i in tqdm(range(0, h - patch_size + 1, stride), desc="裁剪rows"):
            for j in range(0, w - patch_size + 1, stride):
                # GT patch
                gt_patch = self.gt_data[:, i:i+patch_size, j:j+patch_size]  # (C, patch_size, patch_size)
                gt_patches.append(gt_patch)
                
                # HRMSI patch
                hrmsi_patch = self.hrmsi_data[:, i:i+patch_size, j:j+patch_size]  # (3, patch_size, patch_size)
                hrmsi_patches.append(hrmsi_patch)
        
        print(f"   实际生成patch数量: {len(gt_patches)}")
        
        return gt_patches, hrmsi_patches
    
    def crop_test_patches(self):
        """裁剪测试patches - 从左上角2048x2048区域裁出不重叠的1024x1024四张"""
        print("✂️ 裁剪测试patches (2048x2048区域的四个1024x1024)...")
        
        test_region_size = self.config['test_region_size']
        test_patch_size = self.config['test_patch_size']
        
        c, h, w = self.gt_data.shape
        
        # 检查尺寸
        if h < test_region_size or w < test_region_size:
            raise ValueError(f"图像尺寸 {h}×{w} 小于测试区域尺寸 {test_region_size}×{test_region_size}")
        
        print(f"   测试区域: 左上角 {test_region_size} × {test_region_size}")
        print(f"   测试patch尺寸: {test_patch_size} × {test_patch_size}")
        
        # 提取测试区域
        test_region_gt = self.gt_data[:, :test_region_size, :test_region_size]
        test_region_hrmsi = self.hrmsi_data[:, :test_region_size, :test_region_size]
        
        # 裁剪四个不重叠的patches
        gt_patches = []
        hrmsi_patches = []
        
        # 四个patch的左上角坐标
        positions = [
            (0, 0),                                    # 左上
            (0, test_patch_size),                      # 右上  
            (test_patch_size, 0),                      # 左下
            (test_patch_size, test_patch_size)         # 右下
        ]
        
        for i, (start_h, start_w) in enumerate(positions):
            end_h = start_h + test_patch_size
            end_w = start_w + test_patch_size
            
            gt_patch = test_region_gt[:, start_h:end_h, start_w:end_w]
            hrmsi_patch = test_region_hrmsi[:, start_h:end_h, start_w:end_w]
            
            gt_patches.append(gt_patch)
            hrmsi_patches.append(hrmsi_patch)
            
            print(f"   Patch {i+1}: [{start_h}:{end_h}, {start_w}:{end_w}] -> {gt_patch.shape}")
        
        return gt_patches, hrmsi_patches
    
    def generate_training_set(self):
        """生成训练集 - 边处理边存储版本"""
        print("\n" + "="*60)
        print("1️⃣ 生成Chikusei训练集（边处理边存储）")
        print("="*60)
        
        patch_size = self.config['patch_size']
        stride = self.config['stride']
        downsample_factors = self.config['downsample_factors']
        
        # 估算patches数量
        c, h, w = self.gt_data.shape
        n_patches_h = (h - patch_size) // stride + 1
        n_patches_w = (w - patch_size) // stride + 1
        total_estimated_patches = n_patches_h * n_patches_w
        
        print(f"原始图像尺寸: {h} × {w}")
        print(f"估计patches数量: {total_estimated_patches}")
        
        # 首先批量裁剪所有patches
        print("\n✂️ 批量裁剪patches...")
        gt_patches = []
        hrmsi_patches = []
        
        for i in tqdm(range(0, h - patch_size + 1, stride), desc="裁剪patches"):
            for j in range(0, w - patch_size + 1, stride):
                gt_patch = self.gt_data[:, i:i+patch_size, j:j+patch_size]
                hrmsi_patch = self.hrmsi_data[:, i:i+patch_size, j:j+patch_size]
                
                gt_patches.append(gt_patch)
                hrmsi_patches.append(hrmsi_patch)
        
        print(f"实际patches数量: {len(gt_patches)}")
        
        # 创建输出H5文件
        train_file = os.path.join(
            self.config['output_dir'], 
            f'Chikusei_train_patches_stride{stride}_size{patch_size}.h5'
        )
        
        # 边处理边存储
        with h5py.File(train_file, 'w') as f:
            compression_opts = self.config['compression_level']
            
            # 首先保存GT和HRMSI（这些已经准备好了）
            print("\n💾 保存GT和HRMSI...")
            gt_stack = np.stack(gt_patches)
            hrmsi_stack = np.stack(hrmsi_patches)
            
            f.create_dataset('GT', data=gt_stack,
                            #   compression='gzip', compression_opts=compression_opts
                              )
            f.create_dataset('HRMSI', data=hrmsi_stack,
                            #   compression='gzip', compression_opts=compression_opts
                              )
            
            # 释放内存
            del gt_stack, hrmsi_stack
            import gc
            gc.collect()
            
            # 逐个倍率处理LRHSI和LMS
            for factor in downsample_factors:
                print(f"\n🔄 处理下采样倍数 {factor}x...")
                
                # 单独处理当前倍率
                lrhsi_data, lms_data = self.generate_single_factor_data(
                    gt_patches, factor, (patch_size, patch_size)
                )
                
                # 立即保存当前倍率的数据
                print(f"💾 保存 {factor}x 倍率数据...")
                f.create_dataset(f'LRHSI_{factor}', 
                               data=np.stack(lrhsi_data), 
                            #    compression='gzip', compression_opts=compression_opts
                               )
                f.create_dataset(f'lms_{factor}', 
                               data=np.stack(lms_data), 
                            #    compression='gzip', compression_opts=compression_opts
                               )
                
                # 立即释放内存
                del lrhsi_data, lms_data
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"✅ {factor}x 倍率处理完成")
            
            # 保存元数据
            f.attrs['patch_size'] = patch_size
            f.attrs['stride'] = stride
            f.attrs['total_patches'] = len(gt_patches)
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['original_image_shape'] = self.gt_data.shape
    
        file_size_mb = os.path.getsize(train_file) / (1024**2)
        print(f"\n✅ 训练集保存完成:")
        print(f"   文件: {train_file}")
        print(f"   大小: {file_size_mb:.1f} MB")
        print(f"   实际patches: {len(gt_patches)}")
        
        return train_file

    def generate_single_factor_data(self, images, factor, original_size):
        """
        生成单个下采样倍率的LRHSI和LMS数据
        """
        lrhsi_data = []
        lms_data = []
        
        print(f"   总patches数量: {len(images)}")
        
        # 减小批处理大小以节省内存
        batch_size = 64 if self.device.type == 'cuda' else 32
        
        # 分批处理
        for i in tqdm(range(0, len(images), batch_size), desc=f"处理{factor}x下采样"):
            batch_images = images[i:i+batch_size]
            
            try:
                # 批量处理当前批次
                batch_lrhsi = []
                batch_lms = []
                
                # 进一步减小子批次大小
                sub_batch_size = 16
                for j in range(0, len(batch_images), sub_batch_size):
                    sub_batch = batch_images[j:j+sub_batch_size]
                    
                    for image in sub_batch:
                        # 转换为tensor
                        if isinstance(image, np.ndarray):
                            img_tensor = torch.from_numpy(image).float()
                        else:
                            img_tensor = image
                        
                        # 下采样
                        lrhsi_tensor = anti_aliasing_downsample(img_tensor, factor, self.device)
                        lrhsi = lrhsi_tensor.cpu().numpy()
                        batch_lrhsi.append(lrhsi)
                        
                        # 上采样生成LMS
                        lrhsi_tensor = lrhsi_tensor.to(self.device)
                        lrhsi_batch = lrhsi_tensor.unsqueeze(0)
                        lms_tensor = F.interpolate(lrhsi_batch, size=original_size, 
                                                 mode='bilinear', align_corners=False)
                        lms = lms_tensor.squeeze(0).cpu().numpy()
                        batch_lms.append(lms)
                        
                        # 立即清理GPU内存
                        del img_tensor, lrhsi_tensor, lrhsi_batch, lms_tensor
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # 保存批次结果
                lrhsi_data.extend(batch_lrhsi)
                lms_data.extend(batch_lms)
                
                # 清理批次内存
                del batch_lrhsi, batch_lms
                
            except Exception as e:
                print(f"批处理失败 (factor={factor}, batch {i//batch_size}): {e}")
                # 降级处理
                for image in batch_images:
                    try:
                        if factor > 1:
                            lrhsi = image[:, ::factor, ::factor]
                        else:
                            lrhsi = image
                        lrhsi_data.append(lrhsi)
                        
                        # 简单上采样
                        if original_size is not None and factor > 1:
                            from scipy.ndimage import zoom
                            zoom_factors = (1, original_size[0]/lrhsi.shape[-2], original_size[1]/lrhsi.shape[-1])
                            lms = zoom(lrhsi, zoom_factors, order=1)
                        else:
                            lms = lrhsi
                        lms_data.append(lms)
                    except Exception as e2:
                        print(f"降级处理失败: {e2}")
        
        # 每批处理完强制垃圾回收
        import gc
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
        return lrhsi_data, lms_data
    
    def generate_test_set(self):
        """生成测试集 - 边处理边存储版本"""
        print("\n" + "="*60)
        print("2️⃣ 生成测试集（边处理边存储）")
        print("="*60)
        
        # 裁剪测试patches
        gt_patches, hrmsi_patches = self.crop_test_patches()
        
        # 创建输出H5文件
        test_file = os.path.join(self.config['output_dir'], 'Chikusei_test_patches.h5')
        
        with h5py.File(test_file, 'w') as f:
            compression_opts = self.config['compression_level']
            
            # 首先保存GT和HRMSI
            print("\n💾 保存测试集GT和HRMSI...")
            gt_stack = np.stack(gt_patches)
            hrmsi_stack = np.stack(hrmsi_patches)
            
            f.create_dataset('GT', data=gt_stack)
            f.create_dataset('HRMSI', data=hrmsi_stack)
            
            # 释放内存
            del gt_stack, hrmsi_stack
            import gc
            gc.collect()
            
            # 逐个倍率处理
            downsample_factors = self.config['downsample_factors']
            test_patch_size = self.config['test_patch_size']
            
            for factor in downsample_factors:
                print(f"\n🔄 处理测试集 {factor}x 倍率...")
                
                # 单独处理当前倍率
                lrhsi_data, lms_data = self.generate_single_factor_data(
                    gt_patches, factor, (test_patch_size, test_patch_size)
                )
                
                # 立即保存
                print(f"💾 保存测试集 {factor}x 倍率数据...")
                f.create_dataset(f'LRHSI_{factor}', data=np.stack(lrhsi_data))
                f.create_dataset(f'lms_{factor}', data=np.stack(lms_data))
                
                # 立即释放内存
                del lrhsi_data, lms_data
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"✅ 测试集 {factor}x 倍率处理完成")
            
            # 保存元数据
            f.attrs['test_patch_size'] = test_patch_size
            f.attrs['test_region_size'] = self.config['test_region_size']
            f.attrs['total_test_patches'] = len(gt_patches)
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['original_image_shape'] = self.gt_data.shape
    
        file_size_mb = os.path.getsize(test_file) / (1024**2)
        print(f"\n✅ 测试集保存完成:")
        print(f"   文件: {test_file}")
        print(f"   大小: {file_size_mb:.1f} MB")
        
        return test_file
    
    def verify_datasets(self, train_file, test_file):
        """验证生成的数据集"""
        print("\n" + "="*60)
        print("3️⃣ 验证数据集")
        print("="*60)
        
        # 验证训练集
        print("🔍 验证训练集:")
        with h5py.File(train_file, 'r') as f:
            print(f"   数据集键值: {list(f.keys())}")
            for key in ['GT', 'HRMSI'] + [f'LRHSI_{factor}' for factor in self.config['downsample_factors']] + [f'lms_{factor}' for factor in self.config['downsample_factors']]:
                if key in f:
                    data = f[key]
                    print(f"   {key}: {data.shape}, dtype={data.dtype}")
            print(f"   属性: {dict(f.attrs)}")
        
        # 验证测试集
        print("\n🔍 验证测试集:")
        with h5py.File(test_file, 'r') as f:
            print(f"   数据集键值: {list(f.keys())}")
            for key in ['GT', 'HRMSI'] + [f'LRHSI_{factor}' for factor in self.config['downsample_factors']] + [f'lms_{factor}' for factor in self.config['downsample_factors']]:
                if key in f:
                    data = f[key]
                    print(f"   {key}: {data.shape}, dtype={data.dtype}")
            print(f"   属性: {dict(f.attrs)}")
        
        print("\n✅ 数据集验证完成!")
    
    def run(self):
        """运行完整的数据集生成流程"""
        print("🚀 开始生成Chikusei数据集")
        print("="*80)
        
        try:
            # 1. 加载数据
            self.load_chikusei_data()
            
            # 2. 生成训练集
            train_file = self.generate_training_set()
            
            # 3. 生成测试集
            test_file = self.generate_test_set()
            
            # 4. 验证数据集
            self.verify_datasets(train_file, test_file)
            
            # 5. 总结
            print("\n" + "="*80)
            print("🎉 Chikusei数据集生成完成!")
            print("="*80)
            print("📁 输出文件:")
            print(f"   训练集: {train_file}")
            print(f"   测试集: {test_file}")
            print()
            print("✅ 数据集特点:")
            print("   • GT: 128通道高光谱数据")
            print("   • HRMSI: 3通道RGB数据")
            print("   • LRHSI_X: 使用Interp23Tap抗混叠下采样的低分辨率高光谱数据")
            print("   • lms_X: 使用双线性插值上采样回原尺寸的低分辨率多光谱数据")
            print("   • 训练集: 128×128 patches，重叠裁切")
            print("   • 测试集: 四个1024×1024 patches，从左上角2048×2048区域不重叠裁切")
            print("="*80)
            
            return train_file, test_file
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Chikusei数据集生成器')
    
    parser.add_argument('--chikusei_file', type=str, 
                       default=DEFAULT_CONFIG['chikusei_file'],
                       help='Chikusei原始数据文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='输出目录')
    
    parser.add_argument('--patch_size', type=int,
                       default=DEFAULT_CONFIG['patch_size'],
                       help='训练patch尺寸')
    
    parser.add_argument('--test_patch_size', type=int,
                       default=DEFAULT_CONFIG['test_patch_size'],
                       help='测试patch尺寸')
    
    parser.add_argument('--stride', type=int,
                       default=DEFAULT_CONFIG['stride'],
                       help='训练patch步长')
    
    parser.add_argument('--downsample_factors', nargs='+', type=int,
                       default=DEFAULT_CONFIG['downsample_factors'],
                       help='下采样倍数列表')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建生成器并运行
    generator = ChikuseiDatasetGenerator(config)
    train_file, test_file = generator.run()
    
    if train_file and test_file:
        print(f"\n✅ 数据集生成成功!")
        print(f"可以使用以下代码加载数据:")
        print(f"")
        print(f"import h5py")
        print(f"with h5py.File('{train_file}', 'r') as f:")
        print(f"    gt = f['GT'][:]")
        print(f"    lrhsi_4 = f['LRHSI_4'][:]")
        print(f"    lms_4 = f['lms_4'][:]")
    else:
        print(f"\n❌ 数据集生成失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()