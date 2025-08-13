#!/usr/bin/env python3
"""
PaviaCenter数据集生成脚本
生成标准格式的训练集和测试集，包含GT, HRMSI, LRHSI_X, lms_X数据

数据流程:
1. GT (Ground Truth) - 原始102通道高光谱数据
2. HRMSI - 3通道RGB数据
3. LRHSI_X - GT通过Interp23Tap抗混叠下采样X倍得到的低分辨率高光谱数据
4. lms_X - LRHSI_X通过双线性插值上采样回原尺寸得到的低分辨率多光谱数据

训练集：1096x1096图像，重叠裁切128x128 patches
测试集：左上角1024x1024区域，不重叠裁切4个512x512 patches

作者: Assistant
日期: 2025-08-11
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

# Botswana配置参数
PAVIA_CONFIG = {
    'pavia_file': '/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/data/Botswana/Botswana_RGB.h5',
    'output_dir': '/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/data/Botswana/datasets',
    
    # 训练集参数
    'train_patch_size': 128,
    'train_stride': 16,  # 重叠裁切
    
    # 测试集参数
    'test_crop_width': 256,   # 左上角裁切宽度
    'test_crop_height': 1280,   # 左上角裁切高度
    'test_patch_size': 256,    # 测试patch尺寸 (1024/2=512, 1×2=2个patches)
    
    'downsample_factors': [4, 8, 16, 32],
    'compression_level': 9,
    
    # Botswana特定参数
    'original_size': (1476, 256),
    'num_bands': 145,  # PaviaU有103个光谱通道
}

class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.
    (复用Harvard代码中的实现)
    """

    def __init__(self, ratio: int, pad_mode: str = "replicate"):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))
        self.pad_mode = pad_mode

        # Define the 23-tap filter coefficients (CDF23 from MATLAB code)
        cdf23_coeffs = 2.0 * np.array([
            0.5, 0.305334091185, 0.0, -0.072698593239, 0.0, 0.021809577942,
            0.0, -0.005192756653, 0.0, 0.000807762146, 0.0, -0.000060081482,
        ])
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
        """Forward pass for interpolation."""
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
            # Apply horizontal filter
            kernel_w_grouped = self.kernel_w.repeat(c, 1, 1, 1)
            filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)

            # Pad for vertical filter (height)
            padded_h = F.pad(
                filtered_w, (0, 0, self.padding, self.padding), mode="circular"
            )
            # Apply vertical filter
            kernel_h_grouped = self.kernel_h.repeat(c, 1, 1, 1)
            filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)

            current_img = filtered_h  # Update image for next iteration

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
    kernel_h = base_coeffs_t.view(1, 1, -1, 1)  # 垂直滤波核
    kernel_w = base_coeffs_t.view(1, 1, 1, -1)  # 水平滤波核
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

class PaviaCenterDatasetGenerator:
    """PaviaCenter数据集生成器"""
    
    def __init__(self, config=None, gpu_ids=[0]):
        """初始化生成器"""
        self.config = config or PAVIA_CONFIG.copy()
        
        # 设置GPU支持
        self.gpu_ids = gpu_ids if torch.cuda.is_available() else []
        if self.gpu_ids:
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"使用GPU: {gpu_ids}, 主设备: {self.device}")
        else:
            self.device = torch.device('cpu')
            print("使用CPU")
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_pavia_data(self):
        """加载PaviaU数据"""
        print("📂 加载PaviaU数据...")
        
        pavia_file = self.config['pavia_file']
        if not os.path.exists(pavia_file):
            raise FileNotFoundError(f"PaviaU文件不存在: {pavia_file}")
        
        with h5py.File(pavia_file, 'r') as f:
            # 检查文件内容
            print(f"   文件中的数据集: {list(f.keys())}")
            
            # 加载数据 - PaviaCenter只有一张图像
            self.gt_data = f['gt'][:]        # (C, H, W) - 102通道
            self.hrmsi_data = f['HRMSI'][:]  # (3, H, W) - RGB
            
            # 检查数据形状
            print(f"   GT数据: {self.gt_data.shape}")
            print(f"   HRMSI数据: {self.hrmsi_data.shape}")
            
            # 确保是单张图像格式
            if len(self.gt_data.shape) != 3:
                raise ValueError(f"期望GT数据为(C,H,W)格式，实际为{self.gt_data.shape}")
            
            # 验证尺寸
            expected_size = self.config['original_size']
            actual_size = self.gt_data.shape[-2:]
            if actual_size != expected_size:
                print(f"   警告: 图像尺寸 {actual_size} 与期望尺寸 {expected_size} 不匹配")
                self.config['original_size'] = actual_size
            
            # 验证通道数
            expected_bands = self.config['num_bands']
            actual_bands = self.gt_data.shape[0]
            if actual_bands != expected_bands:
                print(f"   警告: 光谱通道数 {actual_bands} 与期望通道数 {expected_bands} 不匹配")
                self.config['num_bands'] = actual_bands
            
            print(f"   数值范围: GT[{self.gt_data.min():.6f}, {self.gt_data.max():.6f}]")
            print(f"   图像尺寸: {actual_size} (H x W)")
            print(f"   光谱通道: {actual_bands}")
            
            # 保存数据集信息
            if 'dataset_name' in f.attrs:
                self.dataset_name = f.attrs['dataset_name']
            else:
                self.dataset_name = 'PaviaCenter'
            
            print(f"   数据集名称: {self.dataset_name}")
        
    def crop_patches_overlapping(self, image, patch_size, stride):
        """使用重叠滑动窗口裁剪patches"""
        if len(image.shape) == 3:  # (C, H, W)
            c, h, w = image.shape
        else:  # (H, W)
            h, w = image.shape
        
        patches = []
        positions = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if len(image.shape) == 3:  # (C, H, W)
                    patch = image[:, y:y+patch_size, x:x+patch_size]
                else:  # (H, W)
                    patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))
        
        return patches, positions
    
    def crop_test_patches_non_overlapping(self, image, crop_width, crop_height, patch_size):
        """
        从左上角crop_width×crop_height区域不重叠裁剪patch_size的patches
        # 对于1024×512裁剪512×512，得到1×2=2个patches
        """
        # 先裁剪左上角区域
        if len(image.shape) == 3:  # (C, H, W)
            cropped = image[:, :crop_height, :crop_width]
        else:  # (H, W)
            cropped = image[:crop_height, :crop_width]
        
        patches = []
        positions = []
        
        # 计算patches的行列数
        patches_per_row = crop_height // patch_size   # 512 // 512 = 1
        patches_per_col = crop_width // patch_size    # 1024 // 512 = 2
        
        for row in range(patches_per_row):
            for col in range(patches_per_col):
                y = row * patch_size
                x = col * patch_size
                
                if len(image.shape) == 3:  # (C, H, W)
                    patch = cropped[:, y:y+patch_size, x:x+patch_size]
                else:  # (H, W)
                    patch = cropped[y:y+patch_size, x:x+patch_size]
                
                patches.append(patch)
                positions.append((y, x))
        
        return patches, positions
    
    def generate_lrhsi_and_lms(self, images, downsample_factors, original_size=None):
        """生成LRHSI和LMS数据 - 内存优化版本"""
        lrhsi_data = {}
        lms_data = {}
        
        for factor in downsample_factors:
            lrhsi_data[f'LRHSI_{factor}'] = []
            lms_data[f'lms_{factor}'] = []
        
        print(f"🔄 生成LRHSI和LMS数据（内存优化）...")
        
        for factor in downsample_factors:
            print(f"   处理下采样倍数 {factor}x...")
            
            for i, image in enumerate(images):
                try:
                    # 转换为tensor
                    if isinstance(image, np.ndarray):
                        img_tensor = torch.from_numpy(image).float()
                    else:
                        img_tensor = image
                    
                    # 下采样
                    lrhsi_tensor = anti_aliasing_downsample(img_tensor, factor, self.device)
                    
                    # 移动到CPU并保存LRHSI
                    lrhsi = lrhsi_tensor.cpu().numpy()
                    lrhsi_data[f'LRHSI_{factor}'].append(lrhsi)
                    
                    # 生成LMS（上采样）
                    target_size = original_size if original_size is not None else image.shape[-2:]
                    lrhsi_tensor = lrhsi_tensor.to(self.device)
                    lrhsi_batch = lrhsi_tensor.unsqueeze(0)
                    lms_tensor = F.interpolate(lrhsi_batch, size=target_size, 
                                             mode='bilinear', align_corners=False)
                    lms = lms_tensor.squeeze(0).cpu().numpy()
                    lms_data[f'lms_{factor}'].append(lms)
                    
                    # 立即清理GPU内存
                    del img_tensor, lrhsi_tensor, lrhsi_batch, lms_tensor
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"处理失败 (factor={factor}, image {i}): {e}")
                    # 使用numpy后备方案
                    if factor > 1:
                        lrhsi = image[:, ::factor, ::factor]
                    else:
                        lrhsi = image
                    lrhsi_data[f'LRHSI_{factor}'].append(lrhsi)
                    
                    # 简单上采样
                    from scipy.ndimage import zoom
                    if original_size is not None:
                        zoom_factors = (1, original_size[0]/lrhsi.shape[-2], original_size[1]/lrhsi.shape[-1])
                        lms = zoom(lrhsi, zoom_factors, order=1)
                    else:
                        lms = lrhsi
                    lms_data[f'lms_{factor}'].append(lms)
        
        return lrhsi_data, lms_data
    
    def generate_training_set(self):
        """生成训练集"""
        print("\n" + "="*60)
        print("1️⃣ 生成PaviaU训练集")
        print("="*60)
        
        patch_size = self.config['train_patch_size']
        stride = self.config['train_stride']
        downsample_factors = self.config['downsample_factors']
        
        print(f"训练patch尺寸: {patch_size}x{patch_size}")
        print(f"裁剪步长: {stride} (重叠裁剪)")
        print(f"原始图像尺寸: {self.config['original_size']}")
        
        # 裁剪训练patches
        print("🔨 裁剪训练patches...")
        gt_patches, positions = self.crop_patches_overlapping(self.gt_data, patch_size, stride)
        hrmsi_patches, _ = self.crop_patches_overlapping(self.hrmsi_data, patch_size, stride)
        
        print(f"总训练patches: {len(gt_patches)}")
        print(f"每个patch尺寸: GT {gt_patches[0].shape}, HRMSI {hrmsi_patches[0].shape}")
        
        # 生成LRHSI和LMS
        print("🔄 生成LRHSI和LMS...")
        lrhsi_data, lms_data = self.generate_lrhsi_and_lms(
            gt_patches, downsample_factors, original_size=(patch_size, patch_size)
        )
        
        # 创建训练集H5文件
        train_file = os.path.join(
            self.config['output_dir'], 
            f'Botswana_train_patches_stride{stride}_size{patch_size}.h5'
        )
        
        print(f"💾 保存训练集到: {train_file}")
        
        with h5py.File(train_file, 'w') as f:
            # 保存GT和HRMSI
            gt_array = np.stack(gt_patches)      # (N, C, H, W)
            hrmsi_array = np.stack(hrmsi_patches)  # (N, 3, H, W)
            
            f.create_dataset('GT', data=gt_array, 
                            #  compression='gzip', 
                        #    compression_opts=self.config['compression_level']
                           )
            f.create_dataset('HRMSI', data=hrmsi_array, 
                        #      compression='gzip',
                        #    compression_opts=self.config['compression_level']
                           )
            
            # 保存LRHSI和LMS
            for factor in downsample_factors:
                lrhsi_array = np.stack(lrhsi_data[f'LRHSI_{factor}'])
                lms_array = np.stack(lms_data[f'lms_{factor}'])
                
                f.create_dataset(f'LRHSI_{factor}', data=lrhsi_array, 
                            #    compression='gzip', compression_opts=self.config['compression_level']
                               )
                f.create_dataset(f'lms_{factor}', data=lms_array,
                            #    compression='gzip', compression_opts=self.config['compression_level']
                            )
            
            # 保存元数据
            f.attrs['dataset_name'] = self.dataset_name
            f.attrs['patch_size'] = patch_size
            f.attrs['stride'] = stride
            f.attrs['total_patches'] = len(gt_patches)
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['original_image_size'] = list(self.config['original_size'])
            f.attrs['num_bands'] = self.config['num_bands']
            f.attrs['data_type'] = 'train'
        
        file_size_mb = os.path.getsize(train_file) / (1024**2)
        print(f"✅ 训练集保存完成! 文件大小: {file_size_mb:.1f} MB")
        
        return train_file
    
    def generate_test_set(self):
        """生成测试集"""
        print("\n" + "="*60)
        print("2️⃣ 生成PaviaCenter测试集")
        print("="*60)
        
        crop_width = self.config['test_crop_width']
        crop_height = self.config['test_crop_height']
        patch_size = self.config['test_patch_size']
        downsample_factors = self.config['downsample_factors']
        
        print(f"测试区域裁剪: 左上角{crop_width}×{crop_height}")
        print(f"测试patch尺寸: {patch_size}×{patch_size}")
        
        # 验证参数
        if crop_width % patch_size != 0:
            raise ValueError(f"裁剪宽度{crop_width}必须能被patch尺寸{patch_size}整除")
        if crop_height % patch_size != 0:
            raise ValueError(f"裁剪高度{crop_height}必须能被patch尺寸{patch_size}整除")
        
        patches_per_row = crop_height // patch_size
        patches_per_col = crop_width // patch_size
        total_patches = patches_per_row * patches_per_col
        print(f"测试patches: {patches_per_row}×{patches_per_col} = {total_patches}个")
        
        # 裁剪测试patches
        print("🔨 裁剪测试patches...")
        gt_patches, positions = self.crop_test_patches_non_overlapping(
            self.gt_data, crop_width, crop_height, patch_size)
        hrmsi_patches, _ = self.crop_test_patches_non_overlapping(
            self.hrmsi_data, crop_width, crop_height, patch_size)
        
        print(f"实际得到patches: {len(gt_patches)}")
        
        # 生成LRHSI和LMS
        print("🔄 生成LRHSI和LMS...")
        lrhsi_data, lms_data = self.generate_lrhsi_and_lms(
            gt_patches, downsample_factors, original_size=(patch_size, patch_size)
        )
        
        # 创建测试集H5文件
        test_file = os.path.join(
            self.config['output_dir'], 
            f'Botswana_test_crop{crop_width}x{crop_height}_patch{patch_size}.h5'
        )
        
        print(f"💾 保存测试集到: {test_file}")
        
        with h5py.File(test_file, 'w') as f:
            # 保存GT和HRMSI
            gt_array = np.stack(gt_patches)      # (N, C, H, W)
            hrmsi_array = np.stack(hrmsi_patches)  # (N, 3, H, W)
            
            f.create_dataset('GT', data=gt_array)
            f.create_dataset('HRMSI', data=hrmsi_array)
            
            # 保存LRHSI和LMS
            for factor in downsample_factors:
                lrhsi_array = np.stack(lrhsi_data[f'LRHSI_{factor}'])
                lms_array = np.stack(lms_data[f'lms_{factor}'])
                
                f.create_dataset(f'LRHSI_{factor}', data=lrhsi_array)
                f.create_dataset(f'lms_{factor}', data=lms_array)
            
            # 保存patch位置信息
            positions_array = np.array(positions)
            f.create_dataset('patch_positions', data=positions_array)
            
            # 保存元数据
            f.attrs['dataset_name'] = self.dataset_name
            f.attrs['crop_width'] = crop_width
            f.attrs['crop_height'] = crop_height
            f.attrs['patch_size'] = patch_size
            f.attrs['total_patches'] = len(gt_patches)
            f.attrs['patches_per_row'] = patches_per_row
            f.attrs['patches_per_col'] = patches_per_col
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['original_image_size'] = list(self.config['original_size'])
            f.attrs['num_bands'] = self.config['num_bands']
            f.attrs['data_type'] = 'test'
    
    def verify_datasets(self, train_file, test_file):
        """验证生成的数据集"""
        print("\n" + "="*60)
        print("3️⃣ 验证数据集")
        print("="*60)
        
        # 验证训练集
        print("🔍 验证训练集:")
        with h5py.File(train_file, 'r') as f:
            print(f"   数据集键值: {list(f.keys())}")
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key]
                    print(f"   {key}: {data.shape}, dtype={data.dtype}")
                    # 检查数据范围
                    sample_data = data[0] if len(data) > 0 else None
                    if sample_data is not None:
                        print(f"      数值范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")
            
            print(f"   属性: {dict(f.attrs)}")
        
        # 验证测试集
        print("\n🔍 验证测试集:")
        with h5py.File(test_file, 'r') as f:
            print(f"   数据集键值: {list(f.keys())}")
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key]
                    print(f"   {key}: {data.shape}, dtype={data.dtype}")
                    # 检查数据范围
                    sample_data = data[0] if len(data) > 0 else None
                    if sample_data is not None:
                        print(f"      数值范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")
            
            print(f"   属性: {dict(f.attrs)}")
        
        print("\n✅ 数据集验证完成!")
    
    def visualize_samples(self, train_file, test_file):
        """可视化数据样本"""
        print("\n" + "="*60)
        print("4️⃣ 可视化数据样本")
        print("="*60)
        
        try:
            import matplotlib.pyplot as plt
            
            # 可视化训练集样本
            with h5py.File(train_file, 'r') as f:
                gt_train = f['GT'][0]        # 第一个训练patch
                hrmsi_train = f['HRMSI'][0]  # 第一个训练patch
                
                print(f"训练样本形状: GT {gt_train.shape}, HRMSI {hrmsi_train.shape}")
            
            # 可视化测试集样本
            with h5py.File(test_file, 'r') as f:
                gt_test = f['GT'][0]         # 第一个测试patch
                hrmsi_test = f['HRMSI'][0]   # 第一个测试patch
                
                print(f"测试样本形状: GT {gt_test.shape}, HRMSI {hrmsi_test.shape}")
            
            # 创建可视化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('PaviaCenter数据集样本可视化', fontsize=16)
            
            # 训练集可视化
            # GT伪彩色 (选择几个代表性波段)
            gt_train_rgb = np.stack([
                gt_train[80],   # 红色通道
                gt_train[50],   # 绿色通道  
                gt_train[20]    # 蓝色通道
            ], axis=2)
            gt_train_rgb = (gt_train_rgb - gt_train_rgb.min()) / (gt_train_rgb.max() - gt_train_rgb.min())
            
            axes[0, 0].imshow(gt_train_rgb)
            axes[0, 0].set_title('训练集-GT伪彩色')
            axes[0, 0].axis('off')
            
            # HRMSI RGB
            hrmsi_train_display = hrmsi_train.transpose(1, 2, 0)  # CHW -> HWC
            hrmsi_train_display = (hrmsi_train_display - hrmsi_train_display.min()) / (hrmsi_train_display.max() - hrmsi_train_display.min())
            
            axes[0, 1].imshow(hrmsi_train_display)
            axes[0, 1].set_title('训练集-HRMSI RGB')
            axes[0, 1].axis('off')
            
            # 下采样示例
            if f'LRHSI_4' in f:
                with h5py.File(train_file, 'r') as f:
                    lrhsi_4 = f['LRHSI_4'][0]
                
                # 上采样用于显示
                from scipy.ndimage import zoom
                lrhsi_4_upsampled = zoom(lrhsi_4, (1, 4, 4), order=1)
                lrhsi_4_rgb = np.stack([
                    lrhsi_4_upsampled[80],
                    lrhsi_4_upsampled[50],
                    lrhsi_4_upsampled[20]
                ], axis=2)
                lrhsi_4_rgb = (lrhsi_4_rgb - lrhsi_4_rgb.min()) / (lrhsi_4_rgb.max() - lrhsi_4_rgb.min())
                
                axes[0, 2].imshow(lrhsi_4_rgb)
                axes[0, 2].set_title('训练集-LRHSI_4 (4倍下采样)')
                axes[0, 2].axis('off')
            
            # 测试集可视化
            gt_test_rgb = np.stack([
                gt_test[80],
                gt_test[50],
                gt_test[20]
            ], axis=2)
            gt_test_rgb = (gt_test_rgb - gt_test_rgb.min()) / (gt_test_rgb.max() - gt_test_rgb.min())
            
            axes[1, 0].imshow(gt_test_rgb)
            axes[1, 0].set_title('测试集-GT伪彩色')
            axes[1, 0].axis('off')
            
            hrmsi_test_display = hrmsi_test.transpose(1, 2, 0)
            hrmsi_test_display = (hrmsi_test_display - hrmsi_test_display.min()) / (hrmsi_test_display.max() - hrmsi_test_display.min())
            
            axes[1, 1].imshow(hrmsi_test_display)
            axes[1, 1].set_title('测试集-HRMSI RGB')
            axes[1, 1].axis('off')
            
            if f'LRHSI_4' in f:
                with h5py.File(test_file, 'r') as f:
                    lrhsi_4_test = f['LRHSI_4'][0]
                
                lrhsi_4_test_upsampled = zoom(lrhsi_4_test, (1, 4, 4), order=1)
                lrhsi_4_test_rgb = np.stack([
                    lrhsi_4_test_upsampled[80],
                    lrhsi_4_test_upsampled[50],
                    lrhsi_4_test_upsampled[20]
                ], axis=2)
                lrhsi_4_test_rgb = (lrhsi_4_test_rgb - lrhsi_4_test_rgb.min()) / (lrhsi_4_test_rgb.max() - lrhsi_4_test_rgb.min())
                
                axes[1, 2].imshow(lrhsi_4_test_rgb)
                axes[1, 2].set_title('测试集-LRHSI_4 (4倍下采样)')
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ 可视化完成!")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")
    
    def run(self):
        """运行完整的数据集生成流程"""
        print("🚀 开始生成PaviaU数据集")
        print("="*80)
        
        try:
            # 1. 加载数据
            self.load_pavia_data()
            
            # 2. 生成训练集
            train_file = self.generate_training_set()
            
            # 3. 生成测试集
            test_file = self.generate_test_set()
            
            # 4. 验证数据集
            self.verify_datasets(train_file, test_file)
            
            # 5. 可视化样本
            self.visualize_samples(train_file, test_file)
            
            # 6. 总结
            print("\n" + "="*80)
            print("🎉 PaviaU数据集生成完成!")
            print("="*80)
            print("📁 输出文件:")
            print(f"   训练集: {train_file}")
            print(f"   测试集: {test_file}")
            print()
            print("✅ 数据集特点:")
            print(f"   • GT: {self.config['num_bands']}通道高光谱数据")
            print("   • HRMSI: 3通道RGB数据")
            print("   • LRHSI_X: 使用Interp23Tap抗混叠下采样的低分辨率高光谱数据")
            print("   • lms_X: 使用双线性插值上采样回原尺寸的低分辨率多光谱数据")
            print(f"   • 训练集: 重叠裁切{self.config['train_patch_size']}x{self.config['train_patch_size']} patches (步长{self.config['train_stride']})")
            print(f"   • 测试集: 左上角{self.config['test_crop_width']}x{self.config['test_crop_height']}裁切4个{self.config['test_patch_size']}x{self.config['test_patch_size']} patches")
            print("="*80)
            
            return train_file, test_file
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PaviaCenter数据集生成器')
    
    parser.add_argument('--pavia_file', type=str, 
                       default=PAVIA_CONFIG['pavia_file'],
                       help='PaviaCenter数据文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default=PAVIA_CONFIG['output_dir'],
                       help='输出目录')
    
    parser.add_argument('--train_patch_size', type=int,
                       default=PAVIA_CONFIG['train_patch_size'],
                       help='训练patch尺寸')
    
    parser.add_argument('--train_stride', type=int,
                       default=PAVIA_CONFIG['train_stride'],
                       help='训练patch步长')
    
    parser.add_argument('--test_crop_width', type=int,
                       default=PAVIA_CONFIG['test_crop_width'],
                       help='测试集裁剪宽度')
    
    parser.add_argument('--test_crop_height', type=int,
                       default=PAVIA_CONFIG['test_crop_height'],
                       help='测试集裁剪高度')
    
    parser.add_argument('--test_patch_size', type=int,
                       default=PAVIA_CONFIG['test_patch_size'],
                       help='测试patch尺寸')
    
    parser.add_argument('--downsample_factors', nargs='+', type=int,
                       default=PAVIA_CONFIG['downsample_factors'],
                       help='下采样倍数列表')
    
    parser.add_argument('--gpu_ids', nargs='+', type=int,
                       default=[0],
                       help='使用的GPU ID列表')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建配置
    config = PAVIA_CONFIG.copy()
    gpu_ids = args.gpu_ids
    del args.gpu_ids
    config.update(vars(args))
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  gpu_ids: {gpu_ids}")
    print()
    
    # 创建生成器并运行
    generator = PaviaCenterDatasetGenerator(config, gpu_ids=gpu_ids)
    train_file, test_file = generator.run()
    
    if train_file and test_file:
        print(f"\n✅ PaviaCenter数据集生成成功!")
    else:
        print(f"\n❌ PaviaCenter数据集生成失败!")
        sys.exit(1)

if __name__ == '__main__':
    main()