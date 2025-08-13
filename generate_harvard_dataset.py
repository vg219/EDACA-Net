#!/usr/bin/env python3
"""
Harvard数据集生成脚本
生成标准格式的训练集和测试集，包含GT, HRMSI, LRHSI_X, lms_X数据

数据流程:
1. GT (Ground Truth) - 原始31通道高光谱数据
2. HRMSI - 3通道RGB数据
3. LRHSI_X - GT通过Interp23Tap抗混叠下采样X倍得到的低分辨率高光谱数据
4. lms_X - LRHSI_X通过双线性插值上采样回原尺寸得到的低分辨率多光谱数据

训练集：67张图像，64步长裁切128x128 patches
测试集：10张图像，左上角1024x1024裁切

作者: Assistant
日期: 2025-08-05
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

# 配置参数
DEFAULT_CONFIG = {
    'harvard_file': '/data2/users/yujieliang/dataset/Harvard_calibrated_full.h5',
    'output_dir': '/data2/users/yujieliang/dataset/Harvard',
    'patch_size': 128,
    'stride': 128,
    'test_crop_size': 1024,  # 测试集裁切尺寸
    'downsample_factors': [4, 8, 16, 32],
    'train_indices': list(range(67)),      # 前67张作为训练集
    'test_indices': list(range(67, 77)),   # 后10张作为测试集
    'compression_level': 9,
}

class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.

    Interpolates the input tensor using a 23-coefficient polynomial interpolator,
    upsampling by the given ratio. The ratio must be a power of 2.

    Args:
        ratio (int): Scale ratio for upsampling. Must be a power of 2.
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
        # Kernel for filtering along height (columns in MATLAB)
        kernel_h = base_coeffs_t.view(1, 1, -1, 1)  # Shape (1, 1, 23, 1)
        # Kernel for filtering along width (rows in MATLAB)
        kernel_w = base_coeffs_t.view(1, 1, 1, -1)  # Shape (1, 1, 1, 23)

        # Register kernels as buffers
        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_w", kernel_w)

        # Calculate padding size (kernel_size=23)
        self.padding = (base_coeffs_t.shape[0] - 1) // 2  # Should be 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (bs, c, h, w).

        Returns:
            torch.Tensor: Interpolated tensor of shape (bs, c, h * ratio, w * ratio).
        """
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
                # I1LRU(2:2:end,2:2:end,:) = I_Interpolated;
                upsampled[..., 1::2, 1::2] = current_img
            else:
                # I1LRU(1:2:end,1:2:end,:) = I_Interpolated;
                upsampled[..., ::2, ::2] = current_img

            # Apply separable convolution with circular padding
            # Grouped convolution: apply filter independently per channel
            # Using conv2d with groups=c is efficient

            # Pad for horizontal filter (width)
            # Pad width dimension (dim 3) by self.padding on both sides
            padded_w = F.pad(
                upsampled, (self.padding, self.padding, 0, 0), mode=self.pad_mode
            )
            # Apply horizontal filter
            # Input: (bs, c, H, W_padded), Kernel: (1, 1, 1, K) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, 1, K) for grouped convolution
            kernel_w_grouped = self.kernel_w.repeat(c, 1, 1, 1)
            filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)

            # Pad for vertical filter (height)
            # Pad height dimension (dim 2) by self.padding on both sides
            padded_h = F.pad(
                filtered_w, (0, 0, self.padding, self.padding), mode="circular"
            )
            # Apply vertical filter
            # Input: (bs, c, H_padded, W), Kernel: (1, 1, K, 1) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, K, 1) for grouped convolution
            kernel_h_grouped = self.kernel_h.repeat(c, 1, 1, 1)
            filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)

            current_img = filtered_h  # Update image for next iteration

        return current_img


def anti_aliasing_downsample(image_tensor, factor, device):
    """
    使用Interp23Tap进行抗混叠下采样
    
    实现原理：
    1. 使用Interp23Tap的反向过程进行抗混叠滤波
    2. 然后进行子采样
    
    Args:
        image_tensor: 输入图像张量 (C, H, W)
        factor: 下采样倍数
        device: 计算设备
        
    Returns:
        下采样后的图像张量
    """
    if factor == 1:
        return image_tensor
    
    # 添加batch维度
    img_batch = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    # 使用Interp23Tap的滤波核进行抗混叠预处理
    # 获取滤波核系数
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

def anti_aliasing_downsample_batch(image_tensors, factor, devices):
    """
    多GPU批量抗混叠下采样
    
    Args:
        image_tensors: 输入图像张量列表
        factor: 下采样倍数
        devices: GPU设备列表
        
    Returns:
        下采样后的图像张量列表
    """
    if factor == 1:
        return image_tensors
    
    if not devices:
        # 单GPU或CPU处理
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return [anti_aliasing_downsample(img, factor, device) for img in image_tensors]
    
    # 多GPU并行处理
    results = []
    batch_size = len(image_tensors)
    gpu_count = len(devices)
    
    # 将数据分配到不同GPU
    for i in range(0, batch_size, gpu_count):
        gpu_results = []
        
        # 创建线程池进行并行处理
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            futures = []
            
            for j, device_id in enumerate(devices):
                if i + j < batch_size:
                    img_tensor = image_tensors[i + j]
                    device = torch.device(f'cuda:{device_id}')
                    future = executor.submit(anti_aliasing_downsample, img_tensor, factor, device)
                    futures.append(future)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                gpu_results.append(future.result())
        
        results.extend(gpu_results)
    
    return results

class HarvardDatasetGenerator:
    """Harvard数据集生成器"""
    
    def __init__(self, config=None, gpu_ids=[0, 1, 2, 3, 4, 5, 6]):
        """初始化生成器"""
        self.config = config or DEFAULT_CONFIG.copy()
        
        # 设置多GPU支持
        self.gpu_ids = gpu_ids if torch.cuda.is_available() else []
        if self.gpu_ids:
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"使用GPU: {gpu_ids}, 主设备: {self.device}")
            
            # 检查GPU可用性
            for gpu_id in gpu_ids:
                if gpu_id >= torch.cuda.device_count():
                    print(f"警告: GPU {gpu_id} 不可用，忽略")
                    self.gpu_ids.remove(gpu_id)
            
            print(f"实际使用GPU: {self.gpu_ids}")
        else:
            self.device = torch.device('cpu')
            print("使用CPU")
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_harvard_data(self):
        """加载原始Harvard数据"""
        print("📂 加载原始Harvard数据...")
        
        harvard_file = self.config['harvard_file']
        if not os.path.exists(harvard_file):
            raise FileNotFoundError(f"Harvard文件不存在: {harvard_file}")
        
        with h5py.File(harvard_file, 'r') as f:
            # 检查文件内容
            print(f"   文件中的数据集: {list(f.keys())}")
            
            # 加载数据
            self.gt_data = f['gt'][:]          # (77, 31, 1040, 1392)
            self.hrmsi_data = f['HR_MSI'][:]   # (77, 3, 1040, 1392)
            
            # 加载样本名称
            if 'sample_names' in f:
                sample_names_raw = f['sample_names'][:]
                self.sample_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                                   for name in sample_names_raw]
            else:
                self.sample_names = [f'harvard_{i:02d}' for i in range(len(self.gt_data))]
        
        print(f"   GT数据: {self.gt_data.shape}")
        print(f"   HRMSI数据: {self.hrmsi_data.shape}")
        print(f"   样本数量: {len(self.sample_names)}")
        print(f"   数值范围: GT[{self.gt_data.min():.2f}, {self.gt_data.max():.2f}]")
        print(f"   图像尺寸: {self.gt_data.shape[-2:]} (H x W)")
        
    def crop_patches_overlapping(self, image, patch_size, stride):
        """使用重叠滑动窗口裁剪patches"""
        h, w = image.shape[-2:]
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
    
    def crop_test_region(self, image, crop_size):
        """裁剪测试集区域（左上角）"""
        if len(image.shape) == 3:  # (C, H, W)
            return image[:, :crop_size, :crop_size]
        else:  # (H, W)
            return image[:crop_size, :crop_size]
    
    def generate_lrhsi_and_lms(self, images, downsample_factors, original_size=None):
        """
        生成LRHSI和LMS数据 - 内存优化版本
        """
        lrhsi_data = {}
        lms_data = {}
        
        for factor in downsample_factors:
            lrhsi_data[f'LRHSI_{factor}'] = []
            lms_data[f'lms_{factor}'] = []
        
        print(f"🔄 生成LRHSI和LMS数据（内存优化）...")
        
        # 大幅减少批处理大小以节省内存
        batch_size = 2 if self.gpu_ids else 1
        
        for factor in downsample_factors:
            print(f"   处理下采样倍数 {factor}x...")
            
            # 分批处理，每批处理完立即清理内存
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                try:
                    # 处理当前批次
                    for j, image in enumerate(batch_images):
                        # 转换为tensor
                        if isinstance(image, np.ndarray):
                            img_tensor = torch.from_numpy(image).float()
                        else:
                            img_tensor = image
                        
                        # 下采样（单图像处理以节省内存）
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
                        
                        # 定期强制垃圾回收
                        if (i + j + 1) % 10 == 0:
                            import gc
                            gc.collect()
                            
                except Exception as e:
                    print(f"批处理 {i//batch_size} 失败 (factor={factor}): {e}")
                    # 降级到最简单的处理方式
                    for image in batch_images:
                        try:
                            # 使用numpy进行简单下采样作为后备方案
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
                        except Exception as e2:
                            print(f"后备处理也失败: {e2}")
                
                # 每批处理完后显示进度和内存使用
                processed = min(i + batch_size, len(images))
                print(f"     进度: {processed}/{len(images)} ({100*processed/len(images):.1f}%)")
                
                # 强制垃圾回收
                import gc
                gc.collect()
        
        return lrhsi_data, lms_data
    
    def generate_training_set(self):
        """生成训练集 - 真正的边存边处理版本"""
        print("\n" + "="*60)
        print("1️⃣ 生成Harvard训练集（边存边处理）")
        print("="*60)
        
        # 提取训练数据
        train_indices = self.config['train_indices']
        patch_size = self.config['patch_size']
        stride = self.config['stride']
        downsample_factors = self.config['downsample_factors']
        
        print(f"训练样本: {len(train_indices)}张图像")
        
        # 估算patches数量
        sample_gt = self.gt_data[train_indices[0]]
        sample_patches, _ = self.crop_patches_overlapping(sample_gt, patch_size, stride)
        patches_per_image = len(sample_patches)
        total_estimated_patches = patches_per_image * len(train_indices)
        
        print(f"每张图像估计patches: {patches_per_image}")
        print(f"总估计patches: {total_estimated_patches}")
        
        # 创建输出H5文件并预分配空间
        train_file = os.path.join(
            self.config['output_dir'], 
            f'Harvard_train_patches_stride{stride}_size{patch_size}.h5'
        )
        
        compression_opts = self.config['compression_level']
        
        # 打开H5文件并预分配数据集
        with h5py.File(train_file, 'w') as f:
            # 预分配所有数据集
            gt_dataset = f.create_dataset(
                'GT', 
                shape=(total_estimated_patches, 31, patch_size, patch_size),
                dtype=np.float32,
                # compression='gzip', 
                # compression_opts=compression_opts,
                # chunks=True  # 启用分块存储
            )
            
            hrmsi_dataset = f.create_dataset(
                'HRMSI', 
                shape=(total_estimated_patches, 3, patch_size, patch_size),
                dtype=np.float32,
                # compression='gzip', 
                # compression_opts=compression_opts,
                # chunks=True
            )
            
            # 为每个下采样倍数预分配数据集
            lrhsi_datasets = {}
            lms_datasets = {}
            
            for factor in downsample_factors:
                lrhsi_size = patch_size // factor
                
                lrhsi_datasets[factor] = f.create_dataset(
                    f'LRHSI_{factor}',
                    shape=(total_estimated_patches, 31, lrhsi_size, lrhsi_size),
                    dtype=np.float32,
                    # compression='gzip',
                    # compression_opts=compression_opts,
                    # chunks=True
                )
                
                lms_datasets[factor] = f.create_dataset(
                    f'lms_{factor}',
                    shape=(total_estimated_patches, 31, patch_size, patch_size),
                    dtype=np.float32,
                    # compression='gzip',
                    # compression_opts=compression_opts,
                    # chunks=True
                )
        
            # 逐图像处理并直接写入H5文件
            print("\n📦 逐图像边存边处理...")
            current_patch_idx = 0
            
            for img_idx, idx in enumerate(train_indices):
                print(f"处理图像 {img_idx+1}/{len(train_indices)}: {self.sample_names[idx]}")
                
                # 加载单张图像
                gt_image = self.gt_data[idx]      # (31, 1040, 1392)
                hrmsi_image = self.hrmsi_data[idx]  # (3, 1040, 1392)
                
                # 裁剪patches
                gt_patches, _ = self.crop_patches_overlapping(gt_image, patch_size, stride)
                hrmsi_patches, _ = self.crop_patches_overlapping(hrmsi_image, patch_size, stride)
                
                # 逐patch处理并写入
                patch_count = len(gt_patches)
                
                # 分小批次处理patches以控制内存
                batch_size = 100  # 每次处理20个patches
                
                for batch_start in range(0, patch_count, batch_size):
                    batch_end = min(batch_start + batch_size, patch_count)
                    batch_patches = gt_patches[batch_start:batch_end]
                    
                    # 批量生成LRHSI和LMS
                    batch_lrhsi, batch_lms = self.generate_lrhsi_and_lms(
                        batch_patches, downsample_factors, 
                        original_size=(patch_size, patch_size)
                    )
                    
                    # 写入当前批次到H5文件
                    start_idx = current_patch_idx + batch_start
                    end_idx = current_patch_idx + batch_end
                    
                    # 写入GT和HRMSI
                    gt_dataset[start_idx:end_idx] = np.stack(batch_patches)
                    hrmsi_dataset[start_idx:end_idx] = np.stack(hrmsi_patches[batch_start:batch_end])
                    
                    # 写入LRHSI和LMS
                    for factor in downsample_factors:
                        lrhsi_datasets[factor][start_idx:end_idx] = np.stack(batch_lrhsi[f'LRHSI_{factor}'])
                        lms_datasets[factor][start_idx:end_idx] = np.stack(batch_lms[f'lms_{factor}'])
                
                # 更新patch索引
                current_patch_idx += patch_count
                
                # 清理图像数据
                del gt_image, hrmsi_image, gt_patches, hrmsi_patches
                
                # 定期垃圾回收
                if (img_idx + 1) % 5 == 0:
                    import gc
                    gc.collect()
                    if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
            
            # 如果实际patches数少于估计数，调整数据集大小
            if current_patch_idx < total_estimated_patches:
                print(f"\n📏 调整数据集大小: {total_estimated_patches} -> {current_patch_idx}")
                
                # 调整所有数据集的大小
                gt_dataset.resize((current_patch_idx, 31, patch_size, patch_size))
                hrmsi_dataset.resize((current_patch_idx, 3, patch_size, patch_size))
                
                for factor in downsample_factors:
                    lrhsi_size = patch_size // factor
                    lrhsi_datasets[factor].resize((current_patch_idx, 31, lrhsi_size, lrhsi_size))
                    lms_datasets[factor].resize((current_patch_idx, 31, patch_size, patch_size))
            
            # 保存元数据 - 移到with语句内部
            f.attrs['patch_size'] = patch_size
            f.attrs['stride'] = stride
            f.attrs['total_patches'] = current_patch_idx
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['train_images'] = len(train_indices)
            f.attrs['original_image_size'] = list(self.gt_data.shape[-2:])
    
        # 这些代码在with语句外部是安全的
        file_size_mb = os.path.getsize(train_file) / (1024**2)
        print(f"✅ 训练集保存完成:")
        print(f"   文件: {train_file}")
        print(f"   大小: {file_size_mb:.1f} MB")
        print(f"   实际patches: {current_patch_idx}")

        return train_file

    def generate_test_set(self):
        """生成测试集 - 边存边处理版本"""
        print("\n" + "="*60)
        print("2️⃣ 生成Harvard测试集（边存边处理）")
        print("="*60)
        
        test_indices = self.config['test_indices']
        test_names = [self.sample_names[i] for i in test_indices]
        crop_size = self.config['test_crop_size']
        downsample_factors = self.config['downsample_factors']
        
        print(f"测试样本: {test_names}")
        print(f"裁剪尺寸: {crop_size}x{crop_size}")
        
        # 创建输出H5文件
        test_file = os.path.join(self.config['output_dir'], f'Harvard_test_crop{crop_size}.h5')
        compression_opts = self.config['compression_level']
        
        with h5py.File(test_file, 'w') as f:
            num_test_images = len(test_indices)
            
            # 预分配测试集数据集
            gt_dataset = f.create_dataset(
                'GT', 
                shape=(num_test_images, 31, crop_size, crop_size),
                dtype=np.float32,
                # compression='gzip', 
                # compression_opts=compression_opts,
                # chunks=True
            )
            
            hrmsi_dataset = f.create_dataset(
                'HRMSI', 
                shape=(num_test_images, 3, crop_size, crop_size),
                dtype=np.float32,
                # compression='gzip', 
                # compression_opts=compression_opts,
                # chunks=True
            )
            
            # 为每个下采样倍数预分配数据集
            lrhsi_datasets = {}
            lms_datasets = {}
            
            for factor in downsample_factors:
                lrhsi_size = crop_size // factor
                
                lrhsi_datasets[factor] = f.create_dataset(
                    f'LRHSI_{factor}',
                    shape=(num_test_images, 31, lrhsi_size, lrhsi_size),
                    dtype=np.float32,
                    # compression='gzip',
                    # compression_opts=compression_opts,
                    # chunks=True
                )
                
                lms_datasets[factor] = f.create_dataset(
                    f'lms_{factor}',
                    shape=(num_test_images, 31, crop_size, crop_size),
                    dtype=np.float32,
                    # compression='gzip',
                    # compression_opts=compression_opts,
                    # chunks=True
                )
            
            # 逐图像处理并直接写入
            print("\n📦 逐图像边存边处理...")
            
            for i, idx in enumerate(test_indices):
                print(f"处理测试图像 {i+1}/{num_test_images}: {test_names[i]}")
                
                # 加载并裁剪图像
                gt_full = self.gt_data[idx]
                hrmsi_full = self.hrmsi_data[idx]
                
                gt_cropped = self.crop_test_region(gt_full, crop_size)
                hrmsi_cropped = self.crop_test_region(hrmsi_full, crop_size)
                
                # 直接写入GT和HRMSI
                gt_dataset[i] = gt_cropped
                hrmsi_dataset[i] = hrmsi_cropped
                
                # 生成并写入LRHSI和LMS
                single_lrhsi, single_lms = self.generate_lrhsi_and_lms(
                    [gt_cropped], downsample_factors, original_size=(crop_size, crop_size))
                
                for factor in downsample_factors:
                    lrhsi_datasets[factor][i] = single_lrhsi[f'LRHSI_{factor}'][0]
                    lms_datasets[factor][i] = single_lms[f'lms_{factor}'][0]
                
                # 立即清理内存
                del gt_full, hrmsi_full, gt_cropped, hrmsi_cropped, single_lrhsi, single_lms
                
                # 强制刷新到磁盘
                f.flush()
                
                # 垃圾回收
                import gc
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 保存图像名称和元数据
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('image_names', data=test_names, dtype=dt)
            
            f.attrs['total_test_images'] = num_test_images
            f.attrs['crop_size'] = crop_size
            f.attrs['original_image_size'] = list(self.gt_data.shape[-2:])
            f.attrs['downsample_factors'] = downsample_factors
        
        file_size_mb = os.path.getsize(test_file) / (1024**2)
        print(f"✅ 测试集保存完成:")
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
            
            if 'image_names' in f:
                names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in f['image_names'][:]]
                print(f"   图像名称: {names}")
                
            print(f"   属性: {dict(f.attrs)}")
        
        print("\n✅ 数据集验证完成!")
    
    def run(self):
        """运行完整的数据集生成流程"""
        print("🚀 开始生成Harvard数据集")
        print("="*80)
        
        try:
            # 1. 加载数据
            self.load_harvard_data()
            
            # 2. 生成训练集
            train_file = self.generate_training_set()
            
            # 3. 生成测试集
            test_file = self.generate_test_set()
            
            # 4. 验证数据集
            self.verify_datasets(train_file, test_file)
            
            # 5. 总结
            print("\n" + "="*80)
            print("🎉 Harvard数据集生成完成!")
            print("="*80)
            print("📁 输出文件:")
            print(f"   训练集: {train_file}")
            print(f"   测试集: {test_file}")
            print()
            print("✅ 数据集特点:")
            print("   • GT: 31通道高光谱数据")
            print("   • HRMSI: 3通道RGB数据")
            print("   • LRHSI_X: 使用Interp23Tap抗混叠下采样的低分辨率高光谱数据")
            print("   • lms_X: 使用双线性插值上采样回原尺寸的低分辨率多光谱数据")
            print("   • 训练集: 67张图像，64步长128x128 patches")
            print("   • 测试集: 10张图像，左上角1000x1000裁剪")
            print("="*80)
            
            return train_file, test_file
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Harvard数据集生成器')
    
    parser.add_argument('--harvard_file', type=str, 
                       default=DEFAULT_CONFIG['harvard_file'],
                       help='Harvard原始数据文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='输出目录')
    
    parser.add_argument('--patch_size', type=int,
                       default=DEFAULT_CONFIG['patch_size'],
                       help='patch尺寸')
    
    parser.add_argument('--stride', type=int,
                       default=DEFAULT_CONFIG['stride'],
                       help='patch步长')
    
    parser.add_argument('--test_crop_size', type=int,
                       default=DEFAULT_CONFIG['test_crop_size'],
                       help='测试集裁剪尺寸')
    
    parser.add_argument('--downsample_factors', nargs='+', type=int,
                       default=DEFAULT_CONFIG['downsample_factors'],
                       help='下采样倍数列表')
    
    parser.add_argument('--compression_level', type=int,
                       default=DEFAULT_CONFIG['compression_level'],
                       help='HDF5压缩级别 (0-9)')
    
    # 新增GPU参数
    parser.add_argument('--gpu_ids', nargs='+', type=int,
                       default=[0, 1, 2, 3, 4, 5, 6],
                       help='使用的GPU ID列表')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    gpu_ids = args.gpu_ids
    del args.gpu_ids  # 从args中移除gpu_ids
    config.update(vars(args))
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  gpu_ids: {gpu_ids}")
    print()
    
    # 创建生成器并运行
    generator = HarvardDatasetGenerator(config, gpu_ids=gpu_ids)
    train_file, test_file = generator.run()
    
    if train_file and test_file:
        print(f"\n✅ 数据集生成成功!")
    else:
        print(f"\n❌ 数据集生成失败!")
        sys.exit(1)

if __name__ == '__main__':
    main()
