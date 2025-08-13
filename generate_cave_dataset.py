#!/usr/bin/env python3
"""
CAVE数据集生成脚本
生成标准格式的训练集和测试集，包含GT, HRMSI, LRHSI_X, lms_X数据

数据流程:
1. GT (Ground Truth) - 原始31通道高光谱数据
2. HRMSI - 3通道RGB数据
3. LRHSI_X - GT通过Interp23Tap抗混叠下采样X倍得到的低分辨率高光谱数据
4. lms_X - LRHSI_X通过双线性插值上采样回原尺寸得到的低分辨率多光谱数据

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
    'cave_file': '/data2/users/yujieliang/dataset/CAVE/CAVE_processed.h5',
    'output_dir': '/data2/users/yujieliang/dataset/CAVE',
    'patch_size': 128,
    'stride': 32,
    'downsample_factors': [4, 8, 16, 32],
    'train_indices': list(range(10, 32)),  # 后22张作为训练集
    'test_indices': list(range(10)),       # 前10张作为测试集
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

class CAVEDatasetGenerator:
    """CAVE数据集生成器"""
    
    def __init__(self, config=None):
        """初始化生成器"""
        self.config = config or DEFAULT_CONFIG.copy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_cave_data(self):
        """加载原始CAVE数据"""
        print("📂 加载原始CAVE数据...")
        
        cave_file = self.config['cave_file']
        if not os.path.exists(cave_file):
            raise FileNotFoundError(f"CAVE文件不存在: {cave_file}")
        
        with h5py.File(cave_file, 'r') as f:
            # 检查文件内容
            print(f"   文件中的数据集: {list(f.keys())}")
            
            # 加载数据
            self.gt_data = f['gt'][:]          # (32, 31, 512, 512)
            self.hrmsi_data = f['HR_MSI'][:]   # (32, 3, 512, 512)
            
            # 加载样本名称
            if 'sample_names' in f:
                sample_names_raw = f['sample_names'][:]
                self.sample_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                   for name in sample_names_raw]
            else:
                self.sample_names = [f'cave_{i:02d}' for i in range(len(self.gt_data))]
        
        print(f"   GT数据: {self.gt_data.shape}")
        print(f"   HRMSI数据: {self.hrmsi_data.shape}")
        print(f"   样本数量: {len(self.sample_names)}")
        print(f"   数值范围: GT[{self.gt_data.min():.2f}, {self.gt_data.max():.2f}]")
        
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
    
    def generate_lrhsi_and_lms(self, images, downsample_factors, is_patch=True):
        """
        生成LRHSI和LMS数据
        使用Interp23Tap抗混叠下采样生成LRHSI，双线性插值上采样生成LMS
        Args:
            images: 输入图像列表或数组
            downsample_factors: 下采样倍数列表
            is_patch: 是否为patch数据
        Returns:
            lrhsi_data: dict of LRHSI数据
            lms_data: dict of LMS数据
        """
        lrhsi_data = {}
        lms_data = {}
        
        for factor in downsample_factors:
            lrhsi_data[f'LRHSI_{factor}'] = []
            lms_data[f'lms_{factor}'] = []
        
        print(f"🔄 生成LRHSI和LMS数据（使用Interp23Tap抗混叠下采样）...")
        
        for factor in downsample_factors:
            print(f"   处理下采样倍数 {factor}x...")
            
            for i, image in enumerate(tqdm(images, desc=f"生成LRHSI/LMS {factor}x")):
                try:
                    # 转换为tensor
                    if isinstance(image, np.ndarray):
                        img_tensor = torch.from_numpy(image).float().to(self.device)  # (C, H, W)
                    else:
                        img_tensor = image.to(self.device)
                    
                    # 步骤1: 使用抗混叠下采样生成LRHSI
                    lrhsi_tensor = anti_aliasing_downsample(img_tensor, factor, self.device)  # (C, H//factor, W//factor)
                    lrhsi = lrhsi_tensor.cpu().numpy()
                    lrhsi_data[f'LRHSI_{factor}'].append(lrhsi)
                    
                    # 步骤2: 使用双线性插值上采样生成LMS
                    original_size = image.shape[-2:] if is_patch else (512, 512)
                    lrhsi_batch = lrhsi_tensor.unsqueeze(0)  # 添加batch维度
                    lms_tensor = F.interpolate(lrhsi_batch, size=original_size, 
                                             mode='bilinear', align_corners=False)
                    lms = lms_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
                    lms_data[f'lms_{factor}'].append(lms)
                    
                    # 清理GPU内存
                    del img_tensor, lrhsi_tensor, lrhsi_batch, lms_tensor
                    
                except Exception as e:
                    print(f"处理图像 {i} 时出错 (factor={factor}): {e}")
                    print("降级到简单平均池化下采样...")
                    
                    # 降级到简单的平均池化下采样
                    if isinstance(image, np.ndarray):
                        img_tensor = torch.from_numpy(image).float().to(self.device)
                    else:
                        img_tensor = image.to(self.device)
                    
                    img_batch = img_tensor.unsqueeze(0)
                    lrhsi_tensor = F.avg_pool2d(img_batch, kernel_size=factor, stride=factor)
                    lrhsi = lrhsi_tensor.squeeze(0).cpu().numpy()
                    lrhsi_data[f'LRHSI_{factor}'].append(lrhsi)
                    
                    # 使用双线性插值生成LMS
                    original_size = image.shape[-2:] if is_patch else (512, 512)
                    lms_tensor = F.interpolate(lrhsi_tensor, size=original_size, 
                                             mode='bilinear', align_corners=False)
                    lms = lms_tensor.squeeze(0).cpu().numpy()
                    lms_data[f'lms_{factor}'].append(lms)
                    
                    del img_tensor, img_batch, lrhsi_tensor, lms_tensor
            
            # 强制垃圾回收
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return lrhsi_data, lms_data
    
    def generate_training_set(self):
        """生成训练集"""
        print("\n" + "="*60)
        print("1️⃣ 生成训练集")
        print("="*60)
        
        # 提取训练数据
        train_indices = self.config['train_indices']
        train_gt = self.gt_data[train_indices]
        train_hrmsi = self.hrmsi_data[train_indices]
        train_names = [self.sample_names[i] for i in train_indices]
        
        print(f"训练集数据: GT{train_gt.shape}, HRMSI{train_hrmsi.shape}")
        print(f"训练样本: {train_names[:3]}...（共{len(train_names)}个）")
        
        # 生成patches
        print("\n📦 生成patches...")
        all_gt_patches = []
        all_hrmsi_patches = []
        
        patch_size = self.config['patch_size']
        stride = self.config['stride']
        
        for i in tqdm(range(len(train_gt)), desc="裁剪训练patches"):
            gt_image = train_gt[i]      # (31, 512, 512)
            hrmsi_image = train_hrmsi[i]  # (3, 512, 512)
            
            # 裁剪GT patches
            gt_patches, _ = self.crop_patches_overlapping(gt_image, patch_size, stride)
            all_gt_patches.extend(gt_patches)
            
            # 裁剪HRMSI patches
            hrmsi_patches, _ = self.crop_patches_overlapping(hrmsi_image, patch_size, stride)
            all_hrmsi_patches.extend(hrmsi_patches)
        
        print(f"   总patches数: {len(all_gt_patches)}")
        
        # 生成LRHSI和LMS
        downsample_factors = self.config['downsample_factors']
        train_lrhsi, train_lms = self.generate_lrhsi_and_lms(
            all_gt_patches, downsample_factors, is_patch=True)
        
        # 保存训练集
        print("\n💾 保存训练集...")
        train_file = os.path.join(
            self.config['output_dir'], 
            f'CAVE_train_patches_stride{stride}_size{patch_size}.h5'
        )
        
        with h5py.File(train_file, 'w') as f:
            # 保存主要数据
            compression_opts = self.config['compression_level']
            gt_stack = np.stack(all_gt_patches)
            hrmsi_stack = np.stack(all_hrmsi_patches)
            
            f.create_dataset('GT', data=gt_stack, compression='gzip', compression_opts=compression_opts)
            f.create_dataset('HRMSI', data=hrmsi_stack, compression='gzip', compression_opts=compression_opts)
            
            # 保存LRHSI和LMS
            for factor in downsample_factors:
                f.create_dataset(f'LRHSI_{factor}', 
                               data=np.stack(train_lrhsi[f'LRHSI_{factor}']), 
                               compression='gzip', compression_opts=compression_opts)
                f.create_dataset(f'lms_{factor}', 
                               data=np.stack(train_lms[f'lms_{factor}']), 
                               compression='gzip', compression_opts=compression_opts)
            
            # 保存元数据
            f.attrs['patch_size'] = patch_size
            f.attrs['stride'] = stride
            f.attrs['total_patches'] = len(all_gt_patches)
            f.attrs['downsample_factors'] = downsample_factors
            f.attrs['train_images'] = len(train_indices)
        
        file_size_mb = os.path.getsize(train_file) / (1024**2)
        print(f"✅ 训练集保存完成:")
        print(f"   文件: {train_file}")
        print(f"   大小: {file_size_mb:.1f} MB")
        
        return train_file
    
    def generate_test_set(self):
        """生成测试集"""
        print("\n" + "="*60)
        print("2️⃣ 生成测试集")
        print("="*60)
        
        # 提取测试数据
        test_indices = self.config['test_indices']
        test_gt = self.gt_data[test_indices]
        test_hrmsi = self.hrmsi_data[test_indices]
        test_names = [self.sample_names[i] for i in test_indices]
        
        print(f"测试集数据: GT{test_gt.shape}, HRMSI{test_hrmsi.shape}")
        print(f"测试样本: {test_names}")
        
        # 生成LRHSI和LMS（全尺寸图像）
        downsample_factors = self.config['downsample_factors']
        test_gt_list = [test_gt[i] for i in range(len(test_gt))]
        test_lrhsi, test_lms = self.generate_lrhsi_and_lms(
            test_gt_list, downsample_factors, is_patch=False)
        
        # 保存测试集
        print("\n💾 保存测试集...")
        test_file = os.path.join(self.config['output_dir'], 'CAVE_test_fullsize.h5')
        
        with h5py.File(test_file, 'w') as f:
            # 保存主要数据
            compression_opts = self.config['compression_level']
            f.create_dataset('GT', data=test_gt, compression='gzip', compression_opts=compression_opts)
            f.create_dataset('HRMSI', data=test_hrmsi, compression='gzip', compression_opts=compression_opts)
            
            # 保存LRHSI和LMS
            for factor in downsample_factors:
                f.create_dataset(f'LRHSI_{factor}', 
                               data=np.stack(test_lrhsi[f'LRHSI_{factor}']), 
                               compression='gzip', compression_opts=compression_opts)
                f.create_dataset(f'lms_{factor}', 
                               data=np.stack(test_lms[f'lms_{factor}']), 
                               compression='gzip', compression_opts=compression_opts)
            
            # 保存图像名称
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('image_names', data=test_names, dtype=dt)
            
            # 保存元数据
            f.attrs['total_test_images'] = len(test_indices)
            f.attrs['image_size'] = [512, 512]
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
                names = [name.decode('utf-8') for name in f['image_names'][:]]
                print(f"   图像名称: {names}")
        
        print("\n✅ 数据集验证完成!")
    
    def run(self):
        """运行完整的数据集生成流程"""
        print("🚀 开始生成CAVE数据集")
        print("="*80)
        
        try:
            # 1. 加载数据
            self.load_cave_data()
            
            # 2. 生成训练集
            train_file = self.generate_training_set()
            
            # 3. 生成测试集
            test_file = self.generate_test_set()
            
            # 4. 验证数据集
            self.verify_datasets(train_file, test_file)
            
            # 5. 总结
            print("\n" + "="*80)
            print("🎉 CAVE数据集生成完成!")
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
            print("   • 训练集: patch格式，便于训练")
            print("   • 测试集: 全尺寸图像，便于评估")
            print("="*80)
            
            return train_file, test_file
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CAVE数据集生成器')
    
    parser.add_argument('--cave_file', type=str, 
                       default=DEFAULT_CONFIG['cave_file'],
                       help='CAVE原始数据文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='输出目录')
    
    parser.add_argument('--patch_size', type=int,
                       default=DEFAULT_CONFIG['patch_size'],
                       help='patch尺寸')
    
    parser.add_argument('--stride', type=int,
                       default=DEFAULT_CONFIG['stride'],
                       help='patch步长')
    
    parser.add_argument('--downsample_factors', nargs='+', type=int,
                       default=DEFAULT_CONFIG['downsample_factors'],
                       help='下采样倍数列表')
    
    parser.add_argument('--compression_level', type=int,
                       default=DEFAULT_CONFIG['compression_level'],
                       help='HDF5压缩级别 (0-9)')
    
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
    generator = CAVEDatasetGenerator(config)
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
