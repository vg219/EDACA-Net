#!/usr/bin/env python3
"""
CAVE数据集快速生成脚本 - 简化版本
专注于快速生成标准格式的CAVE数据集

使用方法:
    python generate_cave_simple.py

输出:
    - CAVE_train_patches_stride40_size128.h5  (训练集)
    - CAVE_test_fullsize.h5                   (测试集)
"""

import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F

def main():
    print("🚀 CAVE数据集快速生成器")
    print("="*50)
    
    # 配置
    cave_file = '/data2/users/yujieliang/dataset/CAVE/CAVE_processed.h5'
    output_dir = '/data2/users/yujieliang/dataset'
    patch_size = 128
    stride = 40
    downsample_factors = [4, 8, 16, 32]
    
    # 检查输入文件
    if not os.path.exists(cave_file):
        print(f"❌ 找不到CAVE数据文件: {cave_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 加载数据: {cave_file}")
    
    # 1. 加载原始数据
    with h5py.File(cave_file, 'r') as f:
        gt_data = f['gt'][:]        # (32, 31, 512, 512)
        hrmsi_data = f['HR_MSI'][:]  # (32, 3, 512, 512)
        sample_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                       for name in f['sample_names'][:]]
    
    print(f"   GT: {gt_data.shape}")
    print(f"   HRMSI: {hrmsi_data.shape}")
    
    # 2. 数据分割
    train_indices = list(range(10, 32))  # 后22张作为训练集
    test_indices = list(range(10))       # 前10张作为测试集
    
    print(f"   训练集: {len(train_indices)} 张图像")
    print(f"   测试集: {len(test_indices)} 张图像")
    
    def crop_patches(image, patch_size, stride):
        """裁剪patches"""
        h, w = image.shape[-2:]
        patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if len(image.shape) == 3:
                    patch = image[:, y:y+patch_size, x:x+patch_size]
                else:
                    patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        return patches
    
    def generate_lrhsi_lms(images, factors):
        """生成LRHSI和LMS数据"""
        lrhsi_data = {}
        lms_data = {}
        
        for factor in factors:
            lrhsi_data[f'LRHSI_{factor}'] = []
            lms_data[f'lms_{factor}'] = []
        
        for factor in factors:
            print(f"   处理factor {factor}...")
            for img in tqdm(images, desc=f"Factor {factor}"):
                # 转换为tensor
                img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, C, H, W)
                
                # 下采样
                lrhsi_tensor = F.avg_pool2d(img_tensor, kernel_size=factor, stride=factor)
                lrhsi = lrhsi_tensor.squeeze(0).numpy()
                lrhsi_data[f'LRHSI_{factor}'].append(lrhsi)
                
                # 上采样
                original_size = img.shape[-2:]
                lms_tensor = F.interpolate(lrhsi_tensor, size=original_size, 
                                         mode='bilinear', align_corners=False)
                lms = lms_tensor.squeeze(0).numpy()
                lms_data[f'lms_{factor}'].append(lms)
        
        return lrhsi_data, lms_data
    
    # 3. 生成训练集
    print("\n📦 生成训练集...")
    
    train_gt = gt_data[train_indices]
    train_hrmsi = hrmsi_data[train_indices]
    
    # 裁剪patches
    print("   裁剪patches...")
    all_gt_patches = []
    all_hrmsi_patches = []
    
    for i in tqdm(range(len(train_gt)), desc="裁剪"):
        gt_patches = crop_patches(train_gt[i], patch_size, stride)
        hrmsi_patches = crop_patches(train_hrmsi[i], patch_size, stride)
        all_gt_patches.extend(gt_patches)
        all_hrmsi_patches.extend(hrmsi_patches)
    
    print(f"   总patches: {len(all_gt_patches)}")
    
    # 生成LRHSI和LMS
    print("   生成LRHSI和LMS...")
    train_lrhsi, train_lms = generate_lrhsi_lms(all_gt_patches, downsample_factors)
    
    # 保存训练集
    print("   保存训练集...")
    train_file = os.path.join(output_dir, f'CAVE_train_patches_stride{stride}_size{patch_size}.h5')
    
    with h5py.File(train_file, 'w') as f:
        # 保存数据
        f.create_dataset('GT', data=np.stack(all_gt_patches), compression='gzip', compression_opts=9)
        f.create_dataset('HRMSI', data=np.stack(all_hrmsi_patches), compression='gzip', compression_opts=9)
        
        for factor in downsample_factors:
            f.create_dataset(f'LRHSI_{factor}', data=np.stack(train_lrhsi[f'LRHSI_{factor}']), 
                           compression='gzip', compression_opts=9)
            f.create_dataset(f'lms_{factor}', data=np.stack(train_lms[f'lms_{factor}']), 
                           compression='gzip', compression_opts=9)
        
        # 保存元数据
        f.attrs['patch_size'] = patch_size
        f.attrs['stride'] = stride
        f.attrs['total_patches'] = len(all_gt_patches)
        f.attrs['downsample_factors'] = downsample_factors
    
    train_size = os.path.getsize(train_file) / (1024**2)
    print(f"   ✅ 训练集: {train_file} ({train_size:.1f} MB)")
    
    # 4. 生成测试集
    print("\n📦 生成测试集...")
    
    test_gt = gt_data[test_indices]
    test_hrmsi = hrmsi_data[test_indices]
    test_names = [sample_names[i] for i in test_indices]
    
    # 生成LRHSI和LMS
    print("   生成测试集LRHSI和LMS...")
    test_gt_list = [test_gt[i] for i in range(len(test_gt))]
    test_lrhsi, test_lms = generate_lrhsi_lms(test_gt_list, downsample_factors)
    
    # 保存测试集
    print("   保存测试集...")
    test_file = os.path.join(output_dir, 'CAVE_test_fullsize.h5')
    
    with h5py.File(test_file, 'w') as f:
        # 保存数据
        f.create_dataset('GT', data=test_gt, compression='gzip', compression_opts=9)
        f.create_dataset('HRMSI', data=test_hrmsi, compression='gzip', compression_opts=9)
        
        for factor in downsample_factors:
            f.create_dataset(f'LRHSI_{factor}', data=np.stack(test_lrhsi[f'LRHSI_{factor}']), 
                           compression='gzip', compression_opts=9)
            f.create_dataset(f'lms_{factor}', data=np.stack(test_lms[f'lms_{factor}']), 
                           compression='gzip', compression_opts=9)
        
        # 保存图像名称
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('image_names', data=test_names, dtype=dt)
        
        # 保存元数据
        f.attrs['total_test_images'] = len(test_indices)
        f.attrs['image_size'] = [512, 512]
        f.attrs['downsample_factors'] = downsample_factors
    
    test_size = os.path.getsize(test_file) / (1024**2)
    print(f"   ✅ 测试集: {test_file} ({test_size:.1f} MB)")
    
    # 5. 验证数据
    print("\n🔍 验证数据集...")
    
    with h5py.File(train_file, 'r') as f:
        print(f"   训练集键值: {list(f.keys())}")
        print(f"   GT shape: {f['GT'].shape}")
        print(f"   LRHSI_4 shape: {f['LRHSI_4'].shape}")
        print(f"   lms_4 shape: {f['lms_4'].shape}")
    
    with h5py.File(test_file, 'r') as f:
        print(f"   测试集键值: {list(f.keys())}")
        print(f"   GT shape: {f['GT'].shape}")
        print(f"   测试图像: {[name.decode('utf-8') for name in f['image_names'][:3]]}...")
    
    print("\n" + "="*50)
    print("🎉 CAVE数据集生成完成!")
    print("="*50)
    print("📁 输出文件:")
    print(f"   训练集: {train_file}")
    print(f"   测试集: {test_file}")
    print()
    print("✅ 数据结构:")
    print("   GT - 31通道高光谱数据")
    print("   HRMSI - 3通道RGB数据")
    print("   LRHSI_4/8/16/32 - 下采样的低分辨率高光谱")
    print("   lms_4/8/16/32 - 上采样的低分辨率多光谱")
    print("="*50)

if __name__ == '__main__':
    main()
