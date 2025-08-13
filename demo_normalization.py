#!/usr/bin/env python3
"""
演示CAVE数据集归一化使用方法
"""

import h5py
import numpy as np

def demo_normalization():
    """演示如何使用最大值归一化"""
    print("📊 CAVE数据集归一化示例")
    print("="*50)
    
    # 假设的数据文件路径（替换为实际路径）
    train_file = "/data2/users/yujieliang/dataset/CAVE/CAVE_train_patches_stride32_size128.h5"
    test_file = "/data2/users/yujieliang/dataset/CAVE/CAVE_test_fullsize.h5"
    
    print("📂 加载训练集数据...")
    try:
        with h5py.File(train_file, 'r') as f:
            print("可用的数据集:")
            for key in f.keys():
                print(f"   {key}: {f[key].shape}")
            
            print("\n可用的归一化参数:")
            for attr_name in f.attrs.keys():
                if attr_name.endswith('_max') or attr_name.endswith('_min'):
                    print(f"   {attr_name}: {f.attrs[attr_name]:.4f}")
            
            # 加载一些数据进行演示
            gt_data = f['GT'][:5]  # 只取前5个patches
            lrhsi_4_data = f['LRHSI_4'][:5]
            lms_4_data = f['lms_4'][:5]
            
            # 获取归一化参数
            gt_max = f.attrs['gt_max']
            lrhsi_4_max = f.attrs['LRHSI_4_max']
            lms_4_max = f.attrs['lms_4_max']
            
            print(f"\n🔢 原始数据范围:")
            print(f"   GT: [{gt_data.min():.4f}, {gt_data.max():.4f}]")
            print(f"   LRHSI_4: [{lrhsi_4_data.min():.4f}, {lrhsi_4_data.max():.4f}]")
            print(f"   LMS_4: [{lms_4_data.min():.4f}, {lms_4_data.max():.4f}]")
            
            # 进行归一化
            gt_normalized = gt_data / gt_max
            lrhsi_4_normalized = lrhsi_4_data / lrhsi_4_max
            lms_4_normalized = lms_4_data / lms_4_max
            
            print(f"\n✨ 归一化后数据范围:")
            print(f"   GT: [{gt_normalized.min():.4f}, {gt_normalized.max():.4f}]")
            print(f"   LRHSI_4: [{lrhsi_4_normalized.min():.4f}, {lrhsi_4_normalized.max():.4f}]")
            print(f"   LMS_4: [{lms_4_normalized.min():.4f}, {lms_4_normalized.max():.4f}]")
            
            print(f"\n📋 归一化公式:")
            print(f"   normalized_data = original_data / max_value")
            print(f"   恢复公式: original_data = normalized_data * max_value")
            
    except FileNotFoundError:
        print(f"❌ 训练文件不存在: {train_file}")
        print("请先运行 generate_cave_dataset.py 生成数据集")
    
    print("\n" + "="*50)
    print("✅ 归一化演示完成!")
    
    print("\n💡 在PyTorch训练中的使用:")
    print("""
import torch
from torch.utils.data import Dataset

class CAVEDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.gt_max = f.attrs['gt_max']
            self.lrhsi_4_max = f.attrs['LRHSI_4_max']
            self.lms_4_max = f.attrs['lms_4_max']
            self.length = f['GT'].shape[0]
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            gt = torch.from_numpy(f['GT'][idx]).float() / self.gt_max
            lrhsi = torch.from_numpy(f['LRHSI_4'][idx]).float() / self.lrhsi_4_max
            lms = torch.from_numpy(f['lms_4'][idx]).float() / self.lms_4_max
            
        return {'gt': gt, 'lrhsi': lrhsi, 'lms': lms}
    
    def __len__(self):
        return self.length

# 使用示例
dataset = CAVEDataset('CAVE_train_patches.h5')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
""")

if __name__ == '__main__':
    demo_normalization()
