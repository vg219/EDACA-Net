#!/usr/bin/env python3
"""
简化版CAVE数据集生成测试
"""

import os
import h5py
import numpy as np

def test_simplified_dataset():
    """测试简化版数据集生成"""
    print("🧪 测试简化版CAVE数据集")
    print("="*50)
    
    # 假设的数据文件路径
    train_file = "/data2/users/yujieliang/dataset/CAVE/CAVE_train_patches_stride32_size128.h5"
    test_file = "/data2/users/yujieliang/dataset/CAVE/CAVE_test_fullsize.h5"
    
    print("📂 检查生成的数据集文件...")
    
    # 检查训练集
    if os.path.exists(train_file):
        print(f"✅ 训练集存在: {train_file}")
        with h5py.File(train_file, 'r') as f:
            print("   训练集内容:")
            for key in f.keys():
                print(f"     {key}: {f[key].shape}, dtype={f[key].dtype}")
            
            print("   训练集属性:")
            for attr_name, attr_value in f.attrs.items():
                print(f"     {attr_name}: {attr_value}")
                
            # 检查数据范围
            gt_data = f['GT'][:]
            print(f"\n   数据范围检查:")
            print(f"     GT: [{gt_data.min():.4f}, {gt_data.max():.4f}]")
            
            if 'LRHSI_4' in f:
                lrhsi_data = f['LRHSI_4'][:]
                print(f"     LRHSI_4: [{lrhsi_data.min():.4f}, {lrhsi_data.max():.4f}]")
            
            if 'lms_4' in f:
                lms_data = f['lms_4'][:]
                print(f"     LMS_4: [{lms_data.min():.4f}, {lms_data.max():.4f}]")
    else:
        print(f"❌ 训练集不存在: {train_file}")
    
    # 检查测试集
    if os.path.exists(test_file):
        print(f"\n✅ 测试集存在: {test_file}")
        with h5py.File(test_file, 'r') as f:
            print("   测试集内容:")
            for key in f.keys():
                if key != 'image_names':
                    print(f"     {key}: {f[key].shape}, dtype={f[key].dtype}")
                else:
                    names = [name.decode('utf-8') for name in f[key][:]]
                    print(f"     {key}: {names}")
            
            print("   测试集属性:")
            for attr_name, attr_value in f.attrs.items():
                print(f"     {attr_name}: {attr_value}")
    else:
        print(f"❌ 测试集不存在: {test_file}")
    
    print("\n" + "="*50)
    print("✅ 简化版数据集测试完成!")
    
    print("\n💡 使用示例 (无需归一化):")
    print("""
import h5py
import torch
from torch.utils.data import Dataset

class CAVEDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.length = f['GT'].shape[0]
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            gt = torch.from_numpy(f['GT'][idx]).float()
            lrhsi = torch.from_numpy(f['LRHSI_4'][idx]).float()
            lms = torch.from_numpy(f['lms_4'][idx]).float()
            
        return {'gt': gt, 'lrhsi': lrhsi, 'lms': lms}
    
    def __len__(self):
        return self.length

# 使用示例
dataset = CAVEDataset('CAVE_train_patches.h5')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    gt = batch['gt']      # (B, 31, 128, 128)
    lrhsi = batch['lrhsi'] # (B, 31, 32, 32) for factor=4
    lms = batch['lms']     # (B, 31, 128, 128)
    break
""")

if __name__ == '__main__':
    test_simplified_dataset()
