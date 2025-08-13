#!/usr/bin/env python3
"""
测试抗混叠下采样功能
"""

import torch
import numpy as np
from generate_cave_dataset import anti_aliasing_downsample
import matplotlib.pyplot as plt

def test_anti_aliasing_downsample():
    """测试抗混叠下采样功能"""
    print("🧪 测试抗混叠下采样功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据 - 模拟CAVE数据
    test_image = torch.randn(31, 128, 128).to(device)  # (C, H, W) - 31通道高光谱数据
    print(f"输入图像shape: {test_image.shape}")
    print(f"输入数据范围: [{test_image.min().item():.4f}, {test_image.max().item():.4f}]")
    
    # 测试不同的下采样倍数
    factors = [2, 4, 8, 16]
    
    for factor in factors:
        print(f"\n📊 测试下采样倍数 {factor}x...")
        
        try:
            # 进行抗混叠下采样
            downsampled = anti_aliasing_downsample(test_image, factor, device)
            
            expected_h = 128 // factor
            expected_w = 128 // factor
            expected_shape = (31, expected_h, expected_w)
            
            print(f"   输入: {test_image.shape}")
            print(f"   输出: {downsampled.shape}")
            print(f"   期望: {expected_shape}")
            print(f"   ✅ 形状正确: {downsampled.shape == expected_shape}")
            
            # 检查数值范围
            print(f"   输出范围: [{downsampled.min().item():.4f}, {downsampled.max().item():.4f}]")
            
            # 计算压缩比
            original_pixels = np.prod(test_image.shape)
            downsampled_pixels = np.prod(downsampled.shape)
            compression_ratio = original_pixels / downsampled_pixels
            print(f"   压缩比: {compression_ratio:.1f}x")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🔄 测试完整下采样+上采样流程...")
    
    # 测试完整流程：GT -> LRHSI -> LMS
    original = torch.randn(31, 128, 128).to(device)
    print(f"原始GT: {original.shape}")
    
    factor = 4
    
    # 步骤1: 抗混叠下采样生成LRHSI
    lrhsi = anti_aliasing_downsample(original, factor, device)
    print(f"LRHSI: {lrhsi.shape}")
    
    # 步骤2: 双线性插值上采样生成LMS
    lrhsi_batch = lrhsi.unsqueeze(0)  # 添加batch维度
    lms = torch.nn.functional.interpolate(
        lrhsi_batch, 
        size=(128, 128), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    print(f"LMS: {lms.shape}")
    
    # 检查尺寸恢复
    print(f"✅ 尺寸恢复正确: {lms.shape == original.shape}")
    
    # 计算重建误差
    mse = torch.mean((original - lms) ** 2).item()
    print(f"重建MSE: {mse:.6f}")
    
    print("\n✅ 抗混叠下采样测试完成!")

if __name__ == '__main__':
    test_anti_aliasing_downsample()
