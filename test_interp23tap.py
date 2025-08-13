#!/usr/bin/env python3
"""
测试Interp23Tap功能
"""

import torch
import numpy as np
from generate_cave_dataset import Interp23Tap
import matplotlib.pyplot as plt

def test_interp23tap():
    """测试Interp23Tap类"""
    print("🧪 测试Interp23Tap类...")
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建一个简单的测试图像 (1, 3, 32, 32)
    test_image = torch.randn(1, 3, 32, 32).to(device)
    print(f"输入图像shape: {test_image.shape}")
    
    # 测试不同的上采样倍数
    factors = [2, 4, 8]
    
    for factor in factors:
        print(f"\n📊 测试上采样倍数 {factor}x...")
        
        try:
            # 创建Interp23Tap实例
            interp = Interp23Tap(ratio=factor).to(device)
            
            # 进行上采样
            upsampled = interp(test_image)
            
            expected_shape = (1, 3, 32 * factor, 32 * factor)
            print(f"   输入: {test_image.shape}")
            print(f"   输出: {upsampled.shape}")
            print(f"   期望: {expected_shape}")
            print(f"   ✅ 形状正确: {upsampled.shape == expected_shape}")
            
            # 检查数值范围
            print(f"   输入范围: [{test_image.min().item():.4f}, {test_image.max().item():.4f}]")
            print(f"   输出范围: [{upsampled.min().item():.4f}, {upsampled.max().item():.4f}]")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    print("\n🔄 测试下采样+上采样流程...")
    
    # 模拟完整的下采样+上采样流程
    original = torch.randn(1, 31, 128, 128).to(device)  # 模拟CAVE patch
    print(f"原始图像: {original.shape}")
    
    factor = 4
    
    # 下采样 (模拟LRHSI生成)
    downsampled = torch.nn.functional.avg_pool2d(original, kernel_size=factor, stride=factor)
    print(f"下采样后: {downsampled.shape}")
    
    # 上采样 (模拟LMS生成)
    interp = Interp23Tap(ratio=factor).to(device)
    upsampled = interp(downsampled)
    print(f"上采样后: {upsampled.shape}")
    
    # 检查是否恢复到原始尺寸
    print(f"尺寸恢复正确: {upsampled.shape == original.shape}")
    
    print("\n✅ Interp23Tap测试完成!")

if __name__ == '__main__':
    test_interp23tap()
