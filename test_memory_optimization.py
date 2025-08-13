#!/usr/bin/env python3
"""
测试Hermite RBF的内存优化效果
"""

import torch
import sys
import time
import psutil
import os
sys.path.append('.')
from model.MHIIF_J2 import MHIIF_J2

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_memory_scaling():
    """测试不同配置下的内存使用"""
    
    print("=== Hermite RBF 内存优化测试 ===\n")
    
    configs = [
        {"n_kernel": 16, "hermite_order": 1, "name": "小配置 (16核, 1阶)"},
        {"n_kernel": 32, "hermite_order": 2, "name": "中配置 (32核, 2阶)"},
        {"n_kernel": 64, "hermite_order": 2, "name": "大配置 (64核, 2阶)"},
        {"n_kernel": 128, "hermite_order": 2, "name": "超大配置 (128核, 2阶)"},
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    for config in configs:
        print(f"--- {config['name']} ---")
        
        try:
            # 创建模型
            model = MHIIF_J2(
                hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
                use_hermite_rbf=True,
                hermite_order=config['hermite_order'],
                n_kernel=config['n_kernel'],
                hermite_weight=0.5
            ).to(device)
            
            # 获取模型信息
            info = model.get_rbf_info()
            print(f"  内核数量: {info['n_kernels']}")
            print(f"  Hermite维度: {info['hermite_dim']}")
            print(f"  估算内存: {info['estimated_memory_mb']:.1f} MB")
            print(f"  内存策略: {info['memory_strategy']}")
            print(f"  模型参数: {info['rbf_parameters']:,}")
            
            # 测试不同图像大小
            image_sizes = [(32, 32), (64, 64), (128, 128)]
            
            for H, W in image_sizes:
                try:
                    # 创建测试数据
                    B = 1
                    HR_MSI = torch.randn([B, 3, H, W]).to(device)
                    lms = torch.randn([B, 31, H, W]).to(device)
                    LR_HSI = torch.randn([B, 31, H // 4, W // 4]).to(device)
                    
                    # 清理缓存
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 记录开始内存
                    mem_start = get_memory_usage()
                    if device == 'cuda':
                        gpu_mem_start = torch.cuda.memory_allocated() / 1024 / 1024
                    
                    # 执行前向传播
                    start_time = time.time()
                    with torch.no_grad():
                        output = model.sharpening_val_step(lms, LR_HSI, HR_MSI, lms)
                    
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # 记录结束内存
                    mem_end = get_memory_usage()
                    if device == 'cuda':
                        gpu_mem_end = torch.cuda.memory_allocated() / 1024 / 1024
                    
                    # 输出结果
                    print(f"    {H}×{W}: 成功 - {end_time - start_time:.3f}s", end="")
                    if device == 'cuda':
                        print(f", GPU内存: +{gpu_mem_end - gpu_mem_start:.1f}MB")
                    else:
                        print(f", CPU内存: +{mem_end - mem_start:.1f}MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"    {H}×{W}: 内存不足")
                        break
                    else:
                        print(f"    {H}×{W}: 错误 - {str(e)}")
                        break
            
            print()
            
        except Exception as e:
            print(f"  配置失败: {str(e)}\n")
            continue
        
        finally:
            # 清理
            if 'model' in locals():
                del model
            if device == 'cuda':
                torch.cuda.empty_cache()

def test_initialization_quality():
    """测试初始化质量"""
    print("=== 初始化质量测试 ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MHIIF_J2(
        hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
        use_hermite_rbf=True,
        hermite_order=2,
        n_kernel=32,
        hermite_weight=0.5
    ).to(device)
    
    try:
        quality = model.test_initialization_quality(test_samples=100)
        print("初始化质量报告:")
        for k, v in quality.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"初始化质量测试失败: {str(e)}")

if __name__ == "__main__":
    test_memory_scaling()
    test_initialization_quality()
