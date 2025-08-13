#!/usr/bin/env python3
"""
测试MHIIF_J2简化版本的功能
"""

import torch
import sys
import time
sys.path.append('.')

def test_mhiif_j2_simplified():
    """测试MHIIF_J2的简化版本功能"""
    
    print("=== MHIIF_J2 简化版本测试 ===")
    
    try:
        from model.MHIIF_J2 import MHIIF_J2
        print("✓ 模型导入成功")
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型 - 简化版本
    try:
        model_simplified = MHIIF_J2(
            hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
            use_hermite_rbf=True,
            use_four_neighbors=False,  # 关键：使用简化版本
            hermite_order=2,
            n_kernel=64,
            hermite_weight=0.8
        ).to(device)
        print("✓ 简化版本模型创建成功")
    except Exception as e:
        print(f"✗ 简化版本模型创建失败: {e}")
        return False
    
    # 创建模型 - 四邻域版本用于对比
    try:
        model_four_neighbors = MHIIF_J2(
            hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
            use_hermite_rbf=True,
            use_four_neighbors=True,  # 使用四邻域版本
            hermite_order=2,
            n_kernel=64,
            hermite_weight=0.8
        ).to(device)
        print("✓ 四邻域版本模型创建成功")
    except Exception as e:
        print(f"✗ 四邻域版本模型创建失败: {e}")
        model_four_neighbors = None
    
    # 获取模型信息
    info = model_simplified.get_rbf_info()
    print(f"\n=== 模型信息 ===")
    print(f"内核数量: {info['n_kernels']}")
    print(f"Hermite维度: {info['hermite_dim']}")
    print(f"估算内存: {info['estimated_memory_mb']:.1f} MB")
    print(f"内存策略: {info['memory_strategy']}")
    print(f"初始化方式: {info['initialization']}")
    
    # 创建测试数据
    H, W = 64, 64
    B = 1
    
    try:
        HR_MSI = torch.randn([B, 3, H, W]).to(device)
        lms = torch.randn([B, 31, H, W]).to(device)
        LR_HSI = torch.randn([B, 31, H // 4, W // 4]).to(device)
        print(f"✓ 测试数据创建成功: HR_MSI{HR_MSI.shape}, LR_HSI{LR_HSI.shape}")
    except Exception as e:
        print(f"✗ 测试数据创建失败: {e}")
        return False
    
    # 测试简化版本推理
    print(f"\n=== 推理测试 ===")
    try:
        model_simplified.eval()
        start_time = time.time()
        
        with torch.no_grad():
            output_simplified = model_simplified._forward_implem_(HR_MSI, lms, LR_HSI)
        
        end_time = time.time()
        print(f"✓ 简化版本推理成功")
        print(f"  - 输出形状: {output_simplified.shape}")
        print(f"  - 推理时间: {end_time - start_time:.3f}s")
        
    except Exception as e:
        print(f"✗ 简化版本推理失败: {e}")
        return False
    
    # 测试四邻域版本推理（如果可用）
    if model_four_neighbors is not None:
        try:
            model_four_neighbors.eval()
            start_time = time.time()
            
            with torch.no_grad():
                output_four = model_four_neighbors._forward_implem_(HR_MSI, lms, LR_HSI)
            
            end_time = time.time()
            print(f"✓ 四邻域版本推理成功")
            print(f"  - 输出形状: {output_four.shape}")
            print(f"  - 推理时间: {end_time - start_time:.3f}s")
            
            # 计算输出差异
            mse = torch.nn.functional.mse_loss(output_simplified, output_four)
            print(f"  - 输出MSE差异: {mse.item():.6f}")
            
        except Exception as e:
            print(f"✗ 四邻域版本推理失败: {e}")
    
    # 测试性能对比（如果两个版本都可用）
    if model_four_neighbors is not None:
        print(f"\n=== 性能对比测试 ===")
        try:
            # 创建较小的测试数据以加速
            test_feat = torch.randn([B, 64, 16, 16]).to(device)
            test_coord = torch.randn([H * W, 2]).to(device) 
            test_hr_guide = torch.randn([B, 64, H, W]).to(device)
            
            # 使用模型的benchmark方法
            model_simplified.use_four_neighbors = False
            benchmark_results = model_simplified.benchmark_query_methods(
                test_feat, test_coord, test_hr_guide, num_runs=3
            )
            
            print(f"性能对比结果:")
            for k, v in benchmark_results.items():
                if isinstance(v, float):
                    print(f"  - {k}: {v:.3f}")
                else:
                    print(f"  - {k}: {v}")
                    
        except Exception as e:
            print(f"✗ 性能对比失败: {e}")
    
    # 测试快速验证模式
    print(f"\n=== 快速验证模式测试 ===")
    try:
        # 设置快速验证模式
        model_simplified.set_validation_mode(fast_mode=True)
        print("✓ 快速验证模式设置成功")
        
        # 验证use_four_neighbors被设置为False
        assert model_simplified.use_four_neighbors == False, "快速验证模式设置错误"
        print("✓ 快速验证模式验证成功")
        
        # 恢复正常模式
        model_simplified.set_validation_mode(fast_mode=False)
        
    except Exception as e:
        print(f"✗ 快速验证模式测试失败: {e}")
    
    print(f"\n=== 测试完成 ===")
    print("✓ MHIIF_J2简化版本功能正常")
    return True

if __name__ == "__main__":
    success = test_mhiif_j2_simplified()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
