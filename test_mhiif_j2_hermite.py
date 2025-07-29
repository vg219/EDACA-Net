#!/usr/bin/env python3
"""
MHIIF_J2 + Hermite RBF 测试脚本
验证集成后的模型是否正常工作
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.MHIIF_J2_Hermite import MHIIF_J2_Hermite


def test_model_creation():
    """测试模型创建"""
    print("=== 测试模型创建 ===")
    
    # 测试不使用Hermite RBF
    print("1. 创建原始MHIIF_J2模型...")
    model_original = MHIIF_J2_Hermite(
        hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
        use_hermite_rbf=False
    )
    print(f"   参数量: {sum(p.numel() for p in model_original.parameters()):,}")
    print(f"   RBF信息: {model_original.get_rbf_info()}")
    
    # 测试使用Hermite RBF
    print("2. 创建Hermite RBF版本...")
    model_hermite = MHIIF_J2_Hermite(
        hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
        use_hermite_rbf=True,
        hermite_order=2,
        n_kernel=128,
        rbf_hidden_dim=32,
        hermite_weight=0.3
    )
    print(f"   参数量: {sum(p.numel() for p in model_hermite.parameters()):,}")
    print(f"   RBF信息: {model_hermite.get_rbf_info()}")
    
    return model_original, model_hermite


def test_forward_pass(model, model_name):
    """测试前向传播"""
    print(f"\n=== 测试{model_name}前向传播 ===")
    
    # 创建测试数据
    B, C, H, W = 2, 31, 32, 32
    scale = 4
    
    # 输入数据
    lms = torch.randn(B, C, H, W)  # 低分辨率HSI上采样
    lr_hsi = torch.randn(B, C, H // scale, W // scale)  # 低分辨率HSI
    hr_msi = torch.randn(B, 3, H, W)  # 高分辨率MSI
    gt = torch.randn(B, C, H, W)  # 真值
    
    print(f"输入形状:")
    print(f"  lms: {lms.shape}")
    print(f"  lr_hsi: {lr_hsi.shape}")
    print(f"  hr_msi: {hr_msi.shape}")
    print(f"  gt: {gt.shape}")
    
    try:
        # 测试验证步骤
        model.eval()
        with torch.no_grad():
            pred = model.sharpening_val_step(lms, lr_hsi, hr_msi, gt)
            print(f"验证输出形状: {pred.shape}")
            print(f"输出值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        # 测试训练步骤
        model.train()
        criterion = torch.nn.L1Loss()
        pred_train, loss = model.sharpening_train_step(lms, lr_hsi, hr_msi, gt, criterion)
        print(f"训练输出形状: {pred_train.shape}")
        print(f"训练损失: {loss.item():.6f}")
        
        # 测试梯度计算
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"梯度范数: {grad_norm:.6f}")
        
        print(f"{model_name} 测试通过 ✓")
        return True
        
    except Exception as e:
        print(f"{model_name} 测试失败 ✗")
        print(f"错误: {e}")
        return False


def test_hermite_features(model_hermite):
    """测试Hermite RBF特有功能"""
    print("\n=== 测试Hermite RBF特有功能 ===")
    
    if not model_hermite.use_hermite_rbf:
        print("模型未启用Hermite RBF")
        return
    
    try:
        # 测试RBF信息获取
        rbf_info = model_hermite.get_rbf_info()
        print(f"RBF信息: {rbf_info}")
        
        # 测试核心修剪
        original_kernels = rbf_info['n_kernels']
        pruned_count = model_hermite.prune_rbf_kernels(threshold=1e-6)
        new_rbf_info = model_hermite.get_rbf_info()
        
        print(f"修剪前核心数: {original_kernels}")
        print(f"修剪的核心数: {pruned_count}")
        print(f"修剪后核心数: {new_rbf_info['n_kernels']}")
        
        # 测试核心重要性
        importance = model_hermite.hermite_rbf.get_kernel_importance()
        print(f"核心重要性统计:")
        print(f"  平均值: {importance.mean().item():.6f}")
        print(f"  标准差: {importance.std().item():.6f}")
        print(f"  最大值: {importance.max().item():.6f}")
        print(f"  最小值: {importance.min().item():.6f}")
        
        print("Hermite RBF功能测试通过 ✓")
        return True
        
    except Exception as e:
        print(f"Hermite RBF功能测试失败 ✗")
        print(f"错误: {e}")
        return False


def test_cuda_compatibility():
    """测试CUDA兼容性"""
    print("\n=== 测试CUDA兼容性 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过GPU测试")
        return True
    
    try:
        device = torch.device('cuda:0')
        
        # 创建模型并移到GPU
        model = MHIIF_J2_Hermite(
            hsi_dim=31, msi_dim=3, feat_dim=32, guide_dim=32,
            use_hermite_rbf=True,
            hermite_order=1,
            n_kernel=64,
            rbf_hidden_dim=16
        ).to(device)
        
        # 创建GPU数据
        B, C, H, W = 1, 31, 16, 16
        scale = 4
        
        lms = torch.randn(B, C, H, W, device=device)
        lr_hsi = torch.randn(B, C, H // scale, W // scale, device=device)
        hr_msi = torch.randn(B, 3, H, W, device=device)
        gt = torch.randn(B, C, H, W, device=device)
        
        # 测试GPU前向传播
        model.eval()
        with torch.no_grad():
            pred = model.sharpening_val_step(lms, lr_hsi, hr_msi, gt)
            print(f"GPU推理成功，输出形状: {pred.shape}")
        
        # 测试GPU训练
        model.train()
        criterion = torch.nn.L1Loss()
        pred_train, loss = model.sharpening_train_step(lms, lr_hsi, hr_msi, gt, criterion)
        loss.backward()
        
        print(f"GPU训练成功，损失: {loss.item():.6f}")
        print("CUDA兼容性测试通过 ✓")
        return True
        
    except Exception as e:
        print(f"CUDA兼容性测试失败 ✗")
        print(f"错误: {e}")
        return False


def performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 测试参数
    B, C, H, W = 1, 31, 64, 64
    scale = 4
    
    # 创建测试数据
    lms = torch.randn(B, C, H, W)
    lr_hsi = torch.randn(B, C, H // scale, W // scale)
    hr_msi = torch.randn(B, 3, H, W)
    gt = torch.randn(B, C, H, W)
    
    models = {
        "原始MHIIF_J2": MHIIF_J2_Hermite(
            hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
            use_hermite_rbf=False
        ),
        "Hermite RBF (阶数=1)": MHIIF_J2_Hermite(
            hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
            use_hermite_rbf=True, hermite_order=1, n_kernel=128
        ),
        "Hermite RBF (阶数=2)": MHIIF_J2_Hermite(
            hsi_dim=31, msi_dim=3, feat_dim=64, guide_dim=64,
            use_hermite_rbf=True, hermite_order=2, n_kernel=128
        )
    }
    
    print(f"{'模型':<20} {'参数量':<15} {'内存使用(MB)':<15}")
    print("-" * 50)
    
    for name, model in models.items():
        # 参数量
        params = sum(p.numel() for p in model.parameters())
        
        # 内存使用估算
        model.eval()
        with torch.no_grad():
            pred = model.sharpening_val_step(lms, lr_hsi, hr_msi, gt)
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        print(f"{name:<20} {params:<15,} {memory_mb:<15.1f}")


def main():
    """主测试函数"""
    print("开始MHIIF_J2 + Hermite RBF集成测试\n")
    
    # 测试结果
    test_results = []
    
    # 1. 测试模型创建
    try:
        model_original, model_hermite = test_model_creation()
        test_results.append(("模型创建", True))
    except Exception as e:
        print(f"模型创建失败: {e}")
        test_results.append(("模型创建", False))
        return
    
    # 2. 测试前向传播
    test_results.append(("原始模型前向传播", test_forward_pass(model_original, "原始模型")))
    test_results.append(("Hermite模型前向传播", test_forward_pass(model_hermite, "Hermite模型")))
    
    # 3. 测试Hermite特有功能
    test_results.append(("Hermite特有功能", test_hermite_features(model_hermite)))
    
    # 4. 测试CUDA兼容性
    test_results.append(("CUDA兼容性", test_cuda_compatibility()))
    
    # 5. 性能对比
    try:
        performance_comparison()
        test_results.append(("性能对比", True))
    except Exception as e:
        print(f"性能对比失败: {e}")
        test_results.append(("性能对比", False))
    
    # 总结测试结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！MHIIF_J2 + Hermite RBF集成成功！")
    else:
        print("⚠️  部分测试失败，请检查问题")


if __name__ == "__main__":
    main()
