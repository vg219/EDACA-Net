#!/usr/bin/env python3
"""
MHIIF_J2 + Hermite RBF 训练脚本
基于原始MHIIF_J2架构集成Hermite RBF
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.MHIIF_J2_Hermite import MHIIF_J2_Hermite


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MHIIF_J2 + Hermite RBF Training')
    
    parser.add_argument('--config', type=str, 
                       default='./configs/mhiif_j2_hermite_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅进行评估')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                       help='GPU ID列表')
    parser.add_argument('--disable_hermite', action='store_true',
                       help='禁用Hermite RBF（对比实验）')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, disable_hermite: bool = False) -> MHIIF_J2_Hermite:
    """创建模型"""
    model_config = config['model'].copy()
    
    # 如果禁用Hermite，覆盖配置
    if disable_hermite:
        model_config['use_hermite_rbf'] = False
        print("Hermite RBF已禁用（对比实验模式）")
    
    model = MHIIF_J2_Hermite(**model_config)
    
    return model


def setup_experiment(config: dict, disable_hermite: bool = False) -> tuple:
    """设置实验环境"""
    # 创建实验目录
    exp_name = config['experiment']['name']
    if disable_hermite:
        exp_name += "_NoHermite"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 设置随机种子
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 保存配置文件
    save_config = config.copy()
    if disable_hermite:
        save_config['model']['use_hermite_rbf'] = False
    
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(save_config, f, default_flow_style=False)
    
    print(f"实验目录: {exp_dir}")
    print(f"配置: {save_config['model']}")
    
    return exp_dir, checkpoint_dir


def train_epoch(model, train_loader, optimizer, criterion, config, epoch, device):
    """训练一个epoch"""
    model.train()
    epoch_losses = []
    hermite_losses = []
    
    for batch_idx, batch_data in enumerate(train_loader):
        # 假设数据格式 (需要根据实际数据加载器调整)
        lms = batch_data['lms'].to(device)
        lr_hsi = batch_data['lr_hsi'].to(device)
        hr_msi = batch_data['hr_msi'].to(device)
        gt = batch_data['gt'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（包含Hermite损失）
        pred, loss = model.sharpening_train_step(lms, lr_hsi, hr_msi, gt, criterion)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if 'gradient_clip' in config['training']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
        # 打印训练信息
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.6f}")
            
            # 如果使用Hermite RBF，打印RBF信息
            if model.use_hermite_rbf and config['logging'].get('log_rbf_info', False):
                if batch_idx % (config['logging'].get('log_rbf_frequency', 20) * 50) == 0:
                    rbf_info = model.get_rbf_info()
                    print(f"RBF信息: {rbf_info}")
    
    return np.mean(epoch_losses)


def validate_epoch(model, val_loader, device):
    """验证一个epoch"""
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            lms = batch_data['lms'].to(device)
            lr_hsi = batch_data['lr_hsi'].to(device)
            hr_msi = batch_data['hr_msi'].to(device)
            gt = batch_data['gt'].to(device)
            
            # 验证
            pred = model.sharpening_val_step(lms, lr_hsi, hr_msi, gt)
            
            # 计算评估指标（需要实现具体的指标计算）
            # psnr = calculate_psnr(pred, gt)
            # ssim = calculate_ssim(pred, gt)
            # sam = calculate_sam(pred, gt)
            
            if batch_idx == 0:
                print(f"验证批次 {batch_idx}: 输出形状 {pred.shape}")
    
    return {"PSNR": 0.0, "SSIM": 0.0, "SAM": 0.0}  # 占位符


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }
    
    # 如果使用Hermite RBF，保存RBF信息
    if model.use_hermite_rbf:
        checkpoint['rbf_info'] = model.get_rbf_info()
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_metric', 0.0)


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
        print(f"使用GPU: {args.gpu_ids}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 设置实验
    exp_dir, checkpoint_dir = setup_experiment(config, args.disable_hermite)
    
    # 创建模型
    model = create_model(config, args.disable_hermite).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    if model.use_hermite_rbf:
        print(f"RBF信息: {model.get_rbf_info()}")
    else:
        print("未使用Hermite RBF")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['eta_min']
    )
    
    # 创建损失函数
    criterion = torch.nn.L1Loss()
    
    # 加载数据集（需要实现具体的数据加载逻辑）
    # train_loader, val_loader = load_dataset(config)
    print("注意：需要实现具体的数据加载逻辑")
    
    # 恢复训练
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"从epoch {start_epoch}恢复训练，最佳PSNR: {best_psnr:.4f}")
    
    # 仅评估模式
    if args.eval_only:
        print("开始评估...")
        # val_metrics = validate_epoch(model, val_loader, device)
        # print(f"评估完成: {val_metrics}")
        return
    
    # 训练循环
    print("开始训练...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # 训练
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
              f"LR={current_lr:.8f}")
        
        # 验证
        if (epoch + 1) % config['evaluation']['eval_frequency'] == 0:
            # val_metrics = validate_epoch(model, val_loader, device)
            
            # 保存最佳模型
            # if val_metrics['PSNR'] > best_psnr:
            #     best_psnr = val_metrics['PSNR']
            #     save_checkpoint(model, optimizer, scheduler, epoch + 1, best_psnr,
            #                   checkpoint_dir / "best_model.pth")
            #     print(f"保存最佳模型，PSNR: {best_psnr:.4f}")
            pass
        
        # 定期保存检查点
        if (epoch + 1) % config['evaluation']['save_frequency'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_psnr,
                          checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        # 核心修剪
        if (model.use_hermite_rbf and 
            config['training']['kernel_pruning']['enabled'] and
            (epoch + 1) in config['training']['kernel_pruning']['prune_epochs']):
            pruned_count = model.prune_rbf_kernels(
                config['training']['kernel_pruning']['threshold']
            )
            print(f"修剪了 {pruned_count} 个RBF核心")
            print(f"当前RBF信息: {model.get_rbf_info()}")
    
    print(f"训练完成！最佳PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()
