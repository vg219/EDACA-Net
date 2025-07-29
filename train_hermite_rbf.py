#!/usr/bin/env python3
"""
Hermite RBF训练脚本
用于启动使用Hermite径向基函数改进的MHIIF网络训练
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

from model.mhiif_hermite import create_hermite_mhiif_system, HermiteTrainer
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hermite RBF MHIIF Training')
    
    parser.add_argument('--config', type=str, 
                       default='./configs/hermite_rbf_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅进行评估')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                       help='GPU ID列表')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: dict) -> tuple:
    """设置实验环境"""
    # 创建实验目录
    exp_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 设置日志
    logger = setup_logger(exp_dir / "train.log", config['logging']['log_level'])
    
    # 设置随机种子
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 保存配置文件
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"配置: {config}")
    
    return exp_dir, checkpoint_dir, logger


def create_model_and_trainer(config: dict, device: torch.device) -> tuple:
    """创建模型和训练器"""
    # 创建系统
    system = create_hermite_mhiif_system(
        hsi_channels=config['model']['hsi_channels'],
        scale_factor=config['model']['scale_factor'],
        hermite_order=config['model']['hermite_order'],
        n_kernel=config['model']['n_kernel'],
        learning_rate=config['training']['learning_rate']
    )
    
    model = system['model'].to(device)
    optimizer = system['optimizer']
    scheduler = system['scheduler']
    
    # 创建训练器
    trainer = HermiteTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        max_hermite_order=config['model']['hermite_order'],
        progressive_training=config['training']['progressive_training']
    )
    
    return model, trainer, optimizer, scheduler


def load_dataset(config: dict):
    """加载数据集"""
    # 这里需要根据你的具体数据集实现
    # 示例：
    from utils.dataset import load_cave_dataset  # 假设你有这个函数
    
    train_loader, val_loader = load_cave_dataset(
        data_path=config['dataset']['data_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        augmentation=config['dataset']['augmentation']
    )
    
    return train_loader, val_loader


def train_epoch(trainer: HermiteTrainer, 
                train_loader, 
                epoch: int, 
                logger) -> dict:
    """训练一个epoch"""
    epoch_losses = {
        'total': [], 'recon': [], 'grad': [], 
        'smooth': [], 'sparsity': []
    }
    
    for batch_idx, (hsi_lr, msi_hr, hsi_hr_gt) in enumerate(train_loader):
        # 训练步骤
        loss_dict = trainer.train_step(hsi_lr, msi_hr, hsi_hr_gt, epoch)
        
        # 记录损失
        for key in epoch_losses:
            if key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
        
        # 打印训练信息
        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                       f"Loss {loss_dict['total']:.6f}")
    
    # 计算平均损失
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
    return avg_losses


def validate_epoch(trainer: HermiteTrainer, 
                  val_loader, 
                  logger) -> dict:
    """验证一个epoch"""
    all_metrics = {
        'PSNR': [], 'SSIM': [], 'SAM': [],
        'total': [], 'recon': []
    }
    
    for batch_idx, (hsi_lr, msi_hr, hsi_hr_gt) in enumerate(val_loader):
        # 评估步骤
        metrics = trainer.evaluate_step(hsi_lr, msi_hr, hsi_hr_gt)
        
        # 记录指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    
    logger.info(f"验证结果: PSNR={avg_metrics['PSNR']:.4f}, "
               f"SSIM={avg_metrics['SSIM']:.4f}, "
               f"SAM={avg_metrics['SAM']:.4f}")
    
    return avg_metrics


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
    exp_dir, checkpoint_dir, logger = setup_experiment(config)
    
    # 创建模型和训练器
    model, trainer, optimizer, scheduler = create_model_and_trainer(config, device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    logger.info(f"RBF信息: {model.get_rbf_info()}")
    
    # 加载数据集
    train_loader, val_loader = load_dataset(config)
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 恢复训练
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        logger.info(f"从epoch {start_epoch}恢复训练，最佳PSNR: {best_psnr:.4f}")
    
    # 仅评估模式
    if args.eval_only:
        logger.info("开始评估...")
        val_metrics = validate_epoch(trainer, val_loader, logger)
        logger.info(f"评估完成: {val_metrics}")
        return
    
    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # 训练
        train_losses = train_epoch(trainer, train_loader, epoch, logger)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录训练信息
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
                   f"Train Loss={train_losses['total']:.6f}, "
                   f"LR={current_lr:.8f}")
        
        # 验证
        if (epoch + 1) % config['evaluation']['eval_frequency'] == 0:
            val_metrics = validate_epoch(trainer, val_loader, logger)
            
            # 保存最佳模型
            if val_metrics['PSNR'] > best_psnr:
                best_psnr = val_metrics['PSNR']
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                    'val_metrics': val_metrics,
                    'config': config
                }, checkpoint_dir / "best_model.pth")
                logger.info(f"保存最佳模型，PSNR: {best_psnr:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config['evaluation']['save_frequency'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'config': config
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        # 核心修剪
        if config['training']['kernel_pruning']['enabled']:
            if (epoch + 1) in config['training']['kernel_pruning']['prune_epochs']:
                pruned_count = trainer.prune_kernels(
                    config['training']['kernel_pruning']['threshold']
                )
                logger.info(f"修剪了 {pruned_count} 个RBF核心")
                logger.info(f"当前RBF信息: {model.get_rbf_info()}")
    
    logger.info(f"训练完成！最佳PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()
