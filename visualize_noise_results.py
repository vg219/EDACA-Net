import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_noise_comparison_table():
    """创建噪声鲁棒性对比表格"""
    
    # 模拟的实验数据 - 您需要用实际实验结果替换
    results_data = {
        'Method': ['3DT-Net', 'DSPNet', 'BDT', 'MIMO-SST', 'EDACA-Net (Ours)'],
        'Clean': [41.25, 42.18, 40.87, 43.12, 44.92],
        'σ=5': [37.82, 38.45, 36.92, 39.28, 41.18],
        'σ=10': [34.15, 35.21, 33.67, 36.45, 38.47],
        'σ=15': [31.68, 32.89, 30.94, 33.82, 36.15],
        'σ=20': [29.45, 30.67, 28.52, 31.24, 34.23],
        'σ=25': [27.83, 28.91, 26.78, 29.15, 32.58]
    }
    
    # SAM结果
    sam_data = {
        'Method': ['3DT-Net', 'DSPNet', 'BDT', 'MIMO-SST', 'EDACA-Net (Ours)'],
        'Clean': [3.52, 3.28, 3.78, 3.15, 2.94],
        'σ=5': [4.89, 4.62, 5.24, 4.38, 3.97],
        'σ=10': [6.48, 6.15, 6.92, 5.74, 4.86],
        'σ=15': [7.95, 7.58, 8.47, 7.12, 5.72],
        'σ=20': [9.32, 8.89, 9.87, 8.45, 6.58],
        'σ=25': [10.58, 10.12, 11.15, 9.72, 7.41]
    }
    
    return pd.DataFrame(results_data), pd.DataFrame(sam_data)

def plot_noise_robustness_curves():
    """绘制噪声鲁棒性曲线"""
    
    # 数据
    noise_levels = [0, 5, 10, 15, 20, 25]
    methods = ['3DT-Net', 'DSPNet', 'BDT', 'MIMO-SST', 'EDACA-Net (Ours)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', '*']
    
    # PSNR数据
    psnr_data = {
        '3DT-Net': [41.25, 37.82, 34.15, 31.68, 29.45, 27.83],
        'DSPNet': [42.18, 38.45, 35.21, 32.89, 30.67, 28.91],
        'BDT': [40.87, 36.92, 33.67, 30.94, 28.52, 26.78],
        'MIMO-SST': [43.12, 39.28, 36.45, 33.82, 31.24, 29.15],
        'EDACA-Net (Ours)': [44.92, 41.18, 38.47, 36.15, 34.23, 32.58]
    }
    
    # SAM数据
    sam_data = {
        '3DT-Net': [3.52, 4.89, 6.48, 7.95, 9.32, 10.58],
        'DSPNet': [3.28, 4.62, 6.15, 7.58, 8.89, 10.12],
        'BDT': [3.78, 5.24, 6.92, 8.47, 9.87, 11.15],
        'MIMO-SST': [3.15, 4.38, 5.74, 7.12, 8.45, 9.72],
        'EDACA-Net (Ours)': [2.94, 3.97, 4.86, 5.72, 6.58, 7.41]
    }
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # PSNR曲线
    ax1 = axes[0, 0]
    for i, method in enumerate(methods):
        line_style = '--' if 'Ours' in method else '-'
        line_width = 3 if 'Ours' in method else 2
        ax1.plot(noise_levels, psnr_data[method], marker=markers[i], 
                linewidth=line_width, linestyle=line_style, 
                color=colors[i], label=method, markersize=8)
    
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(25, 46)
    
    # SAM曲线
    ax2 = axes[0, 1]
    for i, method in enumerate(methods):
        line_style = '--' if 'Ours' in method else '-'
        line_width = 3 if 'Ours' in method else 2
        ax2.plot(noise_levels, sam_data[method], marker=markers[i], 
                linewidth=line_width, linestyle=line_style,
                color=colors[i], label=method, markersize=8)
    
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('SAM (degrees)', fontsize=12)
    ax2.set_title('SAM vs Noise Level', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(2, 12)
    
    # 性能保持率
    ax3 = axes[1, 0]
    for i, method in enumerate(methods):
        clean_psnr = psnr_data[method][0]
        retention_rate = [p/clean_psnr * 100 for p in psnr_data[method]]
        line_style = '--' if 'Ours' in method else '-'
        line_width = 3 if 'Ours' in method else 2
        ax3.plot(noise_levels, retention_rate, marker=markers[i], 
                linewidth=line_width, linestyle=line_style,
                color=colors[i], label=method, markersize=8)
    
    ax3.set_xlabel('Noise Level (σ)', fontsize=12)
    ax3.set_ylabel('Performance Retention (%)', fontsize=12)
    ax3.set_title('PSNR Retention Rate', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(60, 102)
    
    # 性能降解柱状图
    ax4 = axes[1, 1]
    degradation = []
    method_names = []
    for method in methods:
        clean_val = psnr_data[method][0]
        noisy_val = psnr_data[method][-1]  # σ=25
        deg = (clean_val - noisy_val) / clean_val * 100
        degradation.append(deg)
        method_names.append(method.replace(' (Ours)', ''))
    
    bars = ax4.bar(method_names, degradation, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax4.set_title('PSNR Degradation at σ=25', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('noise_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_statistical_analysis():
    """创建统计分析结果"""
    
    print("=== Noise Robustness Statistical Analysis ===\n")
    
    # 性能降解数据
    degradation_data = {
        '3DT-Net': (41.25 - 27.83) / 41.25 * 100,     # 32.5%
        'DSPNet': (42.18 - 28.91) / 42.18 * 100,      # 31.5%
        'BDT': (40.87 - 26.78) / 40.87 * 100,         # 34.5%
        'MIMO-SST': (43.12 - 29.15) / 43.12 * 100,    # 32.4%
        'EDACA-Net (Ours)': (44.92 - 32.58) / 44.92 * 100  # 27.5%
    }
    
    print("**Performance Degradation Analysis (σ=0→25):**")
    for method, deg in degradation_data.items():
        print(f"- **{method}:** {deg:.1f}% degradation")
    
    print(f"\n**Noise Resistance Improvement:**")
    our_deg = degradation_data['EDACA-Net (Ours)']
    improvements = []
    for method, deg in degradation_data.items():
        if 'Ours' not in method:
            improvement = deg - our_deg
            improvements.append(improvement)
            print(f"- vs {method}: {improvement:.1f}% better retention")
    
    print(f"- **Average improvement: {np.mean(improvements):.1f}% better retention**")
    
    print(f"\n**At σ=15 (moderate noise):**")
    sigma15_retention = {
        '3DT-Net': 31.68/41.25*100,
        'DSPNet': 32.89/42.18*100,
        'BDT': 30.94/40.87*100,
        'MIMO-SST': 33.82/43.12*100,
        'EDACA-Net (Ours)': 36.15/44.92*100
    }
    
    for method, retention in sigma15_retention.items():
        print(f"- **{method}:** {retention:.1f}% retention")

if __name__ == "__main__":
    # 创建表格
    psnr_df, sam_df = create_noise_comparison_table()
    print("PSNR Results:")
    print(psnr_df.to_string(index=False))
    print("\nSAM Results:")
    print(sam_df.to_string(index=False))
    
    # 绘制图表
    plot_noise_robustness_curves()
    
    # 统计分析
    create_statistical_analysis()