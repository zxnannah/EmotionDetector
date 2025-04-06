import matplotlib.pyplot as plt
import numpy as np

def plot_training_losses(loss_history, save_path=None):
    """
    绘制多任务训练损失曲线
    
    参数：
        loss_history (dict): 包含各损失历史的字典
        save_path (str): 图片保存路径（可选）
    """
    # 创建画布和子图布局
    plt.figure(figsize=(12, 8))
    
    # 主坐标轴（左侧）
    ax1 = plt.gca()
    
    # 设置颜色和线型
    colors = {
        'mim': 'tab:blue',
        'recon': 'tab:green',
        'emotion': 'tab:red',
        'vad': 'tab:purple'
    }
    
    # 绘制所有损失曲线
    epochs = np.arange(1, len(loss_history['total'])+1)
    
    # MIM Loss（使用左侧坐标轴）
    ax1.plot(epochs, loss_history['mim'], 
            label='MIM Loss', 
            color=colors['mim'],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=5)
    
    # Reconstruction Loss
    # ax1.plot(epochs, loss_history['recon'], 
    #         label='Recon Loss', 
    #         color=colors['recon'],
    #         linestyle='--',
    #         linewidth=2,
    #         marker='s',
    #         markersize=5)
    
    # 创建右侧坐标轴
    ax2 = ax1.twinx()
    
    # Emotion Classification Loss
    ax2.plot(epochs, loss_history['emotion'], 
            label='Emotion Loss', 
            color=colors['emotion'],
            linestyle='-.',
            linewidth=2,
            marker='^',
            markersize=7)
    
    # VAD Regression Loss
    ax2.plot(epochs, loss_history['vad'], 
            label='VAD Loss', 
            color=colors['vad'],
            linestyle=':',
            linewidth=2,
            marker='d',
            markersize=6)
    
    # 设置坐标轴标签
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MIM/Recon Loss', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Emotion/VAD Loss', fontsize=12, fontweight='bold')
    
    # 设置刻度样式
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, 
             loc='upper center',
             bbox_to_anchor=(0.5, -0.15),
             ncol=4,
             fontsize=10)
    
    # 设置网格线
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    # 设置标题
    plt.title("Training Loss Curves", 
             fontsize=14, 
             fontweight='bold', 
             pad=20)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")
    else:
        plt.show()
        
    # 关闭当前图表
    plt.close()

# 使用示例
if __name__ == "__main__":
    # 模拟损失数据
    sample_history = {
        'total': [5.2, 4.1, 3.5, 3.0, 2.8],
        'mim': [1.8, 1.2, 0.9, 0.7, 0.6],
        'recon': [2.5, 1.9, 1.6, 1.3, 1.1],
        'emotion': [0.9, 0.7, 0.5, 0.4, 0.3],
        'vad': [0.8, 0.6, 0.5, 0.4, 0.35]
    }
    plot_training_losses(sample_history, save_path="loss_curves.png")