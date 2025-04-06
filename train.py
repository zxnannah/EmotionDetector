import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import EmotionPerceptionModel
from src.dataset import EmotionDataset
from src.utils import plot_training_losses

def train_model(features_dir, num_epochs=10, batch_size=32, learning_rate=1e-4):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集和数据加载器
    dataset = EmotionDataset(features_dir)
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                        #   num_workers=4,
                          pin_memory=True)

    # 初始化模型
    model = EmotionPerceptionModel().to(device)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 损失记录字典
    loss_history = {
        'total': [],
        'mim': [],
        'recon': [],
        'emotion': [],
        'vad': []
    }

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in loss_history}
        
        for batch_idx, batch in enumerate(dataloader):
            # 数据移动到设备
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            motion = batch['motion'].to(device)
            emotion_labels = batch['emotion'].to(device)
            vad_values = batch['vad'].to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model({
                'audio': audio,
                'text': text,
                'motion': motion,
                'emotion': emotion_labels,
                'vad': vad_values
            })

            # 计算总损失
            total_loss = outputs['total_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 参数更新
            optimizer.step()

            # 记录损失
            epoch_losses['total'] += total_loss.item()
            epoch_losses['mim'] += outputs['mim_loss'].item()
            epoch_losses['recon'] += outputs['recon_loss'].item()
            epoch_losses['emotion'] += outputs.get('emotion_loss', 0).item()
            # epoch_losses['vad'] += outputs.get('vad_loss', 0).item()

            # 每10个batch打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                      f'Total Loss: {total_loss.item():.4f}')

        # 计算epoch平均损失
        for k in epoch_losses:
            loss_history[k].append(epoch_losses[k]/len(dataloader))
        
        # 打印epoch总结
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Total Loss: {loss_history["total"][-1]:.4f}')
        print(f'MIM Loss: {loss_history["mim"][-1]:.4f}')
        print(f'Recon Loss: {loss_history["recon"][-1]:.4f}')
        print(f'Emotion Loss: {loss_history["emotion"][-1]:.4f}')
        print(f'VAD Loss: {loss_history["vad"][-1]:.4f}\n')

    return model, loss_history

# 使用示例
if __name__ == "__main__":
    # 设置训练参数
    FEATURES_DIR = "data/Feature/features" 
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LR = 1e-4

    # 开始训练
    trained_model, losses = train_model(
        features_dir=FEATURES_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR
    )

    # 绘制损失曲线
    plot_training_losses(losses)

    # 保存模型
    torch.save(trained_model.state_dict(), "output/emotion_model.pth")