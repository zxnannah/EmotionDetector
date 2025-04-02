import torch
import numpy as np
from models.model import EmotionPerceptionModel
import os
from tqdm import tqdm

def test_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = EmotionPerceptionModel().to(device)
    model.eval()  # 设置为评估模式
    
    # 特征文件路径
    features_dir = "data/Feature/features/Session1"
    print(f"特征目录: {features_dir}")
    
    # 情绪标签到索引的映射
    emotion_to_idx = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    
    # 加载特征
    print("正在加载特征...")
    audio_features = torch.FloatTensor(np.load(os.path.join(features_dir, 'audio_features.npy'))).to(device)
    text_features = torch.FloatTensor(np.load(os.path.join(features_dir, 'text_features.npy'))).to(device)
    motion_features = torch.FloatTensor(np.load(os.path.join(features_dir, 'motion_features.npy'))).to(device)
    
    # 加载情绪标签并转换为索引
    emotion_labels_str = np.load(os.path.join(features_dir, 'emotion_labels.npy'))
    emotion_labels = torch.LongTensor([emotion_to_idx[emotion] for emotion in emotion_labels_str]).to(device)
    
    vad_values = torch.FloatTensor(np.load(os.path.join(features_dir, 'vad_values.npy'))).to(device)
    
    # 准备测试数据
    batch_size = 32
    num_samples = len(emotion_labels)
    print(f"总样本数: {num_samples}")
    
    # 测试循环
    total_loss = 0
    correct = 0
    total = 0
    
    print("\n开始测试...")
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="测试进度"):
            # 准备批次数据
            end_idx = min(i + batch_size, num_samples)
            batch = {
                'audio': audio_features[i:end_idx],
                'text': text_features[i:end_idx],
                'motion': motion_features[i:end_idx],
                'emotion': emotion_labels[i:end_idx],
                'vad': vad_values[i:end_idx]
            }
            
            # 前向传播
            outputs = model(batch)
            
            # 计算准确率
            _, predicted = torch.max(outputs['emotion_logits'], 1)
            total += emotion_labels[i:end_idx].size(0)
            correct += (predicted == emotion_labels[i:end_idx]).sum().item()
            
            # 累加损失
            total_loss += outputs['total_loss'].item()
    
    # 打印结果
    avg_loss = total_loss / (num_samples / batch_size)
    accuracy = 100 * correct / total
    
    print("\n测试结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.2f}%")
    
    # 打印各个模态的特征维度
    print("\n特征维度:")
    print(f"音频特征: {audio_features.shape}")
    print(f"文本特征: {text_features.shape}")
    print(f"动作特征: {motion_features.shape}")
    
    # 打印预测示例
    print("\n预测示例:")
    sample_idx = 0
    idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
    print(f"真实情绪: {idx_to_emotion[emotion_labels[sample_idx].item()]}")
    print(f"预测情绪: {idx_to_emotion[predicted[sample_idx].item()]}")
    print(f"真实VAD: {vad_values[sample_idx].cpu().numpy()}")
    print(f"预测VAD: {outputs['vad_pred'][sample_idx].cpu().numpy()}")

if __name__ == "__main__":
    test_model() 