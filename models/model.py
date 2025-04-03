import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model
import numpy as np
import os

class SharedMLP(nn.Module):
    """共享的多模态投影层"""
    def __init__(self, input_dim, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class VariationalMIM(nn.Module):
    """变分互信息最大化模块"""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(2*feat_dim, 4*feat_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4*feat_dim, 1)
        )
    
    def forward(self, x, y):
        # 计算变分下界
        pos = self.T(torch.cat([x, y], dim=-1))
        perm = torch.randperm(y.size(0))
        neg = self.T(torch.cat([x, y[perm]], dim=-1))
        return -torch.mean(F.logsigmoid(pos - neg))

class SharedAutoencoder(nn.Module):
    """共享的自编码器结构"""
    def __init__(self):
        super().__init__()
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # 共享解码基座
        self.decoder_base = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        
        # 模态特定输出头
        self.audio_head = nn.Linear(512, 768)  # 音频特征重建
        self.text_head = nn.Linear(512, 1536)   # 文本特征重建
        self.motion_head = nn.Linear(512, 24)  # 头部动作特征重建
        
    def forward(self, x, modality):
        latent = self.encoder(x)
        shared = self.decoder_base(latent)
        
        if modality == 'audio':
            return self.audio_head(shared)
        elif modality == 'text':
            return self.text_head(shared)
        elif modality == 'motion':
            return self.motion_head(shared)
        else:
            raise ValueError("Unsupported modality")

class AudioEncoder(nn.Module):
    """音频特征编码器"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module):
    """文本特征编码器"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MotionEncoder(nn.Module):
    """动作特征编码器"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class EmotionClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=4):
        super(EmotionClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(input_dim=512)
        self.text_encoder = TextEncoder(input_dim=512)
        self.motion_encoder = MotionEncoder(input_dim=512)
        
        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 情感分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # VAD预测器
        self.vad_predictor = nn.Linear(hidden_dim, 3)  # 3个VAD值
        
    def forward(self, audio_features, text_features, motion_features):
        # 编码各个模态
        audio_encoded = self.audio_encoder(audio_features)
        text_encoded = self.text_encoder(text_features)
        motion_encoded = self.motion_encoder(motion_features)
        
        # 特征融合
        """此处只验证特征对齐效果，暂不修改融合算法"""
        combined = torch.cat([audio_encoded, text_encoded, motion_encoded], dim=1)
        fused = self.fusion(combined)
        
        emotion_logits = self.classifier(fused)
        vad_values = self.vad_predictor(fused)
        
        return emotion_logits, vad_values

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.sessions = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
        self.emotion_to_idx = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3} 
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        session_path = os.path.join(self.features_dir, session)
        
        # 加载特征
        audio_features = torch.FloatTensor(np.load(os.path.join(session_path, 'audio_features.npy')))
        text_features = torch.FloatTensor(np.load(os.path.join(session_path, 'text_features.npy')))
        motion_features = torch.FloatTensor(np.load(os.path.join(session_path, 'motion_features.npy')))
        emotion_labels = np.load(os.path.join(session_path, 'emotion_labels.npy'))
        vad_values = torch.FloatTensor(np.load(os.path.join(session_path, 'vad_values.npy')))
        
        # 将情绪标签转换为索引
        emotion_indices = torch.LongTensor([self.emotion_to_idx[emotion] for emotion in emotion_labels])
        
        return {
            'audio': audio_features,
            'text': text_features,
            'motion': motion_features,
            'emotion': emotion_indices,
            'vad': vad_values
        }

class EmotionPerceptionModel(nn.Module):
    """情绪感知多模态模型"""
    def __init__(self):
        super().__init__()
        
        # 共享组件
        self.audio_proj = SharedMLP(input_dim=768)  # 音频输入768维
        self.text_proj = SharedMLP(input_dim=1536)  # 文本输入1536维
        self.motion_proj = SharedMLP(input_dim=24)  # 动作输入24维

        self.mim_module = VariationalMIM()
        self.autoencoder = SharedAutoencoder()
        
        # 情绪分类头
        self.emotion_classifier = EmotionClassifier()
        
        # VAD回归头
        self.vad_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # VAD三个维度
        )
    
    def encode_audio(self, audio_input):
        return self.audio_proj(audio_input)
    
    def encode_text(self, text_input):
        return self.text_proj(text_input)
    
    def encode_motion(self, motion_input):
        return self.motion_proj(motion_input)
    
    def forward(self, batch):
        """完整前向传播
        Args:
            batch: 包含多模态输入的字典
                - 'audio': 音频特征
                - 'text': 文本特征
                - 'motion': 动作特征
                - 'emotion': 情绪标签
                - 'vad': VAD值
        Returns:
            包含所有输出和损失的字典
        """
        outputs = {}
        
        # 特征投影
        audio_feat = self.encode_audio(batch['audio'])
        text_feat = self.encode_text(batch['text'])
        motion_feat = self.encode_motion(batch['motion'])
        
        outputs['audio_features'] = audio_feat
        outputs['text_features'] = text_feat
        outputs['motion_features'] = motion_feat
            
        # 互信息计算
        mim_loss = 0
        mim_loss += self.mim_module(audio_feat, text_feat)
        mim_loss += self.mim_module(audio_feat, motion_feat)
        mim_loss += self.mim_module(text_feat, motion_feat)
        outputs['mim_loss'] = mim_loss / 3  # 平均跨模态损失
        
        # 自编码器重建
        recon_loss = 0
        recon_audio = self.autoencoder(audio_feat, 'audio')
        outputs['recon_audio'] = recon_audio
        recon_loss += F.mse_loss(recon_audio, batch['audio'])
            
        recon_text = self.autoencoder(text_feat, 'text')
        outputs['recon_text'] = recon_text
        recon_loss += F.mse_loss(recon_text, batch['text'])
            
        recon_motion = self.autoencoder(motion_feat, 'motion')
        outputs['recon_motion'] = recon_motion
        recon_loss += F.mse_loss(recon_motion, batch['motion'])
            
        outputs['recon_loss'] = recon_loss
        
        # 情绪分类和VAD回归
        # 融合特征
        fused_feature = (audio_feat + text_feat + motion_feat) / 3
        
        # 情绪分类
        emotion_logits, vad_values = self.emotion_classifier(fused_feature, fused_feature, fused_feature)
        outputs['emotion_logits'] = emotion_logits
        if 'emotion' in batch:
            emotion_loss = F.cross_entropy(emotion_logits, batch['emotion'])
            outputs['emotion_loss'] = emotion_loss
        
        # VAD回归
        vad_pred = self.vad_regressor(fused_feature)
        outputs['vad_pred'] = vad_pred
        if 'vad' in batch:
            vad_loss = F.mse_loss(vad_pred, batch['vad'])
            outputs['vad_loss'] = vad_loss
        
        # 总损失
        total_loss = (
            0.3 * outputs['mim_loss'] +
            0.3 * outputs['recon_loss'] +
            0.2 * outputs.get('emotion_loss', 0) +
            0.2 * outputs.get('vad_loss', 0)
        )
        outputs['total_loss'] = total_loss
        
        return outputs

if __name__ == '__main__':
    model = EmotionPerceptionModel()
    
    # 模拟输入
    batch = {
        'audio': torch.randn(32, 768),
        'text': torch.randn(32, 1536),  # 匹配修改后的输入
        'motion': torch.randn(32, 24)
    }

    outputs = model(batch)
    print(f"Total loss: {outputs['total_loss']:.4f}")
    print(f"Reconstructed audio shape: {outputs['recon_audio'].shape}")
    print(f"Reconstructed text logits shape: {outputs['recon_text'].shape}")