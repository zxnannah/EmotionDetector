import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model
import numpy as np
import os


class SharedMLP(nn.Module):
    """处理三维时序输入的共享投影层"""
    def __init__(self, input_dim, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim)  # 归一化特征维度
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x) 
        x = x.permute(0, 2, 1)
        x = self.gelu(x)
        x = self.ln(x)  
        x = x.permute(0, 2, 1) 
        x = self.conv2(x) 
        x = x.permute(0, 2, 1)
        return x


class VariationalMIM(nn.Module):
    """变分互信息最大化模块"""

    def __init__(self, feat_dim=512):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(2 * feat_dim, 4 * feat_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * feat_dim, 1)
        )

    def forward(self, x, y):
        # 计算变分下界
        batch_size, seq_len, _ = x.shape
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
        self.text_head = nn.Linear(512, 1536)  # 文本特征重建
        self.motion_head = nn.Linear(512, 6)  # 头部动作特征重建

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
        encoded = self.encoder(x) 
        return encoded


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
        encoded = self.encoder(x) 
        return encoded


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
        encoded = self.encoder(x) 
        return encoded


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
        combined = torch.cat([audio_encoded, text_encoded, motion_encoded], dim=2)
        fused = self.fusion(combined)

        fused_pooled = fused.mean(dim=1)  # 全局池化        

        emotion_logits = self.classifier(fused_pooled)
        vad_values = self.vad_predictor(fused_pooled)

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
    
class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        features1: (N, L, D)
        features2: (N, L, D)
        return: scalar loss
        """

        features1 = F.normalize(features1, dim=1).reshape(features1.size(0), -1)  # (N, D)
        features2 = F.normalize(features2, dim=1).reshape(features2.size(0), -1)  # (N, D)

        # print(features1.shape, features2.shape)
        # print("*"*10)
        logits_per1 = features1 @ features2.T  # (N, N)
        logits_per2 = features2 @ features1.T   # (N, N)

        # 对角矩阵
        batch_size = features1.size(0)
        labels = torch.arange(batch_size, device=features1.device)

        logits_per1 /= self.temperature
        logits_per1 /= self.temperature

        # Cross-entropy loss
        loss1 = F.cross_entropy(logits_per1, labels)
        loss2 = F.cross_entropy(logits_per2, labels)

        return (loss1 + loss2) / 2


class EmotionPerceptionModel(nn.Module):
    """情绪感知多模态模型"""

    def __init__(self):
        super().__init__()

        # 共享组件
        self.audio_proj = SharedMLP(input_dim=768)  # 音频输入768维
        self.text_proj = SharedMLP(input_dim=1536)  # 文本输入1536维
        self.motion_proj = SharedMLP(input_dim=6)  # 动作输入6维

        self.mim_module = VariationalMIM()
        self.autoencoder = SharedAutoencoder()

        self.infoNCE = CLIPLoss()

        # 改进的多模态融合 - 添加注意力机制
        self.modal_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

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

    def attention_fusion(self, features_list):
        """
        注意力融合机制

        Args:
            features_list (list): 特征列表 [audio_feat, text_feat, motion_feat]

        Returns:
            torch.Tensor: 注意力加权融合的特征
        """
        # 堆叠特征以计算注意力权重
        stacked_features = torch.stack(features_list, dim=1)  # [batch_size, 3, feat_dim]

        # 计算每个模态的注意力权重
        attention_weights = []
        for i in range(stacked_features.size(1)):
            feat = stacked_features[:, i, :]
            weight = self.modal_attention(feat)
            attention_weights.append(weight)

        # 将权重拼接
        attention_weights = torch.cat(attention_weights, dim=1)  # [batch_size, 3]

        # Softmax归一化
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(-1)  # [batch_size, 3, 1]

        # 加权求和
        weighted_sum = torch.sum(stacked_features * attention_weights, dim=1)  # [batch_size, feat_dim]

        return weighted_sum


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
        infoNCE_loss = 0
        infoNCE_loss += self.infoNCE(audio_feat, text_feat)
        infoNCE_loss += self.infoNCE(audio_feat, motion_feat)
        infoNCE_loss += self.infoNCE(text_feat, motion_feat)
        outputs['infoNCE_loss'] = infoNCE_loss / 3  # 平均跨模态损失

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
                # 系数待定
                0.2 * outputs['mim_loss'] +
                0.2 * outputs['infoNCE_loss'] +
                0.2 * outputs['recon_loss'] +
                0.2 * outputs.get('emotion_loss', 0) +
                0.2 * outputs.get('vad_loss', 0)
        )
        outputs['total_loss'] = total_loss

        return outputs


if __name__ == '__main__':
    # 设置种子，这样大家的测试数据比较统一
    torch.manual_seed(11)

    model = EmotionPerceptionModel()

    # 模拟输入
    batch = {
        'audio': torch.randn(32,1, 768),
        'text': torch.randn(32,1, 1536),  # 匹配修改后的输入
        'motion': torch.randn(32,1, 6)
    }
    outputs = model(batch)
    print(f"Total loss: {outputs['total_loss']:.4f}")
    print(f"Reconstructed audio shape: {outputs['recon_audio'].shape}")
    print(f"Reconstructed text logits shape: {outputs['recon_text'].shape}")
# if __name__ == '__main__':
#     model = EmotionPerceptionModel()
    
#     # 模拟输入
#     batch = {
#         'audio': torch.randn(32, 768),
#         'text': torch.randn(32, 1536),  # 匹配修改后的输入
#         'motion': torch.randn(32, 24)
#     }

#     outputs = model(batch)
#     print(f"Total loss: {outputs['total_loss']:.4f}")
#     print(f"Reconstructed audio shape: {outputs['recon_audio'].shape}")
#     print(f"Reconstructed text logits shape: {outputs['recon_text'].shape}")