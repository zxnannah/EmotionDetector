import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端，避免中文字体问题
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import seaborn as sns
from sklearn.manifold import TSNE


class MPCTransformer(nn.Module):
    """
    模型预测控制Transformer
    用于预测未来状态序列
    """
    # def __init__(self, feat_dim=256, nhead=8, num_layers=3):
    #     super().__init__()
    #     self.transformer = nn.TransformerEncoder(
    #         nn.TransformerEncoderLayer(feat_dim, nhead),
    #         num_layers
    #     )
    #     self.proj = nn.Linear(feat_dim, feat_dim)
    #
    # def forward(self, history):
    #     return self.proj(self.transformer(history)[-1])

    #优化状态转移方程，利用时序信息（条件变分自编码器）
    # latent_dim: 潜在空间的维度，用于VAE的编码部分。
    def __init__(self, feat_dim=256, nhead=8, num_layers=3, latent_dim=64):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(feat_dim, nhead),
            num_layers
        )

        # 编码器: 历史状态到潜在空间
        # 将Transformer的输出隐状态映射到潜在空间的均值和对数方差（用于VAE重参数化）
        self.encoder_mean = nn.Linear(feat_dim, latent_dim)
        self.encoder_logvar = nn.Linear(feat_dim, latent_dim)

        # 解码器: 潜在空间到下一状态
        # 将潜在空间中的向量（latent_dim）映射回原始特征空间（feat_dim）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, feat_dim)
        )

        # 动力学模型: 预测潜在空间中的转移 --做一个非线性的状态转移函数
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(), #非线性变换
            nn.Linear(latent_dim, latent_dim)
        )

        # 增加一个残差连接，防止梯度消失
        self.residual_proj = nn.Linear(feat_dim, feat_dim)

        # 不确定性估计
        self.uncertainty_estimator = nn.Linear(feat_dim, 1)

    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, history, sample_latent=True):
        """
        增强的前向传播，返回预测状态和不确定性估计

        Args:
            history: 历史状态序列 [seq_len, batch, feat_dim]
            sample_latent: 是否从潜在分布中采样

        Returns:
            next_state: 预测的下一状态 [batch, feat_dim]
            uncertainty: 预测的不确定性 [batch, 1]
        """
        # Transformer编码
        trans_out = self.transformer(history)
        last_hidden = trans_out[-1]  # 获取最后一个时间步的隐状态

        # 变分编码
        mu = self.encoder_mean(last_hidden)
        logvar = self.encoder_logvar(last_hidden)

        # 通过潜在空间
        # 如果 sample_latent 为真，则通过重参数化技巧从高斯分布中采样潜在变量 z；否则，使用均值 μ 作为潜在变量
        if sample_latent:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # 应用动力学模型
        z_next = self.dynamics(z)

        # 解码回原始特征特征空间
        decoded = self.decoder(z_next)

        # 残差连接
        residual = self.residual_proj(last_hidden)
        next_state = decoded + residual

        # 估计不确定性
        uncertainty = torch.sigmoid(self.uncertainty_estimator(next_state))

        return next_state, uncertainty, mu, logvar

    def kl_loss(self, mu, logvar):
        """计算KL散度损失
           用于训练CVAE
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class FutureEncoder(nn.Module):
    """编码未来状态序列"""
    def __init__(self, feat_dim):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, feat_dim, batch_first=True)

    def forward(self, x):  # x: [B, T, D]
        _, h_n = self.rnn(x)  # h_n: [1, B, D]
        return h_n.squeeze(0)  # [B, D]



class ModalityProjector(nn.Module):
    """
    特征投影器：将不同模态的特征投影到统一的特征空间
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projector(x)

# 继承控制的想法，优化注意力机制
# 情绪状态通常具有时间惯性，不会在短时间内发生剧烈变化。利用这一特性，我们可以设计一个时序一致性约束机制，使得注意力机制能够考虑情绪的时间连续性，从而减少噪声影响。
class TemporalConsistencyAttention(nn.Module):
    """
    带有时序一致性约束的跨模态通道注意力机制
    """

    def __init__(self, feat_dim, num_modes=3, history_len=5, smooth_factor=0.8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_modes = num_modes
        self.history_len = history_len
        self.smooth_factor = smooth_factor  # 时序平滑因子

        # 保持原有的模态特征投影器
        self.projectors = nn.ModuleList([
            ModalityProjector(feat_dim, feat_dim)
            for _ in range(num_modes)
        ])

        # 可学习的缩放参数gamma，控制注意力强度
        self.gamma = nn.Parameter(torch.zeros(num_modes, num_modes))

        # 模态间注意力偏置
        self.modal_bias = nn.Parameter(torch.zeros(num_modes, num_modes))

        # 通道重要性学习
        self.channel_importance = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, feat_dim),
                nn.Sigmoid()
            ) for _ in range(num_modes)
        ])

        # 情绪状态连续性编码器
        self.emotion_continuity_encoder = nn.ModuleList([
            nn.GRU(feat_dim, feat_dim, batch_first=True)
            for _ in range(num_modes)
        ])

        # 情绪状态约束参数（定义可能的情绪变化范围）
        self.emotion_constraint = nn.Parameter(torch.ones(num_modes) * 0.2)  # 初始化为0.2的约束范围

        # 历史注意力权重存储
        self.attention_history = {i: deque(maxlen=history_len) for i in range(num_modes)}
        for i in range(num_modes):
            for _ in range(history_len):
                self.attention_history[i].append(torch.zeros(feat_dim, feat_dim))

    def compute_temporal_constrained_attention(self, query, key, modal_i, modal_j):
        """计算带有时序约束的注意力权重"""
        # 基础通道注意力计算
        query = query.unsqueeze(2)  # [B, C, 1]
        key = key.unsqueeze(1)  # [B, 1, C]
        current_attn = torch.matmul(query, key)  # [B, C, C]
        current_attn = current_attn / (self.feat_dim ** 0.5)

        # 提取历史注意力
        if len(self.attention_history[modal_i]) > 0:
            # 计算历史注意力的平均值
            hist_attn = torch.stack(list(self.attention_history[modal_i])).mean(0)

            # 约束当前注意力不要偏离历史太多
            constraint_value = self.emotion_constraint[modal_i]

            # 应用约束：新注意力 = 历史注意力 + 有限范围的变化
            constrained_attn = hist_attn + torch.clamp(
                current_attn - hist_attn,
                min=-constraint_value,
                max=constraint_value
            )

            # 平滑过渡
            final_attn = self.smooth_factor * constrained_attn + (1 - self.smooth_factor) * current_attn
        else:
            final_attn = current_attn

        # 更新历史
        self.attention_history[modal_i].append(final_attn.detach().mean(0))  # 存储batch平均值

        # 应用softmax归一化
        return F.softmax(final_attn, dim=-1)

    def process_modal_sequence(self, feature_sequence, modal_idx):
        """处理模态的时序特征"""
        # feature_sequence: [B, T, D]
        output, _ = self.emotion_continuity_encoder[modal_idx](feature_sequence)
        # 返回最后一个时间步的特征，但保留时序编码信息
        return output[:, -1, :]

    def forward(self, features_list, feature_sequences=None):
        """
        多模态特征融合，带有时序约束

        Args:
            features_list: 当前时刻的模态特征列表 [text_feat, audio_feat, video_feat]
                          每个特征的形状为 [B, D]
            feature_sequences: 可选，历史序列特征 [[B,T,D], [B,T,D], [B,T,D]]

        Returns:
            融合后的特征 [B, D]
            注意力权重 [B, num_modes]
        """
        batch_size = features_list[0].shape[0]

        # 如果提供了历史序列，则进行时序处理
        if feature_sequences is not None:
            processed_features = []
            for i, seq in enumerate(feature_sequences):
                if seq is not None and len(seq.shape) == 3:  # [B, T, D]
                    # 处理时序特征
                    processed_feat = self.process_modal_sequence(seq, i)
                    # 将时序编码的特征与当前特征结合
                    features_list[i] = features_list[i] + 0.3 * processed_feat  # 0.3是时序信息的权重
                processed_features.append(features_list[i])
        else:
            processed_features = features_list

        # 投影所有模态特征到同一空间
        projected_features = [
            proj(feat) for feat, proj in zip(processed_features, self.projectors)
        ]

        # 计算通道重要性权重
        channel_weights = [
            imp(feat) for feat, imp in zip(projected_features, self.channel_importance)
        ]

        # 应用通道重要性
        weighted_features = [
            feat * weight for feat, weight in zip(projected_features, channel_weights)
        ]

        # 存储所有模态间的交互结果
        interaction_features = []
        attention_scores = []

        # 计算模态间的交互特征
        for i in range(self.num_modes):
            modal_interactions = []
            modal_attns = []

            for j in range(self.num_modes):
                if i != j:  # 不同模态间的交互
                    # 计算带有时序约束的通道注意力
                    attn = self.compute_temporal_constrained_attention(
                        weighted_features[i],  # query
                        weighted_features[j],  # key
                        i, j  # 模态索引
                    )  # [B, C, C]

                    # 添加模态偏置
                    attn = attn + self.modal_bias[i, j]

                    # 调整注意力强度
                    gamma = torch.sigmoid(self.gamma[i, j])

                    # 特征交互: 应用注意力进行加权
                    value = weighted_features[j].unsqueeze(2)  # [B, C, 1]
                    attended_feat = torch.bmm(attn, value)  # [B, C, 1]
                    attended_feat = attended_feat.squeeze(-1)  # [B, C]

                    # 残差连接
                    interaction = gamma * attended_feat + weighted_features[i]

                    modal_interactions.append(interaction)
                    modal_attns.append(gamma.item())

            if modal_interactions:
                # 汇总该模态与其他模态的交互
                interaction_feature = torch.stack(modal_interactions).mean(0)
                interaction_features.append(interaction_feature)
                attention_scores.append(sum(modal_attns) / len(modal_attns))

        # 融合所有交互特征
        if interaction_features:
            fused_features = torch.stack(interaction_features).mean(0)
            attention_weights = F.softmax(torch.tensor(attention_scores), dim=0)
        else:
            # 如果没有交互特征（例如只有一个模态），直接返回原特征
            fused_features = projected_features[0]
            attention_weights = torch.ones(1) / 1

        return fused_features, attention_weights


class RLPolicy(nn.Module):
    """
      学习策略网络，通道注意力机制
    """
    def __init__(self, feat_dim, history_len=5):
        super().__init__()
        self.future_encoder = FutureEncoder(feat_dim)

        # 通道注意力机制
        self.cross_modal_attn = TemporalConsistencyAttention(
            feat_dim,
            num_modes=3,
            history_len=history_len,
            smooth_factor=0.8
        )

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        # 存储历史状态序列
        self.history_states = {
            'text': deque(maxlen=history_len),
            'audio': deque(maxlen=history_len),
            'video': deque(maxlen=history_len)
        }

    def update_history(self, current_states):
        """更新历史状态"""
        for i, modality in enumerate(['text', 'audio', 'video']):
                self.history_states[modality].append(current_states[i])

        # self.attn_param_net = nn.Sequential(
        #     nn.Linear(feat_dim * 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3 * feat_dim)
        # )

        # self.multihead_attn = nn.MultiheadAttention(feat_dim, 8)

        # self.net = nn.Sequential(
        #     nn.Linear(feat_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )

    def forward(self, current_states, future_states):
        """
        前向传播

        Args:
            current_states: [3, B, D] -- text/audio/video
            future_states: [3, B, T, D]或 [3, B, D] 取决于输入

        Returns:
            weights: 模态融合权重 [B, 3]
            attn_patterns: 注意力模式
        """
        batch_size = current_states.shape[1]

        # 更新历史状态
        self.update_history(current_states)

        # 编码未来状态
        future_encoded = []
        for i in range(3):
            if len(future_states[i].shape) == 3:  # [B, T, D]
                future = self.future_encoder(future_states[i])  # [B, D]
            else:  # [B, D]
                future = future_states[i]  # 已经是编码后的形式
            future_encoded.append(future)

        # 结合当前状态和未来预测
        combined_features = []
        for i in range(3):
            # 按通道连接当前和未来
            combined = current_states[i] + 0.5 * future_encoded[i]  # [B, D]
            combined_features.append(combined)

            # 构建历史序列特征
            history_sequences = []
            for i, modality in enumerate(['text', 'audio', 'video']):
                if len(self.history_states[modality]) > 0:
                    # 将历史状态转换为张量
                    seq = torch.stack(list(self.history_states[modality])).unsqueeze(0)
                    # 复制到batch size
                    seq = seq.repeat(batch_size, 1, 1)  # [B, T, D]
                    history_sequences.append(seq)
                else:
                    history_sequences.append(None)


        # 使用跨模态通道注意力融合特征
        fused_features, attention_patterns = self.cross_modal_attn(
            combined_features,
            history_sequences
        )

        # 计算价值估计
        state_value = self.value_net(fused_features)  # [B, 1]

        # 生成模态融合策略
        modal_weights = self.policy_net(fused_features)  # [B, 3]
        weights = F.softmax(modal_weights, dim=-1)

        return weights, attention_patterns

class MultimodalSystem:
    def __init__(self, feat_dim=256, pred_horizon=5, hist_len=3, num_classes=10, gamma=0.99, lambda_penalty=0.1):
        self.mpc = MPCTransformer(feat_dim)
        self.policy = RLPolicy(feat_dim, history_len=hist_len)
        self.task_model = nn.Linear(feat_dim, num_classes)

        # 情绪平滑系数
        self.emotion_smoothing = 0.8

        # 存储上一时刻的情绪预测结果
        self.prev_emotion = None

        self.history = {
            'text': deque(maxlen=hist_len),
            'audio': deque(maxlen=hist_len),
            'video': deque(maxlen=hist_len)
        }

        # 存储情绪变化率约束
        self.emotion_change_constraint = 0.2


        self.pred_horizon = pred_horizon
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty

        self.optim = torch.optim.Adam([ 
            {'params': self.mpc.parameters()},
            {'params': self.policy.parameters()},
            {'params': self.task_model.parameters()}
        ], lr=1e-4)
        
    def _mpc_predict(self, modality):
        pred_states = []
        # 考虑不确定性和KL损失
        uncertainties = []
        kl_losses = []

        hist = torch.stack(list(self.history[modality]))
        hist = hist.unsqueeze(1)

        for _ in range(self.pred_horizon):
            next_state, uncertainty, mu, logvar = self.mpc(hist)
            pred_states.append(next_state)
            uncertainties.append(uncertainty)
            kl_losses.append(self.mpc.kl_loss(mu, logvar))
            hist = torch.cat([hist[1:], next_state.unsqueeze(0)], dim=0)


        return {
            'states': torch.stack(pred_states),         # [horizon, batch, feat_dim]
            'uncertainties': torch.stack(uncertainties),  # [horizon, batch, 1]
            'kl_loss': sum(kl_losses) / len(kl_losses)    # 平均KL损失
        }

    def _calc_reward(self, acc, consistency, weight_change,uncertainty=None):
        # 计算奖励函数：包括分类准确率、模态一致性、权重波动
        # 考虑预测不确定性
        if uncertainty is not None:
            # 降低高不确定性预测的奖励
            certainty_factor = 1.0 - self.uncertainty_weight * uncertainty
            # 综合奖励
            return (acc + self.gamma * consistency) * certainty_factor - self.lambda_penalty * weight_change
        else:
            return acc + self.gamma * consistency - self.lambda_penalty * weight_change

    def process(self, feats_dict, target=None, explore_prob=0.1):
        for k in self.history:
            self.history[k].append(feats_dict[k])

            # 同时更新策略网络的历史状态
            if hasattr(self.policy, 'history_states'):
                self.policy.history_states[k].append(feats_dict[k])

        if len(self.history['text']) < self.history['text'].maxlen:
            return None

            # MPC预测
        future_preds = [self._mpc_predict(k) for k in ['text', 'audio', 'video']]
        current_states = [self.history[k][-1] for k in ['text', 'audio', 'video']]

        # 汇总不确定性用于奖励计算
        mean_uncertainty = torch.mean(torch.cat([
            future_preds[k]['uncertainties'].mean() for k in ['text', 'audio', 'video']
        ]))

        # RL策略生成权重
        if torch.rand(1) < explore_prob:
            weights = F.softmax(torch.rand(3), dim=0)
            attn_patterns = None
        else:
            weights, attn_patterns = self.policy(
                torch.stack(current_states),
                torch.stack([future_preds[k]['states'].mean(0) for k in ['text', 'audio', 'video']])
            )

        # 应用情绪状态平滑约束
        if self.prev_emotion is not None:
            # 限制权重变化幅度
            weight_change = weights - self.prev_emotion
            constrained_change = torch.clamp(
                weight_change,
                min=-self.emotion_change_constraint,
                max=self.emotion_change_constraint
            )
            weights = self.prev_emotion + constrained_change

            # 确保权重和为1
            weights = F.softmax(weights, dim=-1)

        # 更新历史情绪状态
        self.prev_emotion = weights.clone().detach()

        # 模态融合
        fused = sum(w * feats_dict[k] for w, k in zip(weights, ['text', 'audio', 'video']))

        # 分类和训练
        logits = self.task_model(fused)

        if target is not None:
            # 计算准确率
            acc = (logits.argmax(-1) == target).float()
            # 计算模态一致性，考虑不确定性加权
            consistency = (
                                  torch.cosine_similarity(feats_dict['text'], feats_dict['audio']) +
                                  torch.cosine_similarity(feats_dict['audio'], feats_dict['video']) +
                                  torch.cosine_similarity(feats_dict['video'], feats_dict['text'])
                          ) / 3.0

            weight_change = torch.norm(weights - weights.detach(), p=2).item()

            reward = self._calc_reward(acc, consistency, weight_change, mean_uncertainty)
            # VAE损失(KL散度)
            vae_loss = sum(future_preds[k]['kl_loss'] for k in ['text', 'audio', 'video'])

            # 总损失
            loss = -reward + F.cross_entropy(logits, target) + 0.01 * vae_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return {
            'prediction': logits.argmax(dim=-1).item(),
            'weights': weights.detach().cpu().numpy(),
            'uncertainty': mean_uncertainty.item(),
            'attention_patterns': attn_patterns.detach().cpu().numpy() if attn_patterns is not None else None
        }
# ————————————————————————测试代码——————————————————————————————
# 存在张量维度问题，到时候根据具体的输入特征再修改
# def test_temporal_consistency_attention():
#     """测试时序一致性注意力机制"""
#     print("\n测试时序一致性注意力机制...")
#
#     # 设置随机种子
#     torch.manual_seed(42)
#     np.random.seed(42)
#
#     # 创建随机特征和时序特征
#     batch_size = 2
#     feat_dim = 64
#     seq_len = 5
#
#     # 当前特征
#     features = [
#         torch.rand(batch_size, feat_dim) for _ in range(3)  # 文本、音频、视频模态
#     ]
#
#     # 历史序列特征 (模拟情绪变化过程)
#     history_sequences = []
#     for i in range(3):
#         # 创建一个基础情绪状态
#         base_emotion = torch.rand(1, feat_dim)
#         # 添加小的随机变化，模拟情绪的微小变化
#         noise_scale = 0.05 if i == 0 else 0.2  # 文本模态更稳定
#         sequence = torch.stack([
#             base_emotion + noise_scale * torch.randn(1, feat_dim) * (t / seq_len)
#             for t in range(seq_len)
#         ]).squeeze(1)
#         # 复制到batch size
#         sequence = sequence.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, T, D]
#         history_sequences.append(sequence)
#
#     # 初始化注意力模块
#     attention = TemporalConsistencyAttention(feat_dim, num_modes=3, history_len=5, smooth_factor=0.8)
#
#     # 多次前向传播，模拟连续处理
#     all_weights = []
#     all_attentions = []
#
#     for epoch in range(10):  # 模拟10个时间步
#         # 添加随机噪声，模拟传感器噪声
#         noisy_features = [
#             f + 0.2 * torch.randn_like(f) for f in features
#         ]
#
#         # 前向传播
#         fused_features, attention_weights = attention(noisy_features, history_sequences)
#
#         # 保存权重
#         all_weights.append(attention_weights.detach().numpy())
#
#         # 更新历史序列（在实际应用中，这会是新的观测值）
#         for i in range(3):
#             # 移除最早的时间步，添加新的时间步
#             history_sequences[i] = torch.cat([
#                 history_sequences[i][:, 1:, :],
#                 noisy_features[i].unsqueeze(1)
#             ], dim=1)
#
#     # 打印结果
#     print(f"融合特征形状: {fused_features.shape}")
#
#     # 绘制模态融合权重随时间的变化
#     plt.figure(figsize=(10, 6))
#     all_weights = np.array(all_weights)
#     for i, modality in enumerate(['文本', '音频', '视频']):
#         plt.plot(all_weights[:, i], label=modality, marker='o')
#     plt.title('模态融合权重随时间的变化')
#     plt.xlabel('时间步')
#     plt.ylabel('融合权重')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('temporal_fusion_weights.png')
#     plt.close()
#
#     # 可视化模态间的gamma参数
#     gamma_values = torch.sigmoid(attention.gamma).detach().numpy()
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(gamma_values, annot=True, cmap='coolwarm',
#                 xticklabels=['文本', '音频', '视频'],
#                 yticklabels=['文本', '音频', '视频'])
#     plt.title('模态间注意力强度 (Gamma)')
#     plt.tight_layout()
#     plt.savefig('modal_attention_gamma_temporal.png')
#     plt.close()
#
#     # 可视化情绪约束参数
#     plt.figure(figsize=(8, 4))
#     constraint_values = attention.emotion_constraint.detach().numpy()
#     plt.bar(['文本', '音频', '视频'], constraint_values)
#     plt.title('情绪变化约束参数')
#     plt.ylabel('约束大小')
#     plt.grid(axis='y')
#     plt.tight_layout()
#     plt.savefig('emotion_constraint_values.png')
#     plt.close()
#
#     print("时序一致性注意力机制测试完成，可视化结果已保存")
#     return attention
#
#
# def test_mpc_with_uncertainty():
#     """测试带不确定性估计的MPC模型"""
#     print("\n测试带不确定性估计的MPC模型...")
#
#     # 创建随机序列
#     seq_len = 3
#     batch_size = 1
#     feat_dim = 64
#     history = torch.rand(seq_len, batch_size, feat_dim)
#
#     # 初始化MPC模型
#     mpc = MPCTransformer(feat_dim, latent_dim=16)
#
#     # 前向传播
#     next_state, uncertainty, mu, logvar = mpc(history)
#
#     print(f"下一状态形状: {next_state.shape}")
#     print(f"不确定性估计: {uncertainty.item():.4f}")
#     print(f"KL损失: {mpc.kl_loss(mu, logvar).item():.4f}")
#
#     # 生成多步预测，逐渐增加噪声
#     predictions = []
#     uncertainties = []
#     curr_history = history.clone()
#
#     for i in range(10):  # 预测10步
#         # 随着预测步骤增加，添加更多噪声
#         noise_level = 0.02 * i
#         curr_history = curr_history + noise_level * torch.randn_like(curr_history)
#
#         next_state, uncertainty, mu, logvar = mpc(curr_history)
#         predictions.append(next_state.detach())
#         uncertainties.append(uncertainty.item())
#
#         # 更新历史
#         curr_history = torch.cat([curr_history[1:], next_state.unsqueeze(0)], dim=0)
#
#     # 可视化不确定性随噪声增加的变化
#     plt.figure(figsize=(10, 6))
#     plt.plot(uncertainties, marker='o')
#     plt.title('预测不确定性随噪声增加的变化')
#     plt.xlabel('预测步骤')
#     plt.ylabel('不确定性估计')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('prediction_uncertainty_noise.png')
#     plt.close()
#
#     # 测试不同潜在空间维度对KL损失的影响
#     latent_dims = [8, 16, 32, 64, 128]
#     kl_losses = []
#
#     for dim in latent_dims:
#         model = MPCTransformer(feat_dim, latent_dim=dim)
#         _, _, mu, logvar = model(history)
#         kl_loss = model.kl_loss(mu, logvar).item()
#         kl_losses.append(kl_loss)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(latent_dims, kl_losses, marker='s')
#     plt.title('潜在空间维度对KL损失的影响')
#     plt.xlabel('潜在空间维度')
#     plt.ylabel('KL损失')
#     plt.xscale('log', base=2)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('latent_dim_kl_loss.png')
#     plt.close()
#
#     print("带不确定性估计的MPC模型测试完成，可视化结果已保存")
#     return mpc
#
#
# def test_integrated_system():
#     """测试集成了时序一致性约束的多模态系统"""
#     print("\n测试集成系统...")
#
#     # 配置参数
#     feat_dim = 64
#     pred_horizon = 3
#     hist_len = 5
#     num_classes = 4  # 'ang', 'hap', 'neu', 'sad'
#
#     # 初始化系统
#     system = MultimodalSystem(
#         feat_dim=feat_dim,
#         pred_horizon=pred_horizon,
#         hist_len=hist_len,
#         num_classes=num_classes
#     )
#
#     # 创建模拟情绪序列数据
#     emotion_centers = {
#         0: {'text': 0.2, 'audio': 0.7, 'video': 0.1},  # 愤怒：音频主导
#         1: {'text': 0.3, 'audio': 0.2, 'video': 0.5},  # 高兴：视频主导
#         2: {'text': 0.4, 'audio': 0.3, 'video': 0.3},  # 中性：均衡
#         3: {'text': 0.6, 'audio': 0.3, 'video': 0.1}  # 悲伤：文本主导
#     }
#
#     # 模拟数据
#     num_samples = 80
#     # 初始化results列表
#     results = []
#
#     # 创建4个情绪状态的转换序列
#     emotion_sequence = []
#     for i in range(num_samples):
#         segment = i // 20  # 每20个样本切换一次情绪
#         current_emotion = segment % 4
#         next_emotion = (segment + 1) % 4
#
#         position = i % 20
#         transition_weight = position / 20.0
#
#         blended_centers = {}
#         for modality in ['text', 'audio', 'video']:
#             blended_centers[modality] = (1 - transition_weight) * emotion_centers[current_emotion][modality] + \
#                                         transition_weight * emotion_centers[next_emotion][modality]
#
#         emotion_sequence.append((current_emotion, blended_centers))
#
#     # 处理样本序列
#     for i, (emotion, centers) in enumerate(emotion_sequence):
#         # 创建模拟特征
#         feats_dict = {
#             'text': torch.tensor([centers['text']]) + 0.1 * torch.randn(1),
#             'audio': torch.tensor([centers['audio']]) + 0.1 * torch.randn(1),
#             'video': torch.tensor([centers['video']]) + 0.1 * torch.randn(1)
#         }
#
#         # 扩展到特征维度 - 修复形状
#         for k in feats_dict:
#             # 确保形状是 [1, feat_dim]
#             feats_dict[k] = torch.ones(1, feat_dim) * feats_dict[k].item()
#
#         # 处理样本
#         if i >= hist_len - 1:  # 需要足够的历史数据
#             try:
#                 output = system.process(feats_dict, torch.tensor(emotion))
#                 if output is not None:
#                     results.append({
#                         'sample': i,
#                         'target': emotion,
#                         'prediction': output['prediction'],
#                         'weights': output['weights'],
#                         'uncertainty': output['uncertainty']
#                     })
#                     print(f"样本 {i}: 目标={emotion}, 预测={output['prediction']}, "
#                           f"权重={np.round(output['weights'], 3)}, "
#                           f"不确定性={output['uncertainty']:.4f}")
#             except Exception as e:
#                 print(f"处理样本 {i} 时出错: {e}")
#                 continue
#         else:
#             # 只添加到历史
#             try:
#                 system.process(feats_dict, None)
#             except Exception as e:
#                 print(f"添加历史样本 {i} 时出错: {e}")
#                 continue
#
#     # 分析结果
#     if results:
#         # 计算准确率
#         correct = sum(1 for r in results if r['target'] == r['prediction'])
#         accuracy = correct / len(results)
#         print(f"\n情感识别准确率: {accuracy:.4f}")
#
#         # 可视化模态权重随时间的变化
#         plt.figure(figsize=(12, 8))
#         samples = [r['sample'] for r in results]
#         weights_array = np.array([r['weights'] for r in results])
#
#         for i, modal in enumerate(['Text', 'Audio', 'Video']):  # 使用英文标签
#             plt.plot(samples, weights_array[:, i], label=modal, linewidth=2)
#
#         # 标记情绪转换点
#         for seg in range(1, 4):
#             plt.axvline(x=seg * 20, color='gray', linestyle='--')
#
#         # 添加情绪标签
#         emotion_names = ['Anger', 'Happy', 'Neutral', 'Sad']  # 使用英文标签
#         for seg in range(4):
#             plt.text(seg * 20 + 10, 0.9, emotion_names[seg],
#                      horizontalalignment='center', size=12,
#                      bbox=dict(facecolor='white', alpha=0.8))
#
#         plt.title('Modal Fusion Weights over Emotion States')
#         plt.xlabel('Sample Index')
#         plt.ylabel('Modal Weight')
#         plt.legend(loc='upper right')
#         plt.grid(True, alpha=0.3)
#         plt.ylim(0, 1)
#         plt.tight_layout()
#         plt.savefig('emotion_modal_weights.png')
#         plt.close()
#
#         print("集成系统测试完成，可视化结果已保存")
#     else:
#         print("没有足够的结果进行分析")
#
#     return system, results
#
#
# if __name__ == "__main__":
#     # 设置随机种子
#     torch.manual_seed(42)
#     np.random.seed(42)
#
#     # 测试各个组件
#     attention = test_temporal_consistency_attention()
#     mpc = test_mpc_with_uncertainty()
#     system, results = test_integrated_system()
#
#     print("\n所有测试完成!")