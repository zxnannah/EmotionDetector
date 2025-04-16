import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class MPCTransformer(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=3):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(feat_dim, nhead),
            num_layers
        )
        self.proj = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, history):
        return self.proj(self.transformer(history)[-1])

class FutureEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, feat_dim, batch_first=True)

    def forward(self, x):  # x: [B, T, D]
        _, h_n = self.rnn(x)  # h_n: [1, B, D]
        return h_n.squeeze(0)  # [B, D]

class RLPolicy(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.future_encoder = FutureEncoder(feat_dim)

        self.attn_param_net = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * feat_dim)
        )

        self.multihead_attn = nn.MultiheadAttention(feat_dim, 8)

        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, current_states, future_states):
        """
        current_states: [3, B, D] -- text/audio/video
        future_states: [3, B, T, D]
        """
        batch_size, feat_dim = current_states.shape[1], current_states.shape[2]
        weights = []

        for i in range(3):
            current = current_states[i]              # [B, D]
            future = self.future_encoder(future_states[i])  # [B, D]

            x = torch.cat([current, future], dim=1)  # [B, 2D]
            params = self.attn_param_net(x)          # [B, 3D]
            Wq, Wk, Wv = params.chunk(3, dim=1)

            query = (current.unsqueeze(1) @ Wq.view(batch_size, feat_dim, 1)).permute(1, 0, 2)  # [1, B, D]
            key   = (future.unsqueeze(1)  @ Wk.view(batch_size, feat_dim, 1)).permute(1, 0, 2)  # [1, B, D]
            value = (future.unsqueeze(1)  @ Wv.view(batch_size, feat_dim, 1)).permute(1, 0, 2)  # [1, B, D]

            attn_output, _ = self.multihead_attn(query, key, value)
            score = self.net(attn_output.squeeze(0))  # [B, 3]
            weights.append(score)

        # sum three scores and softmax to get fusion weight
        weights = torch.stack(weights, dim=1).sum(dim=0)  # [B, 3]
        return F.softmax(weights, dim=-1)

class MultimodalSystem:
    def __init__(self, feat_dim=256, pred_horizon=5, hist_len=3, num_classes=10, gamma=0.99, lambda_penalty=0.1):
        self.mpc = MPCTransformer(feat_dim)
        self.policy = RLPolicy(feat_dim, hist_len)
        self.task_model = nn.Linear(feat_dim, num_classes)

        self.history = {
            'text': deque(maxlen=hist_len),
            'audio': deque(maxlen=hist_len),
            'video': deque(maxlen=hist_len)
        }
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
        hist = torch.stack(list(self.history[modality]))
        hist = hist.unsqueeze(1)
        for _ in range(self.pred_horizon):
            next_state = self.mpc(hist)
            pred_states.append(next_state)
            hist = torch.cat([hist[1:], next_state.unsqueeze(0)], dim=0)
        return torch.stack(pred_states)

    def _calc_reward(self, acc, consistency, weight_change):
        # 计算奖励函数：包括分类准确率、模态一致性、权重波动
        return acc + self.gamma * consistency - self.lambda_penalty * weight_change

    def process(self, feats_dict, target=None, explore_prob=0.1):
        for k in self.history:
            self.history[k].append(feats_dict[k])

        if len(self.history['text']) < self.history['text'].maxlen:
            return None

        # MPC预测
        future_preds = [self._mpc_predict(k) for k in ['text', 'audio', 'video']]
        current_states = [self.history[k][-1] for k in ['text', 'audio', 'video']]

        # RL策略生成权重
        if torch.rand(1) < explore_prob:
            weights = torch.rand(3)
        else:
            weights = self.policy(
                torch.stack(current_states),
                torch.stack([p.mean(0) for p in future_preds])
            )

        # 模态融合
        # 通过强化学习生成的权重进行加权融合
        fused = sum(w * feats_dict[k] for w, k in zip(weights, ['text', 'audio', 'video']))

        # 分类和训练
        logits = self.task_model(fused)
        if target is not None:
            acc = (logits.argmax(-1) == target).float()
            consistency = torch.cosine_similarity(feats_dict['text'], feats_dict['audio']) + \
                         torch.cosine_similarity(feats_dict['audio'], feats_dict['video'])
            weight_change = torch.norm(weights - weights.detach(), p=2).item()
            reward = self._calc_reward(acc, consistency, weight_change)
            
            loss = -reward + F.cross_entropy(logits, target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return logits.argmax(dim=-1).item()
