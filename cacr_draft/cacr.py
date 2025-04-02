"""
多模态情绪感知模型
支持文本、音频和头部运动三种模态
基于UNITER架构扩展
"""

import copy
import json
import logging
from io import open
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
import torch.nn.functional as F
from .layer import BertLayer, BertPooler

logger = logging.getLogger(__name__)

class MultimodalConfig(object):
    """多模态模型配置类"""
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=3,  # 修改为3种模态
                 initializer_range=0.02,
                 audio_dim=1024,  # 音频特征维度
                 motion_dim=256):  # 头部运动特征维度
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.audio_dim = audio_dim
            self.motion_dim = motion_dim
        else:
            raise ValueError("First argument must be either a vocabulary size "
                           "(int) or the path to a pretrained model config "
                           "file (str)")

class MultimodalTextEmbeddings(nn.Module):
    """文本嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                          config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                              config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                config.hidden_size)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                     + position_embeddings
                     + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AudioEmbeddings(nn.Module):
    """音频嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.audio_linear = nn.Linear(config.audio_dim, config.hidden_size)
        self.audio_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, audio_feat, audio_masks=None):
        if audio_masks is not None:
            audio_feat = audio_feat * audio_masks.unsqueeze(-1)
        
        audio_embeddings = self.audio_layer_norm(self.audio_linear(audio_feat))
        audio_embeddings = self.LayerNorm(audio_embeddings)
        audio_embeddings = self.dropout(audio_embeddings)
        return audio_embeddings

class MotionEmbeddings(nn.Module):
    """头部运动嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.motion_linear = nn.Linear(config.motion_dim, config.hidden_size)
        self.motion_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, motion_feat, motion_masks=None):
        if motion_masks is not None:
            motion_feat = motion_feat * motion_masks.unsqueeze(-1)
        
        motion_embeddings = self.motion_layer_norm(self.motion_linear(motion_feat))
        motion_embeddings = self.LayerNorm(motion_embeddings)
        motion_embeddings = self.dropout(motion_embeddings)
        return motion_embeddings

class MultimodalEncoder(nn.Module):
    """多模态编码器"""
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                  for _ in range(config.num_hidden_layers)])
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def get_attention_probs(self, layer_module, hidden_states, attn_mask, row_b, row_l, col_b=None, col_l=None):
        attn = layer_module.attention.self.get_attention_probs(hidden_states, attn_mask)
        attn = torch.mul(attn, attn_mask)
        attn = torch.narrow(attn, 2, row_b, row_l)
        if col_b is None and col_l is None:
            col_b, col_l = row_b, row_l
        attn = torch.narrow(attn, 3, col_b, col_l)
        attn = torch.mean(attn, dim=1)
        return attn

    def iais_soft(self, attn1, attn2, cross_attn, modal):
        softmax = nn.Softmax(dim=-1)
        
        if modal == 'T':  # Text
            pseudo_attn = cross_attn.detach() @ attn2 @ torch.transpose(cross_attn.detach(), 0, 1)
            attn1, pseudo_attn = softmax(attn1), softmax(pseudo_attn)
            iais_loss = self.KLDivLoss(attn1.log(), pseudo_attn) + self.KLDivLoss(pseudo_attn.log(), attn1)
        elif modal == 'A':  # Audio
            pseudo_attn = cross_attn.detach() @ attn1 @ torch.transpose(cross_attn.detach(), 0, 1)
            attn2, pseudo_attn = softmax(attn2), softmax(pseudo_attn)
            iais_loss = self.KLDivLoss(attn2.log(), pseudo_attn) + self.KLDivLoss(pseudo_attn.log(), attn2)
        elif modal == 'M':  # Motion
            pseudo_attn = cross_attn.detach() @ attn1 @ torch.transpose(cross_attn.detach(), 0, 1)
            attn2, pseudo_attn = softmax(attn2), softmax(pseudo_attn)
            iais_loss = self.KLDivLoss(attn2.log(), pseudo_attn) + self.KLDivLoss(pseudo_attn.log(), attn2)
        else:
            raise ValueError('error modal')
        return iais_loss

    def forward(self, input_, attention_mask, txt_attn_mask=None, audio_attn_mask=None, motion_attn_mask=None,
                t2a_attn_mask=None, a2t_attn_mask=None, t2m_attn_mask=None, m2t_attn_mask=None,
                a2m_attn_mask=None, m2a_attn_mask=None, max_tl=0, max_audio=0, max_motion=0,
                output_all_encoded_layers=True, IAIS=False, pairs_num=3):
        all_encoder_layers = []
        self_attn_loss_per_layer = {}
        hidden_states = input_

        for i, layer_module in enumerate(self.layer):
            if IAIS and i == len(self.layer) - 1:
                gt_indices = torch.tensor(list(range(0, hidden_states.size(0), pairs_num)),
                                       dtype=torch.long, device=hidden_states.device)
                hidden_states_gt = hidden_states.index_select(0, gt_indices)
                
                # 获取各模态的注意力
                txt_attn = self.get_attention_probs(layer_module, hidden_states_gt, txt_attn_mask, 1, max_tl - 2)
                audio_attn = self.get_attention_probs(layer_module, hidden_states_gt, audio_attn_mask, max_tl, max_audio)
                motion_attn = self.get_attention_probs(layer_module, hidden_states_gt, motion_attn_mask, max_tl + max_audio, max_motion)

                # 获取跨模态注意力
                t2a_attn = self.get_attention_probs(layer_module, hidden_states_gt, t2a_attn_mask, 1, max_tl - 2, max_tl, max_audio)
                a2t_attn = self.get_attention_probs(layer_module, hidden_states_gt, a2t_attn_mask, max_tl, max_audio, 1, max_tl - 2)
                t2m_attn = self.get_attention_probs(layer_module, hidden_states_gt, t2m_attn_mask, 1, max_tl - 2, max_tl + max_audio, max_motion)
                m2t_attn = self.get_attention_probs(layer_module, hidden_states_gt, m2t_attn_mask, max_tl + max_audio, max_motion, 1, max_tl - 2)
                a2m_attn = self.get_attention_probs(layer_module, hidden_states_gt, a2m_attn_mask, max_tl, max_audio, max_tl + max_audio, max_motion)
                m2a_attn = self.get_attention_probs(layer_module, hidden_states_gt, m2a_attn_mask, max_tl + max_audio, max_motion, max_tl, max_audio)

                self_attn_loss_layer_i = torch.tensor(0, dtype=hidden_states.dtype, device=hidden_states.device)
                
                for j, (input_len, audio_len, motion_len) in enumerate(
                    zip(txt_attn_mask[:, 0, 1, :].sum(1),
                        audio_attn_mask[:, 0, max_tl, :].sum(1),
                        motion_attn_mask[:, 0, max_tl + max_audio, :].sum(1))):
                    
                    input_len, audio_len, motion_len = int(input_len.item()), int(audio_len.item()), int(motion_len.item())

                    if IAIS == 'T-soft':
                        iais_loss = self.iais_soft(txt_attn[j, :input_len, :input_len],
                                                 audio_attn[j, :audio_len, :audio_len],
                                                 t2a_attn[j, :input_len, :audio_len], 'T')
                    elif IAIS == 'A-soft':
                        iais_loss = self.iais_soft(audio_attn[j, :audio_len, :audio_len],
                                                 txt_attn[j, :input_len, :input_len],
                                                 a2t_attn[j, :audio_len, :input_len], 'A')
                    elif IAIS == 'M-soft':
                        iais_loss = self.iais_soft(motion_attn[j, :motion_len, :motion_len],
                                                 txt_attn[j, :input_len, :input_len],
                                                 m2t_attn[j, :motion_len, :input_len], 'M')
                    else:
                        raise ValueError("IAIS must in ['T-soft', 'A-soft', 'M-soft']")

                    self_attn_loss_layer_i += iais_loss

                self_attn_loss_per_layer['self_attn_loss/layer_%s' % i] = self_attn_loss_layer_i / gt_indices.size(0)
                self_attn_loss_per_layer['self_attn_loss'] = self_attn_loss_per_layer['self_attn_loss/layer_%s' % i]

            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        if IAIS:
            return all_encoder_layers, self_attn_loss_per_layer
        else:
            return all_encoder_layers

class MultimodalModel(nn.Module):
    """多模态情绪感知模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embeddings = MultimodalTextEmbeddings(config)
        self.audio_embeddings = AudioEmbeddings(config)
        self.motion_embeddings = MotionEmbeddings(config)
        self.encoder = MultimodalEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_text_embeddings(self, input_ids, position_ids, token_type_ids=None):
        return self.text_embeddings(input_ids, position_ids, token_type_ids)

    def _compute_audio_embeddings(self, audio_feat, audio_masks=None):
        return self.audio_embeddings(audio_feat, audio_masks)

    def _compute_motion_embeddings(self, motion_feat, motion_masks=None):
        return self.motion_embeddings(motion_feat, motion_masks)

    def _compute_multimodal_embeddings(self, input_ids, position_ids, audio_feat, motion_feat,
                                     gather_index=None, audio_masks=None, motion_masks=None,
                                     txt_type_ids=None):
        txt_emb = self._compute_text_embeddings(input_ids, position_ids, txt_type_ids)
        audio_emb = self._compute_audio_embeddings(audio_feat, audio_masks)
        motion_emb = self._compute_motion_embeddings(motion_feat, motion_masks)

        if gather_index is not None:
            gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            embedding_output = torch.gather(torch.cat([txt_emb, audio_emb, motion_emb], dim=1),
                                         dim=1, index=gather_index)
        else:
            embedding_output = torch.cat([txt_emb, audio_emb, motion_emb], dim=1)
        return embedding_output

    def extend_self_attn_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = torch.matmul(attention_mask.permute(0, 1, 3, 2), attention_mask)
        return attention_mask

    def extend_cross_attn_mask(self, mask1, mask2):
        mask1 = mask1.unsqueeze(1).unsqueeze(2)
        mask1 = mask1.to(dtype=next(self.parameters()).dtype)
        mask2 = mask2.unsqueeze(1).unsqueeze(2)
        mask2 = mask2.to(dtype=next(self.parameters()).dtype)
        attn_mask1 = torch.matmul(mask1.permute(0, 1, 3, 2), mask2)
        attn_mask2 = torch.matmul(mask2.permute(0, 1, 3, 2), mask1)
        return attn_mask1, attn_mask2

    def forward(self, input_ids, position_ids, audio_feat, motion_feat,
               attention_mask, gather_index=None, audio_masks=None, motion_masks=None,
               txt_attn_mask=None, audio_attn_mask=None, motion_attn_mask=None,
               output_all_encoded_layers=True, IAIS=False, txt_type_ids=None):
        # 计算自注意力掩码
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 计算嵌入
        embedding_output = self._compute_multimodal_embeddings(
            input_ids, position_ids, audio_feat, motion_feat,
            gather_index, audio_masks, motion_masks, txt_type_ids)

        if IAIS:
            assert txt_attn_mask is not None and audio_attn_mask is not None and motion_attn_mask is not None
            
            extended_txt_attn_mask = self.extend_self_attn_mask(txt_attn_mask)
            extended_audio_attn_mask = self.extend_self_attn_mask(audio_attn_mask)
            extended_motion_attn_mask = self.extend_self_attn_mask(motion_attn_mask)

            t2a_attn_mask, a2t_attn_mask = self.extend_cross_attn_mask(txt_attn_mask, audio_attn_mask)
            t2m_attn_mask, m2t_attn_mask = self.extend_cross_attn_mask(txt_attn_mask, motion_attn_mask)
            a2m_attn_mask, m2a_attn_mask = self.extend_cross_attn_mask(audio_attn_mask, motion_attn_mask)

            encoded_layers, self_attn_loss_per_layer = self.encoder(
                embedding_output, extended_attention_mask,
                extended_txt_attn_mask, extended_audio_attn_mask, extended_motion_attn_mask,
                t2a_attn_mask, a2t_attn_mask, t2m_attn_mask, m2t_attn_mask,
                a2m_attn_mask, m2a_attn_mask,
                input_ids.size(1), audio_feat.size(1), motion_feat.size(1),
                output_all_encoded_layers=output_all_encoded_layers,
                IAIS=IAIS)
            
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, self_attn_loss_per_layer
        else:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask,
                output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers 