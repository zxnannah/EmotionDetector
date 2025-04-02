import os
import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from tqdm import tqdm

class IEMOCAPFeatureExtractor:
    def __init__(self):
        # 初始化音频模型
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")
        
        # 初始化文本模型
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model.to(self.device)
        self.text_model.to(self.device)
        
        # 设置为评估模式
        self.text_model.eval()
        
    def parse_emotion_file(self, emotion_file):
        """解析情感标注文件，只保留ang、hap、neu、sad四种情绪"""
        segments = []
        with open(emotion_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            current_segment = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('['):
                    # 解析时间段和情感标签
                    time_info = line[1:line.find(']')].split(' - ')
                    start_time = float(time_info[0])
                    end_time = float(time_info[1])
                    
                    # 解析情感标签和VAD值
                    parts = line[line.find(']')+1:].strip().split('\t')
                    turn_name = parts[1]
                    emotion = parts[2]
                    vad = eval(parts[3])  # 解析VAD值
                    
                    # 只保留四种情绪
                    if emotion in ['ang', 'hap', 'neu', 'sad']:
                        current_segment = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'turn_name': turn_name,
                            'emotion': emotion,
                            'vad': vad
                        }
                        segments.append(current_segment)
        
        return segments
    
    def parse_transcript_file(self, transcript_file):
        """解析转录文件，返回时间段对应的文本"""
        segments = {}
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            current_turn = None
            current_text = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('['):
                    # 保存之前的turn
                    if current_turn is not None:
                        segments[current_turn] = ' '.join(current_text)
                        current_text = []
                    
                    # 解析新的turn
                    time_info = line[1:line.find(']')].split(' - ')
                    current_turn = line[line.find(']')+1:].strip().split('\t')[1]
                elif line and current_turn is not None:
                    # 添加文本内容
                    current_text.append(line)
            
            # 保存最后一个turn
            if current_turn is not None and current_text:
                segments[current_turn] = ' '.join(current_text)
        
        return segments
    
    def extract_audio_features(self, audio_path, start_time, end_time):
        """提取指定时间段的音频特征"""
        # 加载音频文件
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 计算时间段的采样点
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 提取指定时间段的音频
        segment = audio[start_sample:end_sample]
        
        # 准备输入
        inputs = self.audio_tokenizer(segment, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            
        return features.cpu().numpy()
    
    def extract_text_features(self, text):
        """提取文本特征"""
        # 准备输入
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # 使用[CLS]标记的输出作为整个句子的表示
            cls_output = outputs.last_hidden_state[:, 0, :]
            # 同时保留平均池化的特征
            mean_output = outputs.last_hidden_state.mean(dim=1)
            # 将两种特征拼接
            features = torch.cat([cls_output, mean_output], dim=1)
            
        return features.cpu().numpy()
    
    def extract_motion_features(self, motion_path, start_time, end_time):
        """提取指定时间段的头部动作特征"""
        # 读取txt格式的头部动作数据
        motion_data = pd.read_csv(motion_path, delim_whitespace=True, skiprows=1)
        
        # 根据时间筛选数据
        mask = (motion_data['Time'] >= start_time) & (motion_data['Time'] <= end_time)
        segment_data = motion_data[mask]
        
        # 提取旋转和平移特征
        rotation_features = segment_data[['pitch', 'roll', 'yaw']].values
        translation_features = segment_data[['tra_x', 'tra_y', 'tra_z']].values
        
        # 计算统计特征
        features = {
            'rotation_mean': np.mean(rotation_features, axis=0),
            'rotation_std': np.std(rotation_features, axis=0),
            'rotation_max': np.max(rotation_features, axis=0),
            'rotation_min': np.min(rotation_features, axis=0),
            'translation_mean': np.mean(translation_features, axis=0),
            'translation_std': np.std(translation_features, axis=0),
            'translation_max': np.max(translation_features, axis=0),
            'translation_min': np.min(translation_features, axis=0)
        }
        
        # 将所有特征合并成一个向量
        feature_vector = np.concatenate([
            features['rotation_mean'],
            features['rotation_std'],
            features['rotation_max'],
            features['rotation_min'],
            features['translation_mean'],
            features['translation_std'],
            features['translation_max'],
            features['translation_min']
        ])
        
        return feature_vector
    
    def process_session(self, session_path):
        """处理单个会话的所有特征，只保留四种情绪的数据"""
        features = {
            'audio': [],
            'text': [],
            'motion': [],
            'emotion': [],
            'vad': []
        }
        
        # 获取dialog目录路径
        dialog_path = os.path.join(session_path, 'dialog')
        
        # 获取所有对话文件（impro和script）
        dialogue_files = []
        for file in os.listdir(os.path.join(dialog_path, 'EmoEvaluation')):
            if file.endswith('.txt'):
                dialogue_files.append(file.replace('.txt', ''))
        
        # 处理每个对话文件
        for dialogue in dialogue_files:
            # 读取情感标注文件
            emotion_file = os.path.join(dialog_path, 'EmoEvaluation', f"{dialogue}.txt")
            if os.path.exists(emotion_file):
                segments = self.parse_emotion_file(emotion_file)
                
                # 读取转录文件
                transcript_file = os.path.join(dialog_path, 'transcriptions', f"{dialogue}.txt")
                if os.path.exists(transcript_file):
                    transcript_segments = self.parse_transcript_file(transcript_file)
                
                for segment in segments:
                    # 只处理四种情绪的数据
                    if segment['emotion'] in ['ang', 'hap', 'neu', 'sad']:
                        # 提取音频特征
                        audio_path = os.path.join(dialog_path, 'wav', f"{dialogue}.wav")
                        if os.path.exists(audio_path):
                            audio_features = self.extract_audio_features(
                                audio_path, 
                                segment['start_time'], 
                                segment['end_time']
                            )
                            features['audio'].append(audio_features)
                        
                        # 提取文本特征
                        if os.path.exists(transcript_file) and segment['turn_name'] in transcript_segments:
                            text = transcript_segments[segment['turn_name']]
                            text_features = self.extract_text_features(text)
                            features['text'].append(text_features)
                        
                        # 提取头部动作特征
                        motion_path = os.path.join(dialog_path, 'MOCAP_head', f"{dialogue}.txt")
                        if os.path.exists(motion_path):
                            motion_features = self.extract_motion_features(
                                motion_path,
                                segment['start_time'],
                                segment['end_time']
                            )
                            features['motion'].append(motion_features)
                        
                        # 保存情感标签和VAD值
                        features['emotion'].append(segment['emotion'])
                        features['vad'].append(segment['vad'])
        
        return features

def main():
    # 初始化特征提取器
    extractor = IEMOCAPFeatureExtractor()
    
    # IEMOCAP数据集根目录
    dataset_root = "..\New folder1"
    
    # 处理所有会话
    for session in tqdm(os.listdir(dataset_root)):
        session_path = os.path.join(dataset_root, session)
        if os.path.isdir(session_path):
            features = extractor.process_session(session_path)
            
            # 保存特征
            save_path = os.path.join("features", session)
            os.makedirs(save_path, exist_ok=True)
            
            np.save(os.path.join(save_path, 'audio_features.npy'), np.array(features['audio']))
            np.save(os.path.join(save_path, 'text_features.npy'), np.array(features['text']))
            np.save(os.path.join(save_path, 'motion_features.npy'), np.array(features['motion']))
            np.save(os.path.join(save_path, 'emotion_labels.npy'), np.array(features['emotion']))
            np.save(os.path.join(save_path, 'vad_values.npy'), np.array(features['vad']))

if __name__ == "__main__":
    main()
