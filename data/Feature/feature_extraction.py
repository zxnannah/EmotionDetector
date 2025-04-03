import os
import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from tqdm import tqdm


pretrained_dir = "../../pretrained_models"

def check_model_files(model_dir, required_files):
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    return missing_files

class IEMOCAPFeatureExtractor:
    def __init__(self):
        wav2vec2_files = [
            'config.json',
            'pytorch_model.bin',
            'preprocessor_config.json',
            'tokenizer_config.json',
            'vocab.json'
        ]
        wav2vec2_dir = os.path.join(pretrained_dir, 'facebook', 'wav2vec2-base')
        missing_wav2vec2 = check_model_files(wav2vec2_dir, wav2vec2_files)
        if missing_wav2vec2:
            raise FileNotFoundError(f"Wav2Vec2模型缺少以下文件: {missing_wav2vec2}")

        bert_files = [
            'config.json',
            'pytorch_model.bin',
            'tokenizer_config.json',
            'vocab.txt',
            'tokenizer.json'
        ]
        bert_dir = os.path.join(pretrained_dir, 'bert-base-uncased')
        missing_bert = check_model_files(bert_dir, bert_files)
        if missing_bert:
            raise FileNotFoundError(f"BERT模型缺少以下文件: {missing_bert}")


        self.audio_model = Wav2Vec2Model.from_pretrained(wav2vec2_dir, local_files_only=True)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_dir, local_files_only=True)
            
        self.text_model = BertModel.from_pretrained(bert_dir, local_files_only=True)
        self.text_tokenizer = BertTokenizer.from_pretrained(bert_dir, local_files_only=True)

        if not os.path.exists(os.path.join(bert_dir, 'pytorch_model.bin')):
            raise FileNotFoundError("BERT模型权重文件未找到")
                
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model.to(self.device)
        self.text_model.to(self.device)
        
        self.audio_model.eval()
        self.text_model.eval()
        
    def parse_emotion_file(self, file_path):
        """只保留ang、hap、neu、sad四种"""
        segments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('%') or line.startswith('C-') or line.startswith('A-'):
                        continue
                        
                    if line.startswith('['):
                        try:
                            time_info = line[1:line.find(']')].split(' - ')
                            start_time = float(time_info[0])
                            end_time = float(time_info[1])
                            
                            rest = line[line.find(']')+1:].strip().split('\t')
                            if len(rest) >= 3:
                                turn_name = rest[0]
                                emotion = rest[1]
                                
                                vad = [2.5, 2.5, 2.5]  # 默认值
                                if len(rest) > 2:
                                    try:
                                        vad_str = rest[2].strip('[]')
                                        vad = [float(x) for x in vad_str.split(',')]
                                    except:
                                        pass
                                
                                if emotion in ['ang', 'hap', 'neu', 'sad']:
                                    segments.append({
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'turn_name': turn_name,
                                        'emotion': emotion,
                                        'vad': vad
                                    })
                        except Exception as e:
                            print(f"解析情感文件出错: {file_path}")
                            print(f"错误行: {line}")
                            print(f"错误信息: {str(e)}")
                            
        except Exception as e:
            print(f"读取情感文件出错: {file_path}")
            print(f"错误信息: {str(e)}")
            
        return segments
    
    def parse_transcript_file(self, file_path):
        """解析转录文件，返回时间段对应的文本"""
        segments = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        turn_name = line.split(' [')[0]
                        text = line.split(': ')[1] if ': ' in line else ''
                        segments[turn_name] = text
                    except Exception as e:
                        print(f"解析转录文件出错: {file_path}")
                        print(f"错误行: {line}")
                        print(f"错误信息: {str(e)}")
                        
        except Exception as e:
            print(f"读取转录文件出错: {file_path}")
            print(f"错误信息: {str(e)}")
            
        return segments
    
    def extract_audio_features(self, audio_path, start_time, end_time):
        """提取指定时间段的音频特征"""
        audio, sr = librosa.load(audio_path, sr=16000)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment = audio[start_sample:end_sample]

        inputs = self.audio_processor(segment, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            
        return features.cpu().numpy()
    
    def extract_text_features(self, text):
        """提取文本特征"""
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            cls_output = outputs.last_hidden_state[:, 0, :]
            mean_output = outputs.last_hidden_state.mean(dim=1)
            features = torch.cat([cls_output, mean_output], dim=1)
            
        return features.cpu().numpy()
    
    def extract_motion_features(self, motion_path, start_time, end_time):
        """提取指定时间段的头部动作特征"""
        try:
            motion_data = []
            with open(motion_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[2:]:  # 跳过前两行标题
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        time = float(parts[1])
                        if start_time <= time <= end_time:
                            motion_data.append([
                                float(parts[2]),  # pitch
                                float(parts[3]),  # roll
                                float(parts[4]),  # yaw
                                float(parts[5]),  # tra_x
                                float(parts[6]),  # tra_y
                                float(parts[7])   # tra_z
                            ])
            
            if not motion_data:
                return None
                
            motion_data = np.array(motion_data).reshape(1,-1,6)

            target_seq_len = 10
            current_seq_len = motion_data.shape[1]

            if current_seq_len>target_seq_len:
                motion_data = motion_data[:,:target_seq_len,:]
            elif current_seq_len<target_seq_len:
                pad_width=target_seq_len-current_seq_len
                motion_data = np.pad(motion_data, ((0,0),(0,pad_width),(0,0)), 'constant', constant_values=0)

            return motion_data
            
        except Exception as e:
            print(f"提取动作特征出错: {motion_path}")
            print(f"错误信息: {str(e)}")
            return None
    
    def process_session(self, session_path):
        """处理单个会话的所有特征，只保留四种情绪的数据"""
        features = {
            'audio': [],
            'text': [],
            'motion': [],
            'emotion': [],
            'vad': []
        }
        
        try:
            dialog_path = os.path.join(session_path, 'dialog')
            if not os.path.exists(dialog_path):
                return features
            
            dialogue_files = []
            emo_eval_path = os.path.join(dialog_path, 'EmoEvaluation')
            if os.path.exists(emo_eval_path):
                for file in os.listdir(emo_eval_path):
                    if file.endswith('.txt'):
                        dialogue_files.append(file.replace('.txt', ''))
            
            for dialogue in tqdm(dialogue_files, desc="处理对话文件"):
                emotion_file = os.path.join(dialog_path, 'EmoEvaluation', f"{dialogue}.txt")
                if os.path.exists(emotion_file):
                    segments = self.parse_emotion_file(emotion_file)
                    
                    transcript_file = os.path.join(dialog_path, 'transcriptions', f"{dialogue}.txt")
                    if os.path.exists(transcript_file):
                        transcript_segments = self.parse_transcript_file(transcript_file)
                    else:
                        continue
                    
                    for segment in segments:
                        if segment['emotion'] in ['ang', 'hap', 'neu', 'sad']:
                            # 提取音频特征
                            audio_path = os.path.join(dialog_path, 'wav', f"{dialogue}.wav")
                            if os.path.exists(audio_path):
                                try:
                                    audio_features = self.extract_audio_features(
                                        audio_path, 
                                        segment['start_time'], 
                                        segment['end_time']
                                    )
                                    features['audio'].append(audio_features)
                                except Exception as e:
                                    print(f"提取音频特征出错: {audio_path}")
                                    print(f"错误信息: {str(e)}")
                            
                            # 提取文本特征
                            if segment['turn_name'] in transcript_segments:
                                try:
                                    text = transcript_segments[segment['turn_name']]
                                    text_features = self.extract_text_features(text)
                                    features['text'].append(text_features)
                                except Exception as e:
                                    print(f"提取文本特征出错: {segment['turn_name']}")
                                    print(f"错误信息: {str(e)}")
                            
                            # 提取头部动作特征
                            motion_path = os.path.join(dialog_path, 'MOCAP_head', f"{dialogue}.txt")
                            if os.path.exists(motion_path):
                                motion_features = self.extract_motion_features(
                                    motion_path,
                                    segment['start_time'],
                                    segment['end_time']
                                )
                                if motion_features is not None:
                                    features['motion'].append(motion_features)
                            
                            features['emotion'].append(segment['emotion'])
                            features['vad'].append(segment['vad'])
            
        except Exception as e:
            print(f"处理会话出错: {session_path}")
            print(f"错误信息: {str(e)}")
        
        return features

def main():

    extractor = IEMOCAPFeatureExtractor()
    
    dataset_root = "data/IEMOCAP"

    
    print("\n目录:")
    for item in os.listdir(dataset_root):
        print(f"- {item}")
    
    features_dir = "data/Feature/features"
    os.makedirs(features_dir, exist_ok=True)
    print(f"\n特征保存目录: {features_dir}")
    
    for session in tqdm(os.listdir(dataset_root)):
        session_path = os.path.join(dataset_root, session)
        # 只处理Session开头的目录，跳过Documentation
        if os.path.isdir(session_path) and session.startswith('Session'):
            print(f"\n处理会话: {session}")
            print(f"会话路径: {session_path}")
            
            features = extractor.process_session(session_path)
            
            # 保存特征
            save_path = os.path.join(features_dir, session)
            os.makedirs(save_path, exist_ok=True)
            
            # 只保存非空特征
            if features['audio']:
                np.save(os.path.join(save_path, 'audio_features.npy'), np.array(features['audio']))
                print(f"保存音频特征: {len(features['audio'])} 个样本")
            else:
                print("警告: 没有音频特征可保存")
                
            if features['text']:
                np.save(os.path.join(save_path, 'text_features.npy'), np.array(features['text']))
                print(f"保存文本特征: {len(features['text'])} 个样本")
            else:
                print("警告: 没有文本特征可保存")
                
            if features['motion']:
                np.save(os.path.join(save_path, 'motion_features.npy'), np.array(features['motion']))
                print(f"保存动作特征: {len(features['motion'])} 个样本")
            else:
                print("警告: 没有动作特征可保存")
                
            if features['emotion']:
                np.save(os.path.join(save_path, 'emotion_labels.npy'), np.array(features['emotion']))
                print(f"保存情感标签: {len(features['emotion'])} 个样本")
            else:
                print("警告: 没有情感标签可保存")
                
            if features['vad']:
                np.save(os.path.join(save_path, 'vad_values.npy'), np.array(features['vad']))
                print(f"保存VAD值: {len(features['vad'])} 个样本")
            else:
                print("警告: 没有VAD值可保存")

if __name__ == "__main__":
    main()
