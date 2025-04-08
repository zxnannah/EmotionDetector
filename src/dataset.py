import torch
import numpy as np
import os

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.sessions = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
        self.emotion_to_idx = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}

        audio_features = []
        text_features = []
        motion_features = []
        emotion_labels = []
        vad_values = []
        for session in self.sessions:
            session_path = os.path.join(features_dir, session)
            audio_features.append(torch.FloatTensor(np.load(os.path.join(session_path, 'audio_features.npy'))))
            text_features.append(torch.FloatTensor(np.load(os.path.join(session_path, 'text_features.npy'))))
            motion_features.append(torch.FloatTensor(np.load(os.path.join(session_path, 'motion_features.npy'))))
            emotion_label = np.load(os.path.join(session_path, 'emotion_labels.npy'))
            vad_values.append(torch.FloatTensor(np.load(os.path.join(session_path, 'vad_values.npy'))))
            # 将情绪标签转换为索引
            emotion_labels.append(torch.LongTensor([self.emotion_to_idx[emotion] for emotion in emotion_label]))

        # 将所有5个session片段数据，合并在一起，方便后面dataloader按batch读取
        self.audio_features = torch.cat(audio_features,dim=0)
        self.text_features = torch.cat(text_features,dim=0)
        self.motion_features = torch.cat(motion_features,dim=0)
        self.emotion_labels = torch.cat(emotion_labels,dim=0)
        self.vad_values = torch.cat(vad_values,dim=0)

        # print(f"audio_features shape: {self.audio_features.shape}")
        # print(f"text_features shape: {self.text_features.shape}")
        # print(f"motion_features shape: {self.motion_features.shape}")
        # print(f"emotion_labels shape: {self.emotion_labels.shape}")
        # print(f"vad_values shape: {self.vad_values.shape}")
        


    def __len__(self):
        return min(self.audio_features.shape[0], self.text_features.shape[0], 
                   self.motion_features.shape[0], self.emotion_labels.shape[0], self.vad_values.shape[0])

    def __getitem__(self, idx):
        return {
            'audio': self.audio_features[idx],
            'text': self.text_features[idx],
            'motion': self.motion_features[idx],
            'emotion': self.emotion_labels[idx],
            'vad': self.vad_values[idx]
        }