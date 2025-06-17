import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

class RewardDataset(Dataset):
    def __init__(self, data):
        """
        資料集類別，用於載入預測 reward 的資料
        Args:
            data: 包含以下鍵的字典
                - prev_frames: 前兩幀堆疊 (N, 2, H, W)
                - actions: 動作標籤 (N,)
                - rewards: 獎勵值 (N,)
        """
        self.prev_frames = np.array(data["current_frames"], dtype=np.float32)  # (N, 2, H, W)
        self.actions = np.array(data["actions"], dtype=np.int64)  # (N,)
        self.rewards = np.array(data["rewards"], dtype=np.float32)  # (N,)
    
    def __len__(self):
        return len(self.prev_frames)
    
    def __getitem__(self, idx):
        # 獲取前兩幀 (已經堆疊好的)
        prev_frames = self.prev_frames[idx]  # (2, H, W)
        action = self.actions[idx]
        reward = self.rewards[idx]

        # 轉換為 PyTorch Tensor 並歸一化
        prev_frames = torch.tensor(prev_frames, dtype=torch.float32) / 255.0
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        
        return prev_frames, action, reward


class RewardPredictor(nn.Module):
    def __init__(self, input_shape=(2, 192, 192), num_actions=3):
        """
        Reward 預測網路
        Args:
            input_shape: 輸入圖像形狀 (channels, height, width)
            num_actions: 動作數量
        """
        super(RewardPredictor, self).__init__()
        
        # 特徵提取器 (處理連續兩幀)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Flatten()  # 展平特徵
        )
        
        # 計算特徵維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            self.feature_dim = self.feature_extractor(dummy_input).shape[1]
        
        # 動作嵌入層
        self.action_embed = nn.Embedding(num_actions, 128)
        
        # Reward 預測頭
        self.reward_predictor = nn.Sequential(
            nn.Linear(36992, 1024),  # 特徵 + 動作嵌入
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 輸出單一 reward 值
        )
    
    def forward(self, frames, actions):
        """
        Args:
            frames: 前兩幀堆疊 (batch_size, 2, H, W)
            actions: 動作標籤 (batch_size,)
        Returns:
            predicted_reward: 預測的 reward 值 (batch_size,)
        """
        # 提取視覺特徵
        visual_features = self.feature_extractor(frames)  # (batch_size, feature_dim)
        
        # 動作嵌入
        action_embeddings = self.action_embed(actions)  # (batch_size, 128)
        
        # 合併特徵
        combined = torch.cat([visual_features, action_embeddings], dim=1)
        
        # 預測 reward
        predicted_reward = self.reward_predictor(combined).squeeze(-1)
        
        return predicted_reward