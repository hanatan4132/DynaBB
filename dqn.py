import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        # 輸入形狀應該是 (1, H, W) - 從 Maze.render() 得到的灰階圖像
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        # 卷積層設計
        self.conv_layers = nn.Sequential(
            #nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=2, padding=2),
           # nn.ReLU(),
           # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            #nn.ReLU(),

            
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  
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
        )
        
        
        # 計算卷積層輸出尺寸
        conv_output_size = self._get_conv_output_size()
        
        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(36864, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        print(f"卷積層輸出尺寸: {conv_output_size}")
        print(f"網絡結構: {self}")
    
    def _get_conv_output_size(self):
        # 創建一個模擬輸入來獲取卷積層的輸出尺寸
        dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        conv_output = self.conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(conv_output.shape[1:])))  # 乘積所有維度（除了批次維度）
    
    
    def forward(self, x):
        # 確保輸入有批次維度
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 添加批次維度
            
        # 卷積處理
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全連接層
        x = self.fc_layers(x)
        return x