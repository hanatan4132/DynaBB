import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from breaker import BreakoutEnv
from encoder import Autoencoder
from reward_network import RewardPredictor

class CombinedTester:
    def __init__(self, ae_model_path, reward_model_path):
        self.env = BreakoutEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Autoencoder 模型
        self.ae_model = Autoencoder().to(self.device)
        self.ae_model.load_state_dict(torch.load(ae_model_path))
        self.ae_model.eval()

        # Reward 模型
        self.reward_model = RewardPredictor(input_shape=(2, 192, 192), num_actions=3).to(self.device)
        self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.reward_model.eval()

        self.criterion = nn.MSELoss()

        self.key_actions = {
            ord('a'): 1,
            ord('d'): 2,
            ord(' '): 0,
        }

        self.prev_frame_1 = None
        self.prev_frame_2 = None

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return torch.tensor(frame, device=self.device)

    def run(self):
        print("控制方式: A 左, D 右, 空白鍵不動, ESC 離開")

        self.env.reset()
        self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())
        self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())

        while self.env.game_state != "playing":
            self.env.step(0)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
        
            if key in self.key_actions:
                action = self.key_actions[key]
                action_tensor = torch.tensor([action], device=self.device)
        
                # 實際執行一步遊戲
                _, true_reward, done, _ = self.env.step(action)
                real_frame = self.env._get_frame()
                real_tensor = self.preprocess_frame(real_frame)
        
                # 更新預測輸入幀
                self.prev_frame_2 = self.prev_frame_1
                self.prev_frame_1 = real_tensor
        
                # 開始預測十步
                pred_frames = []
                pred_rewards = []
        
                frame_2 = self.prev_frame_2.clone()
                frame_1 = self.prev_frame_1.clone()
        
                for step in range(10):
                    random_action = np.random.randint(0, 3)
                    action_tensor = torch.tensor([random_action], device=self.device)
        
                    with torch.no_grad():
                        # 預測畫面
                        pred_st1, pred_st2 = self.ae_model(frame_2, frame_1, action_tensor)
                        # 預測 reward（以預測畫面為輸入）
                        reward_input = torch.cat((pred_st1, pred_st2), dim=1)
                        pred_reward = self.reward_model(reward_input, action_tensor).item()
        
                    pred_frames.append(pred_st2.cpu().squeeze().numpy())  # 顯示 t+2 畫面
                    pred_rewards.append(pred_reward)
        
                    # 下一層預測用輸出接續（深度預測）
                    frame_2 = pred_st1
                    frame_1 = pred_st2
        
                # 畫出 10 張圖 + reward
                plt.figure(figsize=(20, 3))
                for i in range(10):
                    plt.subplot(1, 10, i+1)
                    plt.imshow(pred_frames[i], cmap='gray')
                    plt.title(f"{pred_rewards[i]:.2f}", fontsize=8)
                    plt.axis('off')
                plt.suptitle("10-step Prediction w/ Predicted Rewards", fontsize=14)
                plt.tight_layout()
                plt.show()
        
                # 如果遊戲結束，重置
                if done or self.env.dead:
                    self.env.reset()
                    self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())
                    self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())
        
            # 顯示當前遊戲畫面
            cv2.imshow("Breakout Game", self.env._get_frame())


        self.env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ae_model_path = "maze_models/best_model_manual.pth"
    reward_model_path = "reward_models/best_reward_model.pth"
    tester = CombinedTester(ae_model_path, reward_model_path)
    tester.run()
