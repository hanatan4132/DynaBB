import torch
import cv2
import numpy as np
from breaker import BreakoutEnv
from reward_network import RewardPredictor  # 假設已實現 RewardPredictor

class RewardPredictorTester:
    def __init__(self, model_path):
        self.env = BreakoutEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加載 RewardPredictor 模型
        self.model = RewardPredictor(input_shape=(2, 192, 192), num_actions=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.key_actions = {
            ord('a'): 1,  # 左
            ord('d'): 2,  # 右
            ord(' '): 0,  # 不動
        }

        # 儲存最近兩幀用於預測
        self.prev_frame_1 = None  # t-1 幀
        self.prev_frame_2 = None  # t-2 幀

    def preprocess_frame(self, frame):
        """轉換影像為模型輸入格式"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉灰階

        frame = frame.astype(np.float32) / 255.0
        return frame

    def run(self):
        print("控制方式:")
        print("A: 左移  D: 右移  空白鍵: 不動")
        print("ESC: 退出")

        # 初始化遊戲和幀緩存
        self.env.reset()
        self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())
        self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())

        # 等待遊戲開始
        while self.env.game_state != "playing":
            self.env.step(0)  # 模擬等待空白鍵
            
        while True:
            # 顯示當前遊戲畫面
            current_frame = self.env._get_frame()
            cv2.imshow("Breakout Game", current_frame)
            key = cv2.waitKey(1)

            if key == 27:  # ESC 退出
                break

            if key in self.key_actions:
                action = self.key_actions[key]

                # 執行動作並獲取 reward
                _, true_reward, done, _ = self.env.step(action)
                current_frame = self.env._get_frame()
                current_processed = self.preprocess_frame(current_frame)

                # 準備模型輸入 (堆疊前兩幀)
                stacked_frames = np.stack([self.prev_frame_2, self.prev_frame_1], axis=0)
                stacked_tensor = torch.tensor(stacked_frames, device=self.device).unsqueeze(0)  # (1, 2, H, W)
                action_tensor = torch.tensor([action], device=self.device)

                # 預測 reward
                with torch.no_grad():
                    pred_reward = self.model(stacked_tensor, action_tensor).item()

                # 更新幀緩存
                self.prev_frame_2 = self.prev_frame_1
                self.prev_frame_1 = current_processed

                # 顯示預測結果
                info_panel = np.zeros((200, 400, 3), dtype=np.uint8)
                action_names = ["不動", "左移", "右移"]
                
                print(f"true Reward: {true_reward:.2f}")
                print(f"predict Reward: {pred_reward:.2f}")
                print(f"loss: {abs(true_reward - pred_reward):.2f}")
                
                cv2.imshow('Reward Prediction Info', info_panel)

                if done:
                    self.env.reset()
                    # 重置幀緩存
                    self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())
                    self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())

        self.env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "reward_models/best_reward_model.pth"  # 替換為你的模型路徑
    tester = RewardPredictorTester(model_path)
    tester.run()