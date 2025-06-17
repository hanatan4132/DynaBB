import torch
import torch.nn as nn
import cv2
import numpy as np
from breaker import BreakoutEnv
from encoder import Autoencoder
import matplotlib.pyplot as plt

class AutoencoderTester:
    def __init__(self, model_path):
        self.env = BreakoutEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.criterion = nn.MSELoss()

        self.key_actions = {
            ord('a'): 1,
            ord('d'): 2,
            ord(' '): 0,
        }

        self.prev_frame_2 = None
        self.prev_frame_1 = None

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return torch.tensor(frame, device=self.device)

    def postprocess_frame(self, tensor):
        frame = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        return frame

    def run(self):
        print("控制方式:")
        print("A: 左移  D: 右移  空白鍵: 不動")
        print("ESC: 退出")

        self.env.reset()
        self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())
        self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())

        while self.env.game_state != "playing":
            self.env.step(0)

        while True:
            current_frame = self.env._get_frame()
            current_tensor = self.preprocess_frame(current_frame)
            cv2.imshow("Breakout Frame", current_frame)
            cv2.waitKey(1)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break

            if key in self.key_actions:
                action = self.key_actions[key]
                _, _, done, _ = self.env.step(action)
                next_frame_1 = self.env._get_frame()
                next_tensor_1 = self.preprocess_frame(next_frame_1)

                _, _, done, _ = self.env.step(action)
                next_frame_2 = self.env._get_frame()
                next_tensor_2 = self.preprocess_frame(next_frame_2)

                with torch.no_grad():
                    action_tensor = torch.tensor([action], device=self.device)
                    pred_st1, pred_st2 = self.model(self.prev_frame_2, self.prev_frame_1, action_tensor)

                    # 👉 計算 MSE loss
                    loss1 = self.criterion(pred_st1, next_tensor_1)
                    loss2 = self.criterion(pred_st2, next_tensor_2)
                    print(f"[Debug] MSE Loss - st+1: {loss1.item():.6f} | st+2: {loss2.item():.6f}")

                # 顯示圖像
                pred_st1_np = pred_st1.cpu().detach().squeeze().numpy()
                pred_st2_np = pred_st2.cpu().detach().squeeze().numpy()
                actual_st1_np = next_tensor_1.cpu().detach().squeeze().numpy()
                actual_st2_np = next_tensor_2.cpu().detach().squeeze().numpy()

                plt.figure(figsize=(12, 6))
                plt.subplot(2, 2, 1)
                plt.imshow(actual_st1_np, cmap="gray")
                plt.title("Actual st+1")
                plt.colorbar()
                plt.axis('off')

                plt.subplot(2, 2, 2)
                plt.imshow(pred_st1_np, cmap="gray")
                plt.title("Predicted st+1")
                plt.colorbar()
                plt.axis('off')

                plt.subplot(2, 2, 3)
                plt.imshow(actual_st2_np, cmap="gray")
                plt.title("Actual st+2")
                plt.colorbar()
                plt.axis('off')

                plt.subplot(2, 2, 4)
                plt.imshow(pred_st2_np, cmap="gray")
                plt.title("Predicted st+2")
                plt.colorbar()
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                self.prev_frame_2 = next_tensor_1
                self.prev_frame_1 = next_tensor_2

                if done:
                    self.env.reset()
                    self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())
                    self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())

        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"maze_models/200_model_manual.pth"
    tester = AutoencoderTester(model_path)
    tester.run()
