import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 假設 RewardDataset 和 RewardPredictor 已經實現
from reward_network import RewardDataset, RewardPredictor

def train_reward_predictor(model, dataloader, optimizer, criterion, num_epochs, device, save_dir="reward_models", patience=10):
    """
    訓練 reward 預測模型並加入 Early Stopping 機制
    Args:
        model: RewardPredictor 實例
        dataloader: 資料加載器
        optimizer: 優化器
        criterion: 損失函數
        num_epochs: 訓練週期數
        device: 運算設備
        save_dir: 模型保存目錄
        patience: Early Stopping 耐心值
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    train_losses = []
    best_loss = float('inf')
    no_improvement_count = 0
    plt.ion()  # 開啟互動模式

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        
        for frames, actions, rewards in dataloader:
            #print("Frames:", frames.shape)
            frames = frames.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            #print("Frames:", frames.shape)
            optimizer.zero_grad()
            pred_rewards = model(frames, actions)
            loss = criterion(pred_rewards, rewards)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_reward_model.pth'))
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early Stopping 檢查
        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # 定期保存檢查點
        if epoch % 70 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        # 可視化訓練過程
        if (epoch + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                # 取第一個批次的數據
                frames_example = frames[0].unsqueeze(0)
                action_example = actions[0].unsqueeze(0)
                reward_example = rewards[0].item()
                pred_reward = model(frames_example, action_example).item()

                # 顯示當前幀和預測結果
                plt.figure(figsize=(10, 5))
                
                # 顯示連續兩幀
                for i in range(2):
                    plt.subplot(1, 3, i+1)
                    frame = frames_example[0, i].cpu().numpy()
                    plt.imshow(frame, cmap='gray')
                    plt.title(f'Frame t-{2-i}')
                    plt.axis('off')
                
                # 顯示 reward 資訊
                plt.subplot(1, 3, 3)
                action_names = ["Stay", "Left", "Right"]
                plt.text(0.1, 0.6, 
                         f"Action: {action_names[action_example.item()]}\n"
                         f"True Reward: {reward_example:.2f}\n"
                         f"Pred Reward: {pred_reward:.2f}\n"
                         f"Error: {abs(reward_example - pred_reward):.2f}",
                         fontsize=12)
                plt.axis('off')
                
                plt.suptitle(f'Epoch {epoch+1} | Loss: {avg_loss:.4f}')
                plt.draw()
                plt.pause(0.001)
            
            model.train()

    # 保存最終模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_reward_model.pth'))
    plt.ioff()
    return train_losses

if __name__ == "__main__":
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入資料 (假設已經有 reward 資料集)
    data = np.load("breakout_auto_reward_dataset.npz")
    dataset = RewardDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化模型
    model = RewardPredictor(input_shape=(2, 192, 192), num_actions=3)
    
    # 使用 SmoothL1Loss 對 reward 預測更穩定
    criterion = nn.SmoothL1Loss()  
    
    # 使用較小的學習率
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #model = RewardPredictor()
    dummy_frames = torch.randn(1, 2, 192, 192)  # 假設輸入尺寸
    dummy_actions = torch.tensor([1])
    print("Input shapes:")
    print("Frames:", dummy_frames.shape)
    print("Actions:", dummy_actions.shape)

    #output = model(dummy_frames, dummy_actions)
    #print("Output shape:", output.shape)
    # 訓練模型
    losses = train_reward_predictor(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=1000,
        device=device,
        save_dir="reward_models",
        patience=40
    )

    # 繪製損失曲線
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()