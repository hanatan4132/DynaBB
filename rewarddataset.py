import numpy as np
import cv2
import random
from breaker import BreakoutEnv
from variables import screen_width, screen_height, fps
import pygame
from pygame.locals import QUIT

# 初始化打磚塊遊戲環境
env = BreakoutEnv()

# 定義資料集存儲變數
current_frames = []  # 存儲前兩幀 (stacked)
actions = []         # 存儲動作
rewards = []         # 存儲獎勵值

def get_frame(env):
    """獲取當前遊戲畫面，轉換為灰階並縮放"""
    frame = env._get_frame()  # 獲取當前畫面 (RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉為灰階
    frame = cv2.resize(frame, (screen_width, screen_height))  # 確保解析度一致
    frame = frame.astype(np.uint8)  # 確保數據類型一致
    return frame

def press_space_to_start(env):
    """模擬按下空白鍵，使遊戲進入 'playing' 狀態"""
    while env.game_state != "playing":
        env.step(0)  # 執行不動作，模擬等待空白鍵
        cv2.imshow("Breakout Frame", get_frame(env))  # 顯示畫面
        cv2.waitKey(1)  # 等待畫面刷新

def collect_data_with_auto_strategy(env, total_steps):
    """使用自動策略收集資料並過濾 reward=0 的樣本"""
    press_space_to_start(env)
    prev_frame_1 = get_frame(env)
    prev_frame_2 = get_frame(env)
    
    collected_count = 0
    attempted_steps = 0
    
    # 定義多種自動策略
    strategies = [
        "追蹤中心", "追蹤左側", "追蹤右側", "追蹤左極端", "追蹤右極端", "隨機位置", "隨機位置2"
    ]

    current_strategy = random.choice(strategies)
    next_strategy_change = 0
    random_target_x = None
    
    while collected_count < total_steps:
        env.clock.tick(fps)
        
        # 處理退出事件
        for event in pygame.event.get():
            if event.type == QUIT:
                env.close()
                return
        
        # 實現自動策略控制
        if attempted_steps >= next_strategy_change or env.dead:
            current_strategy = random.choice(strategies)
            next_strategy_change = attempted_steps + random.randint(50, 100)
            random_target_x = random.randint(env.player_paddle.width//2, 
                                         screen_width - env.player_paddle.width//2)
            print(f"⚡ 策略改變: {current_strategy}")
            
        ball_x = env.ball.rect.centerx
        paddle_x = env.player_paddle.rect.centerx
        paddle_width = env.player_paddle.width
        
        # 根據當前策略確定目標位置
        target_x = paddle_x
        if current_strategy == "追蹤中心":
            target_x = ball_x
        elif current_strategy == "追蹤左側":
            target_x = ball_x + paddle_width * 0.3
        elif current_strategy == "追蹤右側":
            target_x = ball_x - paddle_width * 0.3
        elif current_strategy == "追蹤左極端":
            target_x = ball_x + paddle_width * 0.45
        elif current_strategy == "追蹤右極端":
            target_x = ball_x - paddle_width * 0.45
        elif current_strategy == "隨機位置":
            target_x = random_target_x
        elif current_strategy == "隨機位置2":
            target_x = random_target_x
            
        # 確保目標在畫面範圍內
        target_x = max(paddle_width//2, min(target_x, screen_width - paddle_width//2))
        
        # 決定動作
        if abs(target_x - paddle_x) < 10:
            action = 0  # 不動
        elif target_x < paddle_x:
            action = 1  # 向左
        else:
            action = 2  # 向右

        # 執行動作
        current_frame = get_frame(env)
        _, reward, done, _ = env.step(action)
        current_frame2 = get_frame(env)
        _, reward, done, _ = env.step(action)
        attempted_steps += 1
        
        # 決定是否保留該樣本
        keep_sample = True
        if reward == 0:
            if random.random() < 0.95:  # 95% 機率丟棄零獎勵樣本
                keep_sample = False
        
        if keep_sample:
            try:
                stacked_frames = np.stack([prev_frame_2, prev_frame_1], axis=0)
                current_frames.append(stacked_frames)
                actions.append(action)
                rewards.append(reward)
                collected_count += 1
            except ValueError as e:
                print(f"❌ 發生錯誤：{e}")
            
            # 進度輸出
            if collected_count % 100 == 0:
                print(f"✅ 已收集 {collected_count}/{total_steps} 筆資料 | 嘗試次數: {attempted_steps} | 最新獎勵: {reward}")

        # 更新前兩幀
        prev_frame_2 = current_frame
        prev_frame_1 = current_frame2

        # 遊戲結束處理
        if done or env.dead:
            env.reset()
            prev_frame_1 = get_frame(env)
            prev_frame_2 = get_frame(env)
            press_space_to_start(env)

        # 顯示畫面並添加策略信息
        frame_with_text = cv2.putText(
            current_frame.copy(),
            f"Strategy: {current_strategy}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        cv2.imshow("Breakout Frame", frame_with_text)
        if cv2.waitKey(1) == 27:  # ESC 退出
            break

    cv2.destroyAllWindows()
    return attempted_steps

# 收集資料 (保持總資料量 5000 筆)
print("開始自動收集獎勵資料...")
attempted_steps = collect_data_with_auto_strategy(env, 10000)

# 分析資料分布
zero_rewards = sum(1 for r in rewards if r == 0)
non_zero_rewards = len(rewards) - zero_rewards
print(f"\n資料分布分析:")
print(f"總收集樣本: {len(rewards)}")
print(f"零獎勵樣本: {zero_rewards} ({zero_rewards/len(rewards)*100:.1f}%)")
print(f"非零獎勵樣本: {non_zero_rewards} ({non_zero_rewards/len(rewards)*100:.1f}%)")
print(f"總嘗試次數: {attempted_steps}")

# 儲存資料集
try:
    np.savez("breakout_auto_reward_dataset.npz",
             current_frames=np.array(current_frames),
             actions=np.array(actions),
             rewards=np.array(rewards))
    print("\n🎉 自動收集的獎勵資料集已成功儲存！")
except ValueError as e:
    print(f"\n❌ 儲存失敗：{e}")