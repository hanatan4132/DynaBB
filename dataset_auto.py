import numpy as np
import cv2
import random
from breaker import BreakoutEnv
from variables import screen_width, screen_height, fps
import pygame
from pygame.locals import QUIT

# 初始化打磚塊遊戲環境
env = BreakoutEnv()

def get_frame(env):
    frame = env._get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (screen_width, screen_height))
    return frame.astype(np.uint8)

def press_space_to_start(env):
    while env.game_state != "playing":
        env.step(0)
        cv2.imshow("Breakout Frame", get_frame(env))
        cv2.waitKey(1)

def collect_data_with_random_strategy(env, target_steps):
    press_space_to_start(env)
    prev_frame_1 = get_frame(env)
    prev_frame_2 = get_frame(env)

    st_pairs = []
    actions = []
    next_pairs = []
    rewards = []

    collected_count = 0
    attempted_steps = 0

    strategies = [
        "追蹤中心", "追蹤左側", "追蹤右側", "追蹤左極端", "追蹤右極端", "隨機位置", "隨機位置2", "隨機位置3", "隨機位置4", "隨機位置5", "隨機位置6"
    ]

    current_strategy = random.choice(strategies)
    next_strategy_change = 0
    random_target_x = None

    while collected_count < target_steps:
        env.clock.tick(fps)
        #current_frame = get_frame(env)

        for event in pygame.event.get():
            if event.type == QUIT:
                env.close()
                return st_pairs, actions, next_pairs, rewards

        ball_x = env.ball.rect.centerx
        paddle_x = env.player_paddle.rect.centerx
        paddle_width = env.player_paddle.width

        if attempted_steps >= next_strategy_change or env.dead:
            current_strategy = random.choice(strategies)
            next_strategy_change = attempted_steps + random.randint(50, 100)
            random_target_x = random.randint(paddle_width//2, screen_width - paddle_width//2)
            print(f"⚡ 策略改變: {current_strategy}")

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
        elif current_strategy == "隨機位置3":
            target_x = random_target_x
        elif current_strategy == "隨機位置4":
            target_x = random_target_x
        elif current_strategy == "隨機位置5":
            target_x = random_target_x
        elif current_strategy == "隨機位置6":
            target_x = random_target_x

        target_x = max(paddle_width//2, min(target_x, screen_width - paddle_width//2))

        if abs(target_x - paddle_x) < 10:
            action = 0
        elif target_x < paddle_x:
            action = 1
        else:
            action = 2

        frame_st_plus_1, reward, done, dead = env.step(action)
        attempted_steps += 1

        frame_st_plus_1 = cv2.cvtColor(frame_st_plus_1, cv2.COLOR_RGB2GRAY)
        frame_st_plus_1 = cv2.resize(frame_st_plus_1, (screen_width, screen_height))

        frame_st_plus_2, reward, done, dead = env.step(action)  # 接下來一幀，不做動作
        frame_st_plus_2 = cv2.cvtColor(frame_st_plus_2, cv2.COLOR_RGB2GRAY)
        frame_st_plus_2 = cv2.resize(frame_st_plus_2, (screen_width, screen_height))

        if reward != 0 or random.random() < 0.5:
            st_pairs.append(np.stack([prev_frame_2, prev_frame_1], axis=0))
            next_pairs.append(np.stack([frame_st_plus_1, frame_st_plus_2], axis=0))
            actions.append(action)
            rewards.append(reward)
            collected_count += 1

            if collected_count % 100 == 0:
                print(f"✅ 已收集 {collected_count}/{target_steps} 筆資料 | 嘗試次數: {attempted_steps} | 最新獎勵: {reward}")

        prev_frame_2 = frame_st_plus_1
        prev_frame_1 = frame_st_plus_2

        if done or dead:
            env.reset()
            prev_frame_1 = get_frame(env)
            prev_frame_2 = get_frame(env)
            press_space_to_start(env)

        frame_with_text = cv2.putText(
            frame_st_plus_2.copy(),
            f"Strategy: {current_strategy}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        cv2.imshow("Breakout Frame", frame_with_text)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    return st_pairs, actions, next_pairs, rewards

print("開始自動收集資料...")
st_pairs, actions, next_pairs, rewards = collect_data_with_random_strategy(env, 20000)

np.savez("breakout_ststpp_dataset.npz", 
         st_pairs=np.array(st_pairs),
         actions=np.array(actions),
         next_pairs=np.array(next_pairs),
         rewards=np.array(rewards))

print(f"🎉 資料集已儲存！共 {len(actions)} 筆資料")
print(f"獎勵統計：正獎勵: {np.sum(np.array(rewards) > 0)}, 零獎勵: {np.sum(np.array(rewards) == 0)}, 負獎勵: {np.sum(np.array(rewards) < 0)}")
