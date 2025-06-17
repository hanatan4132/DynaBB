import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from dqn import CNN_DQN
from variables import *
from encoder import Autoencoder
import matplotlib.pyplot as plt
from breaker import BreakoutEnv
import cv2
from reward_network import RewardPredictor
import os
import json
import time
class Dyna:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

class RewardPredict:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RewardPredictor(input_shape=(2, 192, 192), num_actions=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

def calculate_learning_rate(episode, start_lr=0.0001, min_lr=0.00001, decay_period=200):
    decay_factor = max(0, 1 - episode / decay_period)
    current_lr = max(min_lr, start_lr * decay_factor + min_lr)
    return current_lr

def calculate_planning_steps(episode, start_steps=15, min_steps=5, decay_period=200):
    decay_factor = max(0, 1 - episode / decay_period)
    return max(min_steps, int(start_steps * decay_factor) + min_steps)

class DDQNAgent:
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, buffer_size=6000, batch_size=64, virtual_batch_size=32, virtual_lr=0.00003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.real_memory = deque(maxlen=buffer_size)
        self.virtual_memory = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.virtual_optimizer = optim.Adam(self.q_network.parameters(), lr=virtual_lr)
        print(f"Initialized with real lr={lr}, virtual lr={virtual_lr}")

        self.gamma = gamma
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.05

    def select_action(self, state):
        state = state.to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values, dim=1).item()

    def store_experience(self, frames, action, reward, next_frames, done, use_virtual=False):
        frames = [f.to(self.device) for f in frames]
        next_frames = [f.to(self.device) for f in next_frames]
        memory = self.virtual_memory if use_virtual else self.real_memory
        memory.append((frames, action, reward, next_frames, done))

    def fake_exp(self, prev_frame_2, prev_frame_1):
        state_input = torch.cat((prev_frame_2, prev_frame_1), dim=1)
        action = random.randint(0, 2)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_st1p, pred_st2p = dyna.model(prev_frame_2, prev_frame_1, action_tensor)
            predicted_reward = rewardPredict.model(state_input, action_tensor).item()

        self.store_experience([prev_frame_2.squeeze(0), prev_frame_1.squeeze(0)], action, predicted_reward, [pred_st1p.squeeze(0), pred_st2p.squeeze(0)], False, use_virtual=True)
        return pred_st1p, pred_st2p
        
    def train(self, virtual_lr=0.00005, use_virtual=False):
        memory = self.virtual_memory if use_virtual else self.real_memory
        batch_size = self.virtual_batch_size if use_virtual else self.batch_size
        optimizer = self.virtual_optimizer if use_virtual else self.optimizer
        
        # 更新虚拟优化器的学习率
        if use_virtual and virtual_lr is not None:
            for param_group in self.virtual_optimizer.param_groups:
                param_group['lr'] = virtual_lr
            
            # 打印当前虚拟优化器的学习率
            current_lr = self.virtual_optimizer.param_groups[0]['lr']
            #print(f"Current virtual optimizer LR: {current_lr}")

        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        st_2, st_1 = zip(*states)
        st1p, st2p = zip(*next_states)
        if not use_virtual:
            st_2 = torch.stack(st_2).squeeze(1).to(self.device)
            st_1 = torch.stack(st_1).squeeze(1).to(self.device)
            st1p = torch.stack(st1p).squeeze(1).to(self.device)
            st2p = torch.stack(st2p).squeeze(1).to(self.device)
        else:
            st_2 = torch.stack(st_2).to(self.device)
            st_1 = torch.stack(st_1).to(self.device)
            st1p = torch.stack(st1p).to(self.device)
            st2p = torch.stack(st2p).to(self.device)

        state_input = torch.cat((st_2, st_1), dim=1)
        next_input = torch.cat((st1p, st2p), dim=1)

        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_network(state_input).gather(1, actions)
        next_actions = torch.argmax(self.q_network(next_input), dim=1, keepdim=True)
        next_q_values = self.target_network(next_input).gather(1, next_actions)

        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return torch.tensor(frame, device=self.device)

def run_experiment():
    EPSILON_DECAY_STEPS = 800000
    steps = 0
    total_episodes = 500
    virtual_lr_value = 0.00005
    MAX_TREE_DEPTH = 15  # 最大深度是15层
    
    agent = DDQNAgent(input_shape=(2, screen_height, screen_width), num_actions=3)

    steps_per_episode = []
    total_steps = 0
    successes = 0
    success_history = deque(maxlen=10)
    goal = 0
    avg_score = 0

    for episode in range(total_episodes):
        state = env.reset()
        prev_frame_2 = agent.preprocess_frame(env._get_frame())
        prev_frame_1 = agent.preprocess_frame(env._get_frame())
        steps = 0
        
        # 初始化深度-广度预测树
        prediction_tree = [None] * MAX_TREE_DEPTH  # 15个位置的列表
        current_position = 0  # 当前在树中的位置
        
        for step in range(4000):
            current_frame = env._get_frame()

            state_input = torch.cat((prev_frame_2, prev_frame_1), dim=1)
            action = agent.select_action(state_input)

            next_frame_1, reward, done, dead = env.step(action)
            next_tensor_1 = agent.preprocess_frame(next_frame_1)
            prev_frame_2 = prev_frame_1
            prev_frame_1 = next_tensor_1
            state_input = torch.cat((prev_frame_2, prev_frame_1), dim=1)
            action = agent.select_action(state_input)
            next_frame_2, _, _, _ = env.step(action)
            next_tensor_2 = agent.preprocess_frame(next_frame_2)

            # 存储真实经验
            agent.store_experience([prev_frame_2, prev_frame_1], action, reward, [next_tensor_1, next_tensor_2], done, use_virtual=False)
            agent.train(use_virtual=False)
            
            # 深度与广度融合的预测树策略
            virtual_lr = calculate_learning_rate(episode)
            
            # 用真实画面更新预测树的当前位置
            prediction_tree[current_position] = (prev_frame_2, prev_frame_1, next_tensor_1, next_tensor_2)
            
            # 更新所有已存在的预测路径
            with torch.no_grad():
                for idx in range(len(prediction_tree)):
                    if prediction_tree[idx] is not None:
                        # 对每个预测路径进行更新预测
                        p_prev_frame_2, p_prev_frame_1, _, _ = prediction_tree[idx]
                        
                        
                        # 预测下一个状态
                        pred_st1p, pred_st2p = agent.fake_exp(p_prev_frame_2, p_prev_frame_1)
                        
                        # 更新预测树中的路径
                        prediction_tree[idx] = (pred_st1p, pred_st2p, None, None)
                        
                        # 将预测经验加入经验池
                        
            
            # 更新当前位置，循环使用预测树
            current_position = (current_position + 1) % MAX_TREE_DEPTH
            
            # 每20步训练一次虚拟经验
            if step % 20 == 0:
                agent.train(virtual_lr, use_virtual=True)

            steps += 1
            prev_frame_2 = next_tensor_1
            prev_frame_1 = next_tensor_2
            if done:
                goal += 1
                agent.epsilon *= 0.95
                break
            if env.dead:
                break
        
        if episode % 5 == 0:
            agent.update_target_network()

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        avg_score += env.score
        success_history.append(dead)
        success_rate = sum(success_history) / len(success_history)
        if success_rate < 0.2:
            agent.epsilon = max(agent.epsilon * 0.9, agent.epsilon_min)

        if episode % 5 == 0:
            avg_score /= 5
            steps_per_episode.append(avg_score)
            avg_score = 0

        if episode % 499 == 1:
            if 'pred_st1p' in locals():
                pred_st1_np = pred_st1p.cpu().detach().squeeze().numpy()
                pred_st2_np = pred_st2p.cpu().detach().squeeze().numpy()
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

            plt.plot(steps_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("Steps per Episode over Training")
            plt.show()

        print(f"Episode {episode} | Steps: {steps} | Action: {action if 'action' in locals() else 'N/A'} | Reward: {env.score} | Done: {done if 'done' in locals() else False} | Epsilon: {agent.epsilon}")

    return steps_per_episode
all_runs = []
model_path = r"maze_models/best_model_manual.pth"
reward_model_path = "reward_models/best_reward_model.pth"
rewardPredict = RewardPredict(reward_model_path)
dyna = Dyna(model_path)
env = BreakoutEnv()
for run in range(5):
    print(f"\n⚙️ 第 {run + 1} 次實驗開始...")
    steps_per_episode = run_experiment()  # 你需將原訓練流程包裝成此函式
    all_runs.append(steps_per_episode)

# 讓所有長度一致
min_len = min(len(x) for x in all_runs)
all_runs = [x[:min_len] for x in all_runs]

all_runs = np.array(all_runs)
mean_scores = np.mean(all_runs, axis=0)
std_scores = np.std(all_runs, axis=0)

episodes = np.arange(min_len)

# 繪圖
plt.figure(figsize=(10, 5))
plt.plot(episodes, mean_scores, label='Average Reward')
plt.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, alpha=0.3, label='Std Dev')
plt.title("Average Performance over 5 Runs")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.show()