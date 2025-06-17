import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt
from dqn import CNN_DQN
from breaker import BreakoutEnv
from variables import *

# =============================
# Prioritized Replay Buffer
# =============================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.position = 0

    def push(self, transition, td_error, bellman_error, w1=0.5, w2=0.5):
        priority = (w1 * abs(td_error) + w2 * abs(bellman_error) + 1e-6)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors, bellman_errors, w1=0.5, w2=0.5):
        for i, td, be in zip(indices, td_errors, bellman_errors):
            self.priorities[i] = w1 * abs(td) + w2 * abs(be) + 1e-6

    def __len__(self):
        return len(self.buffer)

# =============================
# DDQN Agent
# =============================
class DDQNAgent:
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, buffer_size=10000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        # 修正初始權重為 0.5, 0.5
        self.w1 = 0.2
        self.w2 = 0.8
        self.update_interval = 10

    def select_action(self, state):
        state = state.to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, frames, action, reward, next_frames, done):
        st_2, st_1 = [f.to(self.device) for f in frames]
        st1p, st2p = [f.to(self.device) for f in next_frames]

        state = torch.cat((st_2, st_1), dim=1)

        next_state = torch.cat((st1p, st2p), dim=1)

        with torch.no_grad():
            q_sa = self.q_network(state)[0, action].item()
            td_target = reward + (1 - done) * self.gamma * torch.max(self.target_network(next_state)).item()
            td_error = td_target - q_sa
            probs = torch.ones(self.q_network(next_state).shape[-1], device=self.device)
            probs = probs / probs.sum()
            bellman_expect = (self.q_network(next_state).squeeze() * probs).sum().item()
            bellman_error = reward + self.gamma * bellman_expect - q_sa

        transition = (frames, action, reward, next_frames, done)
        self.replay_buffer.push(transition, td_error, bellman_error, self.w1, self.w2)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        st_2, st_1 = zip(*states)
        st1p, st2p = zip(*next_states)

        st_2 = torch.stack(st_2).squeeze(1).to(self.device)
        st_1 = torch.stack(st_1).squeeze(1).to(self.device)
        st1p = torch.stack(st1p).squeeze(1).to(self.device)
        st2p = torch.stack(st2p).squeeze(1).to(self.device)

        state_input = torch.cat((st_2, st_1), dim=1)
        next_input = torch.cat((st1p, st2p), dim=1)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = weights.unsqueeze(1).to(self.device)

        q_values = self.q_network(state_input).gather(1, actions)
        next_actions = self.q_network(next_input).argmax(dim=1, keepdim=True)
        next_q_values = self.target_network(next_input).gather(1, next_actions)

        td_target = rewards + (1 - dones) * self.gamma * next_q_values
        td_error = td_target - q_values
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            probs = torch.ones_like(self.q_network(next_input)).to(self.device)
            probs = probs / probs.sum(dim=1, keepdim=True)
            bellman_expect = (self.q_network(next_input) * probs).sum(dim=1, keepdim=True)
            bellman_errors = (rewards + self.gamma * bellman_expect - q_values).squeeze().tolist()

        self.replay_buffer.update_priorities(indices, td_error.detach().squeeze().tolist(), bellman_errors, self.w1, self.w2)

    def update_eet_weights(self, zeta=1.2, lr=0.003):
        ratio = self.w1 / (self.w2 + 1e-8)
        dw1 = (zeta - ratio) * (-2 / (self.w1 + self.w2 + 1e-8) ** 2)
        dw2 = -dw1
        self.w1 = max(0.0, self.w1 - lr * dw1)
        self.w2 = max(0.0, self.w2 - lr * dw2)
        total = self.w1 + self.w2
        self.w1 /= total
        self.w2 /= total
        print("self.w1",self.w1,self.w2)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return torch.tensor(frame, device=self.device)

# =============================
# Training Loop
# =============================

num_runs = 5
total_episodes = 500
all_rewards = []

for run in range(num_runs):
    print(f"\n=== 開始第 {run+1} 次訓練 ===")
    env = BreakoutEnv()
    agent = DDQNAgent(input_shape=(2, screen_height, screen_width), num_actions=3)
    steps_per_episode = []
    avg_score = 0
    success_history = deque(maxlen=10)

    for episode in range(total_episodes):
        env.reset()
        prev_frame_2 = agent.preprocess_frame(env._get_frame())
        prev_frame_1 = agent.preprocess_frame(env._get_frame())
        steps = 0

        for step in range(4000):
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

            agent.store_experience([prev_frame_2, prev_frame_1], action, reward,
                                   [next_tensor_1, next_tensor_2], done)
            agent.train()

            steps += 1
            prev_frame_2 = next_tensor_1
            prev_frame_1 = next_tensor_2
            if done or dead:
                break

        if episode % 5 == 0:
            agent.update_target_network()

        # 確保每個更新間隔都調用 update_eet_weights
        if episode % agent.update_interval == 0:
            agent.update_eet_weights()

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        avg_score += env.score
        success_history.append(dead)

        if episode % 5 == 0:
            avg_score /= 5
            steps_per_episode.append(avg_score)
            avg_score = 0

        # 加入權重信息的打印，方便追踪
        print(f"Run {run+1} | Episode {episode} | Steps: {steps} | Reward: {env.score} | Done: {done} | Epsilon: {agent.epsilon:.4f} | w1: {agent.w1:.4f} | w2: {agent.w2:.4f}")

    all_rewards.append(steps_per_episode)

# ========== 統計與圖表顯示 ==========
all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

episodes = np.arange(0, total_episodes, 5)

plt.figure(figsize=(10, 6))
plt.plot(episodes, mean_rewards, label="平均 Reward", color='blue')
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                 color='blue', alpha=0.2, label="±1 標準差")
plt.xlabel("Episode")
plt.ylabel("Average Reward (每5集)")
plt.title("DQN Reward over 5 Runs")
plt.legend()
plt.grid(True)
plt.show()