import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from dqn import CNN_DQN
from variables import *
from breaker import BreakoutEnv
import matplotlib.pyplot as plt
import cv2


class DDQNAgent:
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, buffer_size=6000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def select_action(self, state):
        state = state.to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, frames, action, reward, next_frames, done):
        frames = [f.to(self.device) for f in frames]
        next_frames = [f.to(self.device) for f in next_frames]
        self.memory.append((frames, action, reward, next_frames, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        st_2, st_1 = zip(*states)
        st1p, st2p = zip(*next_states)

        st_2 = torch.stack(st_2).squeeze(1).to(self.device)
        st_1 = torch.stack(st_1).squeeze(1).to(self.device)
        st1p = torch.stack(st1p).squeeze(1).to(self.device)
        st2p = torch.stack(st2p).squeeze(1).to(self.device)

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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return torch.tensor(frame, device=self.device)


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

            agent.store_experience([prev_frame_2, prev_frame_1], action, reward, [next_tensor_1, next_tensor_2], done)
            agent.train()

            steps += 1
            prev_frame_2 = next_tensor_1
            prev_frame_1 = next_tensor_2
            if done or dead:
                break

        if episode % 5 == 0:
            agent.update_target_network()

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        avg_score += env.score
        success_history.append(dead)
        if episode % 5 == 0:
            avg_score /= 5
            steps_per_episode.append(avg_score)
            avg_score = 0

        print(f"Run {run+1} | Episode {episode} | Steps: {steps} | Reward: {env.score} | Epsilon: {agent.epsilon:.3f}")

    all_rewards.append(steps_per_episode)

# ========== 統計處理 ==========

all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

episodes = np.arange(0, total_episodes, 5)

plt.figure(figsize=(10, 6))
plt.plot(episodes, mean_rewards, label="平均 Reward", color='blue')
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2, label="±1 標準差")
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 5 episodes)")
plt.title("DQN Reward Over 5 Runs")
plt.legend()
plt.grid(True)
plt.show()
