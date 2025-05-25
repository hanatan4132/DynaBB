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
#實驗多次求平均與變異數

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
        
        




class DDQNAgent:
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, buffer_size=6000, batch_size=64, virtual_batch_size=16, virtual_lr=0.00003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network = CNN_DQN((2, *input_shape[1:]), num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.real_memory = deque(maxlen=buffer_size)
        self.virtual_memory = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.virtual_optimizer = optim.Adam(self.q_network.parameters(), lr=virtual_lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

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

    def fake_exp(self,prev_frame_2, prev_frame_1):
        state_input = torch.cat((prev_frame_2, prev_frame_1), dim=1)
        #print(state_input.shape)
        action = random.randint(0, 2)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_st1p, pred_st2p = dyna.model(prev_frame_2, prev_frame_1, action_tensor)
            predicted_reward = rewardPredict.model(state_input, action_tensor).item()

        self.store_experience([prev_frame_2.squeeze(0), prev_frame_1.squeeze(0)], action, predicted_reward, [pred_st1p.squeeze(0), pred_st2p.squeeze(0)], False, use_virtual=True)
        return pred_st1p, pred_st2p
    def train(self,virtual_lr=0.00005 ,use_virtual=False):
        memory = self.virtual_memory if use_virtual else self.real_memory
        batch_size = self.virtual_batch_size if use_virtual else self.batch_size
        optimizer = self.virtual_optimizer if use_virtual else self.optimizer

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


EPSILON_DECAY_STEPS = 800000
steps = 0
total_episodes = 1000
model_path = r"maze_models/best_model_manual.pth"
reward_model_path = "reward_models/best_reward_model.pth"
dyna = Dyna(model_path)
env = BreakoutEnv()
rewardPredict = RewardPredict(reward_model_path)
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
    
    for step in range(4000):
        current_frame = env._get_frame()
        #cv2.imshow("Breakout Frame", current_frame)
        #cv2.waitKey(1)

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

        agent.store_experience([prev_frame_2, prev_frame_1], action, reward, [next_tensor_1, next_tensor_2], done, use_virtual=False)
        agent.train(use_virtual=False)

        
        if step % 15 == 0:
            with torch.no_grad():
                action_tensor = torch.tensor([action], device=dyna.device)
                pred_st1p, pred_st2p = dyna.model(prev_frame_2, prev_frame_1, action_tensor)
            pred_st1p, pred_st2p = agent.fake_exp(pred_st1p, pred_st2p)
        else:
            pred_st1p, pred_st2p = agent.fake_exp(pred_st1p, pred_st2p)

        if step % 20 == 0:
            agent.train(virtual_lr=0.00003,use_virtual=True)

        steps += 1
        prev_frame_2 = next_tensor_1  # t+1 → 現在當 t-2
        prev_frame_1 = next_tensor_2
        if done :
            goal += 1
            agent.epsilon *= 0.95
            break
        if env.dead:
            break
    print(virtual_lr)
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

    if episode % 10 == 1:
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

plt.plot(steps_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Steps per Episode over Training")
plt.show()

