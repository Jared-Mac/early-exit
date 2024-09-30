import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DQL Agent
class DQLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 32
        self.lr = 0.001
        self.replay_buffer = ReplayBuffer(1000)
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state])
                return self.q_network(state_tensor).argmax().item()
        else:
            return random.randint(0, self.action_dim - 1)

    def train(self):

        if len(self.replay_buffer) < self.batch_size:
            return
        size  = len(self.replay_buffer) if len(self.replay_buffer) < self.batch_size else self.batch_size
        batch = self.replay_buffer.sample(size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_values = self.q_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def simpy_process(simpy_env, gym_env, agent):
    obs = gym_env.reset()
    while True:
        yield simpy_env.timeout(1)
        action = agent.select_action(obs)
        next_obs, reward_deferred, done, _ = gym_env.step(action)
        reward = yield reward_deferred
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        agent.train()
        obs = next_obs
        gym_env.render()
        if done:
            obs = gym_env.reset()
