import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import cv2  # This would simulate your camera

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import logging 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class DQNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DQNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * input_shape[0] * input_shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes * 2)  # Two actions: 0 or 1

    def forward(self, x, logits):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        logits = self.relu(torch.tensor(logits))
        x = torch.cat((x, logits), dim=1)
        x = self.fc2(x)
        return x
    
class DQLAgent:
    def __init__(self, input_shape, num_classes, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNet(input_shape, num_classes).to(self.device)
        self.target_model = DQNet(input_shape, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, logits):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state, logits)
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, experience):
        self.memory.add(experience)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        logits = torch.tensor([s['logits'] for s in states], dtype=torch.float32).to(self.device)
        images = torch.tensor([s['image'] for s in states], dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        next_logits = torch.tensor([ns['logits'] for ns in next_states], dtype=torch.float32).to(self.device)
        next_images = torch.tensor([ns['image'] for ns in next_states], dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.model(images, logits).gather(1, actions)
        next_q_values = self.target_model(next_images, next_logits).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
import logging

class EarlyExitEnv(gym.Env):
    def __init__(self, image_shape=(64, 64, 3), num_classes=10, max_steps=10000):
        super(EarlyExitEnv, self).__init__()
        
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_steps = max_steps
        
        # Action space: Decide to exit early or not
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Logits from early exit and the image
        self.observation_space = spaces.Dict({
            'logits': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_classes,), dtype=np.float32),
            'image': spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
        })
        
        self.steps = 0
        self.image = None
        self.logits = None
        self.done = False
        
        # Set up logging
        logging.basicConfig(filename='early_exit_env.log', level=logging.INFO)
        logging.info('Environment initialized')
        
        # Mock neural network (replace with your actual model)
        self.mock_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * self.image_shape[0] * self.image_shape[1], self.num_classes)
        )
    
    def reset(self):
        self.steps = 0
        self.done = False
        self.image = self._generate_image()
        self.logits = self._get_logits(self.image)
        logging.info('Environment reset')
        return {'logits': self.logits.detach().numpy(), 'image': self.image}
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        self.steps += 1
        
        if action == 1 or self.steps >= self.max_steps:
            self.done = True
            
            pred_class = torch.argmax(self.logits).item()
            true_class = self._get_true_label(self.image)
            
            accuracy = (pred_class == true_class)
            reward = 1 if accuracy else -1
            reward += (self.max_steps - self.steps)  # Encourage earlier exits
            
            logging.info(f'Step: {self.steps}, Action: {action}, Prediction: {pred_class}, True Class: {true_class}, Reward: {reward}')
        
        else:
            self.image = self._generate_image()
            self.logits = self._get_logits(self.image)
            reward = 0
            logging.info(f'Step: {self.steps}, Action: {action}, Continuing...')
        
        return {'logits': self.logits.detach().numpy(), 'image': self.image}, reward, self.done, {}

    def _generate_image(self):
        # Simulate capturing an image (replace with actual camera input)
        return np.random.randint(0, 256, self.image_shape, dtype=np.uint8)
    
    def _get_logits(self, image):
        # Preprocess and get logits from mock_model (replace with actual model call)
        image_torch = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) / 255.0
        logits = self.mock_model(image_torch).squeeze(0)
        return logits
    
    def _get_true_label(self, image):
        # Simulate generating a true class label
        return np.random.randint(0, self.num_classes)
    
    def render(self, mode='human'):
        # Visualization for the current state
        cv2.imshow('Image', self.image)
        
        # Create a white image for displaying logits
        logits_display = np.ones((300, 400, 3), dtype=np.uint8) * 255
        for i, logit in enumerate(self.logits.detach().numpy()):
            text = f'Class {i}: {logit:.2f}'
            cv2.putText(logits_display, text, (10, 30 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('Logits', logits_display)
        cv2.waitKey(1)
        
if __name__ == '__main__':
    env = EarlyExitEnv()
    state = env.reset()
    
    num_episodes = 500
    target_update_freq = 10
    agent = DQLAgent(env.image_shape, env.num_classes)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for t in range(env.max_steps):
            action = agent.select_action(state['image'], state['logits'])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            agent.store_experience((state, action, reward, next_state, done))
            agent.update()
            state = next_state
            
            if done:
                break
            
            env.render()
        
        # Update the target model occasionally
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        print(f'Episode {episode}, Total Reward: {total_reward}')
        logging.info(f'Episode {episode}, Total Reward: {total_reward}')
    
    env.close()
    cv2.destroyAllWindows()