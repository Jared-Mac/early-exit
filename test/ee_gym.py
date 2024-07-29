import gymnasium as gym
from gymnasium import spaces
import numpy as np
 # This would simulate your camera

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from collections import deque
import random
import logging 

from early_exit_resnet import *

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
    def __init__(self, num_logits, num_actions):
        super(DQNet, self).__init__()
        # Convolutional layers for CIFAR-10 image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer after convolutional layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        
        # Another fully connected layer to process logits
        self.fc2 = nn.Linear(num_logits + 512, 256)
        
        # Output layer for actions
        self.fc3 = nn.Linear(256, num_actions)
    
    def forward(self, image, logits):
        # Process CIFAR-10 image through convolutional layers
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output from conv layers
        x = x.view(-1, 64 * 8 * 8)

        x = F.relu(self.fc1(x))
        logits = logits.reshape(x.size(0),-1)
        
        # Concatenate the processed image with logits
        combined = torch.cat((x, logits), dim=1)
        
        # Process combined features
        combined = F.relu(self.fc2(combined))
        
        # Output actions
        actions = self.fc3(combined)
        
        return actions
    # def __init__(self, input_shape, num_classes):
    #     super(DQNet, self).__init__()
    #     self.input_shape = input_shape
    #     self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
    #     self.relu = nn.ReLU()
    #     self.fc1 = nn.Linear(32 * input_shape[0], 128)
    #     self.fc2 = nn.Linear(32 * 3 * 128 + num_classes , 2)  # Two actions: 0 or 1


    # def forward(self, x, logits):
    #     x = self.relu(self.conv1(x))
    #     # x = x.reshape(x.size(0), -1)  # Flatten
    #     x = self.relu(self.fc1(x))
    #     x = x.reshape(x.size(0),-1)  # Flatten
    #     logits = logits.reshape(x.size(0),-1)  # Flatten

    #     combined = torch.cat((x, logits), dim=1)  # Concatenate along the feature dimension
    #     x = self.fc2(combined)
    #     return x
    
class DQLAgent:
    def __init__(self, input_shape, num_classes, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNet(num_classes, 2).to(self.device)
        self.target_model = DQNet(num_classes, 2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, image, logits):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            image = torch.tensor(image, dtype=torch.float32).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float32).to(self.device)
            q_values = self.model(image, logits)
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, experience):
        self.memory.add(experience)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Extract images and logits from state dictionaries
        state_images = []
        state_logits = []
        next_state_images = []
        next_state_logits = []
        
        for state in states:
            state_img = torch.tensor(state['image'], dtype=torch.float32).squeeze().to(self.device)
            state_logit = torch.tensor(state['logits'], dtype=torch.float32).to(self.device)
            state_images.append(state_img)
            state_logits.append(state_logit)

        for next_state in next_states:
            next_state_img = torch.tensor(next_state['image'], dtype=torch.float32).squeeze().to(self.device)
            next_state_logit = torch.tensor(next_state['logits'], dtype=torch.float32).to(self.device)
            next_state_images.append(next_state_img)
            next_state_logits.append(next_state_logit)

        # Stack and permute to create batch tensors
        state_images = torch.stack(state_images).permute(0, 1, 2, 3).to(self.device)
        state_logits = torch.stack(state_logits).squeeze(0).to(self.device)
        next_state_images = torch.stack(next_state_images).permute(0, 1, 2, 3).to(self.device)
        next_state_logits = torch.stack(next_state_logits).squeeze(0).to(self.device)

        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.model(state_images, state_logits).gather(1, actions)
        next_q_values = self.target_model(next_state_images, next_state_logits).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
class EarlyExitEnv(gym.Env):
    def __init__(self, dataloader,models, num_classes=10):
        super(EarlyExitEnv, self).__init__()
        
        self.dataloader = dataloader
        self.models = models
        self.num_models = len(models)
        self.num_classes = num_classes
        self.max_steps = self.num_models
        self.iterator = iter(self.dataloader)
        
        # Image shape is determined by the first data point
        first_sample = next(self.iterator)
        self.image_shape = first_sample[0].shape
        # Action space: Decide to exit early or not
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Logits from early exit and the image
        self.observation_space = spaces.Dict({
            'logits': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_classes,), dtype=np.float32),
            'image': spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.float32),  # Values between 0 and 1 from ToTensor
        })
        
        self.steps = 0
        self.image = None
        self.logits = None
        self.label = None
        self.model_index = 0
        self.feature = None

        # Set up logging
        logging.basicConfig(filename='early_exit_env.log', level=logging.INFO)
        logging.info('Environment initialized')
        

    def step(self, action):
        assert self.action_space.contains(action)
        
        print(self.steps)
        # Exit Action & Send Data
        if action == 1 or self.model_index == self.max_steps - 1:
            pred_class = torch.argmax(self.logits).item()
            accuracy = (pred_class == self.label)
            reward = 10 if accuracy else -10
            reward += (self.max_steps - self.steps)  # Encourage earlier exits
            self.done = True
            logging.info(f'Step: {self.steps}, Model Index: {self.model_index}, Action: {action}, Prediction: {pred_class}, True Class: {self.label}, Reward: {reward}')
        # Continue Action
        else:
            self.model_index += 1
            self.logits = self._get_logits()
            reward = 0
            logging.info(f'Step: {self.steps}, Model Index: {self.model_index}, Action: {action}, Continuing...')
        self.steps += 1
        
        return {'logits': self.logits.detach().numpy(), 'image': self.image.numpy()}, reward, self.done, {}

    def reset(self):
        self.steps = 0
        self.done = False
        self.image, self.label = self._get_next_data()
        self.feature = self.image
        self.model_index = 0
        self.logits = self._get_logits()
        logging.info('Environment reset')
        return {'logits': self.logits.detach().numpy(), 'image': self.image.numpy()}
    

    def _get_next_data(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data[0], data[1].item()
    
    def _get_logits(self):
        out = self.models[self.model_index](self.feature)
        if self.model_index < 3:
            self.feature, self.logits = out[0], out[1]      
        else:
            self.logits = out
        return self.logits
    
    def render():
        pass


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),std=1),
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

    # Instantiate models
    block1 = HeadNetworkPart1(block=Bottleneck, in_planes=64, num_blocks=[3], num_classes=10)
    block2 = HeadNetworkPart2(Bottleneck, 256, [4], num_classes=10)
    block3 = HeadNetworkPart3(block=Bottleneck, in_planes=512, num_blocks=[6], num_classes=10)
    block4 = TailNetwork(block=Bottleneck, in_planes=1024, num_blocks=[3, 4, 6, 3], num_classes=10)

    # Load weights
    block1.load_state_dict(torch.load("models/cifar10/head1_resnet50.pth"))
    block2.load_state_dict(torch.load("models/cifar10/head2_resnet50.pth"))
    block3.load_state_dict(torch.load("models/cifar10/head3_resnet50.pth"))
    block4.load_state_dict(torch.load("models/cifar10/tail_resnet50.pth"))

    models = [block1, block2, block3, block4]

    # Create gym environment
    env = EarlyExitEnv(dataloader=dataloader, models=models)

    # Initialize DQL agent
    agent = DQLAgent(input_shape=env.image_shape, num_classes=env.num_classes)

    # Hyperparameters
    num_episodes = 5000
    target_update_freq = 10


    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        for t in range(env.max_steps):
            
            action = agent.select_action(state['image'], state['logits'])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.store_experience((state, action, reward, next_state, done))
            agent.update()
            state = next_state

            if done:
                break


        # Update the target model occasionally
        if episode % target_update_freq == 0:
            agent.update_target_model()

        print(f'Episode {episode}, Total Reward: {total_reward}')
        logging.info(f'Episode {episode}, Total Reward: {total_reward}')

