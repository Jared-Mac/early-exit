import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import simpy

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
    def __init__(self, num_logits, num_exit, num_actions, image_shape):
        super(DQNet, self).__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the flattened feature map
        conv_output_height = image_shape[1] // 4
        conv_output_width = image_shape[2] // 4
        self.conv_output_size = conv_output_height * conv_output_width * 64
        
        # Fully connected layer after convolutional layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        
        # Another fully connected layer to process logits and exit
        self.fc2 = nn.Linear(512 + num_logits + num_exit, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layer for actions
        self.fc4 = nn.Linear(128, num_actions)
    
    def forward(self, exits, image, logits):
        # Process image through convolutional layers
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Ensure logits and exits have the correct dimensions
        logits = logits.view(logits.size(0), -1)
        exits = exits.view(exits.size(0), -1)

        # Concatenate the processed image with logits and exit
        combined = torch.cat((x, logits, exits), dim=1)
        # Process combined features
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        # Output actions
        actions = self.fc4(combined)

        return actions
class DQLAgent:
    def __init__(self, env, num_classes, num_exit, image_shape, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = 3
        self.model = DQNet(num_classes, num_exit, 3, image_shape).to(self.device)  # Changed from 2 to 3
        self.target_model = DQNet(num_classes, num_exit, 3, image_shape).to(self.device)  # Changed from 2 to 3
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.env = env
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_frequency = 10
        self.env.process(self.train())
    def select_action(self, exits, image, logits):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
            logits = logits.unsqueeze(0).to(self.device)
            exits = exits.unsqueeze(0).to(self.device)
            q_values = self.model(exits, image, logits)
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, experience):
        self.memory.add(experience)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Extract images, logits, and exits from state dictionaries
        state_exits, state_images, state_logits = [], [], []
        next_state_exits, next_state_images, next_state_logits = [], [], []

        for state in states:
            state_exits.append(state['exit'])
            state_images.append(torch.tensor(state['image'], dtype=torch.float32).squeeze().to(self.device))
            state_logits.append(torch.tensor(state['logits'], dtype=torch.float32).to(self.device))

        for next_state in next_states:
            next_state_exits.append(next_state['exit'])
            next_state_images.append(torch.tensor(next_state['image'], dtype=torch.float32).squeeze().to(self.device))
            next_state_logits.append(torch.tensor(next_state['logits'], dtype=torch.float32).to(self.device))

        # Stack and permute to create batch tensors
        state_exits = torch.stack(state_exits).to(self.device)
        state_images = torch.stack(state_images).permute(0, 1, 2, 3).to(self.device)
        state_logits = torch.stack(state_logits).to(self.device)
        next_state_exits = torch.stack(next_state_exits).to(self.device)
        next_state_images = torch.stack(next_state_images).permute(0, 1, 2, 3).to(self.device)
        next_state_logits = torch.stack(next_state_logits).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.model(state_exits, state_images, state_logits).gather(1, actions)
        next_q_values = self.target_model(next_state_exits, next_state_images, next_state_logits).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def train(self):
        iteration = 0
        while True:
            self.update()
            if iteration % self.update_frequency == 0:
                self.update_target_model()
            iteration += 1
            yield self.env.timeout(1)