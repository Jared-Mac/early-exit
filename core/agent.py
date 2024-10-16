import simpy
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, num_logits, num_exit, num_actions):
        super(DQNet, self).__init__()
        # Convolutional layers for CIFAR-10 image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer after convolutional layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        
        # Another fully connected layer to process logits and exit
        self.fc2 = nn.Linear(num_logits + num_exit + 512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layer for actions
        self.fc4 = nn.Linear(128, num_actions)
    
    def forward(self, exits, image, logits):
        # Process CIFAR-10 image through convolutional layers
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
    def __init__(self, input_shape, num_classes, num_exit, memory_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNet(num_classes, num_exit, 2).to(self.device)  # Adjust input dimensions for the model
        self.target_model = DQNet(num_classes, num_exit, 2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, exits, image, logits):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(0, 1, 2, 3).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float32).unsqueeze(0).to(self.device)
            exits = torch.tensor(exits, dtype=torch.float32).unsqueeze(0).to(self.device)
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
    
class SimpyAgent:
    def __init__(self, env, rl_agent, update_frequency=5, timeout=1):
        self.env = env
        self.rl_agent = rl_agent
        self.update_frequency = update_frequency
        self.timeout = timeout
        # Start the training process
        self.env.process(self.train())

    def step(self, state):
        action = self.rl_agent.select_action(state['exit'], state['image'], state['logits'])
        next_state, reward, done = self.env.step(action)  # Replace with your environment's step function
        experience = (state, action, reward, next_state, done)
        self.rl_agent.store_experience(experience)
        return next_state, reward, done

    def train(self):
        iteration = 0
        while True:
            self.rl_agent.update()
            if iteration % self.update_frequency == 0:
                self.rl_agent.update_target_model()
            iteration += 1
            yield self.env.timeout(self.timeout)  

if __name__ == "__main__":
    print("here")
    env = simpy.Environment()
    rl_agent = DQLAgent(input_shape=(3, 32, 32), num_classes=10, num_exit=1)
    agent = SimpyAgent(env, rl_agent)
    env.run()