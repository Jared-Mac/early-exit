import gymnasium as gym
from gymnasium import spaces
import numpy as np
 # This would simulate your camera

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from collections import deque
import random
import logging 

from early_exit_resnet import *
from dataset import CacheDataset
import numpy as np

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
    
class EarlyExitEnv(gym.Env):
    def __init__(self, dataloader, num_classes=10):
        super(EarlyExitEnv, self).__init__()
        
        self.dataloader = dataloader
        # self.models = models
        self.num_models = 4
        self.num_exit = 4
        self.num_classes = num_classes
        self.max_steps = self.num_models
        self.iterator = iter(self.dataloader)
        
        # Image shape is determined by the first data point
        first_sample = next(self.iterator)
        self.image_shape = first_sample['image'].shape
        
        # Action space: Decide to exit early (1) or not (0)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Logits from early exit and the image
        self.observation_space = spaces.Dict({
            'logits': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_classes,), dtype=np.float32),
            'exit': spaces.Box(low=0, high=1, shape=(self.num_exit,), dtype=np.float32),  # One-hot vector with num_exit categories
            'image': spaces.Box(low=-1.0, high=1.0, shape=self.image_shape, dtype=np.float32),  # Normalized image values
        })
        
        self.steps = 0
        self.exit = torch.eye(self.num_exit)
        self.image = None
        self.logits = None
        self.label = None
        self.model_index = 0
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Set up logging
        logging.basicConfig(filename='early_exit_env.log', level=logging.INFO)
        logging.info('Environment initialized')
        

    def step(self, action):
        assert self.action_space.contains(action)
        self.logits = self.datum["logits"][self.model_index]
        self.label = self.datum["label"]
        # Exit & Check Logits for accuracy
        if action == 1 or self.model_index == self.max_steps - 1:
            pred_class = int(self.softmax(self.logits).argmax(dim=1))
            accuracy = 1 if (pred_class == self.label) else 0

            reward = 10 if accuracy else -10

            self.done = True

            logging.info(f'Step: {self.steps}, Model Index: {self.model_index}, Action: {action}, Prediction: {pred_class}, True Class: {self.label}, Reward: {reward}')
        
        # Continue and fetch logits from next block
        else:
            reward = -self.steps
            accuracy = None
            logging.info(f'Step: {self.steps}, Model Index: {self.model_index}, Action: {action}, Continuing...')

        self.steps += 1
        
        observation = {
            'exit': self.exit[self.model_index].clone().detach(),
            'logits': self.logits.clone().detach().numpy(),
            'image': self.datum["image"].clone().numpy()
        }
        self.model_index += 1
        return observation, reward, self.done, {'accuracy': accuracy}

    def reset(self):
        self.steps = 0
        self.done = False
        self._get_next_data()
        self.model_index = 0
        self.logits = self.datum["logits"][self.model_index] # Get initial logits from the first model
        self.feature = self.datum["image"]
        logging.info('Environment reset')
        return {
            'exit': self.exit[self.model_index].clone().detach(),
            'logits': self.logits.clone().detach().numpy(),
            'image': self.datum["image"].clone().numpy()
        }

    def _get_next_data(self):
        try:
            data = next(self.iterator)
            self.datum = data
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
            self.datum = data

    
    # def _get_logits(self):
    #     with torch.no_grad():
    #         logits = self.models[self.model_index](self.feature)
    #         if self.model_index == 0:
    #             self.feature, self.logits = logits # Unpacking tuple (features, logits)
    #         else:
    #             self.logits = logits

    #         self.model_index += 1
    #     return self.logits
    
    def render(self):
        pass
        
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    total_samples = len(test_loader.dataset)
    correct = 0 # Assuming three exits

    with torch.no_grad():  # No need to compute gradients
        for inputs, labels in test_loader:
            # print(inputs)
            exit = model(inputs)[1]  # Forward pass
            softmax = torch.nn.Softmax(dim=1)
            exit_soft = softmax(exit)
            predictions = exit_soft.argmax(dim=1)
            print(predictions)
            correct += 1 if (predictions == labels) else 0

    accuracy = correct / total_samples
    return accuracy
    
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)
    # dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

    # Instantiate models
    # block1 = HeadNetworkPart1(block=Bottleneck, in_planes=64, num_blocks=[3], num_classes=10)
    # block2 = HeadNetworkPart2(Bottleneck, 256, [4], num_classes=10)
    # block3 = HeadNetworkPart3(block=Bottleneck, in_planes=512, num_blocks=[6], num_classes=10)
    # block4 = TailNetwork(block=Bottleneck, in_planes=1024, num_blocks=[3], num_classes=10)
    # # Load weights
    # block1.load_state_dict(torch.load("models/cifar10/head1_resnet50.pth"))
    # block2.load_state_dict(torch.load("models/cifar10/head2_resnet50.pth"))
    # block3.load_state_dict(torch.load("models/cifar10/head3_resnet50.pth"))
    # block4.load_state_dict(torch.load("models/cifar10/tail_resnet50.pth"))

    # models = [block1, block2, block3, block4]

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    custom_test_set = CacheDataset(test_set, models=None,compute_logits=False)
    dataloader = DataLoader(custom_test_set, batch_size=1, shuffle=True, num_workers=1)
        
    # Create gym environment
    env = EarlyExitEnv(dataloader=dataloader)

    # Initialize DQL agent
    agent = DQLAgent(input_shape=env.image_shape, num_classes=env.num_classes,num_exit=(env.num_models))



    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)
    # dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

    # accuracy = evaluate_model(block1,dataloader, 'cuda')
    # print(accuracy)

    # Hyperparameters
    num_epochs = 100
    num_episodes = 1000
    target_update_freq = 50
    for epoch in range(num_epochs):
        epoch_reward = 0
        accuracy = 0
        total = 0.0
        count = 0
        correct_predictions = 0
        total_predictions = 0

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state['exit'], state['image'], state['logits'])
                next_state, reward, done, info = env.step(action)
                env.steps += 1
                agent.store_experience((state, action, reward, next_state, done))
                agent.update()
                state = next_state

                if info['accuracy'] is not None:
                    correct_predictions += 1 if info['accuracy'] else 0
                    total_predictions += 1
                epoch_reward += reward
            if episode % target_update_freq == 0:
                agent.update_target_model()
        epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Epoch {epoch}, Reward: {epoch_reward}, Accuracy: {epoch_accuracy}")
        logging.info(f"Epoch {epoch}, Epoch Reward: {epoch_reward}, Accuracy: {epoch_accuracy}")
 