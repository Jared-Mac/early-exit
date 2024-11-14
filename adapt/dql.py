import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
import simpy
import torch.backends.cudnn as cudnn
if torch.cuda.is_available():
    cudnn.benchmark = True  # Enable cudnn auto-tuner

def to_tensor(x, device):
    """Efficiently convert numpy array to tensor on device"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)

def calculate_reward(action, accuracy, latency, block_num, battery_state):
    base_reward = 0
    
    # Reward for accuracy
    if accuracy:
        base_reward += 10
    else:
        base_reward -= 10
    
    # Penalty for latency
    base_reward -= latency
    
    # Reward/penalty based on action
    if action == 0:  # Continue to next block
        base_reward -= 1  # Small penalty for continuing
    elif action == 1:  # Exit without transmitting
        base_reward += 5  # Reward for saving transmission energy
    elif action == 2:  # Exit and transmit
        base_reward -= 2  # Small penalty for transmission energy use
    
    # Adjust reward based on battery state
    battery_soc = battery_state[0]
    if battery_soc < 0.2:  # If battery is low
        if action == 2:  # Penalize transmission more heavily
            base_reward -= 5
        elif action == 1:  # Reward energy-saving actions more
            base_reward += 5
    
    # Adjust reward based on block number
    if block_num < 3 and action != 0:
        base_reward -= (3 - block_num) * 2  # Penalize early exits
    
    return base_reward

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process state dictionaries efficiently
        processed_states = {
            'exit': torch.stack([to_tensor(s['exit'], self.device) for s in states]),
            'image': torch.stack([to_tensor(s['image'], self.device) for s in states]).squeeze(1),
            'logits': torch.stack([to_tensor(s['logits'], self.device) for s in states]),
            'battery': torch.stack([to_tensor(s['battery'], self.device) for s in states])
        }
        
        processed_next_states = {
            'exit': torch.stack([to_tensor(s['exit'], self.device) for s in next_states]),
            'image': torch.stack([to_tensor(s['image'], self.device) for s in next_states]).squeeze(1),
            'logits': torch.stack([to_tensor(s['logits'], self.device) for s in next_states]),
            'battery': torch.stack([to_tensor(s['battery'], self.device) for s in next_states])
        }
        return (processed_states,
                torch.tensor(actions, device=self.device, dtype=torch.long),
                torch.tensor(rewards, device=self.device, dtype=torch.float32),
                processed_next_states,
                torch.tensor(dones, device=self.device, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)
class DQNet(nn.Module):
    def __init__(self, num_logits, num_exit, num_actions, image_shape, battery_features):
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
        
        # Adjust the input size of fc2
        fc2_input_size = 512 + num_logits + num_exit + battery_features
        self.fc2 = nn.Linear(fc2_input_size, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layer for actions
        self.fc4 = nn.Linear(128, num_actions)
    
    def forward(self, exits, image, logits, battery_state):
        # Process image through convolutional layers
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Ensure all inputs have the correct dimensions
        logits = logits.view(logits.size(0), -1)
        exits = exits.view(exits.size(0), -1)
        battery_state = battery_state.view(battery_state.size(0), -1)

        # Concatenate the processed image with logits, exit, and battery state
        combined = torch.cat((x, logits, exits, battery_state), dim=1)
        # Process combined features
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        # Output actions
        actions = self.fc4(combined)

        return actions
class DQLAgent:
    def __init__(self, env, num_classes, num_exit, image_shape, battery_features, memory_size=100000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = 3
        self.model = DQNet(num_classes, num_exit, 3, image_shape, battery_features).to(self.device)
        self.target_model = DQNet(num_classes, num_exit, 3, image_shape, battery_features).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.env = env
        self.memory = ReplayBuffer(memory_size, self.device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_frequency = 10
        self.env.process(self.train())
        self.loss_history = deque(maxlen=100)  # Store recent losses
        self.convergence_threshold = 0.5  # Threshold for convergence
        self.min_iterations = 1000  # Minimum iterations before checking convergence
        self.converged = False
    def select_action(self, exits, image, logits, battery_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            # Batch all inputs at once
            inputs = {
                'exits': exits.unsqueeze(0),
                'image': image.unsqueeze(0),
                'logits': logits.unsqueeze(0),
                'battery_state': battery_state.unsqueeze(0)
            }
            # Move to device in one operation
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            q_values = self.model(inputs['exits'], inputs['image'], 
                                inputs['logits'], inputs['battery_state'])
            return torch.argmax(q_values, dim=1).item()

    def store_experience(self, experience):
        self.memory.add(experience)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # States and next_states are already processed by ReplayBuffer
        with torch.no_grad():
            next_q_values = self.target_model(
                next_states['exit'],
                next_states['image'],
                next_states['logits'],
                next_states['battery']
            ).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        current_q_values = self.model(
            states['exit'],
            states['image'],
            states['logits'],
            states['battery']
        ).gather(1, actions.unsqueeze(1))
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        loss = loss.item()
        self.loss_history.append(loss)
        
        return loss

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

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.model.state_dict(),
            'target_net_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

    def check_convergence(self):
        """Check if training has converged based on recent loss values."""
        if len(self.loss_history) < self.loss_history.maxlen:
            return False
            
        # Calculate average change in loss over recent iterations
        loss_changes = np.diff(list(self.loss_history))
        avg_change = np.abs(loss_changes).mean()
        
        return avg_change, self.convergence_threshold
