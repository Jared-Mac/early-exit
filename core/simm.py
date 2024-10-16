import simpy
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from functools import partial

from dataset import CacheDataset
from ns.packet.packet import Packet
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire


# FAKE DATA FOR TESTING
packet_sizes = {
    'IMAGE': 32*32*3,
    'block0': 256*32*32,
    'block1': 512*16*16,
    'block2': 1024*8*8,
    'block3': 1024*8*8}
block_processing_times = {
    'block0': 0.01,
    'block1': 0.04,
    'block2': 0.06,
    'block3': 0.09}
AGENT_TIME = 0.001

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
class Camera:
    def __init__(self, env, data_loader, interval=1):
        self.env = env
        self.interval = interval
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)
        self.image_shape = next(self.iterator)['image'].shape
        self.out = None
        self.env.process(self.run())

    def run(self):
        while True:
            for item in self.data_loader:
                if self.out:
                    self.out.put(item)
                yield self.env.timeout(self.interval)

class Head:
    def __init__(self, env, element_id, agent, data_tracker, exits=4, debug=False):
        self.env = env
        self.element_id = element_id
        self.agent = agent
        self.exits = torch.eye(exits)
        self.out = None
        self.debug = debug
        self.packets_sent = 0
        self.env.process(self.process_data())
        self.store = simpy.Store(env)
        self.data_tracker = data_tracker
        self.start_times = {}
        self.data_counter = 0  # Add a counter for unique IDs
        self.latencies = {}  # Add this line to store latencies

    def put(self, data):
        return self.store.put(data)

    def process_item(self, data):
        self.data_counter += 1
        data_id = self.data_counter
        start_time = self.env.now
        self.start_times[data_id] = start_time
        cont = True
        block_num = 0
        while cont and block_num < 4:
            reward = 0

            # Run Neural Block
            yield self.env.timeout(block_processing_times[f'block{block_num}'])
            image = torch.tensor(data['image'], dtype=torch.float32)
            logits = torch.tensor(data['logits'][block_num], dtype=torch.float32)
            # Run Agent
            yield self.env.timeout(AGENT_TIME)
            action = self.agent.select_action(self.exits[block_num], image, logits)

            # Calculate latency for the current block
            current_latency = self.env.now - start_time

            # Process Action
            if action == 0:  # Continue to next block
                reward = - current_latency
                # Store Experience
                state = {
                    'exit': self.exits[block_num],
                    'image': data['image'],
                    'logits': data['logits'][block_num]
                }
                next_block_num = min(block_num + 1, 3)
                next_state = {
                    'exit': self.exits[next_block_num],
                    'image': data['image'],
                    'logits': data['logits'][next_block_num]
                }
                self.agent.store_experience((state, action, reward, next_state, False))

            elif action == 1:  # Exit without transmitting
                # Calculate reward and store experience
                accuracy = int(logits.argmax()) == int(data['label'])
                reward = 10 if accuracy else -100
                reward -= current_latency
                cont = False
                state = {
                    'exit': self.exits[block_num],
                    'image': data['image'],
                    'logits': data['logits'][block_num]
                }
                next_block_num = min(block_num + 1, 3)
                next_state = {
                    'exit': self.exits[next_block_num],
                    'image': data['image'],
                    'logits': data['logits'][next_block_num]
                }
                self.agent.store_experience((state, action, reward, next_state, True))
                self.data_tracker.update(accuracy, start_time, block_num, current_latency, action)
            elif action == 2:  # Exit and transmit  
                classification = logits.argmax().item()
                self.send_packet(data, block_num, action, current_latency)
                cont = False
                break
            
            # Update Block Number
            block_num += 1

    def send_packet(self, data, block_num, action, latency):
        packet = Packet(
            time=self.env.now,
            size=packet_sizes[f'block{block_num}'],
            packet_id=self.packets_sent,
            src=self.element_id,
            dst="tail",
            flow_id=0,
        )
        packet.data = data
        packet.block_num = block_num
        packet.action = action
        packet.start_time = self.start_times[self.data_counter]
        packet.latency = latency
        if self.out:
            self.out.put(packet)
        self.packets_sent += 1
        if self.debug:
            print(f"Sent packet {self.packets_sent} at time {self.env.now}")
    def process_data(self):
        while True:
            data = yield self.store.get()
            yield self.env.process(self.process_item(data))


class DataTracker:
    def __init__(self, env, window_size=500):
        self.env = env
        self.total_classifications = 0
        self.correct_classifications = 0
        self.window_size = window_size
        self.accuracy_window = deque(maxlen=window_size)
        self.latency_window = deque(maxlen=window_size)
        self.exit_window = deque(maxlen=window_size)
        self.action_window = deque(maxlen=window_size)
        self.cumulative_exit_numbers = {f'block{i}': 0 for i in range(4)}
        self.cumulative_action_numbers = {i: 0 for i in range(3)}  # 0: continue, 1: exit without transmit, 2: exit and transmit
        self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(500)
            print(f"Rolling accuracy: {self.get_rolling_accuracy():.2f}")
            if self.latency_window:
                print(f"Rolling average latency): {self.get_rolling_latency():.2f}")
            print(f"Rolling exit numbers): {self.get_rolling_exit_numbers()}")
            # print(f"Cumulative exit numbers: {self.cumulative_exit_numbers}")

    def update(self, accuracy, start_time, exit_number, latency, action):
        self.total_classifications += 1
        self.correct_classifications += accuracy
        self.accuracy_window.append(accuracy)
        self.latency_window.append(latency)
        self.exit_window.append(exit_number)
        self.action_window.append(action)
        self.cumulative_exit_numbers[f'block{exit_number}'] += 1
        self.cumulative_action_numbers[action] += 1

    def get_rolling_accuracy(self):
        if not self.accuracy_window:
            return 0
        return sum(self.accuracy_window) / len(self.accuracy_window)

    def get_rolling_latency(self):
        if not self.latency_window:
            return 0
        return sum(self.latency_window) / len(self.latency_window)

    def get_rolling_exit_numbers(self):
        if not self.exit_window:
            return {f'block{i}': 0 for i in range(4)}
        exit_counts = {f'block{i}': self.exit_window.count(i) for i in range(4)}
        return exit_counts

    def get_rolling_action_numbers(self):
        if not self.action_window:
            return {i: 0 for i in range(3)}
        action_counts = {i: self.action_window.count(i) for i in range(3)}
        return action_counts

    def get_current_accuracy(self):
        if self.total_classifications == 0:
            return 0
        return self.correct_classifications / self.total_classifications

    def get_accuracy_history(self):
        return list(self.accuracy_window)

    def get_latency_history(self):
        return list(self.latency_window)

    def get_exit_history(self):
        return list(self.exit_window)

    def get_cumulative_exit_numbers(self):
        return self.cumulative_exit_numbers

    def get_cumulative_action_numbers(self):
        return self.cumulative_action_numbers

class Tail:
    def __init__(self, env, agent, data_tracker, rec_waits=True, debug=False):
        self.env = env
        self.store = simpy.Store(env)
        self.rec_waits = rec_waits
        self.debug = debug
        self.received_classifications = []
        self.data_tracker = data_tracker
        self.exits = torch.eye(4)
        self.agent = agent
        self.env.process(self.run())
    def put(self, packet):
        return self.store.put(packet)

    def run(self):
        while True:
            packet = yield self.store.get()
            self.receive(packet)

    def receive(self, packet):
        if hasattr(packet, 'data'):
            data = packet.data
            block_num = packet.block_num
            action = packet.action
            start_time = packet.start_time
            state = {
                'exit': self.exits[block_num],
                'image': data['image'],
                'logits': data['logits'][block_num]
            }
            next_block_num = 3
            next_state = {
                'exit': self.exits[next_block_num],
                'image': data['image'],
                'logits': data['logits'][next_block_num]
            }
            # Calculate reward based on previous classification accuracy
            previous_logits = data['logits'][block_num]
            previous_prediction = torch.argmax(previous_logits).item()
            true_label = data['label']
            
            latency = packet.latency
            
            if previous_prediction == true_label:
                reward = -5 - latency
            else:
                reward = 10 - latency
            self.agent.store_experience((state, action, reward, next_state, True))
            final_prediction = torch.argmax(data['logits'][3]).item()
            accuracy = final_prediction == true_label
            self.data_tracker.update(accuracy, start_time, block_num, latency, action)
            if self.debug:
                print(f"Received packet with data at time {self.env.now}")
                print(f"Data type: {type(data)}")
                print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        else:
            if self.debug:
                print(f"Received packet without data attribute at time {self.env.now}")


class AlwaysTransmitHead(Head):
    def __init__(self, env, element_id, data_tracker, exits=4, debug=False):
        super().__init__(env, element_id, None, data_tracker, exits, debug)

    def process_item(self, data):
        self.data_counter += 1
        data_id = self.data_counter
        start_time = self.env.now
        self.start_times[data_id] = start_time

        # Always transmit after the first block
        self.send_packet(data, 0, 2, start_time)
    def process_data(self):
        while True:
            data = yield self.store.get()
            self.process_item(data)

class AlwaysTransmitTail(Tail):
    def __init__(self, env, data_tracker, rec_waits=True, debug=False):
        super().__init__(env, None, data_tracker, rec_waits, debug)

    def receive(self, packet):
        if hasattr(packet, 'data'):
            data = packet.data
            block_num = packet.block_num
            start_time = packet.start_time
            latency = self.env.now - packet.latency

            final_prediction = torch.argmax(data['logits'][3]).item()
            true_label = data['label']
            accuracy = final_prediction == true_label
            latency = self.env.now - start_time
            self.data_tracker.update(accuracy, start_time, block_num, latency, 2)

        if self.debug:
            print(f"Received packet at time {self.env.now}")

class EarlyExitHead(Head):
    def __init__(self, env, element_id, data_tracker, confidence_threshold=0.8, exits=4, debug=False):
        super().__init__(env, element_id, None, data_tracker, exits, debug)
        self.confidence_threshold = confidence_threshold

    def process_item(self, data):
        self.data_counter += 1
        data_id = self.data_counter
        start_time = self.env.now
        self.start_times[data_id] = start_time

        for block_num in range(4):
            # Run Neural Block
            yield self.env.timeout(block_processing_times[f'block{block_num}'])
            logits = torch.tensor(data['logits'][block_num], dtype=torch.float32)
            
            # Check if confidence exceeds threshold
            confidence = torch.softmax(logits, dim=0).max().item()
            if confidence >= self.confidence_threshold:
                latency = self.env.now - start_time
                self.send_packet(data, block_num, 2, latency)
                break

        # If we've gone through all blocks without exceeding threshold, send anyway
        if block_num == 3:
            latency = self.env.now - start_time
            self.send_packet(data, 3, 2, latency)

    def process_data(self):
        while True:
            data = yield self.store.get()
            yield self.env.process(self.process_item(data))

class EarlyExitTail(Tail):
    def __init__(self, env, data_tracker, rec_waits=True, debug=False):
        super().__init__(env, None, data_tracker, rec_waits, debug)

    def receive(self, packet):
        if hasattr(packet, 'data'):
            data = packet.data
            block_num = packet.block_num
            start_time = packet.start_time
            latency = packet.latency

            final_prediction = torch.argmax(data['logits'][block_num]).item()
            true_label = data['label']
            accuracy = final_prediction == true_label
            self.data_tracker.update(accuracy, start_time, block_num, latency, 2)

        if self.debug:
            print(f"Received packet at time {self.env.now}")

class Simulation:
    def __init__(self, strategy='dql'):
        self.env = simpy.Environment()
        self.debug = False
        self.cache_dataset = CacheDataset()
        self.camera = Camera(self.env, self.cache_dataset, 1)
        image_shape = self.cache_dataset.image_shape
        self.data_tracker = DataTracker(self.env)
        
        if strategy == 'dql':
            self.agent = DQLAgent(self.env, num_classes=10, num_exit=4, image_shape=image_shape)
            self.head = Head(self.env, "head", self.agent, self.data_tracker, debug=self.debug)
            self.tail = Tail(self.env, self.agent, self.data_tracker, debug=self.debug)
        elif strategy == 'always_transmit':
            self.head = AlwaysTransmitHead(self.env, "head", self.data_tracker, debug=self.debug)
            self.tail = AlwaysTransmitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_threshold=0.8, debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Wire setup
        self.wire1 = Wire(self.env, partial(random.gauss, 0.15, 0.02), wire_id=1, debug=self.debug)

        self.camera.out = self.head
        self.head.out = self.wire1
        self.wire1.out = self.tail
        
        self.stability_window = 2000
        self.stability_threshold = 0.03
        self.strategy = strategy
        self.evaluation_time = 1000  # Time to run evaluation after training
    
    def is_training_stable(self):
        accuracy_history = self.data_tracker.get_accuracy_history()
        if len(accuracy_history) < self.stability_window:
            print(f"Not enough data for stability check. Current window: {len(accuracy_history)}")
            return False
        
        recent_accuracy = accuracy_history[-self.stability_window:]
        stability = np.std(recent_accuracy) < self.stability_threshold
        print(f"Stability check: std={np.std(recent_accuracy):.4f}, threshold={self.stability_threshold}")
        return stability

    def start(self, max_sim_time=5000):
        self.env.run(until=max_sim_time)

        # Print final results for both strategies
        final_accuracy = self.data_tracker.get_current_accuracy()
        print(f"\nFinal results for {self.strategy.upper()} strategy:")
        print(f"Final accuracy: {final_accuracy:.2f}")
        print(f"Final rolling accuracy: {self.data_tracker.get_rolling_accuracy():.2f}")
        print(f"Final rolling latency: {self.data_tracker.get_rolling_latency():.2f}")
        print(f"Final rolling exit numbers: {self.data_tracker.get_rolling_exit_numbers()}")
        print(f"Cumulative exit numbers: {self.data_tracker.get_cumulative_exit_numbers()}")
        print(f"Cumulative action numbers: {self.data_tracker.get_cumulative_action_numbers()}")

        return {
            'accuracy': final_accuracy,
            'avg_latency': self.data_tracker.get_rolling_latency(),
            'exit_numbers': self.data_tracker.get_cumulative_exit_numbers(),
            'action_numbers': self.data_tracker.get_cumulative_action_numbers()
        }

if __name__ == "__main__":
    strategies = ['dql', 'always_transmit', 'early_exit']
    results = {}

    for strategy in strategies:
        print(f"\nRunning simulation with {strategy} strategy:")
        sim = Simulation(strategy)
        results[strategy] = sim.start(max_sim_time=5000)

    print("\nComparison of results:")
    for strategy, result in results.items():
        print(f"\n{strategy.upper()} Strategy:")
        if result:
            print(f"Accuracy: {result['accuracy']:.2f}")
            print(f"Average latency: {result['avg_latency']:.2f}")
            print(f"Exit numbers: {result['exit_numbers']}")
            print(f"Action numbers: {result['action_numbers']}")
        else:
            print("No evaluation data available")