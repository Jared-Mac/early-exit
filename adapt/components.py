import simpy
import torch
from torch.nn import Softmax
from dql import calculate_reward
from ns.packet.packet import Packet
import yaml
from dataset import CIFAR10DataModule, CIFAR100DataModule, Flame2DataModule, TinyImageNetDataModule

with open('adapt/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Access parameters like this:
packet_sizes = config['packet_sizes']
model = config['model']
dataset = config['dataset']
block_processing_times = config['block_processing_times'][model]
block_cpu_flops = config['block_cpu_flops'][model]
agent_time = config['agent_time']
current_draw = config['current_draw'][model]
idle_current = config['current_draw']['idle']

AGENT_TIME = agent_time
softmax = Softmax(dim=0)

# Function to get the appropriate dataset
def get_dataset():
    if dataset == 'cifar10':
        return CIFAR10DataModule()
    elif dataset == 'cifar100':
        return CIFAR100DataModule()
    elif dataset == 'flame2':
        return Flame2DataModule()
    elif dataset == 'tiny-imagenet':
        return TinyImageNetDataModule()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

class Battery:
    def __init__(self, env):
        battery_config = config['battery']
        self.env = env
        self.charge = battery_config['initial_charge']
        self.capacity = battery_config['capacity']
        self.discharge_rate = battery_config['discharge_rate']
        self.voltage = battery_config['voltage']
        self.temperature = 25.0  # Celsius
        self.last_update_time = env.now  # Track last update timestamp

    def get_state(self):
        soc = self.charge / self.capacity
        return torch.tensor([soc], dtype=torch.float32)

    def update(self, current, duration):
        # First handle idle time since last update
        idle_time = self.env.now - self.last_update_time
        if idle_time > 0:
            self._apply_discharge(idle_current, idle_time)
        
        # Then handle the actual operation
        self._apply_discharge(current, duration)
        self.last_update_time = self.env.now

    def _apply_discharge(self, current, duration):
        # Convert duration from seconds to hours
        duration_hours = duration / 3600
        # Calculate energy consumed in mAh
        charge_consumed = current * duration_hours
        # Add self-discharge
        charge_consumed += self.discharge_rate * duration_hours
        self.charge = max(0, self.charge - charge_consumed)
        # Update temperature (simplified model)
        self.temperature += 0.1 * current * duration_hours
        self.temperature = min(45, max(0, self.temperature))


class Camera:
    def __init__(self, env, cache_dataset, interval=1, batch_size=32):
        self.env = env
        self.interval = interval
        self.data_loader = cache_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.batch_buffer = []
        
        # Get first item to determine shape
        self.iterator = iter(self.data_loader) 
        first_item = next(self.iterator)
        if isinstance(first_item, (list, tuple)):
            self.image_shape = first_item[0].shape
        else:
            self.image_shape = first_item['image'].shape
            
        self.out = None
        self.env.process(self.run())

    def run(self):
        while True:
            # Preload batch to GPU
            batch = []
            for i, item in enumerate(self.data_loader):
                if isinstance(item, (list, tuple)):
                    item = (item[0].to(self.device), item[1])
                else:
                    item['image'] = item['image'].to(self.device)
                    if 'logits' in item:
                        item['logits'] = [l.to(self.device) for l in item['logits']]
                
                batch.append(item)
                
                if len(batch) == self.batch_size:
                    # Process full batch
                    for b in batch:
                        if self.out:
                            self.out.put(b)
                            yield self.env.timeout(self.interval)
                    batch = []
                    
            # Process remaining items
            for b in batch:
                if self.out:
                    self.out.put(b)
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
        self.data_counter = 0
        self.latencies = {}
        self.battery = Battery(env)  # Use the simplified Battery class

    def put(self, data):
        return self.store.put(data)

    def process_item(self, data):
        self.data_counter += 1
        data_id = self.data_counter
        start_time = self.env.now
        self.start_times[data_id] = start_time
        cont = True
        block_num = 0
        total_flops = 0
        
        while cont and block_num < 4:
            reward = 0
            # Run Neural Block
            processing_time = block_processing_times[f'block{block_num}']
            yield self.env.timeout(processing_time)
            
            # Update battery for processing
            self.battery.update(self.get_processing_current(), processing_time)
            image = data['image']
            logits = data['logits'][block_num]
            # Get current battery state
            battery_state = self.battery.get_state()
            data['battery'] = battery_state

            # Get logits and confidence
            confidence = torch.max(torch.softmax(logits, dim=0)).item()

            # Run Agent
            yield self.env.timeout(AGENT_TIME)
            action = self.agent.select_action(self.exits[block_num], image, logits, battery_state)

            # Calculate latency for the current block
            current_latency = self.env.now - start_time

            # Add FLOPS calculation
            total_flops += block_cpu_flops[f'block{block_num}']

            # Process Action
            if action == 0:  # Continue to next block
                accuracy = None  # We don't know the accuracy yet
                reward = calculate_reward(action, accuracy, current_latency, block_num, data['battery'])
                # Store Experience
                state = {
                    'exit': self.exits[block_num],
                    'image': image,
                    'logits': logits,
                    'battery': data['battery']
                }
                next_block_num = min(block_num + 1, 3)
                next_state = {
                    'exit': self.exits[next_block_num],
                    'image': image,
                    'logits': data['logits'][next_block_num]if isinstance(data['logits'][next_block_num], torch.Tensor) else torch.tensor(data['logits'][next_block_num], dtype=torch.float32),
                    'battery': data['battery']
                }
                self.agent.store_experience((state, action, reward, next_state, False))

            elif action == 1:  # Exit without transmitting

                prediction = torch.argmax(softmax(logits))
                accuracy = prediction == data['label']
                with open('results.txt', 'a') as f:
                    f.write(f"prediction: {prediction}, label: {data['label']}\n")
                reward = calculate_reward(action, accuracy, current_latency, block_num, data['battery'])
                cont = False
                state = {
                    'exit': self.exits[block_num],
                    'image': image,
                    'logits': logits,
                    'battery': data['battery']
                }
                next_block_num = min(block_num + 1, 3)
                next_state = {
                    'exit': self.exits[next_block_num],
                    'image': image,
                    'logits': data['logits'][next_block_num],
                    'battery': data['battery']
                }
                self.agent.store_experience((state, action, reward, next_state, True))
                self.data_tracker.update(accuracy, start_time, block_num, current_latency, action, total_flops, battery_state, confidence)
            elif action == 2:  # Exit and transmit  
                accuracy = int(logits.argmax()) == int(data['label'])
                prediction = torch.argmax(softmax(logits))
                with open('results.txt', 'a') as f:
                    f.write(f"prediction: {prediction}, label: {data['label']}\n")
                reward = calculate_reward(action, accuracy, current_latency, block_num, data['battery'])
                self.send_packet(data, block_num, action, current_latency, total_flops)
                cont = False
                
                # Update battery for transmission
                transmission_time = self.get_transmission_time(block_num)
                self.battery.update(self.get_transmission_current(), transmission_time)
                
                break
            
            block_num += 1

    def get_processing_current(self):
        return current_draw['processing']

    def get_transmission_current(self):
        return current_draw['transmission']

    def get_idle_current(self):
        return idle_current

    def get_transmission_time(self, block_num):
        # Assuming a more realistic 250 kbps transmission rate for IoT devices
        return packet_sizes[f'block{block_num}'] / 250e3  # seconds

    def process_data(self):
        while True:
            data = yield self.store.get()
            yield self.env.process(self.process_item(data))

    def send_packet(self, data, block_num, action, latency, total_flops):
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
        packet.total_flops = total_flops  # Add this line
        if self.out:
            self.out.put(packet)
        self.packets_sent += 1
        
        # Update the data tracker with the amount of data sent
        data_sent = packet_sizes[f'block{block_num}']
        self.data_tracker.update_data_sent(data_sent)
        
        if self.debug:
            print(f"Sent packet {self.packets_sent} at time {self.env.now}")


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
                'logits': data['logits'][block_num],
                'battery': data['battery']
            }
            next_block_num = 3
            next_state = {
                'exit': self.exits[next_block_num],
                'image': data['image'],
                'logits': data['logits'][next_block_num],
                'battery': data['battery']
            }
            # Calculate reward based on previous classification accuracy
            previous_logits = data['logits'][block_num]
            previous_prediction = torch.argmax(previous_logits)
            true_label = data['label']
            

            latency = self.env.now - start_time
            total_flops = packet.total_flops  # Add this line
            if previous_prediction == true_label:
                reward = - latency
            else:
                reward = 10
            self.agent.store_experience((state, action, reward, next_state, True))
            final_prediction = torch.argmax(data['logits'][3])
            accuracy = final_prediction == true_label
            
            self.data_tracker.update(accuracy, start_time, 3, latency, action, total_flops,data['battery'])
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
        total_flops = block_cpu_flops['block0']
        
        # Update battery for processing
        processing_time = block_processing_times['block0']
        self.battery.update(self.get_processing_current(), processing_time)
        
        # Get current battery state
        battery_state = self.battery.get_state()
        data['battery'] = battery_state
        
        latency = self.env.now - start_time
        
        # Update battery for transmission
        transmission_time = self.get_transmission_time(0)
        self.battery.update(self.get_transmission_current(), transmission_time)
        
        self.send_packet(data, 0, 2, latency, total_flops)

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
            latency = self.env.now - start_time
            final_prediction = torch.argmax(data['logits'][3])
            true_label = data['label']
            accuracy = final_prediction == true_label
            total_flops = packet.total_flops
            battery_state = data.get('battery', torch.tensor([1.0, 3.7, 25.0]))  # Default values if battery info is missing
            self.data_tracker.update(accuracy, start_time, 3, latency, 2, total_flops, battery_state)

        if self.debug:
            print(f"Received packet at time {self.env.now}")

class EarlyExitHead(Head):
    def __init__(self, env, element_id, data_tracker, confidence_thresholds=None, exits=4, debug=False):
        super().__init__(env, element_id, None, data_tracker, exits, debug)
        if confidence_thresholds is not None:
            self.confidence_thresholds = confidence_thresholds
        else:
            # Use the same threshold for all exits if only one is provided
            self.confidence_thresholds = [confidence_thresholds] * 4

    def process_item(self, data):
        self.data_counter += 1
        data_id = self.data_counter
        start_time = self.env.now
        self.start_times[data_id] = start_time
        total_flops = 0

        for block_num in range(4):
            # Run Neural Block
            processing_time = block_processing_times[f'block{block_num}']
            yield self.env.timeout(processing_time)
            
            # Update battery for processing
            self.battery.update(self.get_processing_current(), processing_time)
            
            logits = torch.tensor(data['logits'][block_num], dtype=torch.float32)
            
            # Add FLOPS calculation
            total_flops += block_cpu_flops[f'block{block_num}']
            
            # Check if confidence exceeds threshold
            confidence = torch.softmax(logits, dim=0).max().detach().cpu().numpy()
            latency = self.env.now - start_time
            
            # Get current battery state
            battery_state = self.battery.get_state()
            data['battery'] = battery_state
            
            if confidence >= self.confidence_thresholds[block_num] or block_num == 3:
                self.send_to_data_tracker(data, block_num, latency, total_flops, battery_state)
                break

            # Update battery for idle time
            idle_time = self.env.now - start_time - sum(block_processing_times[f'block{i}'] for i in range(block_num+1))
            if idle_time > 0:
                self.battery.update(self.get_idle_current(), idle_time)

    def send_to_data_tracker(self, data, block_num, latency, total_flops, battery_state):
        true_label = data['label']
        prediction = torch.argmax(data['logits'][block_num]).detach().cpu().numpy()
        accuracy = prediction == true_label
        self.data_tracker.update(accuracy, self.start_times[self.data_counter], block_num, latency, 1, total_flops, battery_state)  # Action 1 for exit without transmitting

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
            latency = self.env.now - start_time
            final_prediction = torch.argmax(data['logits'][block_num]).detach().cpu().numpy()
            true_label = data['label']
            accuracy = final_prediction == true_label
            self.data_tracker.update(accuracy, start_time, block_num, latency, 2, total_flops,data['battery'])

        if self.debug:
            print(f"Received packet at time {self.env.now}")















