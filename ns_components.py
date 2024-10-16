import simpy
import torch
from ns.packet.packet import Packet
from ns.packet.dist_generator import DistPacketGenerator
class Camera(Node):
    def __init__(self, env, data_loader, interval=1):
        super().__init__(env, 'Camera')
        self.env = env
        self.interval = interval
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)
        self.out = None
        self.env.process(self.run())

    def run(self):
        while True:
            for item in self.data_loader:
                packet = Packet(item)
                self.out.put(packet)
            yield self.env.timeout(self.interval)

class Head(Node):
    def __init__(self, env, agent, exits=4):
        super().__init__(env, 'Head')
        self.env = env
        self.agent = agent
        self.exits = torch.eye(exits)
        self.tail = None
        self.env.process(self.run())

    def run(self):
        yield self.env.timeout(1)
        running_reward = 0
        while True:
            packet = yield self.get_input()
            data = packet.payload
            cont = True
            block_num = 0

            while cont and block_num < 4:
                image = torch.tensor(data['image'], dtype=torch.float32)
                logits = torch.tensor(data['logits'][block_num], dtype=torch.float32)
                action = self.agent.select_action(self.exits[block_num], image, logits)
                
                if action == 0:  # Continue to next block
                    block_num += 1
                elif action == 1:  # Exit without transmitting
                    cont = False
                elif action == 2:  # Exit and transmit
                    if self.tail:
                        classification = logits.argmax().item()
                        self.tail.put(Packet(classification))
                    cont = False

                # Calculate reward and store experience
                accuracy = int(logits.argmax()) == int(data['label'])
                reward = 10 if accuracy else -10
                running_reward += reward
                
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
                self.agent.store_experience((state, action, reward, next_state, not cont))

                if block_num == 3:
                    break

class Tail(Node):
    def __init__(self, env):
        super().__init__(env, 'Tail')
        self.env = env
        self.received_classifications = []
        self.env.process(self.run())

    def run(self):
        while True:
            packet = yield self.get_input()
            classification = packet.payload
            self.received_classifications.append(classification)
            print(f"Tail received classification: {classification}")

    def get_classifications(self):
        return self.received_classifications

if __name__ == "__main__":
    import simpy
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Set up the environment
    env = simpy.Environment()
    
    # Set up the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize components
    camera = Camera(env, dataloader)
    tail = Tail(env)
    agent = DQLAgent(env, num_classes=10, num_exit=4, image_shape=(3, 32, 32))
    head = Head(env, agent, tail=tail)
    
    # Connect components
    camera.out = head.store
    
    # Run the simulation
    env.run(until=100)  # Run for 100 time steps
    
    # Print results
    print(f"Number of classifications received by Tail: {len(tail.get_classifications())}")
    print(f"Classifications: {tail.get_classifications()[:10]}")  # Print first 10 classifications

