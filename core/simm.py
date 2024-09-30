import simpy
from components import *
import numpy as np
from functools import partial, wraps
import random
from ns.port.wire import Wire
import matplotlib.pyplot as plt
from collections import defaultdict
from agent import Agent
class Simulation:
    def __init__(self):
        # self.conf1 = conf1
        # self.conf2 = conf2
        # self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        # self.data = []
        # self.tail_net = TailNetwork(block=Bottleneck, in_planes=512, num_blocks=[3, 4, 6, 3], num_classes=10)
        # self.tail_net.load_state_dict(torch.load("models/tail_resnet50.pth"))
        # self.tail_net.eval()
        # device_tail = 'cuda'
        # self.tail_net = self.tail_net.to(device_tail)

        # self.heads = []
        # self.cameras = []
        # self.wires = []
        self.env = simpy.Environment()

        # self.tail = Tail(self.env,self.tail_net,partial(random.gauss, 0.02, 0.002),self.data)
        
        self.agent = Agent(self.env, agent = "dql", model = None, optimizer=None, criterion=None)
        self.camera = Camera(self.env,None, 1,None, self.agent.store)
        # trainer_store = self.trainer.store
        # for i in range(1):

            # head_net_part1 = HeadNetworkPart1(block=Bottleneck, in_planes=64, num_blocks=[3, 4], num_classes=10)
            # head_net_part2 = HeadNetworkPart2(block=Bottleneck, in_planes=256, num_blocks=[3, 4], num_classes=10)
            
            # head_net_part1.load_state_dict(torch.load(f"models/head1_resnet50.pth"))
            # head_net_part2.load_state_dict(torch.load(f"models/head2_resnet50.pth"))
            # head_net_part1.eval()
            # head_net_part2.eval()
            
            # device_head = 'cuda'
            # head_net_part1 = head_net_part1.to(device_head)
            # head_net_part2 = head_net_part2.to(device_head)

            # camera = Camera(self.env,DataLoader(self.test_set, batch_size=1, shuffle=True, num_workers=1))
            # wire = Wire(self.env, partial(random.gauss, 0.1, 0.02), wire_id=i+1, debug=False)
            # head = Head(self.env, head_net_part1, head_net_part2, partial(random.gauss, 0.15, 0.01), partial(random.gauss, 0.12, 0.01), self.conf1, self.conf2,self.data)
            
            # self.heads.append(head)
            # self.cameras.append(camera)
            # self.wires.append(wire)
        

    def start(self, sim_time=100):
        # for i, (camera, head, wire) in enumerate(zip(self.cameras, self.heads, self.wires)):
        #     camera.out = head
        #     head.out = wire
        #     wire.out = self.tail  

        self.env.run(until=sim_time)

    def calculate_metrics(self):
        accuracy_per_exit = defaultdict(lambda: {'correct': 0, 'total': 0, 'elapsed_sum': 0.0})
        metrics_per_exit = {}

        for item in self.data:
            correct = 1 if (item.yhat == item.label) else 0
            accuracy_per_exit[item.exit_num]['correct'] += correct
            accuracy_per_exit[item.exit_num]['total'] += 1
            accuracy_per_exit[item.exit_num]['elapsed_sum'] += item.elapsed

        for exit_num, values in accuracy_per_exit.items():
            total_items = values['total']
            correct_items = values['correct']
            elapsed_sum = values['elapsed_sum']
            accuracy = (correct_items / total_items) * 100 if total_items else 0
            avg_elapsed = elapsed_sum / total_items if total_items else 0

            metrics_per_exit[exit_num] = {
                'accuracy': accuracy,
                'total_items': total_items,
                'average_elapsed_time': avg_elapsed
            }
        return metrics_per_exit

if __name__ == "__main__":
    sim = Simulation()
    sim.start()
        