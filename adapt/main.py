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

from dql import DQLAgent
from utils import DataTracker
from components import Camera, Head, Tail, EarlyExitHead, EarlyExitTail, AlwaysTransmitHead, AlwaysTransmitTail
from simulation import Simulation

def evaluate_strategies():
    strategies = ['dql', 'always_transmit', 'early_exit']
    results = {}

    for strategy in strategies:
        print(f"\nEvaluating {strategy} strategy:")
        if strategy == 'dql':
            sim = Simulation(strategy=strategy, load_model='/home/coder/early-exit/models/dql/cifar10/trained_dqn_model.pth')
        else:
            sim = Simulation(strategy=strategy)
        results[strategy] = sim.evaluate(max_sim_time=50000)

    print("\nComparison of results:")
    for strategy, result in results.items():
        print(f"\n{strategy.upper()} Strategy:")
        print(f"Accuracy: {result['accuracy']:.2f}")
        print(f"Average latency: {result['avg_latency']:.2f}")
        print(f"Exit numbers: {result['exit_numbers']}")
        print(f"Action numbers: {result['action_numbers']}")
        print(f"Average data sent: {result['avg_data_sent']:.2f} bytes")
        print(f"Cumulative data sent: {result['cumulative_data_sent']} bytes")
        print(f"Average CPU FLOPS: {result['avg_flops']:.2f}")
        print(f"Cumulative CPU FLOPS: {result['cumulative_flops']}")
        print(f"Final Battery SoC: {result['final_battery_soc']:.2f}")

if __name__ == "__main__":
    evaluate_strategies()
