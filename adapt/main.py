import simpy
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

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
    # Strategies to evaluate
    strategies = ['dql', 'always_transmit', 'early_exit_conservative', 'early_exit_balanced', 'early_exit_aggressive', 'early_exit_very_aggressive']
    results = {}
    
    # Try to load existing results
    csv_path = 'strategy_results.csv'
    try:
        df = pd.read_csv(csv_path)
        results = df.set_index('strategy').to_dict('index')
        print("Loaded existing results from CSV")
    except FileNotFoundError:
        # Run evaluations if no CSV exists
        for strategy in strategies:
            print(f"\nEvaluating {strategy} strategy:")
            model_dir = 'models/cifar10/resnet50'
            cached_data_file = model_dir+'/blocks/cached_logits.pkl'
            if strategy == 'dql':
                sim = Simulation(strategy=strategy, cached_data_file=cached_data_file, load_model=model_dir+'/trained_dqn_model.pth')
            else:
                sim = Simulation(strategy=strategy, cached_data_file=cached_data_file)
            results[strategy] = sim.evaluate(max_sim_time=5000)
        
        # Save results to CSV
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'strategy'
        df.to_csv(csv_path)
        print(f"Saved results to {csv_path}")

    # Create visualization plots
    metrics = {
        'accuracy': 'Accuracy (%)',
        'avg_latency': 'Average Latency (ms)',
        'avg_data_sent': 'Average Data Sent (bytes)',
        'avg_flops': 'Average CPU FLOPS',
        'final_battery_soc': 'Final Battery SoC (%)'
    }

    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    fig.suptitle('Strategy Comparison Results')

    for idx, (metric, ylabel) in enumerate(metrics.items()):
        values = [results[s][metric] for s in strategies]
        ax = axes[idx] if len(metrics) > 1 else axes
        bars = ax.bar(strategies, values)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by Strategy')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()

    # Print numerical results as before
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
