import argparse
import os
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

def evaluate_strategies(dataset, model_type):
    # Define number of classes for each dataset
    dataset_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny-imagenet': 200,
        # Add other datasets as needed
    }
    
    num_classes = dataset_classes.get(dataset, 10)  # Default to 10 if dataset not found
    
    # Strategies to evaluate
    strategies = ['dql', 'always_transmit', 'early_exit_conservative', 'early_exit_balanced', 'early_exit_aggressive', 'early_exit_very_aggressive']
    results = {}
    
    # Construct paths
    base_dir = f'models/{dataset}/{model_type}'
    csv_path = os.path.join(base_dir, 'strategy_results.csv')
    plot_path = os.path.join(base_dir, 'strategy_comparison.png')
    
    # Try to load existing results
    try:
        df = pd.read_csv(csv_path)
        results = df.set_index('strategy').to_dict('index')
        print(f"Loaded existing results from {csv_path}")
    except FileNotFoundError:
        # Run evaluations if no CSV exists
        for strategy in strategies:
            print(f"\nEvaluating {strategy} strategy:")
            cached_data_file = os.path.join(base_dir, 'blocks/cached_logits.pkl')
            if strategy == 'dql':
                sim = Simulation(strategy=strategy, 
                               cached_data_file=cached_data_file,
                               load_model=os.path.join(base_dir, 'trained_dqn_model.pth'),
                               num_classes=num_classes)
            else:
                sim = Simulation(strategy=strategy, 
                               cached_data_file=cached_data_file,
                               num_classes=num_classes)
            results[strategy] = sim.evaluate(max_sim_time=5000)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
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
    plt.savefig(plot_path)
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
    parser = argparse.ArgumentParser(description='Evaluate different strategies for model execution')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       help='Model type (default: resnet50)')
    
    args = parser.parse_args()
    evaluate_strategies(args.dataset, args.model_type)
