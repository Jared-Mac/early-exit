import torch
import pickle
import argparse
from torch.nn import Softmax
import numpy as np

def analyze_cache(cache_path):
    """Analyze cached logits and compute accuracy for each exit."""
    # Load cached data
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    num_exits = len(cached_data[0]['logits'])
    total_samples = len(cached_data)
    correct_per_exit = [0] * num_exits
    softmax = Softmax(dim=0)
    
    # Analyze each sample
    for sample in cached_data:
        label = sample['label']
        logits = sample['logits']
        
        # Check prediction for each exit
        for exit_idx, exit_logits in enumerate(logits):
            prediction = torch.argmax(softmax(exit_logits))
            if prediction == label:
                correct_per_exit[exit_idx] += 1
    
    # Calculate and print accuracies
    print("\nAccuracy for each exit:")
    print("-" * 30)
    for exit_idx in range(num_exits):
        accuracy = (correct_per_exit[exit_idx] / total_samples) * 100
        print(f"Exit {exit_idx + 1}: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze cached dataset accuracies')
    parser.add_argument('--cache_path', type=str, required=True,
                        help='Path to the cached logits file')
    
    args = parser.parse_args()
    analyze_cache(args.cache_path) 