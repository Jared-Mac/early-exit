import torch
import time
import argparse
from split_model import get_model_config, initialize_blocks
from torchstat import stat
from flops_profiler.profiler import get_model_profile
from thop import profile, clever_format
import pandas as pd

def load_blocks(model_type='resnet50', path='models/cifar10', device=torch.device("cpu")):
    """Load all blocks for the specified model type."""
    # Get model configuration
    ModelClass, block_configs, block_type = get_model_config(model_type)
    
    # Initialize blocks
    blocks = initialize_blocks(ModelClass, block_type, block_configs, model_type)
    
    # Load saved state dictionaries
    blocks_dir = f'{path}/{model_type}_blocks'
    
    for block_name, block in blocks.items():
        block_path = f"{blocks_dir}/{block_name}.pth"
        state_dict = torch.load(block_path, map_location=device)
        block.load_state_dict(state_dict)
        block.eval()
    
    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32)
    
    return blocks, x

def measure_block_times(blocks, x, device):
    """Measure execution time for each block."""
    times = {}
    
    with torch.no_grad():
        # Dummy execution
        start = time.time()
        out1, _ = blocks['block1'](x)
        times['dummy'] = time.time() - start

        # Block 1
        start = time.time()
        out1, _ = blocks['block1'](x)
        times['block1'] = time.time() - start
        
        # Block 2
        start = time.time()
        out2, _ = blocks['block2'](out1)
        print(out1.shape)
        times['block2'] = time.time() - start
        
        # Block 3
        start = time.time()
        out3, _ = blocks['block3'](out2)
        print(out2.shape)
        times['block3'] = time.time() - start
        
        # Block 4
        start = time.time()
        _ = blocks['block4'](out3)
        print(out3.shape)
        times['block4'] = time.time() - start
    
    return times

def measrue_idle_voltage_and_currency(blocks, x):
    pass

def measure_flops(blocks, block_num, x, device):
    flops = {}
    input_sizes = [(1, 3, 32, 32), (1, 24, 8, 8), (1, 32, 4, 4), (1, 64, 2, 2)]
    with torch.no_grad():
        for i in range(block_num):
            model = blocks[f"block{i+1}"]
            mocked_input = torch.zeros(input_sizes[i]).to(device)
            flops, params = profile(model.to(device), inputs=(mocked_input, ))
            flops, params = clever_format([flops, params], '%.3f')
            print(flops, params)


def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Measure execution time for model blocks')
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      default='resnet50', help='Model architecture to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                      default='cpu', help='Device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size') 
    parser.add_argument('--path', type=str, default='models/cifar10', help='Path to model blocks')
    parser.add_argument('--host_device', type=str, choices=['rpi', 'nano'],
                        default='rpi', help='Device that run the models')
    parser.add_argument('--metrics', type=str, default='flops', choices=['flops', 'proc_time', 'VA'], nargs='+', help='the metrics to measure')
    parser.add_argument('--block_num', type=int, default=4, help='the number of blocks to run')

    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load blocks and create input
    print(f"\nMeasuring execution times for {args.model}")
    blocks, x = load_blocks(args.model, args.path, device)

    metrics = args.metrics
    block_num = args.block_num

    for metric in metrics:
        if metric == 'proc_time':
            #Measure execution times
            times = measure_block_times(blocks, x, device)
            # Print results
            print("\nExecution times:")
            for block_name, execution_time in times.items():
                print(f"{block_name} execution time: {execution_time*1000:.2f} ms")
        elif metric == 'flops':
            # Measure flops
            measure_flops(blocks, block_num, x, device)
        elif metric == 'VA':
            pass
        else:
            print(f"Unknown metric {metric}!")

if __name__ == "__main__":
    main()
