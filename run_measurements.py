import torch
import time
import argparse
from split_model import get_model_config, initialize_blocks
from torchstat import stat

def load_blocks(model_type='resnet50', path='models/cifar10', device='cpu'):
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
        # Block 1
        start = time.perf_counter()
        out1, _ = blocks['block1'](x)
        times['block1'] = time.perf_counter() - start
        
        # Block 2
        start = time.perf_counter()
        out2, _ = blocks['block2'](out1)
        times['block2'] = time.perf_counter() - start

        # Block 3
        start = time.perf_counter()
        out3, _ = blocks['block3'](out2)
        times['block3'] = time.perf_counter() - start

        # Block 4
        start = time.perf_counter()
        _ = blocks['block4'](out3)
        times['block4'] = time.perf_counter() - start
    
    return times

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Measure execution time for model blocks')
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      default='resnet50', help='Model architecture to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                      default='cpu', help='Device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size') 
    parser.add_argument('--path', type=str, default='models/cifar10', help='Path to model blocks')

    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load blocks and create input
    print(f"\nMeasuring execution times for {args.model}")
    blocks, x = load_blocks(args.model, args.path, device)
    
    # Measure execution times
    times = measure_block_times(blocks, x, device)
    
    # Print results
    print("\nExecution times:")
    for block_name, execution_time in times.items():
        print(f"{block_name} execution time: {execution_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
