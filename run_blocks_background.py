import torch
import time
from split_model import get_model_config, initialize_blocks
import argparse

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

def run_blocks(blocks, x, device, interval=1):
    while True:
        with torch.no_grad():
            # Block 1
            out1, _ = blocks['block1'](x)

            # Block 2
            out2, _ = blocks['block2'](out1)

            # Block 3
            out3, _ = blocks['block3'](out2)

            # Block 4
            _ = blocks['block4'](out3)
        time.sleep(interval)


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
    parser.add_argument('--interval', type=float,
                        default=1, help='Interval for each operation')

    args = parser.parse_args()
    interval = args.interval

    device = torch.device(args.device)
    blocks, x = load_blocks(args.model, args.path, device)

    print(f"\nRunning {args.model} in background with interval {interval} s...")
    run_blocks(blocks, x, device, interval)

if __name__ == "__main__":
    main()

