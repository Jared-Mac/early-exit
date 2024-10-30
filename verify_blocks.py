import os
import torch
import argparse
from split_model import get_model_config, initialize_blocks

def load_and_verify_blocks(model_type):
    """Load split blocks and verify they can produce output."""
    # Get model configuration
    ModelClass, block_configs, block_type = get_model_config(model_type)
    
    # Initialize blocks
    blocks = initialize_blocks(ModelClass, block_type, block_configs, model_type)
    
    # Load and print saved state dictionaries
    blocks_dir = f'models/cifar10/{model_type}_blocks'
    for block_name, block in blocks.items():
        block_path = os.path.join(blocks_dir, f'{block_name}.pth')
        if not os.path.exists(block_path):
            raise FileNotFoundError(f"Block file not found: {block_path}")
        
        state_dict = torch.load(block_path)
        print(f"\n{block_name} State Dictionary:")
        for key, tensor in state_dict.items():
            print(f"{key}: shape {tensor.shape}, dtype {tensor.dtype}")
        
        block.load_state_dict(state_dict)
        block.eval()
    
    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Test forward pass through blocks sequentially
    try:
        with torch.no_grad():
            # Block 1
            out1, _ = blocks['block1'](x)
            print(f"Block 1 output shape: {out1.shape}")
            
            # Block 2
            out2, _ = blocks['block2'](out1)
            print(f"Block 2 output shape: {out2.shape}")
            
            # Block 3
            out3, _ = blocks['block3'](out2)
            print(f"Block 3 output shape: {out3.shape}")
            
            # Block 4
            final_out = blocks['block4'](out3)
            print(f"Final output shape: {final_out.shape}")
            
        print("Successfully verified all blocks!")
        return True
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify split neural network blocks')
    parser.add_argument('--model-type', type=str, default='mobilenetv2',
                      choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      help='Type of model to verify')
    args = parser.parse_args()

    try:
        success = load_and_verify_blocks(args.model_type)
        return 0 if success else 1
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())