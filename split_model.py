import os
import torch
import argparse
import early_exit_resnet as EarlyExitResNet50
import early_exit_resnet18 as EarlyExitResNet18
import early_exit_mobilenetv2 as EarlyExitMobileNetV2

def get_model_config(model_type):
    """Return model class and block configurations based on model type."""
    configs = {
        "resnet50": (
            EarlyExitResNet50,
            [(64, [3]), (256, [4]), (512, [6]), (1024, [3])],
            EarlyExitResNet50.Bottleneck
        ),
        "resnet18": (
            EarlyExitResNet18,
            [(64, [2]), (64, [2]), (128, [2]), (256, [2])],
            EarlyExitResNet18.BasicBlock
        ),
        "mobilenetv2": (
            EarlyExitMobileNetV2,
            [(32, [1]), (16, [2]), (24, [3]), (32, [4])],
            EarlyExitMobileNetV2.InvertedResidual
        )
    }
    
    if model_type not in configs:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return configs[model_type]

def initialize_blocks(ModelClass, block_type, block_configs, model_type):
    """Initialize all blocks for the given model configuration."""
    if model_type == "mobilenetv2":
        blocks = {
            'block1': ModelClass.Block1(num_classes=10),
            'block2': ModelClass.Block2(num_classes=10),
            'block3': ModelClass.Block3(num_classes=10),
            'block4': ModelClass.Block4(num_classes=10)
        }
    else:  # ResNet models
        blocks = {
            'block1': ModelClass.Block1(block_type, block_configs[0][0], block_configs[0][1], num_classes=10, input_channels=3),
            'block2': ModelClass.Block2(block_type, block_configs[1][0], block_configs[1][1], num_classes=10),
            'block3': ModelClass.Block3(block_type, block_configs[2][0], block_configs[2][1], num_classes=10),
            'block4': ModelClass.Block4(block_type, block_configs[3][0], block_configs[3][1], num_classes=10)
        }
    return blocks

def split_state_dict(combined_state_dict):
    """Split combined state dictionary into individual block state dictionaries."""
    block_state_dicts = {f'block{i}': {} for i in range(1, 5)}
    
    for key, value in combined_state_dict['state_dict'].items():
        for block_name in block_state_dicts:
            if key.startswith(block_name):
                block_state_dicts[block_name][key.removeprefix(f'{block_name}.')] = value
                break
    
    return block_state_dicts

def save_block_models(blocks, block_state_dicts, save_dir):
    """Save individual block models to specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    for block_name, block in blocks.items():
        block.load_state_dict(block_state_dicts[block_name])
        save_path = os.path.join(save_dir, f'{block_name}.pth')
        torch.save(block.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser(description='Split neural network model into blocks')
    parser.add_argument('--model-type', type=str, default='mobilenetv2',
                      choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      help='Type of model to split')
    args = parser.parse_args()

    try:
        # Get model configuration
        ModelClass, block_configs, block_type = get_model_config(args.model_type)
        
        # Load combined model state dict
        model_path = f'models/cifar10/{args.model_type}.ckpt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        combined_state_dict = torch.load(model_path)
        
        # Initialize blocks with model_type
        blocks = initialize_blocks(ModelClass, block_type, block_configs, args.model_type)
        
        # Split state dictionary
        block_state_dicts = split_state_dict(combined_state_dict)
        
        # Save individual block models
        save_dir = f'models/cifar10/{args.model_type}_blocks'
        save_block_models(blocks, block_state_dicts, save_dir)
        
        print(f"Successfully split models saved to {save_dir}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
