import os
import torch
import argparse
from split_model import get_model_config, initialize_blocks, split_state_dict, save_block_models
from create_cache import create_cache

def prepare_model(model_path, output_dir, model_type, dataset_name, num_classes, data_dir='./data'):
    """Prepare model by splitting it and creating cache."""
    try:
        # Get model configuration
        ModelClass, block_configs, block_type = get_model_config(model_type)
        
        # Load combined model state dict
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        combined_state_dict = torch.load(model_path)
        
        # Initialize blocks
        blocks = initialize_blocks(ModelClass, block_type, block_configs, model_type, num_classes)
        
        # Split state dictionary
        block_state_dicts = split_state_dict(combined_state_dict)
        
        # Save individual block models
        save_block_models(blocks, block_state_dicts, output_dir + '/blocks')
        print(f"Successfully split models saved to {output_dir}/blocks")
        
        # Create cache
        create_cache(output_dir, model_type, dataset_name, num_classes, data_dir)
        print(f"Successfully created cache in {output_dir}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Split model and create cache')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the combined model checkpoint file')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save the split model blocks and cache')
    parser.add_argument('--model-type', type=str, default='resnet50',
                      choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      help='Type of model to split')
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                      help='Dataset to create cache for')
    parser.add_argument('--num-classes', type=int, required=True,
                      help='Number of output classes for the model')
    parser.add_argument('--data-dir', type=str, default='./data',
                      help='Directory containing the dataset')
    
    args = parser.parse_args()
    
    return prepare_model(
        args.model_path,
        args.output_dir,
        args.model_type,
        args.dataset,
        args.num_classes,
        args.data_dir
    )

if __name__ == "__main__":
    exit(main()) 