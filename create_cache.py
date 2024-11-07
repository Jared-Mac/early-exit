import torch
import torchvision
import torchvision.transforms as transforms
from adapt.dataset import CacheDataset, CIFAR10DataModule, CIFAR100DataModule, TinyImageNetDataModule
import os
import argparse
from split_model import get_model_config, initialize_blocks

def get_dataset(dataset_name, data_dir='./data'):
    """Get the appropriate dataset based on name."""
    if dataset_name == 'cifar10':
        datamodule = CIFAR10DataModule(data_dir=data_dir)
        datamodule.setup('test')
        return datamodule.cifar10_test
    elif dataset_name == 'cifar100':
        datamodule = CIFAR100DataModule(data_dir=data_dir)
        datamodule.setup('test')
        return datamodule.cifar100_test
    elif dataset_name == 'tiny-imagenet':
        datamodule = TinyImageNetDataModule()
        datamodule.setup('test')
        return datamodule.test_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def create_cache(block_dir, model_type, dataset_name, num_classes, data_dir='./data'):
    # Create models directory if it doesn't exist
    os.makedirs(block_dir, exist_ok=True)
    
    # Get model configuration and initialize blocks
    ModelClass, block_configs, block_type = get_model_config(model_type)
    blocks = initialize_blocks(ModelClass, block_type, block_configs, model_type, num_classes)
    
    # Load model weights from the specified directory
    for block_name, block in blocks.items():
        block.load_state_dict(torch.load(os.path.join(block_dir, f"{block_name}.pth")))
        block.eval()

    # Get the appropriate dataset
    test_set = get_dataset(dataset_name, data_dir)
    
    # Create cache file path in the block directory
    cache_file = os.path.join(block_dir, f'cached_logits.pkl')
    
    custom_test_set = CacheDataset(
        test_set, 
        cached_data_file=cache_file,
        models=blocks,
        compute_logits=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cache for blocks')
    parser.add_argument('--block_dir', type=str, default='models/cifar10',
                        help='Directory containing block weights and where to save cache')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'mobilenetv2'],
                        help='Type of model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='Dataset to create cache for')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes for the model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
    args = parser.parse_args()
    
    create_cache(args.block_dir, args.model_type, args.dataset, args.num_classes, args.data_dir)