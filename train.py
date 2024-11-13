import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import argparse

from early_exit_resnet import EarlyExitResNet50
from early_exit_resnet18 import EarlyExitResNet18
from early_exit_mobilenetv2 import EarlyExitMobileNetV2
from adapt.dataset import CIFAR10DataModule, CIFAR100DataModule, VisualWakeWordsDataModule
from tiny_imagenet_loader import TinyImageNetDataModule

def get_model(model_name, num_classes, input_channels, input_height, input_width, loss_weights):
    models = {
        'resnet18': EarlyExitResNet18,
        'resnet50': EarlyExitResNet50,
        'mobilenetv2': EarlyExitMobileNetV2
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    return models[model_name](
        num_classes=num_classes,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        loss_weights=loss_weights
    )

def get_datamodule(dataset_name, batch_size):
    datasets = {
        'cifar10': (CIFAR10DataModule, {'num_classes': 10, 'input_size': 32}),
        'cifar100': (CIFAR100DataModule, {'num_classes': 100, 'input_size': 32}),
        'tiny-imagenet': (TinyImageNetDataModule, {'num_classes': 200, 'input_size': 64}),
        'visualwakewords': (VisualWakeWordsDataModule, {'num_classes': 2, 'input_size': 224})
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(datasets.keys())}")
    
    DataModule, config = datasets[dataset_name]
    return DataModule(batch_size=batch_size), config

def main():
    parser = argparse.ArgumentParser(description='Train early-exit networks on various datasets')
    parser.add_argument('--model', type=str, default='resnet18', 
                      choices=['resnet18', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'cifar100', 'tiny-imagenet', 'visualwakewords'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--loss_weights', type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
                      help='Weights for each exit point loss')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')

    # Initialize data module and get dataset config
    data_module, config = get_datamodule(args.dataset, args.batch_size)

    # Initialize model
    model = get_model(
        model_name=args.model,
        num_classes=config['num_classes'],
        input_channels=3,
        input_height=config['input_size'],
        input_width=config['input_size'],
        loss_weights=args.loss_weights
    )

    # Set up PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=f"models/{args.dataset}/{args.model}"
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

if __name__ == "__main__":
    main() 