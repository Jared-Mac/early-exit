import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import os

from early_exit_resnet import EarlyExitResNet50
from early_exit_resnet18 import EarlyExitResNet18
from early_exit_mobilenetv2 import EarlyExitMobileNetV2

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/tiny-imagenet-200", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(64),  # Tiny ImageNet is 64x64
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            self.transform
        ])

    def setup(self, stage=None):
        # Training data
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        
        # Validation data
        self.val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'val'),
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

def main():
    torch.set_float32_matmul_precision('medium')

    # Initialize data module
    tiny_imagenet_dm = TinyImageNetDataModule(batch_size=64)

    # Initialize model (200 classes for Tiny ImageNet)
    model = EarlyExitResNet50(
        num_classes=200,
        input_channels=3,
        input_height=64,
        input_width=64,
        loss_weights=[0.25, 0.25, 0.25, 0.25]
    )

    # Set up PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,  # Disable progress bar
        enable_model_summary=True, # Disable model summary
        log_every_n_steps=None     # Only log at end of epoch
    )

    # Train the model
    trainer.fit(model, tiny_imagenet_dm)

    # Test the model
    trainer.test(model, tiny_imagenet_dm)

if __name__ == "__main__":
    main()
