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
from tiny_imagenet_loader import TrainTinyImageNetDataset, TestTinyImageNetDataset, id_dict


def main():
    torch.set_float32_matmul_precision('medium')

    # Initialize data module
    tiny_imagenet_dm = TinyImageNetDataModule(batch_size=64)
    model_name = "resnet18"
    dataset_name = "tiny_imagenet"
    # Initialize model (200 classes for Tiny ImageNet)
    model = EarlyExitResNet18(
        num_classes=200,
        input_channels=3,
        input_height=64,
        input_width=64,
        loss_weights=[0, 0, 0, 1]
    )

    # Set up PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,  # Disable progress bar
        enable_model_summary=True, # Disable model summary
        log_every_n_steps=None,
        default_root_dir="models/" + dataset_name + "/" + model_name
    )

    # Train the model
    trainer.fit(model, tiny_imagenet_dm)

    # Test the model
    trainer.test(model, tiny_imagenet_dm)

if __name__ == "__main__":
    main()
