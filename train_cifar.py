import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping

from early_exit_resnet import EarlyExitResNet50
from early_exit_resnet18 import EarlyExitResNet18
from dataset import Flame2DataModule, CIFAR10DataModule, CIFAR100DataModule
from early_exit_mobilenetv2 import EarlyExitMobileNetV2

torch.set_float32_matmul_precision('medium')


cifar10_dm = CIFAR10DataModule(batch_size=350)
# cifar10_dm = CIFAR100DataModule(batch_size=350)

# Initialize the model
model = EarlyExitMobileNetV2(num_classes=10, input_channels=3, input_height=32, input_width=32, loss_weights=[0.25, 0.25, 0.25, 0.25])

# Set up PyTorch Lightning Trainer
# ModelSummary(max_depth=2)
trainer = Trainer(max_epochs=30,callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
# tuner = Tuner(trainer)
# tuner.scale_batch_size(model, datamodule=cifar10_dm, mode="binsearch")

# Train the model
trainer.fit(model, cifar10_dm)

# Test the model
trainer.test(model, cifar10_dm)
