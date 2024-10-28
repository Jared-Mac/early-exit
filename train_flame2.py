import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from early_exit_resnet18 import EarlyExitResNet18
from dataset import Flame2DataModule
from early_exit_mobilenetv2 import EarlyExitMobileNetV2

# Set float32 matmul precision
torch.set_float32_matmul_precision('medium')

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Initialize the Flame2DataModule
flame2_dm = Flame2DataModule(
    image_dir='data/Flame2/Thermal',  # Replace with actual path to Flame2 images
    batch_size=32,
    transform=transform
)
flame2_dm.setup()

# Initialize the model
model = EarlyExitMobileNetV2(num_classes=3, input_channels=3, input_height=224, input_width=224, loss_weights=[0.25, 0.25, 0.25, 0.25])  

# Set up PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=30,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
    accelerator='gpu',
    devices='auto',  # Use all available GPUs
    strategy='ddp'  # Use Distributed Data Parallel
)

# Train the model
trainer.fit(model, flame2_dm)

# Test the model
trainer.test(model, flame2_dm)

# Save the trained model
torch.save(model.state_dict(), 'models/flame2/mobilenetv2.pth')
