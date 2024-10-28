import torch
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torchao.quantization import quantize_, int8
import os

from early_exit_resnet import EarlyExitResNet50
from dataset import Flame2DataModule, CIFAR10DataModule

def quantize_model(dataset='flame2'):
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')
    
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the appropriate DataModule
    if dataset == 'flame2':
        data_module = Flame2DataModule(
            image_dir='data/Flame2/RGB',  # Replace with actual path to Flame2 test images
            batch_size=6,
            transform=transform
        )
    elif dataset == 'cifar10':
        data_module = CIFAR10DataModule(data_dir='./data', batch_size=6)
    else:
        raise ValueError("Invalid dataset choice. Choose 'flame2' or 'cifar10'.")

    # Load the trained model
    model = EarlyExitResNet50.load_from_checkpoint('lightning_logs/version_26/checkpoints/epoch=15-step=24944.ckpt')
    quantize_(model, int8())
    model.eval()

    # Set up PyTorch Lightning Trainer for testing
    trainer = Trainer(accelerator='gpu',devices="auto")

    # Test the quantized model
    print(f"Testing quantized model on {dataset} dataset:")
    trainer.test(model, data_module)

    # Save the quantized model
    save_path = f'models/{dataset}/early_exit_resnet50_quantized.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    quantized_size = os.path.getsize(save_path)

    print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")

if __name__ == "__main__":
    # You can change this to 'cifar10' to test on CIFAR10
    quantize_model(dataset='flame2')
