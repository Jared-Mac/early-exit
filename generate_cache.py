import torchvision
import torchvision.transforms as transforms
from dataset_generator import CachedDatasetGenerator
from early_exit_resnet18 import Block1, Block2, Block3, Block4, BasicBlock

# Setup dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)

# Define model configurations
model_paths = {
    'block1': 'models/cifar10/resnet18_blocks/block1.pth',
    'block2': 'models/cifar10/resnet18_blocks/block2.pth',
    'block3': 'models/cifar10/resnet18_blocks/block3.pth',
    'block4': 'models/cifar10/resnet18_blocks/block4.pth'
}

model_classes = {
    'block1': Block1,
    'block2': Block2,
    'block3': Block3,
    'block4': Block4
}

model_configs = {
    'block1': {'block': BasicBlock, 'in_planes': 64, 'num_blocks': [2], 'num_classes': 10},
    'block2': {'block': BasicBlock, 'in_planes': 64, 'num_blocks': [2], 'num_classes': 10},
    'block3': {'block': BasicBlock, 'in_planes': 128, 'num_blocks': [2], 'num_classes': 10},
    'block4': {'block': BasicBlock, 'in_planes': 256, 'num_blocks': [2], 'num_classes': 10}
}

# Generate cache
model_type = 'resnet18'
generator = CachedDatasetGenerator(test_set, cached_data_file=f'data/cached/{model_type}_cifar10_cache.pkl')
models = generator.load_models(model_paths, model_classes, model_configs)
generator.generate_cache(models)
