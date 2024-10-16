import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from early_exit_resnet import *
from core.dataset import CacheDataset

block1 = HeadNetworkPart1(block=Bottleneck, in_planes=64, num_blocks=[3], num_classes=10)
block2 = HeadNetworkPart2(Bottleneck, 256, [4], num_classes=10)
block3 = HeadNetworkPart3(block=Bottleneck, in_planes=512, num_blocks=[6], num_classes=10)
block4 = TailNetwork(block=Bottleneck, in_planes=1024, num_blocks=[3, 4, 6, 3], num_classes=10)

block1.load_state_dict(torch.load("models/cifar10/block1.pth"))
block2.load_state_dict(torch.load("models/cifar10/block2.pth"))
block3.load_state_dict(torch.load("models/cifar10/block3.pth"))
block4.load_state_dict(torch.load("models/cifar10/block4.pth"))

block1.eval()
block2.eval()
block3.eval()
block4.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
custom_test_set = CacheDataset(test_set, models={'block1': block1, 'block2': block2, 'block3': block3, 'block4': block4},compute_logits=True)