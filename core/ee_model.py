import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os 
from tqdm import tqdm  # Import tqdm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EarlyExitBlock(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(EarlyExitBlock, self).__init__()
        super(EarlyExitBlock, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_planes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class EarlyExitResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(EarlyExitResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.early_exit_1 = EarlyExitBlock(64 * block.expansion, num_classes)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.early_exit_2 = EarlyExitBlock(128 * block.expansion, num_classes)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        # First early exit path
        ee1_out = self.early_exit_1(out)
        out = self.layer2(out)
        
        # Second early exit path
        ee2_out = self.early_exit_2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        final_out = self.linear(out)  # Final classifier output
        
        # Return all outputs
        return ee1_out, ee2_out, final_out
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class EarlyExitResNet50(EarlyExitResNet):
    def __init__(self, num_classes=10):
        super(EarlyExitResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes)

# Define the ResNet architecture with early exits
def EarlyExitResNet18(num_classes):
    return EarlyExitResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
class HeadNetworkPart1(nn.Module):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(HeadNetworkPart1, self).__init__()
        self.in_planes = in_planes

        # Initial convolution and batch normalization
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        # First block
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        
        # Early exit after the first block
        self.early_exit_1 = EarlyExitBlock(64 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        ee1_out = self.early_exit_1(out)
        return out, ee1_out

class HeadNetworkPart2(nn.Module):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(HeadNetworkPart2, self).__init__()
        self.in_planes = in_planes  # Adjust according to the output of Part 1

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.early_exit_2 = EarlyExitBlock(128 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        ee2_out = self.early_exit_2(out)
        return out, ee2_out

class TailNetwork(nn.Module):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(TailNetwork, self).__init__()
        self.in_planes = in_planes  # This should be set to the output planes of the last layer of the head network

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        final_out = self.linear(out)
        return final_out
# Define the transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])