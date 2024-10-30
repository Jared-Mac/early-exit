import early_exit_resnet as resnet_split
import early_exit_mobilenetv2 as mobilenet_split
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch

device = torch.device("cpu")

# block1 = HeadNetworkPart1(block=Bottleneck, in_planes=64, num_blocks=[3], num_classes=10)
# block2 = HeadNetworkPart2(Bottleneck, 256, [4], num_classes=10)
# block3 = HeadNetworkPart3(block=Bottleneck, in_planes=512, num_blocks=[6], num_classes=10)
# block4 = TailNetwork(block=Bottleneck, in_planes=1024, num_blocks=[3, 4, 6, 3], num_classes=10)
#
# model = EarlyExitResNet50(num_classes=3)
#
# block1_state_dict = {}
# block2_state_dict = {}
# block3_state_dict = {}
# block4_state_dict = {}
#
# block1.load_state_dict(torch.load("models/cifar10/block1.pth", map_location=device))
# block2.load_state_dict(torch.load("models/cifar10/block2.pth", map_location=device))
# block3.load_state_dict(torch.load("models/cifar10/block3.pth", map_location=device))
# block4.load_state_dict(torch.load("models/cifar10/block4.pth", map_location=device))
#
# model.head1 = block1
# model.head2 = block2
# model.head3 = block3
# model.tail = block4

# test mobilenet
mobilenet_block1 = mobilenet_split.Block1()
mobilenet_block1.load_state_dict(torch.load("models/cifar10/mobilenetv2_blocks/block1.pth", map_location=device))

def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    total_samples = len(test_loader.dataset)
    correct = 0 # Assuming three exits

    with torch.no_grad():  # No need to compute gradients
        for inputs, labels in test_loader:
            # print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            exit = model(inputs)[1]  # Forward pass
            softmax = torch.nn.Softmax(dim=1)
            exit_soft = softmax(exit)
            predictions = exit_soft.argmax(dim=1)
            # print(inputs.shape)
            correct += 1 if (predictions == labels) else 0

    accuracy = correct / total_samples
    return accuracy

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True,transform=transform)
dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

accuracy = evaluate_model(mobilenet_block1, dataloader, 'cpu')
print(accuracy)
