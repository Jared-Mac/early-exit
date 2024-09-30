import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class Bottleneck(pl.LightningModule):
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

class EarlyExitBlock(pl.LightningModule):
    def __init__(self, in_planes, num_classes):
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

class HeadNetworkPart1(pl.LightningModule):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(HeadNetworkPart1, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.early_exit_1 = EarlyExitBlock(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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

class HeadNetworkPart2(pl.LightningModule):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(HeadNetworkPart2, self).__init__()
        self.in_planes = in_planes
        self.layer2 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.early_exit_2 = EarlyExitBlock(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        ee2_out = self.early_exit_2(out)
        return out, ee2_out

class HeadNetworkPart3(pl.LightningModule):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(HeadNetworkPart3, self).__init__()
        self.in_planes = in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.early_exit_3 = EarlyExitBlock(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        ee3_out = self.early_exit_3(out)
        return out, ee3_out

class TailNetwork(pl.LightningModule):
    def __init__(self, block, in_planes, num_blocks, num_classes=10):
        super(TailNetwork, self).__init__()
        self.in_planes = in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[0], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        final_out = self.linear(out)
        return final_out

class EarlyExitResNet50(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(EarlyExitResNet50, self).__init__()
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.head1 = HeadNetworkPart1(Bottleneck, 64, [3], num_classes)
        self.head2 = HeadNetworkPart2(Bottleneck, 256, [4], num_classes)
        self.head3 = HeadNetworkPart3(Bottleneck, 512, [6], num_classes)
        self.tail = TailNetwork(Bottleneck, 1024, [3], num_classes)
        self.accuracy1 = torchmetrics.Accuracy(num_classes=num_classes,task="multiclass")
        self.accuracy2 = torchmetrics.Accuracy(num_classes=num_classes,task="multiclass")
        self.accuracy3 = torchmetrics.Accuracy(num_classes=num_classes,task="multiclass")
        self.accuracyfinal = torchmetrics.Accuracy(num_classes=num_classes,task="multiclass")

        self.test_step_outputs = []
        self.save_hyperparameters()
    def forward(self, x):
        out, ee1_out = self.head1(x)
        out, ee2_out = self.head2(out)
        out, ee3_out = self.head3(out)
        final_out = self.tail(out)
        return ee1_out, ee2_out, ee3_out, final_out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = ce_loss1 + ce_loss2 + ce_loss3 + ce_loss_final
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = ce_loss1 + ce_loss2 + ce_loss3 + ce_loss_final
        self.log('val_loss', loss.item())
        return loss
    
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = ce_loss1 + ce_loss2 + ce_loss3 + ce_loss_final
        self.log('test_loss', loss.item())

        self.test_step_outputs.append(loss)
        preds1 = torch.argmax(ee1_out, dim=1)
        preds2 = torch.argmax(ee2_out, dim=1)
        preds3 = torch.argmax(ee3_out, dim=1)
        predsfinal = torch.argmax(final_out, dim=1)

        acc1 = self.accuracy1(preds1, targets)
        acc2 = self.accuracy2(preds2, targets)
        acc3 = self.accuracy3(preds3, targets)
        accfinal = self.accuracyfinal(predsfinal, targets)
        
        self.log('test_acc_exit_1', acc1, on_step=True, on_epoch=True)
        self.log('test_acc_exit_2', acc2, on_step=True, on_epoch=True)
        self.log('test_acc_exit_3', acc3, on_step=True, on_epoch=True)
        self.log('test_acc_exit_final', accfinal, on_step=True, on_epoch=True)

        return acc1, acc2, acc3, accfinal
        
    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        self.log("test_epoch_average", epoch_average)
        self.test_step_outputs.clear()  # free memory
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)