import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class InvertedResidual(pl.LightningModule):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EarlyExitBlock(pl.LightningModule):
    def __init__(self, in_planes, num_classes):
        super(EarlyExitBlock, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_planes, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class Block1(pl.LightningModule):
    def __init__(self, num_classes=10, input_channels=3):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.inv_res1 = InvertedResidual(32, 16, 1, 1)
        self.inv_res2 = InvertedResidual(16, 24, 2, 6)
        self.early_exit_1 = EarlyExitBlock(24, num_classes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.inv_res1(out)
        out = self.inv_res2(out)
        ee1_out = self.early_exit_1(out)
        return out, ee1_out

class Block2(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(Block2, self).__init__()
        self.inv_res3 = InvertedResidual(24, 24, 1, 6)
        self.inv_res4 = InvertedResidual(24, 32, 2, 6)
        self.inv_res5 = InvertedResidual(32, 32, 1, 6)
        self.early_exit_2 = EarlyExitBlock(32, num_classes)

    def forward(self, x):
        out = self.inv_res3(x)
        out = self.inv_res4(out)
        out = self.inv_res5(out)
        ee2_out = self.early_exit_2(out)
        return out, ee2_out

class Block3(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(Block3, self).__init__()
        self.inv_res6 = InvertedResidual(32, 64, 2, 6)
        self.inv_res7 = InvertedResidual(64, 64, 1, 6)
        self.inv_res8 = InvertedResidual(64, 64, 1, 6)
        self.early_exit_3 = EarlyExitBlock(64, num_classes)

    def forward(self, x):
        out = self.inv_res6(x)
        out = self.inv_res7(out)
        out = self.inv_res8(out)
        ee3_out = self.early_exit_3(out)
        return out, ee3_out

class Block4(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(Block4, self).__init__()
        self.inv_res9 = InvertedResidual(64, 96, 1, 6)
        self.inv_res10 = InvertedResidual(96, 96, 1, 6)
        self.inv_res11 = InvertedResidual(96, 96, 1, 6)
        self.conv2 = nn.Conv2d(96, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.inv_res9(x)
        out = self.inv_res10(out)
        out = self.inv_res11(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        final_out = self.linear(out)
        return final_out

class EarlyExitMobileNetV2(pl.LightningModule):
    def __init__(self, num_classes=10, input_channels=3, input_height=32, input_width=32, loss_weights=[0.25, 0.25, 0.25, 0.25]):
        super(EarlyExitMobileNetV2, self).__init__()
        self.example_input_array = torch.rand(1, input_channels, input_height, input_width)
        self.block1 = Block1(num_classes, input_channels)
        self.block2 = Block2(num_classes)
        self.block3 = Block3(num_classes)
        self.block4 = Block4(num_classes)
        self.accuracy1 = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.accuracy2 = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.accuracy3 = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.accuracyfinal = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.name = "mobilenetv2"
        self.loss_weights = loss_weights
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        out, ee1_out = self.block1(x)
        out, ee2_out = self.block2(out)
        out, ee3_out = self.block3(out)
        final_out = self.block4(out)
        return ee1_out, ee2_out, ee3_out, final_out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = (self.loss_weights[0] * ce_loss1 + 
                self.loss_weights[1] * ce_loss2 + 
                self.loss_weights[2] * ce_loss3 + 
                self.loss_weights[3] * ce_loss_final)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = (self.loss_weights[0] * ce_loss1 + 
                self.loss_weights[1] * ce_loss2 + 
                self.loss_weights[2] * ce_loss3 + 
                self.loss_weights[3] * ce_loss_final)
        self.log('val_loss', loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        ee1_out, ee2_out, ee3_out, final_out = self(inputs)
        ce_loss1 = F.cross_entropy(ee1_out, targets)
        ce_loss2 = F.cross_entropy(ee2_out, targets)
        ce_loss3 = F.cross_entropy(ee3_out, targets)
        ce_loss_final = F.cross_entropy(final_out, targets)
        loss = (self.loss_weights[0] * ce_loss1 + 
                self.loss_weights[1] * ce_loss2 + 
                self.loss_weights[2] * ce_loss3 + 
                self.loss_weights[3] * ce_loss_final)
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
