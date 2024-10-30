import early_exit_mobilenetv2 as mobilenet
import torch

device = torch.device("cpu")

class MobileNetMeasurement():
    def __init__(self):
        pass

    def run_one_block(self):
        pass
    def run_two_blocks(self):
        pass
    def run_three_blocks(self):
        pass
    def run_four_blocks(self):
        pass


def run_mobilenet_oneblock():
    model = mobilenet.Block1()
    model.block1.load_state_dict(torch.load("../models/cifar10/mobilenetv2_blocks/block1.pth", map_location=device))
