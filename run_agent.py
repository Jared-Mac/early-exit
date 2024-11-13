import torch
from adapt.dql import DQNet
import numpy as np

def test_dqlnet():
    # CIFAR-10 parameters
    image_shape = (3, 32, 32)  # channels, height, width
    num_classes = 10
    num_exits = 4
    battery_features = 1

    # Initialize network
    device = "cpu"
    model = DQNet(
        num_logits=num_classes,
        num_exit=num_exits,
        num_actions=3,
        image_shape=image_shape,
        battery_features=battery_features
    ).to(device)

    # Create dummy inputs
    batch_size = 1
    dummy_exits = torch.zeros(batch_size, num_exits)
    dummy_image = torch.randn(batch_size, *image_shape)
    dummy_logits = torch.randn(batch_size, num_classes)
    dummy_battery = torch.ones(batch_size, battery_features)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_exits, dummy_image, dummy_logits, dummy_battery)
    
    print("Model output shape:", output.shape)
    print("Q-values for actions:", output.squeeze().cpu().numpy())

if __name__ == "__main__":
    test_dqlnet()
