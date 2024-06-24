import torch
if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using the GPU.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch is using the CPU.")