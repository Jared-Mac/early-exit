import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Softmax
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10, CIFAR100
import pickle

class Flame2Data(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        match = re.search(r'\((\d+)\)', os.path.basename(img_path))
        if match:
            frame_number = int(match.group(1))
        else:
            raise ValueError(f"Frame number not found in {img_path}")

        label = self.get_label(frame_number)
        
        if self.transform:
            image = self.transform(image)

        label_map = {'NN': 0, 'YY': 1, 'YN': 2}
        label_idx = label_map[label]

        # Convert label to tensor
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return image, label_tensor
    
    def get_label(self, frame_number):
        ranges = [
            (1, 13700, 'NN'), (13701, 14699, 'YY'), (14700, 15980, 'YN'),
            (15981, 19802, 'YY'), (19803, 19899, 'YN'), (19900, 27183, 'YY'),
            (27184, 27514, 'YN'), (27515, 31294, 'YY'), (31295, 31509, 'YN'),
            (31510, 33597, 'YY'), (33598, 33929, 'YN'), (33930, 36550, 'YY'),
            (36551, 38030, 'YN'), (38031, 38153, 'YY'), (38154, 41642, 'YN'),
            (41642, 45279, 'YY'), (45280, 51206, 'YN'), (51207, 52286, 'YY'),
            (52287, 53451, 'YN')
        ]
        
        for start, end, label in ranges:
            if start <= frame_number <= end:
                return label

        raise ValueError(f"Frame number {frame_number} out of defined ranges")

class Flame2DataModule(pl.LightningDataModule):
    def __init__(self, image_dir, batch_size=32, transform=None, train_val_test_split=(0.7, 0.15, 0.15), seed=42):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.transform = transform
        self.train_val_test_split = train_val_test_split
        self.seed = seed

    def setup(self, stage=None):
        full_dataset = Flame2Data(self.image_dir, self.transform)

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        # Calculate split lengths
        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        
    def evaluate_model(self, model, device):
        model.eval()  # Set model to evaluation mode
        model = model.to(device)
        test_loader = self.test_dataloader()  # Use the datamodule's test dataloader
        total_samples = len(test_loader.dataset)
        correct_predictions = [0, 0, 0, 0]  # Adjust if you have a different number of exits

        with torch.no_grad():  # No need to compute gradients
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                exits = model(inputs)  # Forward pass
                softmax = Softmax(dim=1)
                
                for i, exit in enumerate(exits):
                    predictions = softmax(exit).argmax(dim=1)
                    correct_predictions[i] += (predictions == labels).type(torch.float).sum().item()

        accuracies = [correct / total_samples for correct in correct_predictions]
        return accuracies

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download CIFAR10 dataset
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Transform data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Split the data
        if stage == 'fit' or stage is None:
            cifar10_full = CIFAR10(root=self.data_dir, train=True, transform=transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(root=self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=1)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download CIFAR10 dataset
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Transform data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Split the data
        if stage == 'fit' or stage is None:
            cifar100_full = CIFAR100(root=self.data_dir, train=True, transform=transform)
            self.cifar100_train, self.cifar100_val = random_split(cifar100_full, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar100_test = CIFAR100(root=self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=1)

class CacheDataset(Dataset):
    def __init__(self, base_dataset=None, cached_data_file='data/cached_logits.pkl', models=None, compute_logits=False):
        if base_dataset is not None:
            self.base_dataset = base_dataset
        self.cached_data_file = cached_data_file
        self.models = models
        for model in self.models:
            model.eval()
        if not compute_logits and os.path.exists(self.cached_data_file):
            with open(self.cached_data_file, 'rb') as f:
                self.cached_data = pickle.load(f)
                self.base_dataset = self.cached_data
        else:
            if compute_logits and models:
                print("Computing and saving logits...")
                self.cached_data = self.compute_and_cache_logits()
            else:
                raise ValueError("Compute logits is set to true but no models provided or cached data file not found.")
        
        # Store the image shape
        self.image_shape = self.cached_data[0]['image'].shape

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

    def compute_and_cache_logits(self):
        # Check for available GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move models to GPU
        for key in self.models.keys():
            self.models[key] = self.models[key].to(device)
            
        cache = []
        for idx in range(len(self.base_dataset)):
            image, label = self.base_dataset[idx]
            
            # Move image and label to GPU
            image = image.to(device)
            label = label
            
            x = image.unsqueeze(0)  # unsqueeze to add batch dimension
            features, logit1 = self.models['block1'](x)
            features, logit2 = self.models['block2'](features)
            features, logit3 = self.models['block3'](features)
            logit4 = self.models['block4'](features)
            
            logit1 = logit1.detach().squeeze(0).cpu()
            logit2 = logit2.detach().squeeze(0).cpu()
            logit3 = logit3.detach().squeeze(0).cpu()
            logit4 = logit4.detach().squeeze(0).cpu()
            
            logits = [logit1,logit2,logit3,logit4]
            cache.append({
                'image': image.cpu(),  # move image back to CPU for caching
                'logits': logits,
                'label': label  # move label back to CPU for caching
            })

        with open(self.cached_data_file, 'wb') as f:
            pickle.dump(cache, f)

        return cache