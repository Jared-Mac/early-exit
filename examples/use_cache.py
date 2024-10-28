from core.dataset import CacheDataset
from torch.utils.data import DataLoader

# Load cached dataset
cached_dataset = CacheDataset(cached_data_file='data/cifar10_cache.pkl')
dataloader = DataLoader(cached_dataset, batch_size=1, shuffle=True)
