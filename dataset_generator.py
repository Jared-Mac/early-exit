import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from typing import Dict, Optional, Union, List

class CachedDatasetGenerator:
    """Generates cached datasets from different block models."""
    
    def __init__(self, base_dataset: Dataset, 
                 cached_data_file: str = 'data/cached_logits.pkl',
                 device: Optional[str] = None):
        """
        Initialize the dataset generator.
        
        Args:
            base_dataset: Base dataset to generate cache from
            cached_data_file: Path to save/load cached data
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.base_dataset = base_dataset
        self.cached_data_file = cached_data_file
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_cache(self, models: Dict[str, torch.nn.Module]) -> None:
        """
        Generate and save cache from provided models.
        
        Args:
            models: Dictionary of models with keys like 'block1', 'block2', etc.
        """
        # Move models to appropriate device
        for model in models.values():
            model.to(self.device)
            model.eval()

        cache = []
        with torch.no_grad():
            for idx in range(len(self.base_dataset)):
                image, label = self.base_dataset[idx]
                image = image.to(self.device)
                
                # Forward pass through each block
                features = image.unsqueeze(0)  # Add batch dimension
                logits = []
                
                for model in models.values():
                    if hasattr(model, 'forward') and len(model._forward_pre_hooks) == 0:
                        # Single output model (e.g., final block)
                        features = model(features)
                        logits.append(features[0].squeeze(0).cpu())  # Extract first element if tuple
                    else:
                        # Early exit model
                        features, exit_logits = model(features)
                        logits.append(exit_logits.squeeze(0).cpu())

                cache.append({
                    'image': image.cpu(),
                    'logits': logits,
                    'label': label
                })

                if idx % 1000 == 0:
                    print(f"Processed {idx}/{len(self.base_dataset)} images")

        # Save cache
        os.makedirs(os.path.dirname(self.cached_data_file), exist_ok=True)
        with open(self.cached_data_file, 'wb') as f:
            pickle.dump(cache, f)

    @staticmethod
    def load_models(model_paths: Dict[str, str], 
                   model_classes: Dict[str, type],
                   model_configs: Dict[str, dict]) -> Dict[str, torch.nn.Module]:
        """
        Load models from paths with specified configurations.
        
        Args:
            model_paths: Dictionary mapping block names to model file paths
            model_classes: Dictionary mapping block names to model classes
            model_configs: Dictionary mapping block names to model configurations
            
        Returns:
            Dictionary of loaded models
        """
        models = {}
        for block_name in model_paths:
            # Initialize model
            model_class = model_classes[block_name]
            model = model_class(**model_configs[block_name])
            
            # Load state dict
            state_dict = torch.load(model_paths[block_name], map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            
            models[block_name] = model
            
        return models
