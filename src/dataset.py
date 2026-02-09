"""
Dataset loader for CircuitNet congestion prediction.

Each sample consists of:
  - Feature: (256, 256, 3) numpy array — macro_region, RUDY, RUDY_pin
  - Label:   (256, 256, 1) numpy array — combined H+V congestion overflow

Both are already normalized to [0, 1] by generate_training_set.py.
Files are .npy format (no extension in some cases).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class CircuitNetCongestionDataset(Dataset):
    """
    Loads preprocessed CircuitNet .npy files for congestion prediction.
    
    The generate_training_set.py script already:
      - Resized everything to 256x256
      - Normalized features to [0, 1]
      - Combined H+V overflow into one label
    
    So we just load, convert to tensors, and rearrange to (C, H, W) for PyTorch.
    """
    
    def __init__(self, file_list, feature_dir, label_dir, augment=False):
        """
        Args:
            file_list: List of filenames (e.g., ['1-RISCY-a-1-c2-u0.7-m1-p1-f0.npy', ...])
            feature_dir: Path to the feature directory
            label_dir: Path to the label directory
            augment: Whether to apply data augmentation (random flips/rotations)
        """
        self.file_list = file_list
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Load feature: (256, 256, 3) and label: (256, 256, 1)
        feature = np.load(os.path.join(self.feature_dir, filename)).astype(np.float32)
        label = np.load(os.path.join(self.label_dir, filename)).astype(np.float32)
        
        # Data augmentation (random flips — helps model generalize)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                feature = np.flip(feature, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                feature = np.flip(feature, axis=0).copy()
                label = np.flip(label, axis=0).copy()
        
        # Convert from (H, W, C) → (C, H, W) for PyTorch
        feature = torch.from_numpy(feature).permute(2, 0, 1)  # (3, 256, 256)
        label = torch.from_numpy(label).permute(2, 0, 1)      # (1, 256, 256)
        
        return feature, label


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=4):
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get all .npy files
    all_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith('.npy')])
    
    if len(all_files) == 0:
        raise ValueError(
            f"No .npy files found in '{FEATURE_DIR}'!\n"
            f"Make sure you ran generate_training_set.py first."
        )
    
    print(f"Found {len(all_files)} samples")
    
    # Split into train / val / test
    train_files, temp_files = train_test_split(
        all_files, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files, test_size=(1 - val_ratio_adjusted), random_state=RANDOM_SEED
    )
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets
    train_dataset = CircuitNetCongestionDataset(
        train_files, FEATURE_DIR, LABEL_DIR, augment=True
    )
    val_dataset = CircuitNetCongestionDataset(
        val_files, FEATURE_DIR, LABEL_DIR, augment=False
    )
    test_dataset = CircuitNetCongestionDataset(
        test_files, FEATURE_DIR, LABEL_DIR, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# Quick test — run this file directly to check your data
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CircuitNet Data Loading")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    # Grab one batch
    features, labels = next(iter(train_loader))
    print(f"\nFeature batch shape: {features.shape}")   # Expected: (4, 3, 256, 256)
    print(f"Label batch shape:   {labels.shape}")       # Expected: (4, 1, 256, 256)
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Label range:   [{labels.min():.4f}, {labels.max():.4f}]")
    print("\n✅ Data loading works!")
