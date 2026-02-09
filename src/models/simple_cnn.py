"""
Simple CNN Baseline for congestion prediction.

Architecture: A straightforward stack of conv layers.
This won't be state-of-the-art but gives us a baseline to compare against.

Input:  (batch, 3, 256, 256)  — macro_region, RUDY, RUDY_pin
Output: (batch, 1, 256, 256)  — predicted congestion overflow
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A basic CNN that maintains spatial resolution throughout.
    Uses padding to keep H, W the same at every layer.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: in_channels → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 128 → 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 5: 64 → 32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Output: 32 → out_channels
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] since labels are normalized
        )
    
    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    model = SimpleCNN(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
