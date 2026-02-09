"""
U-Net for congestion prediction.

U-Net is an encoder-decoder architecture with skip connections.
The encoder downsamples to capture context, the decoder upsamples to
restore resolution, and skip connections carry fine-grained spatial
details from encoder to decoder.

Input:  (batch, 3, 256, 256)
Output: (batch, 1, 256, 256)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks (the basic U-Net building block)."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.
    
    Args:
        in_channels: Number of input feature channels (3)
        out_channels: Number of output channels (1)
        features: List of feature sizes for each encoder level
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=True
                )
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
