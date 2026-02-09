"""
GPDL-style Fully Convolutional Network for congestion prediction.

Replicates the architecture from:
"Global Placement with Deep Learning-Enabled Explicit Routability Optimization"

This is an encoder-decoder FCN WITHOUT skip connections (unlike U-Net).
This is the standard baseline used in CircuitNet's experiments.

Input:  (batch, 3, 256, 256)
Output: (batch, 1, 256, 256)
"""
import torch
import torch.nn as nn


class GPDL_FCN(nn.Module):
    """
    GPDL-style FCN encoder-decoder.
    No skip connections — decoder reconstructs purely from bottleneck.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.enc5 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (transposed convolutions for upsampling)
        self.dec5 = self._upconv_block(1024, 512)
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(128, 64)
        self.dec1 = self._upconv_block(64, 32)
        
        # Output
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e5))
        
        # Decoder — NO skip connections
        d5 = self.dec5(b)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        return self.final(d1)


if __name__ == "__main__":
    model = GPDL_FCN(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
