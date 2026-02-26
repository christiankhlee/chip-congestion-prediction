"""
Attention-Augmented U-Net for congestion prediction.
(Lightweight version - optimized for training on Apple MPS)

MOTIVATION (from our experimental analysis):
  - U-Net has the best Pearson R (0.5016) — skip connections preserve local spatial detail
  - ViT has the best SSIM (0.6268) — self-attention captures global chip-wide patterns
  - Neither model dominates both metrics

HYPOTHESIS:
  Adding self-attention to U-Net's bottleneck gives us BOTH:
  - Local precision from skip connections (U-Net's strength)
  - Global context from attention (ViT's strength)

ARCHITECTURE:
  Same encoder-decoder structure as U-Net, but with a Transformer-based
  bottleneck that applies multi-head self-attention to the compressed
  feature map before decoding.

  Encoder (same as U-Net):
    256×256×3 → 128×128×64 → 64×64×128 → 32×32×256 → 16×16×512

  Bottleneck (NEW — Transformer attention):
    16×16×512 → project to 256 tokens × 512 dim
    → 2 layers of Multi-Head Self-Attention + FFN
    → project back and reshape to 16×16×1024

  Decoder (same as U-Net, with skip connections):
    16×16×1024 → 32×32×512 → 64×64×256 → 128×128×128 → 256×256×64

  Output: 256×256×1

Input:  (batch, 3, 256, 256)
Output: (batch, 1, 256, 256)
"""
import torch
import torch.nn as nn
import math


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks (standard U-Net building block)."""

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


class TransformerBottleneck(nn.Module):
    """
    Lightweight Transformer bottleneck.
    
    Projects the high-dimensional bottleneck features down to a smaller
    dimension for attention (cheaper), then projects back up.
    
    16×16×1024 → project to 16×16×512 → attention on 256 tokens × 512 dim → project back to 16×16×1024
    
    This keeps the global attention benefit while being ~4x faster than
    running attention on the full 1024-dim features.
    """

    def __init__(self, in_dim=1024, attn_dim=512, num_heads=8, depth=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()

        self.attn_dim = attn_dim

        # Project down for cheaper attention
        self.proj_in = nn.Linear(in_dim, attn_dim)
        self.proj_out = nn.Linear(attn_dim, in_dim)

        # Learnable position embeddings for the 16×16 = 256 spatial positions
        self.pos_embed = nn.Parameter(torch.randn(1, 256, attn_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(attn_dim),
                'attn': nn.MultiheadAttention(
                    attn_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'norm2': nn.LayerNorm(attn_dim),
                'mlp': nn.Sequential(
                    nn.Linear(attn_dim, int(attn_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(attn_dim * mlp_ratio), attn_dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.norm_out = nn.LayerNorm(attn_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape: (B, C, H, W) → (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)

        # Project down: (B, 256, 1024) → (B, 256, 512)
        tokens = self.proj_in(tokens)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :H*W, :]

        # Apply transformer layers
        for layer in self.layers:
            normed = layer['norm1'](tokens)
            attn_out, _ = layer['attn'](normed, normed, normed)
            tokens = tokens + attn_out
            tokens = tokens + layer['mlp'](layer['norm2'](tokens))

        tokens = self.norm_out(tokens)

        # Project back up: (B, 256, 512) → (B, 256, 1024)
        tokens = self.proj_out(tokens)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


class AttentionUNet(nn.Module):
    """
    Attention-Augmented U-Net (Lightweight).
    
    Standard U-Net encoder-decoder with skip connections, but the bottleneck
    is replaced with a Transformer that applies self-attention across the
    entire compressed feature map.
    
    This version uses a projection trick: project 1024-dim features down to 
    512-dim before attention, then back up. This makes attention ~4x cheaper
    while preserving the global context benefit.
    """

    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512],
                 attn_heads=8, attn_depth=2):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path (same as U-Net)
        ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch, feature))
            ch = feature

        # Bottleneck: Conv to expand channels, then lightweight Transformer
        bottleneck_dim = features[-1] * 2  # 1024
        self.bottleneck_conv = DoubleConv(features[-1], bottleneck_dim)
        self.bottleneck_attn = TransformerBottleneck(
            in_dim=bottleneck_dim,
            attn_dim=512,
            num_heads=attn_heads,
            depth=attn_depth,
        )

        # Decoder path (same as U-Net, with skip connections)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck: conv + attention
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)

        # Decoder with skip connections
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
    model = AttentionUNet(in_channels=3, out_channels=1, attn_heads=8, attn_depth=2)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    enc_params = sum(p.numel() for n, p in model.named_parameters() if 'downs' in n)
    btl_conv = sum(p.numel() for n, p in model.named_parameters() if 'bottleneck_conv' in n)
    attn_params = sum(p.numel() for n, p in model.named_parameters() if 'bottleneck_attn' in n)
    dec_params = sum(p.numel() for n, p in model.named_parameters() if 'ups' in n)
    print(f"\nEncoder:        {enc_params:,}")
    print(f"Bottleneck CNN: {btl_conv:,}")
    print(f"Attention:      {attn_params:,}")
    print(f"Decoder:        {dec_params:,}")
