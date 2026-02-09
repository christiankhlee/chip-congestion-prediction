"""
Vision Transformer (ViT) based model for congestion prediction.

Uses a ViT encoder to capture global relationships across the entire chip,
combined with a CNN decoder to produce the full-resolution output.

Input:  (batch, 3, 256, 256)
Output: (batch, 1, 256, 256)
"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""
    
    def __init__(self, in_channels=3, patch_size=16, embed_dim=384, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block."""
    
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CNNDecoder(nn.Module):
    """Upsample from (B, C, 16, 16) to (B, 1, 256, 256)."""
    
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.decoder(x)


class ViTCongestion(nn.Module):
    """
    Complete ViT-based congestion prediction model.
    
    Args:
        in_channels: Input feature channels (3)
        out_channels: Output channels (1)
        img_size: Input image size (256)
        patch_size: Size of each patch (16)
        embed_dim: Transformer embedding dimension (384)
        depth: Number of transformer blocks (6)
        num_heads: Number of attention heads (6)
    """
    
    def __init__(self, in_channels=3, out_channels=1, img_size=256,
                 patch_size=16, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        self.patch_embed = PatchEmbedding(
            in_channels, patch_size, embed_dim, img_size
        )
        
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        self.decoder = CNNDecoder(embed_dim, out_channels)
    
    def forward(self, x):
        B = x.shape[0]
        
        tokens = self.patch_embed(x)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        
        feature_map = tokens.transpose(1, 2).reshape(
            B, -1, self.grid_size, self.grid_size
        )
        
        output = self.decoder(feature_map)
        return output


if __name__ == "__main__":
    model = ViTCongestion(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
