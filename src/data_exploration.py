"""
Data Exploration — Understand what the data looks like before modeling.

Run from the project root:
    python src/data_exploration.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.dataset import get_dataloaders

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load a batch
# ------------------------------------------------------------------
print("Loading data...")
train_loader, _, _ = get_dataloaders(batch_size=8, num_workers=0)
features, labels = next(iter(train_loader))

print(f"Features: {features.shape}")  # (8, 3, 256, 256)
print(f"Labels:   {labels.shape}")    # (8, 1, 256, 256)

# ------------------------------------------------------------------
# 2. Visualize one sample — all features + label side by side
# ------------------------------------------------------------------
sample_idx = 0
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, name in enumerate(FEATURE_NAMES):
    ax = axes[i]
    im = ax.imshow(features[sample_idx, i].numpy(), cmap='hot', interpolation='nearest')
    ax.set_title(f"INPUT: {name}", fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Label
ax = axes[3]
im = ax.imshow(labels[sample_idx, 0].numpy(), cmap='hot', interpolation='nearest')
ax.set_title("LABEL: Congestion Overflow", fontsize=11)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Sample Visualization: Features → Label", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "data_exploration_sample.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {RESULTS_DIR}/data_exploration_sample.png")

# ------------------------------------------------------------------
# 3. Visualize multiple samples
# ------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(20, 16))

for row in range(4):
    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[row, i]
        im = ax.imshow(features[row, i].numpy(), cmap='hot', interpolation='nearest')
        if row == 0:
            ax.set_title(name, fontsize=11)
        ax.axis('off')
    
    ax = axes[row, 3]
    im = ax.imshow(labels[row, 0].numpy(), cmap='hot', interpolation='nearest')
    if row == 0:
        ax.set_title("Congestion Label", fontsize=11)
    ax.axis('off')

plt.suptitle("Multiple Samples: Features → Labels", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "data_exploration_multi.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {RESULTS_DIR}/data_exploration_multi.png")

# ------------------------------------------------------------------
# 4. Distribution of pixel values
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, name in enumerate(FEATURE_NAMES):
    vals = features[:, i].numpy().flatten()
    axes[i].hist(vals, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
    axes[i].set_title(f"{name}\nmean={vals.mean():.3f}, std={vals.std():.3f}")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

vals = labels[:, 0].numpy().flatten()
axes[3].hist(vals, bins=100, alpha=0.7, color='coral', edgecolor='none')
axes[3].set_title(f"Congestion Label\nmean={vals.mean():.3f}, std={vals.std():.3f}")
axes[3].set_xlabel("Value")

plt.suptitle("Pixel Value Distributions", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "data_exploration_distributions.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {RESULTS_DIR}/data_exploration_distributions.png")

# ------------------------------------------------------------------
# 5. Feature-label correlations
# ------------------------------------------------------------------
print("\n--- Feature-Label Correlations ---")
for i, feat_name in enumerate(FEATURE_NAMES):
    feat_flat = features[:, i].numpy().flatten()
    label_flat = labels[:, 0].numpy().flatten()
    corr = np.corrcoef(feat_flat, label_flat)[0, 1]
    print(f"  {feat_name:20s} ↔ congestion: r = {corr:.4f}")

# ------------------------------------------------------------------
# 6. Label sparsity analysis
# ------------------------------------------------------------------
label_vals = labels[:, 0].numpy()
zero_pct = (label_vals == 0).sum() / label_vals.size * 100
nonzero_vals = label_vals[label_vals > 0]
print(f"\n--- Label Sparsity ---")
print(f"  {zero_pct:.1f}% of pixels are zero (no congestion)")
print(f"  Non-zero pixels: mean={nonzero_vals.mean():.4f}, max={nonzero_vals.max():.4f}")
print(f"  Total samples in dataset: {len(train_loader.dataset) + 1536 + 1536}")

print(f"\n✅ Exploration complete! Check {RESULTS_DIR}/ for plots.")
