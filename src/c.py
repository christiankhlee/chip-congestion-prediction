"""
Train the Attention-Augmented U-Net and compare against previous results.

This script:
  1. Trains the new AttentionUNet model
  2. Loads the previously saved comparison_results.json
  3. Adds the new model's results
  4. Prints an updated comparison table
  5. Generates updated comparison plots

Usage:
    python src/train_attention_unet.py
"""
import os
import sys
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.train import train_model
from src.models.attention_unet import AttentionUNet


def main():
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Create model
    model = AttentionUNet(
        in_channels=NUM_INPUT_CHANNELS,
        out_channels=NUM_OUTPUT_CHANNELS,
        features=[64, 128, 256, 512],
        attn_heads=8,
        attn_depth=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"AttentionUNet parameters: {total_params:,}")

    # Train
    print(f"\n{'#'*60}")
    print(f"  Training: AttentionUNet")
    print(f"{'#'*60}")

    start_time = time.time()
    trained_model, history = train_model(
        model, model_name="attention_unet", device=device
    )
    elapsed = time.time() - start_time

    results = history['test_results']
    results['training_time_min'] = elapsed / 60
    results['num_params'] = total_params

    # Load previous results if they exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prev_results_path = os.path.join(RESULTS_DIR, 'comparison_results.json')

    if os.path.exists(prev_results_path):
        with open(prev_results_path, 'r') as f:
            all_results = json.load(f)
        print("\nLoaded previous results for comparison.")
    else:
        all_results = {}
        print("\nNo previous results found, showing only AttentionUNet.")

    all_results['AttentionUNet'] = results

    # Save updated results
    with open(prev_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*85}")
    print(f" UPDATED MODEL COMPARISON RESULTS")
    print(f"{'='*85}")
    print(f"  {'Model':<18} {'NRMSE↓':>10} {'SSIM↑':>10} {'Pearson↑':>10} {'Params':>14} {'Time(min)':>10}")
    print(f"  {'-'*72}")

    for name, res in all_results.items():
        print(f"  {name:<18} {res['nrmse']:>10.4f} {res['ssim']:>10.4f} "
              f"{res['pearson']:>10.4f} {res['num_params']:>14,} {res['training_time_min']:>10.1f}")

    # Highlight if AttentionUNet wins any metric
    print(f"\n  --- Analysis ---")
    metrics = {'nrmse': 'min', 'ssim': 'max', 'pearson': 'max'}
    for metric, direction in metrics.items():
        if direction == 'min':
            best_model = min(all_results, key=lambda m: all_results[m][metric])
        else:
            best_model = max(all_results, key=lambda m: all_results[m][metric])
        marker = " ← NEW BEST!" if best_model == 'AttentionUNet' else ""
        print(f"  Best {metric}: {best_model} ({all_results[best_model][metric]:.4f}){marker}")

    # Generate updated bar chart
    model_names = list(all_results.keys())
    metrics_plot = ['nrmse', 'ssim', 'pearson']
    metric_labels = ['NRMSE ↓', 'SSIM ↑', 'Pearson R ↑']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (metric, label) in enumerate(zip(metrics_plot, metric_labels)):
        values = [all_results[m][metric] for m in model_names]
        bar_colors = [colors[j % len(colors)] for j in range(len(model_names))]
        # Highlight AttentionUNet
        if 'AttentionUNet' in model_names:
            idx = model_names.index('AttentionUNet')
            bar_colors[idx] = '#F44336'

        bars = axes[i].bar(model_names, values, color=bar_colors)
        axes[i].set_title(label, fontsize=13)
        axes[i].set_ylabel(label.split(' ')[0])
        axes[i].tick_params(axis='x', rotation=20)

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model Comparison — Including Attention-Augmented U-Net (red)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_with_attention.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved updated comparison to {RESULTS_DIR}/")
    print(f"Run visualization: python src/visualize.py --model attention_unet")
    print(f"Run feature importance: python src/feature_importance.py --model attention_unet")


if __name__ == "__main__":
    main()
