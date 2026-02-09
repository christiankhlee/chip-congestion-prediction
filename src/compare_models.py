"""
Train all models and compare their performance.

Usage:
    python src/compare_models.py
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
from src.models.simple_cnn import SimpleCNN
from src.models.unet import UNet
from src.models.gpdl_fcn import GPDL_FCN
from src.models.vit_model import ViTCongestion


def run_comparison(device='cuda'):
    """Train all models and compare."""
    
    models = {
        'SimpleCNN': SimpleCNN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS),
        'UNet': UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS),
        'GPDL_FCN': GPDL_FCN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS),
        'ViT': ViTCongestion(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS,
                              embed_dim=384, depth=6, num_heads=6),
    }
    
    all_results = {}
    all_histories = {}
    
    for name, model in models.items():
        print(f"\n{'#'*60}")
        print(f"  Training: {name}")
        print(f"{'#'*60}")
        
        start_time = time.time()
        trained_model, history = train_model(
            model, model_name=name, device=device
        )
        elapsed = time.time() - start_time
        
        results = history['test_results']
        results['training_time_min'] = elapsed / 60
        results['num_params'] = sum(p.numel() for p in model.parameters())
        
        all_results[name] = results
        all_histories[name] = history
    
    # ---- Save Results ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # ---- Print Comparison Table ----
    print(f"\n{'='*80}")
    print(f" MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<15} {'NRMSE↓':>10} {'SSIM↑':>10} {'Pearson↑':>10} {'Params':>12} {'Time(min)':>10}")
    print(f"  {'-'*67}")
    
    for name, res in all_results.items():
        print(f"  {name:<15} {res['nrmse']:>10.4f} {res['ssim']:>10.4f} "
              f"{res['pearson']:>10.4f} {res['num_params']:>12,} {res['training_time_min']:>10.1f}")
    
    # ---- Plot Training Curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, history in all_histories.items():
        axes[0].plot(history['train_loss'], label=f'{name}', alpha=0.8)
        axes[1].plot(history['val_loss'], label=f'{name}', alpha=0.8)
    
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
    plt.show()
    
    # ---- Bar Chart Comparison ----
    model_names = list(all_results.keys())
    metrics = ['nrmse', 'ssim', 'pearson']
    metric_labels = ['NRMSE ↓', 'SSIM ↑', 'Pearson R ↑']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [all_results[m][metric] for m in model_names]
        bars = axes[i].bar(model_names, values, color=colors[:len(model_names)])
        axes[i].set_title(label, fontsize=13)
        axes[i].set_ylabel(label.split(' ')[0])
        
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Comparison on CircuitNet Congestion Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.show()
    
    return all_results, all_histories


if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    run_comparison(device=device)
