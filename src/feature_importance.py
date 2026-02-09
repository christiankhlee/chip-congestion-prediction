"""
Feature Importance Analysis for Congestion Prediction.

Three methods:
  1. Ablation Study: Zero out one feature channel → measure performance drop
  2. Gradient Saliency: Which pixels does the model look at most?
  3. Channel-wise importance: Average gradient magnitude per channel

Run from project root:
    python src/feature_importance.py --model unet
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model, compute_nrmse


def ablation_study(model, test_loader, device='cuda'):
    """
    Zero out one feature channel at a time and see how much
    performance degrades. Bigger degradation = more important feature.
    """
    print("\n--- Ablation Study ---")
    
    # Baseline: all features
    baseline_results = evaluate_model(model, test_loader, device)
    baseline_nrmse = baseline_results['nrmse']
    print(f"  Baseline NRMSE (all features): {baseline_nrmse:.4f}")
    
    importance = {}
    
    for ch_idx, feat_name in enumerate(FEATURE_NAMES):
        print(f"  Ablating: {feat_name} (channel {ch_idx})...")
        
        model.eval()
        nrmse_scores = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                # Zero out this channel
                features_ablated = features.clone()
                features_ablated[:, ch_idx, :, :] = 0.0
                
                preds = model(features_ablated)
                
                for i in range(preds.shape[0]):
                    nrmse_scores.append(
                        compute_nrmse(
                            preds[i, 0].cpu().numpy(),
                            labels[i, 0].cpu().numpy()
                        )
                    )
        
        ablated_nrmse = np.mean(nrmse_scores)
        degradation = ablated_nrmse - baseline_nrmse
        importance[feat_name] = {
            'nrmse_without': ablated_nrmse,
            'degradation': degradation,
            'relative_importance': degradation / max(baseline_nrmse, 1e-8) * 100
        }
        print(f"    NRMSE without {feat_name}: {ablated_nrmse:.4f} "
              f"(+{degradation:.4f}, {degradation/max(baseline_nrmse, 1e-8)*100:.1f}% worse)")
    
    return importance


def gradient_saliency(model, test_loader, device='cuda', num_samples=50):
    """
    Compute gradient of output w.r.t. input.
    High gradient = model is sensitive to changes in that pixel.
    """
    print("\n--- Gradient Saliency ---")
    model.eval()
    
    accumulated_saliency = None
    count = 0
    
    for features, labels in test_loader:
        if count >= num_samples:
            break
        
        features = features.to(device).requires_grad_(True)
        preds = model(features)
        
        loss = preds.sum()
        loss.backward()
        
        saliency = features.grad.abs().cpu().numpy()
        
        if accumulated_saliency is None:
            accumulated_saliency = saliency.sum(axis=0)
        else:
            accumulated_saliency += saliency.sum(axis=0)
        
        count += features.shape[0]
    
    accumulated_saliency /= count
    print(f"  Computed saliency over {count} samples")
    
    # Channel-wise importance (average saliency per channel)
    print("\n  Channel-wise gradient importance:")
    for i, name in enumerate(FEATURE_NAMES):
        mean_sal = accumulated_saliency[i].mean()
        print(f"    {name}: {mean_sal:.6f}")
    
    return accumulated_saliency


def plot_feature_importance(importance, save_path=None):
    """Plot ablation study results."""
    names = list(importance.keys())
    degradations = [importance[n]['relative_importance'] for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    
    bars = ax.barh(names, degradations, color=colors)
    ax.set_xlabel('Performance Degradation (%)\n(higher = more important)')
    ax.set_title('Feature Importance via Ablation Study\n'
                 '(How much worse is the model without each feature?)')
    
    for bar, val in zip(bars, degradations):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_saliency_maps(saliency, save_path=None):
    """Plot saliency maps for each feature channel."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, FEATURE_NAMES)):
        im = ax.imshow(saliency[i], cmap='magma', interpolation='nearest')
        ax.set_title(f'Saliency: {name}', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Gradient Saliency Maps\n(Brighter = model pays more attention)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unet', choices=['simple_cnn', 'unet', 'gpdl_fcn', 'vit'])
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    # Load model
    if args.model == 'simple_cnn':
        from src.models.simple_cnn import SimpleCNN
        model = SimpleCNN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'unet':
        from src.models.unet import UNet
        model = UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'gpdl_fcn':
        from src.models.gpdl_fcn import GPDL_FCN
        model = GPDL_FCN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'vit':
        from src.models.vit_model import ViTCongestion
        model = ViTCongestion(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    
    # Load trained weights
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ No checkpoint found at {ckpt_path}")
        print(f"   Train the model first: python src/train.py --model {args.model}")
        sys.exit(1)
    
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    print(f"Loaded {args.model} from epoch {checkpoint['epoch']}")
    
    # Get data
    _, _, test_loader = get_dataloaders(batch_size=8, num_workers=0)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run ablation study
    importance = ablation_study(model, test_loader, args.device)
    plot_feature_importance(importance,
                           os.path.join(RESULTS_DIR, f'feature_importance_{args.model}.png'))
    
    # Run gradient saliency
    saliency = gradient_saliency(model, test_loader, args.device)
    plot_saliency_maps(saliency,
                       os.path.join(RESULTS_DIR, f'saliency_maps_{args.model}.png'))
