"""
Visualization utilities for congestion prediction results.

Run from project root:
    python src/visualize.py --model unet
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.dataset import get_dataloaders


def visualize_predictions(model, test_loader, model_name, device='cuda',
                          num_samples=4, save_path=None):
    """
    Show predicted vs actual congestion maps side by side.
    For each sample: Feature channels | Predicted | Actual | Error
    """
    model.eval()
    features, labels = next(iter(test_loader))
    features = features[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        preds = model(features).cpu()
    
    features = features.cpu()
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    
    for i in range(num_samples):
        # Feature channels
        for j, name in enumerate(FEATURE_NAMES):
            axes[i, j].imshow(features[i, j].numpy(), cmap='hot', vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(f'Input: {name}', fontsize=10)
            axes[i, j].axis('off')
        
        # Predicted
        axes[i, 3].imshow(preds[i, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        if i == 0:
            axes[i, 3].set_title('Predicted', fontsize=10)
        axes[i, 3].axis('off')
        
        # Actual
        axes[i, 4].imshow(labels[i, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        if i == 0:
            axes[i, 4].set_title('Actual', fontsize=10)
        axes[i, 4].axis('off')
        
        # Error
        error = np.abs(preds[i, 0].numpy() - labels[i, 0].numpy())
        axes[i, 5].imshow(error, cmap='Reds', vmin=0)
        if i == 0:
            axes[i, 5].set_title('|Error|', fontsize=10)
        axes[i, 5].axis('off')
    
    plt.suptitle(f'Congestion Prediction: {model_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def inference_speed_comparison(models_dict, device='cuda', num_runs=100):
    """Measure inference speed for each model."""
    print("\n--- Inference Speed Comparison ---")
    x = torch.randn(1, 3, 256, 256).to(device)
    
    results = {}
    for name, model in models_dict.items():
        model = model.to(device).eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / num_runs * 1000
        results[name] = elapsed
        print(f"  {name:<15}: {elapsed:.2f} ms per inference")
    
    print(f"\n  Traditional global routing: ~30-120 minutes")
    fastest = min(results.values())
    print(f"  Speedup vs routing: ~{60*60*1000 / fastest:.0f}x faster!")
    
    return results


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
    if args.model == 'unet':
        from src.models.unet import UNet
        model = UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'simple_cnn':
        from src.models.simple_cnn import SimpleCNN
        model = SimpleCNN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'gpdl_fcn':
        from src.models.gpdl_fcn import GPDL_FCN
        model = GPDL_FCN(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    elif args.model == 'vit':
        from src.models.vit_model import ViTCongestion
        model = ViTCongestion(NUM_INPUT_CHANNELS, NUM_OUTPUT_CHANNELS)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ No checkpoint found at {ckpt_path}")
        print(f"   Train the model first: python src/train.py --model {args.model}")
        sys.exit(1)
    
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    _, _, test_loader = get_dataloaders(batch_size=8, num_workers=0)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    visualize_predictions(model, test_loader, args.model, device=args.device,
                          save_path=os.path.join(RESULTS_DIR, f'predictions_{args.model}.png'))
