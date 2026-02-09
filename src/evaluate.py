"""
Evaluation metrics for congestion prediction.

We use multiple metrics to get a complete picture of model performance:
  - NRMSE: Normalized error (lower is better)
  - SSIM: Structural similarity (higher is better, max 1.0)
  - Pearson R: Linear correlation (higher is better, max 1.0)
  - MSE: Mean squared error (lower is better)

NOTE: Uses pure numpy implementations to avoid segfaults with
skimage/scipy on Mac MPS.
"""
import numpy as np
import torch


def compute_nrmse(pred, target):
    """
    Normalized Root Mean Square Error.
    Lower is better. 0 = perfect.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    target_range = target.max() - target.min()
    if target_range < 1e-8:
        return 0.0
    return float(rmse / target_range)


def compute_ssim(pred, target):
    """
    Structural Similarity Index (simplified global implementation).
    Avoids skimage dependency issues on Mac/MPS.
    Higher is better. 1.0 = identical.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    
    # Constants for numerical stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    sigma_pred_sq = np.mean((pred - mu_pred) ** 2)
    sigma_target_sq = np.mean((target - mu_target) ** 2)
    sigma_cross = np.mean((pred - mu_pred) * (target - mu_target))
    
    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred**2 + mu_target**2 + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    if denominator < 1e-12:
        return 1.0
    
    return float(numerator / denominator)


def compute_pearson(pred, target):
    """
    Pearson correlation coefficient (pure numpy).
    Higher is better. 1.0 = perfect linear relationship.
    """
    pred = pred.astype(np.float64).flatten()
    target = target.astype(np.float64).flatten()
    
    pred_std = pred.std()
    target_std = target.std()
    
    if pred_std < 1e-8 or target_std < 1e-8:
        return 0.0
    
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    numerator = np.mean(pred_centered * target_centered)
    denominator = pred_std * target_std
    
    if denominator < 1e-12:
        return 0.0
    
    return float(numerator / denominator)


def evaluate_model(model, test_loader, device='cuda'):
    """
    Run the model on the test set and compute all metrics.
    
    Returns:
        dict with metric names and values (averaged across test set)
    """
    model.eval()
    
    all_nrmse = []
    all_ssim = []
    all_pearson = []
    all_mse = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            
            preds = model(features)
            
            # Move everything to CPU numpy immediately
            preds_np = preds.cpu().float().numpy()
            labels_np = labels.float().numpy()
            
            # Compute per-sample metrics
            for i in range(preds_np.shape[0]):
                pred_map = preds_np[i, 0]
                label_map = labels_np[i, 0]
                
                all_nrmse.append(compute_nrmse(pred_map, label_map))
                all_ssim.append(compute_ssim(pred_map, label_map))
                all_pearson.append(compute_pearson(pred_map, label_map))
                all_mse.append(float(np.mean((pred_map.astype(np.float64) - label_map.astype(np.float64)) ** 2)))
    
    results = {
        'nrmse': float(np.mean(all_nrmse)),
        'ssim': float(np.mean(all_ssim)),
        'pearson': float(np.mean(all_pearson)),
        'mse': float(np.mean(all_mse)),
    }
    
    return results


def print_results(results, model_name="Model"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f" Results: {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'Value':>12}")
    print(f"  {'-'*32}")
    print(f"  {'NRMSE ↓':<20} {results['nrmse']:>12.4f}")
    print(f"  {'SSIM ↑':<20} {results['ssim']:>12.4f}")
    print(f"  {'Pearson R ↑':<20} {results['pearson']:>12.4f}")
    print(f"  {'MSE ↓':<20} {results['mse']:>12.6f}")
    print(f"{'='*50}\n")
