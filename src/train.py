"""
Training loop for congestion prediction models.

Supports any model with (batch, 3, 256, 256) → (batch, 1, 256, 256) interface.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model, print_results


def train_model(model, model_name="model", num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE, device='cuda'):
    """
    Complete training pipeline.
    
    Args:
        model: PyTorch model
        model_name: String name for saving checkpoints
        num_epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda', 'mps', or 'cpu'
    
    Returns:
        model: Trained model
        history: Dict with training metrics over epochs
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    model = model.to(device)
    
    # Data
    print(f"\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(num_workers=0)
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHEDULER_PATIENCE,
        factor=0.5
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'best_val_loss': float('inf'), 'best_epoch': 0
    }
    
    # Early stopping
    patience_counter = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f" Training: {model_name}")
    print(f" Epochs: {num_epochs}, LR: {lr}, Device: {device}")
    print(f" Parameters: {total_params:,}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        # ---- TRAINING ----
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            preds = model(features)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader.dataset)
        
        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                preds = model(features)
                loss = criterion(preds, labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # ---- LOGGING ----
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # ---- SCHEDULER ----
        scheduler.step(val_loss)
        
        # ---- CHECKPOINTING ----
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✅ New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
        
        # ---- EARLY STOPPING ----
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break
    
    # ---- FINAL EVALUATION ----
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth"),
        map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nEvaluating best model (epoch {checkpoint['epoch']})...")
    results = evaluate_model(model, test_loader, device=device)
    print_results(results, model_name)
    
    history['test_results'] = results
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'unet', 'gpdl_fcn', 'vit', 'attention_unet'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'  # Apple Silicon GPU
        else:
            args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Import the right model
    if args.model == 'simple_cnn':
        from src.models.simple_cnn import SimpleCNN
        model = SimpleCNN(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS)
    elif args.model == 'unet':
        from src.models.unet import UNet
        model = UNet(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS)
    elif args.model == 'gpdl_fcn':
        from src.models.gpdl_fcn import GPDL_FCN
        model = GPDL_FCN(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS)
    elif args.model == 'vit':
        from src.models.vit_model import ViTCongestion
        model = ViTCongestion(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS)
    elif args.model == 'attention_unet':
        from src.models.attention_unet import AttentionUNet
        model = AttentionUNet(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS)    
    
    train_model(model, model_name=args.model, num_epochs=args.epochs,
                lr=args.lr, device=args.device)
