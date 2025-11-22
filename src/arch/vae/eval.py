"""
VAE Evaluation Script
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from .model import VAE
from tools.data_utils import get_cifar10_loaders, compute_image_metrics
from tools.package_utils import save_image_grid
from tools.train_utils import get_latest_model


def evaluate(model_path=None, latent_dim=128, batch_size=128, device=None):
    """
    Evaluate VAE model
    
    Args:
        model_path: Path to model checkpoint (if None, uses latest)
        latent_dim: Dimension of latent space
        batch_size: Batch size for evaluation
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Evaluating VAE")
    print("="*60)
    
    # Find model if not specified
    if model_path is None:
        vae_dir = get_latest_model('vae')
        if vae_dir is None:
            print("No trained VAE model found!")
            return None
        model_path = vae_dir / 'model.pth'
    else:
        model_path = Path(model_path)
        vae_dir = model_path.parent
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    # Create evaluation results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = Path('results') / 'evaluation' / 'vae' / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, test_loader = get_cifar10_loaders(batch_size)
    
    # Compute metrics
    mse_scores = []
    ssim_scores = []
    
    print("Computing reconstruction metrics...")
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc='Computing metrics'):
            data = data.to(device)
            recon, _, _ = model(data)
            
            for i in range(min(10, len(data))):
                mse_score, ssim_score = compute_image_metrics(data[i], recon[i])
                mse_scores.append(mse_score)
                ssim_scores.append(ssim_score)
            
            if len(mse_scores) >= 100:
                break
    
    metrics = {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores)
    }
    
    print(f"\nVAE Reconstruction Quality:")
    print(f"  MSE:  {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    with torch.no_grad():
        # Reconstructions
        sample = next(iter(test_loader))[0][:16].to(device)
        recon, _, _ = model(sample)
        comparison = torch.cat([sample, recon])
        save_image_grid(comparison, eval_dir / 'reconstruction_comparison.png', nrow=8)
        
        # Random samples
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decoder(z)
        save_image_grid(samples, eval_dir / 'generated_samples.png', nrow=8)
    
    print(f"\n  → Results saved to {eval_dir}/")
    print("="*60)
    
    return metrics
