"""
DCGAN Evaluation Script
"""

import torch
from pathlib import Path
from datetime import datetime

from .model import DCGANGenerator
from tools.data_utils import get_cifar10_loaders
from tools.package_utils import save_image_grid
from tools.train_utils import get_latest_model


def evaluate(model_path=None, latent_dim=100, batch_size=128, device=None):
    """
    Evaluate DCGAN model
    
    Args:
        model_path: Path to generator checkpoint (if None, uses latest)
        latent_dim: Dimension of latent noise vector
        batch_size: Batch size for evaluation
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Evaluating DCGAN")
    print("="*60)
    
    # Find model if not specified
    if model_path is None:
        dcgan_dir = get_latest_model('dcgan')
        if dcgan_dir is None:
            print("No trained DCGAN model found!")
            return None
        model_path = dcgan_dir / 'generator.pth'
    else:
        model_path = Path(model_path)
        dcgan_dir = model_path.parent
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    # Create evaluation results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = Path('results') / 'evaluation' / 'dcgan' / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading generator from {model_path}...")
    
    # Load model
    netG = DCGANGenerator(latent_dim=latent_dim).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        # Generate multiple sample sets
        for i, num_samples in enumerate([64, 100], 1):
            noise = torch.randn(num_samples, latent_dim, 1, 1).to(device)
            fake = netG(noise)
            # Denormalize from [-1, 1] to [0, 1]
            fake = (fake + 1) / 2
            save_image_grid(fake, eval_dir / f'generated_samples_{i}.png', nrow=8)
    
    print(f"\n  → Results saved to {eval_dir}/")
    print("="*60)
    
    return eval_dir
