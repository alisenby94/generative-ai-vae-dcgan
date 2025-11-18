"""
VAE Training Script
"""

import torch
import torch.optim as optim
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from .model import VAE, weights_init
from tools.data_utils import get_cifar10_loaders, get_class_samples, CIFAR10_CLASSES
from tools.package_utils import plot_vae_losses, save_image_grid, save_training_metrics


def train(epochs=50, latent_dim=128, learning_rate=0.001, batch_size=128, device=None):
    """
    Train VAE model
    
    Args:
        epochs: Number of training epochs
        latent_dim: Dimension of latent space
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        device: Device to train on (cuda/cpu)
        
    Returns:
        history: Dictionary containing training metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Training VAE")
    print("="*60)
    
    # Create timestamped directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path('models') / 'vae' / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped results directory
    results_dir = Path('results') / 'vae' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model = VAE(latent_dim=latent_dim).to(device)
    model.apply(weights_init)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    
    # Get class-specific samples for tracking (using 'dog' class, index 5)
    class_idx = 5  # dog
    class_name = CIFAR10_CLASSES[class_idx]
    print(f"Will track class-specific reconstructions for: {class_name}")
    class_samples = get_class_samples(test_loader.dataset, class_idx, num_samples=64, device=str(device))
    
    # Training loop
    history = {'loss': [], 'recon_loss': [], 'kl_loss': [], 'time': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        num_samples = 0
        start_time = time.time()
        
        # Beta annealing: gradually increase KL weight from 0 to target value
        # This helps the model learn good reconstructions first
        beta = min(0.001, 0.001 * (epoch / 10))  # Linearly increase to 0.001 over 10 epochs
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data, _ in pbar:
            data = data.to(device)
            batch_size_actual = data.size(0)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = model.compute_loss(recon, data, mu, logvar, use_mse=True, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_samples += batch_size_actual
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Recon': f'{recon_loss.item():.3f}',
                'KL': f'{kl_loss.item():.3f}',
                'Beta': f'{beta:.4f}'
            })
        
        # Average losses
        epoch_loss /= num_samples
        epoch_recon /= num_samples
        epoch_kl /= num_samples
        epoch_time = time.time() - start_time
        
        history['loss'].append(epoch_loss)
        history['recon_loss'].append(epoch_recon)
        history['kl_loss'].append(epoch_kl)
        history['time'].append(epoch_time)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Recon={epoch_recon:.4f}, '
              f'KL={epoch_kl:.4f}, Time={epoch_time:.1f}s')
        
        # Save samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Reconstructions
                sample = next(iter(test_loader))[0][:64].to(device)
                recon, _, _ = model(sample)
                comparison = torch.cat([sample, recon])
                save_image_grid(comparison, results_dir / f'recon_epoch_{epoch+1}.png', nrow=8)
                
                # Class-specific reconstructions (dogs)
                if class_samples is not None:
                    class_recon, _, _ = model(class_samples)
                    class_comparison = torch.cat([class_samples, class_recon])
                    save_image_grid(class_comparison, results_dir / f'recon_{class_name}_epoch_{epoch+1}.png', nrow=8)
                
                # Random samples
                z = torch.randn(64, latent_dim).to(device)
                samples = model.decoder(z)
                save_image_grid(samples, results_dir / f'sample_epoch_{epoch+1}.png', nrow=8)
            print(f"  → Saved samples to {results_dir}/")
    
    # Save model with timestamp
    model_path = model_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n  → Model saved to {model_path}")
    
    # Plot losses
    plot_vae_losses(history, results_dir / 'losses.png')
    
    # Save training metrics
    save_training_metrics(history, results_dir)
    
    print(f"\nVAE Training Complete! Total time: {sum(history['time'])/60:.2f} min")
    print(f"  → Results saved to {results_dir}/")
    
    return history
