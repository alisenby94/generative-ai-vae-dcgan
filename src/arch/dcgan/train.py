"""
DCGAN Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from .model import DCGANGenerator, DCGANDiscriminator, weights_init
from tools.data_utils import get_cifar10_loaders
from tools.package_utils import plot_dcgan_losses, save_image_grid


def train(epochs=50, latent_dim=100, learning_rate=0.0002, batch_size=128, device=None):
    """
    Train DCGAN model
    
    Args:
        epochs: Number of training epochs
        latent_dim: Dimension of latent noise vector
        learning_rate: Learning rate for optimizers
        batch_size: Batch size for training
        device: Device to train on (cuda/cpu)
        
    Returns:
        history: Dictionary containing training metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Training DCGAN")
    print("="*60)
    
    # Create timestamped directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path('models') / 'dcgan' / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped results directory
    results_dir = Path('results') / 'dcgan' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    netG = DCGANGenerator(latent_dim=latent_dim).to(device)
    netD = DCGANDiscriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Loss
    criterion = nn.BCELoss()
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)
    
    # Training loop
    history = {'g_loss': [], 'd_loss': [], 'd_x': [], 'd_g_z': [], 'time': []}
    
    for epoch in range(epochs):
        netG.train()
        netD.train()
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_x = 0
        epoch_d_g_z = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for real_images, _ in pbar:
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # Normalize to [-1, 1] for tanh
            real_images = real_images * 2 - 1
            
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            # Train Discriminator
            netD.zero_grad()
            output_real = netD(real_images)
            loss_d_real = criterion(output_real, real_labels)
            d_x = output_real.mean().item()
            
            noise = torch.randn(batch_size_actual, latent_dim, 1, 1).to(device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()
            
            # Train Generator
            netG.zero_grad()
            output_fake = netD(fake_images)
            loss_g = criterion(output_fake, real_labels)
            d_g_z = output_fake.mean().item()
            loss_g.backward()
            optimizerG.step()
            
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            epoch_d_x += d_x
            epoch_d_g_z += d_g_z
            
            pbar.set_postfix({
                'D_loss': f'{loss_d.item():.3f}',
                'G_loss': f'{loss_g.item():.3f}',
                'D(x)': f'{d_x:.3f}',
                'D(G(z))': f'{d_g_z:.3f}'
            })
        
        # Average metrics
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        epoch_d_x /= len(train_loader)
        epoch_d_g_z /= len(train_loader)
        epoch_time = time.time() - start_time
        
        history['g_loss'].append(epoch_g_loss)
        history['d_loss'].append(epoch_d_loss)
        history['d_x'].append(epoch_d_x)
        history['d_g_z'].append(epoch_d_g_z)
        history['time'].append(epoch_time)
        
        print(f'Epoch {epoch+1}: G_Loss={epoch_g_loss:.4f}, D_Loss={epoch_d_loss:.4f}, '
              f'D(x)={epoch_d_x:.4f}, D(G(z))={epoch_d_g_z:.4f}, Time={epoch_time:.1f}s')
        
        # Save samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            netG.eval()
            with torch.no_grad():
                fake = netG(fixed_noise)
                # Denormalize from [-1, 1] to [0, 1]
                fake = (fake + 1) / 2
                save_image_grid(fake, results_dir / f'sample_epoch_{epoch+1}.png', nrow=8)
            print(f"  → Saved samples to {results_dir}/")
    
    # Save models with timestamp
    gen_path = model_dir / 'generator.pth'
    disc_path = model_dir / 'discriminator.pth'
    torch.save(netG.state_dict(), gen_path)
    torch.save(netD.state_dict(), disc_path)
    print(f"\n  → Models saved to {model_dir}/")
    
    # Plot losses
    plot_dcgan_losses(history, results_dir / 'losses.png')
    
    print(f"\nDCGAN Training Complete! Total time: {sum(history['time'])/60:.2f} min")
    print(f"  → Results saved to {results_dir}/")
    
    return history
