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
from tools.package_utils import plot_dcgan_losses, save_image_grid, save_training_metrics


def train(epochs=50, latent_dim=100, learning_rate=0.0002, batch_size=128, device=None):
    """
    Train DCGAN: Generator makes fakes, Discriminator learns to spot them
    They fight until Generator gets good at fooling Discriminator
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Training DCGAN")
    print("="*60)
    
    # Setup: Create folders to save stuff
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path('models') / 'dcgan' / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path('results') / 'dcgan' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build the two neural networks
    netG = DCGANGenerator(latent_dim=latent_dim).to(device)  # Makes fake images
    netD = DCGANDiscriminator().to(device)                    # Spots fake images
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Step 2: Setup learning rates (Discriminator learns 4x faster)
    d_lr = 0.00015  # Fast learner
    g_lr = 0.0001  # Slow learner
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))
    
    # Step 3: Loss function (measures how wrong we are)
    criterion = nn.BCELoss()
    
    # Step 4: Labels (1=real, 0=fake)
    real_label_value = 1.0
    fake_label_value = 0.0
    
    print(f"Learning rates: Discriminator={d_lr} (4x faster), Generator={g_lr}")
    print(f"Architecture: 4 layers each (proven CIFAR-10 design)")
    
    # Step 5: Load CIFAR-10 images
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    
    # Step 6: Create fixed noise (same noise every time = consistent test)
    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)
    
    # Step 7: Start training!
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
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # Convert images from [0,1] to [-1,1] (Generator outputs tanh)
            real_images = real_images * 2 - 1
            
            # Create labels: 1=real, 0=fake
            real_labels = torch.full((batch_size_actual, 1, 1, 1), real_label_value, device=device)
            fake_labels = torch.full((batch_size_actual, 1, 1, 1), fake_label_value, device=device)
            
            # === Train Discriminator: Learn to spot fakes ===
            netD.zero_grad()
            
            # Test on real images (should output 1)
            output_real = netD(real_images)
            loss_d_real = criterion(output_real, real_labels)
            d_x = output_real.mean().item()  # Track average realness score
            
            # Test on fake images (should output 0)
            noise = torch.randn(batch_size_actual, latent_dim, 1, 1).to(device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())  # Don't backprop through Generator
            loss_d_fake = criterion(output_fake, fake_labels)
            
            # Update Discriminator
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()
            
            # === Train Generator: Learn to fool Discriminator ===
            netG.zero_grad()
            
            # Make fakes and try to convince Discriminator they're real
            noise = torch.randn(batch_size_actual, latent_dim, 1, 1).to(device)
            fake_images = netG(noise)
            output_fake = netD(fake_images)
            loss_g = criterion(output_fake, real_labels)  # Want Discriminator to say "real"!
            d_g_z = output_fake.mean().item()  # Track how real our fakes look
            
            # Update Generator
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
        
        # Save samples every 5 epochs
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
    
    # Save training metrics
    save_training_metrics(history, results_dir)
    
    print(f"\nDCGAN Training Complete! Total time: {sum(history['time'])/60:.2f} min")
    print(f"  → Results saved to {results_dir}/")
    
    return history
