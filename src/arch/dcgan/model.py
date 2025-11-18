"""
Deep Convolutional Generative Adversarial Network (DCGAN) for CIFAR-10
Following DCGAN architecture guidelines from the original paper
"""

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    Generator: Makes fake images from random noise
    Think of it like upscaling a tiny pixel into a full image
    """
    
    def __init__(self, latent_dim=100, output_channels=3):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # 4 layers: Start tiny (1x1), grow to full size (32x32)
        # Each layer DOUBLES the image size and HALVES the features
        self.main = nn.Sequential(
            # Layer 1: Random noise -> 512 features at 4x4 pixels
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 512 features -> 256 features at 8x8 pixels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 256 features -> 128 features at 16x16 pixels
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 128 features -> 3 RGB channels at 32x32 pixels (final image!)
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Squish values to [-1, 1]
        )
    
    def forward(self, z):
        # z = random noise, output = fake image
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    Discriminator: Detective that spots fake images
    Think of it like shrinking an image down to a single yes/no decision
    """
    
    def __init__(self, input_channels=3):
        super(DCGANDiscriminator, self).__init__()
        
        # 4 layers: Start with full image (32x32), shrink to single number
        # Each layer HALVES the image size and DOUBLES the features
        self.main = nn.Sequential(
            # Layer 1: RGB image (3 channels) -> 128 features at 16x16 pixels
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 128 features -> 256 features at 8x8 pixels
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 256 features -> 512 features at 4x4 pixels
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 512 features -> 1 number (real=1, fake=0)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Squish to 0-1 probability
        )
    
    def forward(self, x):
        # x = image, output = realness score (0=fake, 1=real)
        return self.main(x)


class DCGAN:
    """
    DCGAN wrapper class containing both Generator and Discriminator
    """
    
    def __init__(self, latent_dim=100, channels=3, device='cuda'):
        """
        Args:
            latent_dim: Size of random noise input (default: 100)
            channels: Number of image channels (3 for RGB)
            device: 'cuda' for GPU or 'cpu'
        """
        self.latent_dim = latent_dim
        self.device = device
        
        # Create generator and discriminator
        self.generator = DCGANGenerator(latent_dim, channels).to(device)
        self.discriminator = DCGANDiscriminator(channels).to(device)
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
    
    def sample_noise(self, batch_size):
        """
        Sample random noise vectors
        
        Args:
            batch_size: Number of noise vectors to generate
        Returns:
            Noise tensor of shape (batch_size, latent_dim, 1, 1)
        """
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
    
    def generate(self, num_samples):
        """
        Generate images from random noise
        
        Args:
            num_samples: Number of images to generate
        Returns:
            Generated images
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.sample_noise(num_samples)
            fake_images = self.generator(noise)
        return fake_images


def weights_init(m):
    """
    Custom weights initialization for DCGAN
    From DCGAN paper: All weights initialized from N(0, 0.02)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
