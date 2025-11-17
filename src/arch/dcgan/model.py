"""
Deep Convolutional Generative Adversarial Network (DCGAN) for CIFAR-10
Following DCGAN architecture guidelines from the original paper
"""

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator network
    Maps random noise vectors to images
    """
    
    def __init__(self, latent_dim=100, output_channels=3, ngf=64):
        """
        Args:
            latent_dim: Dimension of input noise vector
            output_channels: Number of output channels (3 for RGB)
            ngf: Size of feature maps in generator
        """
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Architecture for 32x32 images
        # Input: latent_dim x 1 x 1
        self.main = nn.Sequential(
            # Layer 1: latent_dim -> ngf*4 (4x4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            
            # Layer 2: ngf*8 -> ngf*4 (8x8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            
            # Layer 3: ngf*4 -> ngf*2 (16x16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            
            # Layer 4: ngf*2 -> output_channels (32x32)
            nn.ConvTranspose2d(ngf * 2, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output size: output_channels x 32 x 32
        )
    
    def forward(self, z):
        """
        Args:
            z: Noise vector of shape (batch_size, latent_dim, 1, 1)
        Returns:
            Generated images of shape (batch_size, output_channels, 32, 32)
        """
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator network
    Classifies images as real or fake
    """
    
    def __init__(self, input_channels=3, ndf=64):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            ndf: Size of feature maps in discriminator
        """
        super(DCGANDiscriminator, self).__init__()
        
        # Architecture for 32x32 images
        # Input: input_channels x 32 x 32
        self.main = nn.Sequential(
            # Layer 1: input_channels -> ndf (16x16)
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf x 16 x 16
            
            # Layer 2: ndf -> ndf*2 (8x8)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 8 x 8
            
            # Layer 3: ndf*2 -> ndf*4 (4x4)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 4 x 4
            
            # Layer 4: ndf*4 -> 1 (1x1)
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output size: 1 x 1 x 1
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch_size, input_channels, 32, 32)
        Returns:
            Probability of being real (batch_size, 1, 1, 1)
        """
        return self.main(x)


class DCGAN:
    """
    DCGAN wrapper class containing both Generator and Discriminator
    """
    
    def __init__(self, latent_dim=100, channels=3, ngf=64, ndf=64, device='cuda'):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            channels: Number of image channels
            ngf: Generator feature map size
            ndf: Discriminator feature map size
            device: Device to run on
        """
        self.latent_dim = latent_dim
        self.device = device
        
        # Create generator and discriminator
        self.generator = DCGANGenerator(latent_dim, channels, ngf).to(device)
        self.discriminator = DCGANDiscriminator(channels, ndf).to(device)
        
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
