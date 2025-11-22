"""
Convolutional Variational Auto-Encoder (VAE) for CIFAR-10
Architecture designed for 32x32x3 images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """Encoder network for VAE - maps images to latent distribution parameters"""
    
    def __init__(self, latent_dim=128, input_channels=3):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)  # 32 x 16 x 16
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 64 x 8 x 8
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 128 x 4 x 4
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 256 x 2 x 2
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Compute mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE - reconstructs images from latent vectors"""
    
    def __init__(self, latent_dim=128, output_channels=3):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Transposed convolutional layers (deconvolution)
        # 256 x 2 x 2
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 128 x 4 x 4
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64 x 8 x 8
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32 x 16 x 16
        self.bn3 = nn.BatchNorm2d(32)
        
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # 3 x 32 x 32
    
    def forward(self, z):
        # Expand latent vector
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        
        # Transposed convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        
        # Final layer with sigmoid to get pixel values in [0, 1]
        x = torch.sigmoid(self.deconv4(x))
        
        return x


class VAE(nn.Module):
    """Complete Variational Auto-Encoder"""
    
    def __init__(self, latent_dim=128, input_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(latent_dim, input_channels)
        self.decoder = VAEDecoder(latent_dim, input_channels)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Sample from the latent space and generate images
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples
    
    def compute_loss(self, recon_x, x, mu, logvar, use_mse=True, beta=0.001):
        """
        Compute VAE loss = Reconstruction loss + KL divergence
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            use_mse: If True, use MSE for reconstruction loss (more stable for continuous values)
                     If False, use BCE (traditional VAE loss)
            beta: Weight for KL divergence term (helps balance reconstruction vs regularization)
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        batch_size = x.size(0)
        
        if use_mse:
            # MSE reconstruction loss - use mean instead of sum for better scaling
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        else:
            # Binary Cross Entropy reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        
        # KL divergence loss - normalize by batch size
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss with beta weighting on KL term
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term (beta-VAE)
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def weights_init(m):
    """Initialize weights for convolutional and batch norm layers"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
