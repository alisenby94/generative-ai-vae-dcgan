"""
Variational Auto-Encoder (VAE) package
"""

from .model import VAE, VAEEncoder, VAEDecoder, weights_init

__all__ = ['VAE', 'VAEEncoder', 'VAEDecoder', 'weights_init']
