"""
Deep Convolutional Generative Adversarial Network (DCGAN) package
"""

from .model import DCGANGenerator, DCGANDiscriminator, DCGAN, weights_init

__all__ = ['DCGANGenerator', 'DCGANDiscriminator', 'DCGAN', 'weights_init']
