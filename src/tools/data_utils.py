"""
Data loading and preprocessing utilities for CIFAR-10
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar10_loaders(batch_size=128, data_root='./data', num_workers=0):
    """
    Load CIFAR-10 dataset and create data loaders
    
    Args:
        batch_size: Batch size for training
        data_root: Root directory for dataset storage
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_class_samples(dataset, class_idx, num_samples=64, device='cpu'):
    """
    Get samples from a specific class
    
    Args:
        dataset: CIFAR-10 dataset
        class_idx: Class index (0-9)
        num_samples: Number of samples to get
        device: Device to move samples to
        
    Returns:
        Tensor of images from the specified class
    """
    samples = []
    for img, label in dataset:
        if label == class_idx:
            samples.append(img)
            if len(samples) >= num_samples:
                break
    
    if samples:
        return torch.stack(samples).to(device)
    return None


def compute_image_metrics(img1, img2):
    """
    Compute MSE and SSIM between two images
    
    Args:
        img1: First image tensor (C, H, W) in range [0, 1]
        img2: Second image tensor (C, H, W) in range [0, 1]
        
    Returns:
        mse_score: Mean squared error
        ssim_score: Structural similarity index
    """
    # Convert to numpy and transpose to (H, W, C)
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    # Clip values to valid range
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Compute metrics
    mse_score = mse(img1_np, img2_np)
    ssim_score = ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    
    return mse_score, ssim_score
