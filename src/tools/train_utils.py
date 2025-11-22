"""
Training utility functions
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_latest_model(model_type, base_dir='models'):
    """
    Get the latest model checkpoint for a given type
    
    Args:
        model_type: Type of model ('vae' or 'dcgan')
        base_dir: Base directory where models are stored
        
    Returns:
        Path to latest model directory or None if not found
    """
    model_base = Path(base_dir) / model_type
    if not model_base.exists():
        return None
    
    # Get all timestamp directories
    timestamps = [d for d in model_base.iterdir() if d.is_dir()]
    if not timestamps:
        return None
    
    # Sort and get latest
    latest = sorted(timestamps)[-1]
    return latest


def get_cifar10_dataloader(data_dir='./data', batch_size=128, train=True, num_workers=2):
    """
    Get CIFAR-10 dataloader
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size
        train: If True, load training data; otherwise test data
        num_workers: Number of worker processes for data loading
    Returns:
        DataLoader object
    """
    # Data preprocessing and augmentation
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Input tensor
        mean: Mean used for normalization
        std: Std used for normalization
    Returns:
        Denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def save_image_grid(images, filepath, nrow=8, normalize=True, value_range=None):
    """
    Save a grid of images
    
    Args:
        images: Tensor of images (N, C, H, W)
        filepath: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize images
        value_range: Value range for normalization
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    grid = torchvision.utils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2
    )
    
    # Convert to numpy and save
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(ndarr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def plot_training_curves(history, save_path, model_name="Model"):
    """
    Plot training curves
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
        model_name: Name of the model
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 5))
    
    if len(history) == 1:
        axes = [axes]
    
    for idx, (key, values) in enumerate(history.items()):
        axes[idx].plot(values)
        axes[idx].set_title(f'{model_name} - {key}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model or dictionary of models
        optimizer: Optimizer or dictionary of optimizers
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
    }
    
    # Handle single model or multiple models
    if isinstance(model, dict):
        checkpoint['model_state_dict'] = {k: v.state_dict() for k, v in model.items()}
    else:
        checkpoint['model_state_dict'] = model.state_dict()
    
    # Handle single optimizer or multiple optimizers
    if isinstance(optimizer, dict):
        checkpoint['optimizer_state_dict'] = {k: v.state_dict() for k, v in optimizer.items()}
    else:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model or dictionary of models
        optimizer: Optimizer or dictionary of optimizers (optional)
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath)
    
    # Load model state
    if isinstance(model, dict):
        for k, v in model.items():
            v.load_state_dict(checkpoint['model_state_dict'][k])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        if isinstance(optimizer, dict):
            for k, v in optimizer.items():
                v.load_state_dict(checkpoint['optimizer_state_dict'][k])
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
