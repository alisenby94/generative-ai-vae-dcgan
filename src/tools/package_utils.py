"""
Package and environment utilities
"""

import sys


def check_package_versions():
    """
    Check and print versions of key packages
    """
    import torch
    import torchvision
    import numpy
    import matplotlib
    
    print("\nPackage Versions:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  NumPy: {numpy.__version__}")
    print(f"  Matplotlib: {matplotlib.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA: Not available")


def get_device():
    """
    Get the best available device (CUDA > CPU)
    
    Returns:
        torch.device object
    """
    import torch
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
    
    return device


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"\nRandom seed set to: {seed}")


def plot_vae_losses(history, save_path):
    """
    Plot VAE training losses
    
    Args:
        history: Dictionary with 'loss', 'recon_loss', 'kl_loss' keys
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['recon_loss'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['kl_loss'])
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_dcgan_losses(history, save_path):
    """
    Plot DCGAN training losses
    
    Args:
        history: Dictionary with 'g_loss', 'd_loss', 'd_x', 'd_g_z' keys
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['d_x'], label='D(x)')
    plt.plot(history['d_g_z'], label='D(G(z))')
    plt.title('Discriminator Outputs')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['time'])
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_metrics(history, save_dir):
    """
    Save training metrics to JSON and CSV files for reporting
    
    Args:
        history: Dictionary containing training metrics
        save_dir: Directory to save the metrics files
    """
    import json
    import csv
    from pathlib import Path
    
    save_dir = Path(save_dir)
    
    # Save as JSON
    json_path = save_dir / 'training_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  → Metrics saved to {json_path}")
    
    # Save as CSV for easy analysis in Excel/Pandas
    csv_path = save_dir / 'training_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['epoch'] + list(history.keys()))
        
        # Write data rows
        num_epochs = len(history[list(history.keys())[0]])
        for epoch in range(num_epochs):
            row = [epoch + 1]
            for key in history.keys():
                row.append(history[key][epoch])
            writer.writerow(row)
    print(f"  → Metrics saved to {csv_path}")


def save_image_grid(images, save_path, nrow=8):
    """
    Save a grid of images
    
    Args:
        images: Tensor of images (N, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
    """
    from torchvision.utils import save_image
    save_image(images, save_path, nrow=nrow)


def create_comparison_figure(image_paths, titles, save_path):
    """
    Create a comparison figure from multiple images
    
    Args:
        image_paths: List of paths to images
        titles: List of titles for each image
        save_path: Path to save the comparison figure
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    
    try:
        num_images = len(image_paths)
        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        
        if num_images == 1:
            axes = [axes]
        
        for ax, img_path, title in zip(axes, image_paths, titles):
            img_path = Path(img_path)
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  Could not create comparison figure: {e}")
        return False
