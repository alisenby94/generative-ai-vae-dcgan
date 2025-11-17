"""
Utility functions for model architectures
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """
    Count total number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def print_model_info(model, model_name="Model"):
    """
    Print detailed information about a model
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    num_params = count_parameters(model)
    size_mb = get_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total Parameters: {num_params:,}")
    print(f"  Model Size: {size_mb:.2f} MB")
    print(f"  Architecture:")
    print(model)
