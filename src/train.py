"""
Main training orchestrator - simplified to isolate architecture specific training logic
"""

import torch
import argparse

# Import model-specific training
from arch.vae import train as vae_train
from arch.dcgan import train as dcgan_train  
from arch.vae import eval as vae_eval
from arch.dcgan import eval as dcgan_eval


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train VAE and/or DCGAN on CIFAR-10')
    parser.add_argument('--model', choices=['vae', 'dcgan', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Model: {args.model.upper()}")
    print(f"Epochs: {args.epochs}\n")

    # Start training/evaluation
    if not args.eval_only:
        if args.model in ['vae', 'both']:
            vae_train.train(epochs=args.epochs, batch_size=args.batch_size, device=device)
        
        if args.model in ['dcgan', 'both']:
            dcgan_train.train(epochs=args.epochs, batch_size=args.batch_size, device=device)
    
    if not args.skip_eval or args.eval_only:
        if args.model in ['vae', 'both']:
            vae_eval.evaluate(device=device)
        
        if args.model in ['dcgan', 'both']:
            dcgan_eval.evaluate(device=device)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
