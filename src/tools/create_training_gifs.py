#!/usr/bin/env python3
"""
Create animated GIFs from training epoch images with epoch labels
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re


def extract_epoch_number(filename):
    """
    Extract epoch number from filename
    
    Args:
        filename: Filename string (e.g., 'sample_epoch_5.png')
    
    Returns:
        int: Epoch number or 0 if not found
    """
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0


def add_epoch_label(image, epoch, position='top', font_size=40, padding=20):
    """
    Add epoch label outside the image area
    
    Args:
        image: PIL Image
        epoch: Epoch number
        position: Label position ('top' or 'bottom')
        font_size: Font size for the label
        padding: Padding around text
    
    Returns:
        PIL Image with label added above or below
    """
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = f"Epoch {epoch}"
    
    # Create a temporary image to get text size
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate new image size
    img_width, img_height = image.size
    label_height = int(text_height + padding * 2)
    
    # Create new image with extra space for label
    if position == 'top':
        new_img = Image.new('RGB', (img_width, img_height + label_height), color=(255, 255, 255))
        # Paste original image below label
        new_img.paste(image, (0, label_height))
        text_y = padding
    else:  # bottom
        new_img = Image.new('RGB', (img_width, img_height + label_height), color=(255, 255, 255))
        # Paste original image above label
        new_img.paste(image, (0, 0))
        text_y = img_height + padding
    
    # Draw text centered horizontally
    draw = ImageDraw.Draw(new_img)
    text_x = (img_width - text_width) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    return new_img


def create_gif_from_images(image_dir, output_path, pattern, duration=500, 
                          label_position='top', font_size=40, loop=0):
    """
    Create animated GIF from training images
    
    Args:
        image_dir: Directory containing the images
        output_path: Path to save the GIF
        pattern: Glob pattern to match images (e.g., 'sample_epoch_*.png')
        duration: Duration of each frame in milliseconds
        label_position: Position of epoch label ('top' or 'bottom')
        font_size: Font size for epoch label
        loop: Number of loops (0 = infinite)
    """
    image_dir = Path(image_dir)
    
    # Find all matching images
    image_files = sorted(image_dir.glob(pattern), key=lambda x: extract_epoch_number(x.name))
    
    if not image_files:
        print(f"  ⚠ No images found matching '{pattern}' in {image_dir}")
        return False
    
    print(f"  Found {len(image_files)} images")
    
    # Load and label images
    frames = []
    for img_path in image_files:
        epoch = extract_epoch_number(img_path.name)
        img = Image.open(img_path)
        
        # Convert to RGB if necessary (remove alpha channel)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Add epoch label
        labeled_img = add_epoch_label(img, epoch, label_position, font_size)
        frames.append(labeled_img)
        print(f"    Processed: {img_path.name} (Epoch {epoch})")
    
    # Save as GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    
    print(f"  ✓ GIF saved to {output_path}")
    return True


def process_training_run(results_dir, model_type='vae', duration=500, 
                        label_position='top', font_size=40):
    """
    Process all images from a training run and create GIFs
    
    Args:
        results_dir: Path to the results directory (e.g., results/vae/20251117_123456/)
        model_type: 'vae' or 'dcgan'
        duration: Duration of each frame in milliseconds
        label_position: Position of epoch label ('top' or 'bottom')
        font_size: Font size for epoch label
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    print(f"\nProcessing {model_type.upper()} training images from {results_dir.name}")
    print("=" * 70)
    
    # Process based on model type
    if model_type.lower() == 'vae':
        # Create GIF for reconstructions
        print("\n📹 Creating reconstruction GIF...")
        create_gif_from_images(
            results_dir, 
            results_dir / 'reconstruction_progress.gif',
            'recon_epoch_*.png',
            duration=duration,
            label_position=label_position,
            font_size=font_size
        )
        
        # Create GIF for class-specific reconstructions (e.g., dogs)
        print("\n📹 Creating class-specific reconstruction GIF...")
        dog_pattern = 'recon_dog_epoch_*.png'
        if list(results_dir.glob(dog_pattern)):
            create_gif_from_images(
                results_dir,
                results_dir / 'reconstruction_dog_progress.gif',
                dog_pattern,
                duration=duration,
                label_position=label_position,
                font_size=font_size
            )
        
        # Create GIF for samples
        print("\n📹 Creating samples GIF...")
        create_gif_from_images(
            results_dir,
            results_dir / 'samples_progress.gif',
            'sample_epoch_*.png',
            duration=duration,
            label_position=label_position,
            font_size=font_size
        )
    
    elif model_type.lower() == 'dcgan':
        # Create GIF for samples
        print("\n📹 Creating samples GIF...")
        create_gif_from_images(
            results_dir,
            results_dir / 'samples_progress.gif',
            'sample_epoch_*.png',
            duration=duration,
            label_position=label_position,
            font_size=font_size
        )
    
    print("\n✓ Done!")


def find_latest_run(model_type='vae'):
    """
    Find the latest training run directory
    
    Args:
        model_type: 'vae' or 'dcgan'
    
    Returns:
        Path to latest run or None
    """
    results_base = Path('results') / model_type
    
    if not results_base.exists():
        return None
    
    # Get all timestamp directories
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()], reverse=True)
    
    return run_dirs[0] if run_dirs else None


def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIFs from training epoch images with epoch labels'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Path to results directory (e.g., results/vae/20251117_123456/)'
    )
    parser.add_argument(
        '--model',
        choices=['vae', 'dcgan'],
        help='Model type (required if --dir not specified, uses latest run)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=500,
        help='Duration of each frame in milliseconds (default: 500)'
    )
    parser.add_argument(
        '--position',
        choices=['top', 'bottom'],
        default='top',
        help='Position of epoch label (default: top)'
    )
    parser.add_argument(
        '--font-size',
        type=int,
        default=40,
        help='Font size for epoch label (default: 40)'
    )
    
    args = parser.parse_args()
    
    # Determine which directory to process
    if args.dir:
        results_dir = Path(args.dir)
        # Infer model type from path if not specified
        if args.model:
            model_type = args.model
        elif 'vae' in str(results_dir):
            model_type = 'vae'
        elif 'dcgan' in str(results_dir):
            model_type = 'dcgan'
        else:
            print("Error: Could not determine model type. Please specify --model")
            return
    elif args.model:
        # Find latest run for specified model
        results_dir = find_latest_run(args.model)
        if not results_dir:
            print(f"Error: No training runs found for {args.model}")
            return
        model_type = args.model
        print(f"Using latest run: {results_dir}")
    else:
        print("Error: Must specify either --dir or --model")
        parser.print_help()
        return
    
    # Process the training run
    process_training_run(
        results_dir,
        model_type=model_type,
        duration=args.duration,
        label_position=args.position,
        font_size=args.font_size
    )


if __name__ == '__main__':
    main()
