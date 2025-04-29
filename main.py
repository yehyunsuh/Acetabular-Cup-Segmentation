import os
import argparse
import torch

from utils import customize_seed, str2bool
from model import UNet
from data_loader import dataloader
from train import train


def main(args):
    """
    Main function that initializes and trains the model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet(device)

    train_loader, val_loader = dataloader(args)

    train(args, model, device, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for anatomical landmark segmentation with U-Net."
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility"
    )

    # Data settings
    parser.add_argument(
        "--train_image_dir", type=str, default="./data/train_images_fluoro_cup",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--label_dir", type=str, default="./data/labels",
        help="Directory containing ground truth annotation CSVs"
    )
    parser.add_argument(
        "--train_csv_file", type=str, default="train_annotation_fluoro_cup.csv",
        help="CSV file containing training annotations"
    )
    parser.add_argument(
        "--train_val_split", type=float, default=0.9,
        help="Proportion of data to use for training (0.0 - 1.0)"
    )

    # Image/label settings
    parser.add_argument(
        "--image_resize", type=int, default=512,
        help="Target image size after resizing (must be divisible by 32)"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=12,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="fluoro_cup",
        help="Name for the experiment directory"
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Number of epochs with no improvement before stopping"
    )

    # Visualization options
    parser.add_argument(
        "--gif", action="store_true",
        help="Enable GIF creation of training visuals"
    )

    args = parser.parse_args()

    # Fix randomness
    customize_seed(args.seed)

    # Create necessary directories
    os.makedirs(f"{args.experiment_name}/visualization", exist_ok=True)
    os.makedirs(f"{args.experiment_name}/graph", exist_ok=True)
    os.makedirs(f"weight", exist_ok=True)
    os.makedirs(f"{args.experiment_name}/train_results", exist_ok=True)

    main(args)