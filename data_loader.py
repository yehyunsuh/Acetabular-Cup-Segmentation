"""
data_loader.py

Dataset and dataloader utilities for anatomical landmark segmentation.

Author: Yehyun Suh
"""

import os
import csv
import cv2
import torch
import numpy as np
import albumentations as A

from scipy.ndimage import binary_dilation
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split


class SegmentationDataset(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
            n_landmarks (int): Number of landmarks per image.
            dilation_iters (int): Number of dilation iterations for landmark masks.
        """
        self.image_dir = image_dir
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                image_width = int(row[1])
                image_height = int(row[2])
                coords = list(map(int, row[4:]))
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, image_width, image_height, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, image_width, image_height, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        max_side = max(h, w)

        # Apply resizing and normalization
        transform = A.Compose([
            A.PadIfNeeded(
                min_height=max_side, min_width=max_side,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Resize(512, 512),
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.HorizontalFlip(p=0.25),
            A.RandomBrightnessContrast(p=0.25),
            A.CoarseDropout(
                num_holes_range=(6, 12),
                hole_height_range=(10, 30),
                hole_width_range=(10, 30),
                fill=0,
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})

        # On the mask, draw an ellipse using the landmark coordinates
        # Extract points
        major_pt1 = np.array(landmarks[1])
        major_pt2 = np.array(landmarks[2])
        minor_pt1 = np.array(landmarks[3])
        minor_pt2 = np.array(landmarks[4])

        # Compute ellipse parameters
        center = ((major_pt1 + major_pt2) / 2).astype(np.float32)
        a = 0.5 * np.linalg.norm(major_pt2 - major_pt1)
        b = 0.5 * np.linalg.norm(minor_pt2 - minor_pt1)
        angle = np.degrees(np.arctan2(major_pt2[1] - major_pt1[1], major_pt2[0] - major_pt1[0]))

        # Create mask
        masks = np.zeros((h, w), dtype=np.uint8)

        # Draw ellipse
        axes = (int(round(a)), int(round(b)))
        cv2.ellipse(masks, center=tuple(map(int, center)), axes=axes,
                    angle=angle, startAngle=0, endAngle=360,
                    color=255, thickness=-1)

        # Apply mask transform
        transformed = transform(image=image, mask=masks)
        image = transformed["image"]
        mask = transformed["mask"].float() / 255.0
        mask = mask.unsqueeze(0)  # [1, H, W]

        return image, mask, image_name


def dataloader(args):
    """
    Constructs and returns PyTorch dataloaders for training and validation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all config.

    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = SegmentationDataset(
        csv_path=os.path.join(args.label_dir, args.train_csv_file),
        image_dir=args.train_image_dir,
    )

    # Train-validation split
    train_size = int(args.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")

    return train_loader, val_loader