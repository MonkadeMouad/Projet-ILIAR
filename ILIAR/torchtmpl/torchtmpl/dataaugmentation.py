import numpy as np
import torch
import torchvision.transforms as T
import random
import cv2
import pathlib
import logging
import gc
from enum import Enum
from typing import Union, List, Dict
from collections import defaultdict

class Fold(Enum):
    TRAIN = 0
    VALID = 1

class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        fold: Fold,
        transform=None,
        augment=False,
        valid_ratio=0.2,
        image_size=(128, 128),
    ):
        self.root = pathlib.Path(root) if isinstance(root, str) else root
        self.fold = fold
        self.transform = transform
        self.augment = augment
        self.image_size = image_size
        
        # Store file paths and frame indices instead of actual data
        self.frame_locations: List[Dict] = []
        self.total_frames = 0
        
        # Scan dataset and create index
        logging.info(f"Indexing dataset in {self.root}...")
        self._create_dataset_index(valid_ratio)
        
    def _create_dataset_index(self, valid_ratio: float) -> None:
        """Creates an index of frame locations without loading the actual data."""
        all_files = list(self.root.glob("**/*.npz"))
        
        for file_path in all_files:
            try:
                # Only load metadata to get frame count
                with np.load(file_path, allow_pickle=True) as data:
                    num_frames = len(data['frames'])
                
                split_idx = int((1 - valid_ratio) * num_frames)
                
                if self.fold == Fold.TRAIN:
                    frame_range = range(split_idx)
                else:
                    frame_range = range(split_idx, num_frames)
                
                # Store only file path and frame index
                for frame_idx in frame_range:
                    self.frame_locations.append({
                        'file_path': file_path,
                        'frame_idx': frame_idx
                    })
                    self.total_frames += 1
                
            except Exception as e:
                logging.warning(f"Skipping {file_path} due to error: {e}")
        
        # Shuffle frame locations for training
        if self.fold == Fold.TRAIN:
            random.shuffle(self.frame_locations)
            
        logging.info(f"Indexed {self.total_frames} frames from {len(all_files)} files")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        """Load a single item on demand."""
        location = self.frame_locations[idx]
        
        # Load only the required frame
        with np.load(location['file_path'], allow_pickle=True) as data:
            frame = data['frames'][location['frame_idx']]
            steering = data['steerings'][location['frame_idx']]
        
        # Process images
        if isinstance(frame, dict):
            left, _ = self.preprocess_image(frame["left"], steering)
            front, _ = self.preprocess_image(frame["front"], steering)
            right, steering = self.preprocess_image(frame["right"], steering)
        else:
            raise TypeError("Frames should be a dictionary for multi-camera data.")
            
        return left, front, right, torch.tensor(steering, dtype=torch.float32)

    def preprocess_image(self, image, steering):
        """Applies augmentations to the image."""
        if self.augment:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            hue_factor = random.uniform(-0.05, 0.05)

            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = np.clip(image[:, :, 0] + hue_factor * 255, 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                steering = -steering

            # Random crop
            h, w, _ = image.shape
            x1, y1 = random.randint(0, w // 10), random.randint(0, h // 10)
            x2, y2 = w - random.randint(0, w // 10), h - random.randint(0, h // 10)
            image = image[y1:y2, x1:x2]

        image = cv2.resize(image, self.image_size)
        
        if self.augment:
            # Apply noise and motion blur with lower probability
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.01, image.shape)
                image = np.clip(image + noise, 0, 255)

            if random.random() < 0.2:
                kernel_size = random.choice([3, 5, 7])
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
                image = cv2.filter2D(image, -1, kernel)

        normalized = image / 255.0
        ordered_img = normalized.transpose(2, 0, 1)
        
        if self.transform:
            ordered_img = self.transform(torch.tensor(ordered_img, dtype=torch.float32))
            
        return ordered_img, steering

def get_dataloaders(root, valid_ratio, batch_size, num_workers, augment=False, image_size=(128, 128), use_cuda=False):
    transform = T.Compose([T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    logging.info("Creating training dataset index...")
    train_dataset = MemoryEfficientDataset(
        root=root,
        fold=Fold.TRAIN,
        transform=transform,
        augment=augment,
        valid_ratio=valid_ratio,
        image_size=image_size
    )
    
    logging.info("Creating validation dataset index...")
    valid_dataset = MemoryEfficientDataset(
        root=root,
        fold=Fold.VALID,
        transform=transform,
        augment=False,
        valid_ratio=valid_ratio,
        image_size=image_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    return train_loader, valid_loader
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Argument parser for flexibility
    parser = argparse.ArgumentParser(description="Test MemoryEfficientDataset")
    parser.add_argument("--root", type=str, default="/usr/users/avr/avr_11/ILIAR1/dataset_complet", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Resize images to this size")
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    logging.info("Initializing dataset...")

    train_loader, valid_loader = get_dataloaders(
        root=args.root,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        image_size=tuple(args.image_size),
        use_cuda=args.use_cuda and torch.cuda.is_available()
    )

    logging.info(f"Training dataset: {len(train_loader.dataset)} samples")
    logging.info(f"Validation dataset: {len(valid_loader.dataset)} samples")

    # Test first few batches
    for i, (left, front, right, steering) in enumerate(train_loader):
        if i >= 5:  # Only check first 5 batches
            break

        logging.info(f"Sample {i} - Steering Angle: {steering}")
        logging.info(f"Left View Shape: {left.shape}, Front View Shape: {front.shape}, Right View Shape: {right.shape}")

    logging.info("Dataset initialization and batch testing complete.")
