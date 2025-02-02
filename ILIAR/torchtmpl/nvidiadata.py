import numpy as np
import torch
import torchvision.transforms as T
import random
import cv2
import pathlib
import logging
from enum import Enum
from typing import Union


class Fold(Enum):
    TRAIN = 0
    VALID = 1


class NPZDatasetWithAugmentations(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        fold: Fold,
        transform=None,
        augment=False,
        valid_ratio=0.2,
        image_size=(128, 128),
        steering_offset=0.2,  # Adjust steering for left/right views
    ):
        self.root = pathlib.Path(root) if isinstance(root, str) else root
        self.fold = fold
        self.transform = transform
        self.augment = augment
        self.image_size = image_size
        self.steering_offset = steering_offset
        self.datafiles = []
        self.num_frames = 0

        # Load data
        for f in self.root.glob("**/*.npz"):
            if f.suffix == ".tmp.npz":
                f.unlink()
                continue

            try:
                data = np.load(f, allow_pickle=True)
                frames = data["frames"]  # Dictionary with {"left", "front", "right"}
                steerings = data["steerings"]  # Array of steering angles
            except Exception as e:
                logging.warning(f"Error loading {f}: {e}")
                continue

            num_samples = len(frames)
            split_idx = int((1 - valid_ratio) * num_samples)

            # Train-validation split
            if fold == Fold.TRAIN:
                frames, steerings = frames[:split_idx], steerings[:split_idx]
            elif fold == Fold.VALID:
                frames, steerings = frames[split_idx:], steerings[split_idx:]

            # Store all frames and their corresponding steering values
            for i in range(len(frames)):
                self.datafiles.append({"frames": frames[i], "steering": steerings[i]})

            self.num_frames += len(frames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        def preprocess_image(image):
            """Resize, normalize, and convert image to PyTorch tensor."""
            image = cv2.resize(image, self.image_size)
            image = image / 255.0  # Normalize to range [0,1]
            return torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC → CHW

        def augment_image(image, steering, is_left=False, is_right=False):
            """Apply random augmentations and adjust steering angle for left/right views."""
            transform_list = []

            # Brightness & contrast
            transform_list.append(T.ColorJitter(brightness=0.3, contrast=0.3))

            # Gaussian Blur (50% probability)
            if random.random() > 0.5:
                transform_list.append(T.GaussianBlur(kernel_size=3))

            # Convert to PIL and apply transformations
            transform = T.Compose(transform_list)
            image = transform(T.ToPILImage()(image))
            image = np.array(image)  # Convert back to NumPy

            # Additional augmentations for Left & Right images
            if is_left or is_right:
                # Random rotation (-5° to +5°)
                angle = random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

                # Horizontal shift (-10px to +10px)
                shift = random.randint(-10, 10)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

                # Adjust steering for left/right images
                if is_left:
                    steering += self.steering_offset
                elif is_right:
                    steering -= self.steering_offset

            return image, steering

        # Load frame data
        frame_data = self.datafiles[idx]
        steering = frame_data["steering"]

        # Extract views
        left = preprocess_image(frame_data["frames"]["left"])
        front = preprocess_image(frame_data["frames"]["front"])
        right = preprocess_image(frame_data["frames"]["right"])

        # Apply augmentations (only to left & right)
        if self.augment:
            left, steering_left = augment_image(left, steering, is_left=True)
            right, steering_right = augment_image(right, steering, is_right=True)
        else:
            steering_left = steering_right = steering

        # Convert to PyTorch tensor
        steering_tensor = torch.tensor([steering_left, steering, steering_right], dtype=torch.float32)

        return left, front, right, steering_tensor


def get_dataloaders(root, valid_ratio, batch_size, num_workers, augment=False, image_size=(128, 128), use_cuda=False):
    """Create DataLoader objects for training and validation sets."""
    transform = T.Compose([
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    train_dataset = NPZDatasetWithAugmentations(
        root=root, fold=Fold.TRAIN, transform=transform, augment=augment, valid_ratio=valid_ratio, image_size=image_size
    )

    valid_dataset = NPZDatasetWithAugmentations(
        root=root, fold=Fold.VALID, transform=transform, augment=False, valid_ratio=valid_ratio, image_size=image_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    logging.info(f"Training size: {len(train_dataset)} (augmented)")
    logging.info(f"Validation size: {len(valid_dataset)}")

    return train_loader, valid_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    datapath = "C:/Users/Hamza/Desktop/Modele/dataset_complet"
    valid_ratio = 0.2

    logging.info("Initializing dataset...")

    train_loader, valid_loader = get_dataloaders(
        root=datapath,
        valid_ratio=valid_ratio,
        batch_size=8,
        num_workers=0,
        augment=True,
        image_size=(128, 128),
        use_cuda=torch.cuda.is_available()
    )

    logging.info("Dataset initialized.")
    logging.info(f"Training size: {len(train_loader.dataset)}")
    logging.info(f"Validation size: {len(valid_loader.dataset)}")

    logging.info("Displaying first few samples from validation set...")

    for i, (left, front, right, steering) in enumerate(valid_loader):
        if i >= 5:  # Show only the first 5 samples
            break
        logging.info(f"Sample {i} - Steering Angles: {steering.numpy()}")
        logging.info(f"Left Shape: {left.shape}, Front Shape: {front.shape}, Right Shape: {right.shape}")
