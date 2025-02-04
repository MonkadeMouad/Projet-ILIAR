import numpy as np
import torch
import torchvision.transforms as T
import random
import cv2
import pathlib
import logging
from tqdm import tqdm
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
        image_size=(128, 128)
    ):
        self.root = pathlib.Path(root) if isinstance(root, str) else root
        self.fold = fold
        self.transform = transform
        self.augment = augment
        self.image_size = image_size
        self.datafiles = {}
        self.num_frames = 0

        logging.info(f"Scanning dataset in {self.root}...")
        for f in tqdm(self.root.glob("**/*.npz"), desc="Loading NPZ files"):
            if f.suffix == ".tmp.npz":
                f.unlink()
                continue

            try:
                data = np.load(f, allow_pickle=True)
                frames = data["frames"]
                steerings = data["steerings"]
            except Exception as e:
                logging.warning(f"Error loading {f}: {e}")
                continue

            num_samples = len(frames)
            split_idx = int((1 - valid_ratio) * num_samples)
            if fold == Fold.TRAIN:
                frames, steerings = frames[:split_idx], steerings[:split_idx]
            elif fold == Fold.VALID:
                frames, steerings = frames[split_idx:], steerings[split_idx:]

            self.datafiles[f] = {"frames": frames, "steerings": steerings, "num_frames": len(frames)}
            self.num_frames += len(frames)

        self.cached_datafile = {"filename": None, "frames": None, "steerings": None}

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        def preprocess_image(image):
            """Apply augmentations and normalize the image for MobileNetV2."""
            # 1️⃣ Random Brightness, Contrast, Hue Adjustments
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            hue_factor = random.uniform(-0.05, 0.05)

            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = np.clip(image[:, :, 0] + hue_factor * 255, 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # 2️⃣ Random Crop & Resize
            h, w, _ = image.shape
            x1, y1 = random.randint(0, w // 10), random.randint(0, h // 10)
            x2, y2 = w - random.randint(0, w // 10), h - random.randint(0, h // 10)
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, self.image_size)

            # 3️⃣ Add Gaussian Noise
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 255)

            # 4️⃣ Motion Blur (randomly apply)
            if random.random() < 0.2:
                kernel_size = random.choice([3, 5, 7])
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
                image = cv2.filter2D(image, -1, kernel)

            # 5️⃣ Normalize for Pretrained MobileNetV2
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image / 255.0  # Convert to [0,1] range
            image = (image - mean) / std  # Apply ImageNet normalization

            # Convert HWC → CHW for PyTorch
            image = image.transpose(2, 0, 1)

            return torch.tensor(image, dtype=torch.float32)

        for f, chunk in self.datafiles.items():
            if idx < chunk["num_frames"]:
                if self.cached_datafile["filename"] != f:
                    self.cached_datafile = {
                        "filename": f,
                        "frames": chunk["frames"],
                        "steerings": chunk["steerings"],
                    }

                frames = self.cached_datafile["frames"]
                steerings = self.cached_datafile["steerings"]

                if isinstance(frames[idx], dict):
                    left = preprocess_image(frames[idx]["left"])
                    front = preprocess_image(frames[idx]["front"])
                    right = preprocess_image(frames[idx]["right"])

                    # Ensure steering is returned
                    steering = torch.tensor(steerings[idx] / 2.0, dtype=torch.float32)                
                else:
                    raise TypeError("Frames should be a dictionary for multi-camera data.")

                return left, front, right, steering  # ✅ Steering is explicitly returned

            idx -= chunk["num_frames"]





def get_dataloaders(root, valid_ratio, batch_size, num_workers, augment=False, image_size=(128, 128), use_cuda=False):
    """Create DataLoader objects for training and validation sets with tqdm."""
    transform = T.Compose([
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    logging.info("Loading training dataset...")
    train_dataset = NPZDatasetWithAugmentations(
        root=root, fold=Fold.TRAIN, transform=transform, augment=augment, valid_ratio=valid_ratio, image_size=image_size
    )

    logging.info("Loading validation dataset...")
    valid_dataset = NPZDatasetWithAugmentations(
        root=root, fold=Fold.VALID, transform=transform, augment=False, valid_ratio=valid_ratio, image_size=image_size
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

    logging.info(f"Training size: {len(train_dataset)}")
    logging.info(f"Validation size: {len(valid_dataset)}")

    return train_loader, valid_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    datapath = "C:/Users/Hamza/Downloads/dataset_complet"
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
        if i >= 5:
            break

        logging.info(f"Sample {i} - Steering Angle: {steering}")
        logging.info(f"Left View Shape: {left.shape}, Front View Shape: {front.shape}, Right View Shape: {right.shape}")