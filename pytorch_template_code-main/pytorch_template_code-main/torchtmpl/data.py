# coding: utf-8

# Standard imports
import logging
import pathlib
from enum import Enum

# External imports
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Union

class Fold(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2

# Global dataset
GLOBAL_DATASET = None

class NPZDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        fold: Fold,
        transform=None,
        valid_ratio: float = 0.2,
    ):
        if isinstance(root, str):
            root = pathlib.Path(root)

        self.datafiles = {}
        self.num_frames = 0

        for f in root.glob("**/*.npz"):
            # Skip temporary .npz.tmp files
            if f.suffix == ".tmp.npz":
                f.unlink()  # Delete the temporary file
                continue

            try:
                # Load data from each .npz file with allow_pickle=True
                data = np.load(f, allow_pickle=True)
                frames = data["frames"]  # Shape: (N, 3, H, W) or dict with camera views
                steerings = data["steerings"]  # Shape: (N,)
            except Exception as e:
                logging.warning(f"Error loading {f}: {e}")
                continue  # Skip problematic file and continue

            # Split dataset
            num_samples = len(frames)
            split_idx = int((1 - valid_ratio) * num_samples)
            if fold == Fold.TRAIN:
                frames = frames[:split_idx]
                steerings = steerings[:split_idx]
            elif fold == Fold.VALID:
                frames = frames[split_idx:]
                steerings = steerings[split_idx:]

            self.datafiles[f] = {
                "frames": frames,
                "steerings": steerings,
                "num_frames": len(frames),
            }
            self.num_frames += len(frames)

        self.cached_datafile = {
            "filename": None,
            "frames": None,
            "steerings": None,
        }
        self.transform = transform

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        def preprocess_frames(frame):
            """Preprocess the frame (convert to tensor, normalize, etc.)."""
            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame).float() / 255.0  # Normalization to [0, 1]
                frame = frame.permute(2, 0, 1)  # Convert HWC -> CHW
            else:
                logging.error(f"Expected np.ndarray, but got {type(frame)}.")
                raise TypeError(f"Expected np.ndarray, but got {type(frame)}")
            return frame

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

                # Gestion des frames sous forme de dictionnaire
                if isinstance(frames[idx], dict):
                    # Par défaut, sélectionnons l'image "front" si elle existe
                    selected_frame = frames[idx].get("front", None)
                    if selected_frame is None:
                        raise KeyError(f"'front' key not found in frame dictionary: {frames[idx].keys()}")
                    frame = preprocess_frames(selected_frame)
                elif isinstance(frames[idx], np.ndarray):
                    frame = preprocess_frames(frames[idx])
                else:
                    raise TypeError(f"Unexpected type for frame: {type(frames[idx])}")

                Y = steerings[idx]
                break

            idx -= chunk["num_frames"]

        if self.transform:
            frame = self.transform(frame)

        return frame, Y.astype(np.float32)


def initialize_global_dataset(datapath, valid_ratio):
    global GLOBAL_DATASET

    input_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Assurez-vous que vos frames sont des tenseurs.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Clean up temporary .npz.tmp files in the dataset folder
    for tmp_file in pathlib.Path(datapath).rglob("*.tmp.npz"):
        tmp_file.unlink()

    GLOBAL_DATASET = {
        Fold.TRAIN: NPZDataset(
            root=datapath, fold=Fold.TRAIN, transform=input_transform, valid_ratio=valid_ratio
        ),
        Fold.VALID: NPZDataset(
            root=datapath, fold=Fold.VALID, transform=input_transform, valid_ratio=valid_ratio
        ),
    }

def show_samples():
    global GLOBAL_DATASET

    dataset = GLOBAL_DATASET[Fold.TRAIN]
    nsamples = 5
    fig = plt.figure(constrained_layout=True, figsize=(10, 30))
    fig.suptitle("Samples from the dataset")

    subfigs = fig.subfigures(nrows=nsamples, ncols=1)

    for row, subfig in enumerate(subfigs):
        idx = np.random.randint(0, len(dataset))
        X, y = dataset[idx]
        subfig.suptitle(f"Sample {idx}, Command to predict : {y}")

        axs = subfig.subplots(nrows=1, ncols=1)
        axs.imshow(X.permute(1, 2, 0))  # Assuming X is a tensor of shape (3, H, W)

    plt.show()

def get_dataloaders(batch_size, num_workers, use_cuda):
    global GLOBAL_DATASET

    train_dataset = GLOBAL_DATASET[Fold.TRAIN]
    valid_dataset = GLOBAL_DATASET[Fold.VALID]

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

    logging.info(f"Training size: {len(train_dataset)}")
    logging.info(f"Validation size: {len(valid_dataset)}")

    return train_loader, valid_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    datapath = os.path.expanduser("~/ros2_ws/dataset")
    valid_ratio = 0.2

    logging.info("Initializing dataset...")
    initialize_global_dataset(datapath, valid_ratio)

    logging.info("Dataset initialized.")
    logging.info(f"Training size: {len(GLOBAL_DATASET[Fold.TRAIN])}")
    logging.info(f"Validation size: {len(GLOBAL_DATASET[Fold.VALID])}")

    logging.info("Displaying samples...")
    show_samples()
