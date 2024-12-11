# coding: utf-8

# Standard imports
import pathlib
from enum import Enum

# External imports
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np

from typing import Union  
class Fold(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


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
            # Load data from each .npz file with allow_pickle=True
            data = np.load(f, allow_pickle=True)
            frames = data["frames"]  # Shape: (N, 3, H, W)
            steerings = data["steerings"]  # Shape: (N,)

            # Split dataset
            num_samples = len(frames)
            split_idx = int((1 - valid_ratio) * num_samples)
            if fold == Fold.TRAIN:
                frames = frames[:split_idx]
                steerings = steerings[:split_idx]
            elif fold == Fold.VALID:
                frames = frames[split_idx:]
                steerings = steerings[split_idx:]
            elif fold == Fold.TEST:
                # Use the entire dataset for testing
                pass

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
            frame = torch.from_numpy(frame).float() / 255.0  # Normalize
            frame = frame.permute(2, 0, 1)  # Convert HWC to CHW
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

                # Extract specific frame set
                frame_set = frames[idx]  # This is a dictionary
                Xleft = preprocess_frames(frame_set["left"])
                Xforward = preprocess_frames(frame_set["front"])
                Xright = preprocess_frames(frame_set["right"])
                Y = steerings[idx]  # Steering command
                break

            idx -= chunk["num_frames"]

        if self.transform:
            Xleft = self.transform(Xleft)
            Xforward = self.transform(Xforward)
            Xright = self.transform(Xright)

        return {"left": Xleft, "forward": Xforward, "right": Xright}, Y.astype(np.float32)


def show_samples():

    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ros2_ws/dataset>")
        sys.exit(1)

    datapath = sys.argv[1]
    input_transform = transforms.Compose([transforms.Resize((128, 128))])

    dataset = NPZDataset(
        root=datapath,
        fold=Fold.TRAIN,
        transform=input_transform,
        valid_ratio=0.2,
    )

    nsamples = 5
    fig = plt.figure(constrained_layout=True, figsize=(10, 30))
    fig.suptitle("Samples from the dataset")

    subfigs = fig.subfigures(nrows=nsamples, ncols=1)

    for row, subfig in enumerate(subfigs):
        idx = np.random.randint(0, len(dataset))
        X, y = dataset[idx]
        subfig.suptitle(f"Sample {idx}, Command to predict : {y}")

        axs = subfig.subplots(nrows=1, ncols=3)

        axs[0].imshow(X["left"].permute(1, 2, 0))
        axs[1].imshow(X["forward"].permute(1, 2, 0))
        axs[2].imshow(X["right"].permute(1, 2, 0))

    plt.show()


if __name__ == "__main__":
    show_samples()
