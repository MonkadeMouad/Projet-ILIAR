import torch
import numpy as np
from torchvision.models import resnet18
import torch.nn as nn
import cv2

# Define the custom ResNet18 model
class ResNet18_9channels(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(ResNet18_9channels, self).__init__()
        # Initialize the standard ResNet18
        self.model = resnet18(pretrained=pretrained)

        # Modify the first convolutional layer to accept 9 channels instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels=9,              # Changed from 3 to 9
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize the new conv1 weights
        if pretrained:
            # Load the pretrained weights for the first 3 channels
            pretrained_conv1 = resnet18(pretrained=True).conv1.weight.data
            # Initialize weights for the additional 6 channels
            new_conv1_weight = torch.randn(64, 9, 7, 7) * 0.01  # Small random weights
            # Assign the pretrained weights to the first 3 channels
            new_conv1_weight[:, :3, :, :] = pretrained_conv1
            self.model.conv1.weight = nn.Parameter(new_conv1_weight)

        # Modify the fully connected layer to match the desired number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, left, forward, right):
        # Concatenate the left, forward, and right images along the channel dimension
        x = torch.cat([left, forward, right], dim=1)
        # Pass the concatenated input through the model
        return self.model(x)

def preprocess_image(image, image_size=(128, 128)):
    """Preprocess the image: resize, normalize, and format as CHW tensor."""
    resized = cv2.resize(image, image_size)
    normalized = resized / 255.0
    ordered_img = normalized.transpose(2, 0, 1)  # HWC -> CHW
    return torch.tensor(ordered_img, dtype=torch.float32)

if __name__ == "__main__":
    # Path to the specific .npz file
    npz_file = "/usr/users/avr/avr_11/ILIAR1/dataset_complet/chunk_0.npz"

    # Load the .npz file
    data = np.load(npz_file, allow_pickle=True)
    frames = data["frames"]  # Dictionary containing left, front, right views
    steerings = data["steerings"]  # Steering angles

    # Extract the first sample
    first_sample = frames[0]
    left_img = preprocess_image(first_sample["left"])
    front_img = preprocess_image(first_sample["front"])
    right_img = preprocess_image(first_sample["right"])
    steering_angle = steerings[0]

    # Add batch dimension
    left_img = left_img.unsqueeze(0)  # Shape: (1, 3, H, W)
    front_img = front_img.unsqueeze(0)  # Shape: (1, 3, H, W)
    right_img = right_img.unsqueeze(0)  # Shape: (1, 3, H, W)

    # Instantiate the model
    model = ResNet18_9channels(pretrained=False)

    # Move data and model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    left_img, front_img, right_img = left_img.to(device), front_img.to(device), right_img.to(device)

    # Test the model with the first sample
    output = model(left_img, front_img, right_img)

    # Print the output
    print(f"Model output: {output.item()}")
    print(f"Ground truth steering angle: {steering_angle}")
