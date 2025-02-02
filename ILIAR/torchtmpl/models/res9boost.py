import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(ResNet18, self).__init__()
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
            # Optionally, replicate the pretrained weights for the additional channels
            # new_conv1_weight[:, 3:, :, :] = pretrained_conv1[:, :6, :, :]
            self.model.conv1.weight = nn.Parameter(new_conv1_weight)
        
        # Modify the fully connected layer to match the desired number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
