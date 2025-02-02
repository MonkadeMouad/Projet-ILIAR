
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from .cbam import CBAMBlock
class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        # Charger le modèle ResNet18 préentraîné de torchvision
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Adapter la dernière couche pour une tâche de régression
        num_features = self.model.fc.in_features  # Taille d'entrée de la couche FC
        self.model.fc = nn.Linear(num_features, 1)  # Remplacer avec une sortie unique (régression)

    def forward(self, x):
        return self.model(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18_9channels(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, dropout_prob=0.8):
        super(ResNet18_9channels, self).__init__()  

        # Load standard ResNet18 model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 9 channels
        self.model.conv1 = nn.Conv2d(
            in_channels=9,  # Changed from 3 to 9
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize the new conv1 weights
        if pretrained:
            pretrained_conv1 = resnet18(weights=ResNet18_Weights.DEFAULT).conv1.weight.data
            new_conv1_weight = torch.randn(64, 9, 7, 7) * 0.01  # Small random weights
            new_conv1_weight[:, :3, :, :] = pretrained_conv1  # Copy first 3 channels
            self.model.conv1.weight = nn.Parameter(new_conv1_weight)

        # Add dropout before the fully connected layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Modify the fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, left, forward, right):
        # Concatenate the left, forward, and right images along the channel dimension
        x = torch.cat([left, forward, right], dim=1)

        # Forward pass through ResNet18
        x = self.model(x)

        # Apply dropout
        x = self.dropout(x)

        return x


import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2_9channels(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, dropout_prob=0.3):
        super(MobileNetV2_9channels, self).__init__()

        # Load the standard MobileNetV2
        base_model = mobilenet_v2(pretrained=pretrained)

        # Modify the first convolutional layer to accept 9 channels instead of 3
        self.model = base_model
        self.model.features[0][0] = nn.Conv2d(
            in_channels=9,              # Changed from 3 to 9
            out_channels=32,            # Keep original out_channels
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Initialize weights for new 9-channel conv layer
        if pretrained:
            pretrained_conv1 = mobilenet_v2(pretrained=True).features[0][0].weight.data  # (32, 3, 3, 3)
            new_conv1_weight = torch.zeros(32, 9, 3, 3)  # Initialize with zeros

            # Copy the first 3 channels from the pretrained model
            new_conv1_weight[:, :3, :, :] = pretrained_conv1

            # Fill the extra 6 channels with the mean of the first 3 channels
            new_conv1_weight[:, 3:, :, :] = pretrained_conv1.mean(dim=1, keepdim=True).repeat(1, 6, 1, 1)

            self.model.features[0][0].weight = nn.Parameter(new_conv1_weight)

        # Modify the classifier to match the desired number of output classes
        in_features = self.model.classifier[1].in_features  # Get last FC layer input size
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, left, front, right):
        # Concatenate the left, forward, and right images along the channel dimension
        x = torch.cat([left, front, right], dim=1)
        x = self.model(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(9, 24, kernel_size=5, stride=2)  # Input: 9 channels (left, front, right concatenated)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 100)  # Adjust input size based on image resolution
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)  # Single output for regression (steering angle)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, left, front, right):
        """
        Forward pass that concatenates the three camera views and passes them through the CNN.
        """
        # Concatenate left, front, and right images along the channel dimension (dim=1)
        x = torch.cat([left, front, right], dim=1)

        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Steering angle output

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class NVIDIA_CNN(nn.Module):
    def __init__(self):
        super(NVIDIA_CNN, self).__init__()
        
        # Convolutional layers (assuming 3-channel input, 128x128 resolution)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after all conv layers for a single (128x128) image.
        # Example with stride=2 on three 5x5 layers, then two 3x3 layers:
        #   - After conv1:  (128 - 5)/2 + 1 = 62 --> 62×62 features
        #   - After conv2:  (62 - 5)/2 + 1 = 29 --> 29×29 features
        #   - After conv3:  (29 - 5)/2 + 1 = 13 --> 13×13 features
        #   - After conv4:  13 - 3 + 1 = 11 (stride=1) --> 11×11
        #   - After conv5:  11 - 3 + 1 = 9  --> 9×9
        #
        # 64 channels * 9 * 9 = 5184
        feature_size = 64 * 9 * 9

        # Fully connected layers (only a single image’s features)
        self.fc1 = nn.Linear(feature_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)  # Steering angle output

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass for a single camera image (3-channel).
        """
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Steering angle output

        return x
class ResNet18_9channels_CBAM(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, dropout_prob=0.5):
        super(ResNet18_9channels_CBAM, self).__init__()

        # Load ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify first conv layer to take 9-channel input
        self.model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            pretrained_conv1 = resnet18(weights=ResNet18_Weights.DEFAULT).conv1.weight.data
            new_conv1_weight = torch.randn(64, 9, 7, 7) * 0.01
            new_conv1_weight[:, :3, :, :] = pretrained_conv1  # Copy pre-trained weights for RGB channels
            self.model.conv1.weight = nn.Parameter(new_conv1_weight)

        # Insert CBAM before layer3 for maximum impact
        self.cbam1 = CBAMBlock(channels=64)
        self.cbam2 = CBAMBlock(channels=128)
        self.cbam3 = CBAMBlock(channels=256)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, left, forward, right):
        x = torch.cat([left, forward, right], dim=1)  # Concatenate 3 views

        # Apply ResNet layers with CBAM
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.cbam1(x)  # CBAM added after layer1

        x = self.model.layer2(x)
        x = self.cbam2(x)  # CBAM after layer2

        x = self.model.layer3(x)
        x = self.cbam3(x)  # CBAM after layer3

        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)

        return x
from torchvision.models import efficientnet_b0

class EfficientNetB0_9channels_CBAM(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, dropout_prob=0.5):
        super(EfficientNetB0_9channels_CBAM, self).__init__()

        self.model = efficientnet_b0(weights="IMAGENET1K_V1")

        # Modify first layer for 9 channels
        self.model.features[0][0] = nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Add CBAM Attention
        self.cbam1 = CBAMBlock(32)
        self.cbam2 = CBAMBlock(48)
        self.cbam3 = CBAMBlock(112)

        self.model.classifier[1] = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, left, forward, right):
        x = torch.cat([left, forward, right], dim=1)

        x = self.model.features[0](x)
        x = self.cbam1(x)

        x = self.model.features[1:3](x)
        x = self.cbam2(x)

        x = self.model.features[3:6](x)
        x = self.cbam3(x)

        x = self.model.features[6:](x)
        x = x.mean([2, 3])
        x = self.dropout(x)
        x = self.model.classifier(x)

        return x
