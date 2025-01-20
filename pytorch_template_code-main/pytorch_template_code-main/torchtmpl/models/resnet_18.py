
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        # Charger le modèle ResNet18 préentraîné de torchvision
        self.model = models.resnet18(pretrained=pretrained)
        
        # Adapter la dernière couche pour une tâche de régression
        num_features = self.model.fc.in_features  # Taille d'entrée de la couche FC
        self.model.fc = nn.Linear(num_features, 1)  # Remplacer avec une sortie unique (régression)

    def forward(self, x):
        return self.model(x)
