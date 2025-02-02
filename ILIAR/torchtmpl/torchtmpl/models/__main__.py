# coding: utf-8

# External imports
import torch

# Local imports
from . import build_model


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_resnet18():
    cfg = {"class": "ResNet18", "pretrained": True}  # Assurez-vous que 'pretrained' est bien défini si vous voulez l'utiliser
    input_size = (3, 224, 224)  # Taille d'entrée typique pour ResNet
    batch_size = 64
    num_classes = 1  # Sortie pour régression, donc 1 classe
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)  # Batch d'images
    output = model(input_tensor)
    print(f"Output tensor of size: {output.shape}")

if __name__ == "__main__":
    test_resnet18()

