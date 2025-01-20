# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
from .resnet_18 import ResNet18  # Importer ResNet18 ici


def build_model(cfg, input_size, num_classes):
    # On vérifie que la classe demandée dans 'cfg' correspond bien à ResNet18
    model_class = cfg["class"]
    if model_class == "ResNet18":
        return ResNet18(pretrained=cfg.get("pretrained", True))
    # D'autres modèles peuvent être gérés ici si nécessaire
    return eval(f"{model_class}(cfg, input_size, num_classes)")
