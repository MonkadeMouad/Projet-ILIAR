# coding: utf-8

# External imports
import torch


from .resnet_18 import *  # Importer ResNet18 ici


def build_model(cfg, input_size, num_classes):
    # On vérifie que la classe demandée dans 'cfg' correspond bien à ResNet18
    model_class = cfg["class"]
    if model_class == "ResNet18":
        return ResNet18(pretrained=cfg.get("pretrained", True))
    if model_class == "ResNet18_9channels":
        return ResNet18_9channels(pretrained=cfg.get("pretrained", True))
    if model_class == "mobile":
        return MobileNetV2_9channels(pretrained=cfg.get("pretrained", True),num_classes=cfg.get("num_classes", 1),
        dropout_prob=cfg.get("dropout_prob", 0.3)
)
    if model_class == "cnn":
        return SimpleCNN()
    if model_class == "Nvidia":
        return NVIDIA_CNN( )
    if model_class == "Eff":
        return EfficientNetB0_9channels_CBAM()
    # D'autres modèles peuvent être gérés ici si nécessaire
    return eval(f"{model_class}(cfg, input_size, num_classes)")
