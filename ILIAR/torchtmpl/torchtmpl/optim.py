# coding: utf-8

# External imports
import torch
import torch.nn as nn

def get_loss(cfg):
    loss_class = getattr(torch.nn, cfg["type"], None)
    if not loss_class:
        raise ValueError(f"Unsupported loss type: {cfg['type']}")

    # Debugging statement
    print(f"Instantiating loss {cfg['type']} with params: {cfg.get('paramloss', {})}")
    return loss_class(**cfg.get("paramloss", {}))


def get_optimizer(cfg, params):
    """
    Dynamically creates an optimizer based on the config file.
    
    Args:
        cfg (dict): The optimizer configuration with keys 'algo' and 'params'.
        params (iterable): The parameters to optimize.
    
    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_class = getattr(torch.optim, cfg["algo"], None)  # Dynamically get the optimizer class
    if not optimizer_class:
        raise ValueError(f"Unsupported optimizer: {cfg['algo']}")

    # Return the instantiated optimizer with the parameters and config options
    return optimizer_class(params, **cfg.get("params", {}))

import yaml
import torch
from torch.nn import Linear

# Path to your configuration file
CONFIG_PATH = "/usr/users/avr/avr_11/ILIAR1/configs/tmp4qmvnwmx-config.yml"

if __name__ == "__main__":
    # Load the config file
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    # Test optimizer and loss functions
    print("Testing get_loss and get_optimizer...")

    # Mock model for testing optimizer
    model = Linear(10, 1)  # Example model with simple linear layer

    # Test get_loss
    try:
        loss_fn = get_loss(config["loss"])
        assert isinstance(loss_fn, torch.nn.Module), "Loss function is not a valid torch.nn.Module."
        print(f"Loss function created successfully: {loss_fn}")
    except Exception as e:
        print(f"Error testing get_loss: {e}")

    # Test get_optimizer
    try:
        optimizer = get_optimizer(config["optim"], model.parameters())
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer is not a valid torch.optim.Optimizer."
        print(f"Optimizer created successfully: {optimizer}")

        # Check optimizer parameters
        for param_group in optimizer.param_groups:
            if "lr" in config["optim"]["params"]:
                assert param_group["lr"] == config["optim"]["params"]["lr"], "Learning rate does not match config."
            if "weight_decay" in config["optim"]["params"]:
                assert param_group["weight_decay"] == config["optim"]["params"]["weight_decay"], "Weight decay does not match config."
    except Exception as e:
        print(f"Error testing get_optimizer: {e}")

    print("All tests passed!")

