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
import torch.optim.lr_scheduler as lr_scheduler

import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(cfg, optimizer, train_loader=None, n_epochs=None):
    """
    Dynamically creates a learning rate scheduler based on the config file.

    Args:
        cfg (dict): The scheduler configuration with keys 'type' and additional params.
        optimizer (torch.optim.Optimizer): The optimizer whose LR needs to be scheduled.
        train_loader (DataLoader, optional): Training DataLoader (needed for OneCycleLR).
        n_epochs (int, optional): Number of epochs (needed for OneCycleLR).

    Returns:
        torch.optim.lr_scheduler: The configured scheduler.
    """
    scheduler_class = getattr(lr_scheduler, cfg["type"], None)
    if not scheduler_class:
        raise ValueError(f"Unsupported scheduler: {cfg['type']}")

    scheduler_params = {k: v for k, v in cfg.items() if k != "type"}

    # Handle OneCycleLR separately since it requires total_steps
    if cfg["type"] == "OneCycleLR":
        if "total_steps" not in scheduler_params or scheduler_params["total_steps"] is None:
            if train_loader is None or n_epochs is None:
                raise ValueError("OneCycleLR requires `train_loader` and `n_epochs` to compute `total_steps`.")
            total_steps = len(train_loader) * n_epochs  # Calculate total steps dynamically
            scheduler_params["total_steps"] = total_steps

    return scheduler_class(optimizer, **scheduler_params)


import yaml
import torch
from torch.nn import Linear

# Path to your configuration file
CONFIG_PATH = "/usr/users/avr/avr_11/hammou1/hammou/ILIAR/configs/tmp4qmvnwmx-config.yml"


if __name__ == "__main__":
    # Load the config file
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    print("Testing get_loss, get_optimizer, and get_scheduler...")

    # Mock model for testing optimizer and scheduler
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

    # Test get_scheduler
    # Test get_scheduler
    try:
        class MockDataLoader:
            def __len__(self):
                return 100  # Assume 100 batches per epoch

        train_loader = MockDataLoader()
        n_epochs = config["training"]["epochs"]

        scheduler = get_scheduler(config["scheduler"], optimizer, train_loader, n_epochs)

        # Check if scheduler is valid (OneCycleLR is special)
        valid_schedulers = (lr_scheduler._LRScheduler, lr_scheduler.OneCycleLR)

        if not isinstance(scheduler, valid_schedulers):
            raise TypeError(f"Scheduler {scheduler} is not a valid torch.optim.lr_scheduler.")

        print(f"Scheduler created successfully: {scheduler}")

        # Simulate stepping the scheduler
        for _ in range(5):  # Simulating 5 epochs
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(0.5)  # Dummy validation loss
            else:
                scheduler.step()
            print(f"Scheduler stepped. Current LR: {scheduler.get_last_lr()[0]}")

    except Exception as e:
        print(f"Error testing get_scheduler: {e}")