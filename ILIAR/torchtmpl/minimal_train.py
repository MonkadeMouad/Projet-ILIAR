# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
from torchinfo import summary

# Local imports
from torchtmpl.dataaugmentation import get_dataloaders
from torchtmpl.models import build_model
from torchtmpl.optim import get_optimizer, get_loss
from torchtmpl.utils import train_one_epoch, validate, ModelCheckpoint, generate_unique_logpath
from torchtmpl.models.resnet_18 import ResNet18_9channels, MobileNetV2_9channels , NVIDIA_CNN  # Ensure this path is correct based on your project structure
from torchtmpl.name_generator import generate_cool_name

def train(config):
    run_name = generate_cool_name()
    wandb.init(
        project="YourProjectName",
        config=config,
        name=run_name,
    )

    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Build dataloaders
    logging.info("= Initializing dataset and building dataloaders")
    data_config = config["data"]
    datapath = os.path.expanduser(data_config["trainpath"])
    valid_ratio = data_config.get("valid_ratio", 0.2)
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 4)

    train_loader, valid_loader = get_dataloaders(
        root=datapath,
        valid_ratio=valid_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=data_config.get("augment", False),
        image_size=data_config.get("image_size", (224, 224)),
        use_cuda=use_cuda
    )
    print(f"Dataloaders built with batch_size={batch_size} and num_workers={num_workers}.")

    # Build the model
    logging.info("= Building the model")
    model_config = config["model"]
    num_classes = model_config["num_classes"]
    model = build_model(model_config, input_size=(batch_size, 9, 224, 224), num_classes=num_classes)
    model.to(device)
    print(f"Model built and moved to device {device}.")

    # Configure the loss function
    logging.info("= Configuring the loss function")
    config_loss = config["loss"]
    criterion = get_loss(config_loss)
    print("Loss function configured: MSELoss.")

    # Configure the optimizer
    logging.info("= Configuring the optimizer")
    optim_config = config["optim"]
    optimizer = get_optimizer(optim_config, model.parameters())
    print(f"Optimizer configured: {optimizer}.")

    # Logging directory
    logging_config = config["logging"]
    logdir = generate_unique_logpath(logging_config["logdir"], model_config["class"])
    os.makedirs(logdir, exist_ok=True)
    logging.info(f"Logs will be saved in {logdir}")
    print(f"Logging directory set to: {logdir}")

    # Save configuration
    with open(pathlib.Path(logdir) / "config.yaml", "w") as f:
        yaml.dump(config, f)
    print("Configuration saved to config.yaml.")

    # Watch model with W&B
    wandb.watch(model, criterion, log="all", log_freq=10)
    print("W&B is now watching the model.")

    # Set up the checkpoint for the best model
    checkpoint = ModelCheckpoint(model, savepath=pathlib.Path(logdir) / "best_model.pth")
    print("Checkpointing setup complete.")

    # Define the model dump directory
    model_dump_dir = "/usr/users/avr/avr_11/hammou1/hammou/ILIAR/modeldump"
    os.makedirs(model_dump_dir, exist_ok=True)
    logging.info(f"Model dumps will be saved in {model_dump_dir}")
    print(f"Model dump directory set to: {model_dump_dir}")

    # Training loop
    n_epochs = config["training"]["epochs"]
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Starting Epoch {epoch + 1}/{n_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate
        valid_loss = validate(model, valid_loader, criterion, device)
        print(f"Validation Loss: {valid_loss:.4f}")

        # Save the best model
        is_best = checkpoint.update(valid_loss)
        logging.info(f"Validation Loss: {valid_loss:.4f} {'[BEST]' if is_best else ''}")
        print(f"Validation Loss: {valid_loss:.4f} {'[BEST]' if is_best else ''}")

        # Save the model after the current epoch
        epoch_model_path = os.path.join(model_dump_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        logging.info(f"Saved model for epoch {epoch + 1} at {epoch_model_path}")
        print(f"Model checkpoint saved at: {epoch_model_path}")

       
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })
        print(f"Metrics logged for Epoch {epoch + 1}.")

    logging.info("Training complete.")
    print("Training complete.")
    wandb.finish()



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) not in [3, 4]:
        logging.error(f"Usage: {sys.argv[0]} config.yaml <train|test> [model_path_for_test]")
        print(f"Usage: {sys.argv[0]} config.yaml <train|test> [model_path_for_test]")
        sys.exit(-1)

    config_path = sys.argv[1]
    command = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) == 4 else None

    logging.info(f"Loading configuration from {config_path}")
    print(f"Loading configuration from {config_path}")
    config = yaml.safe_load(open(config_path, "r"))
    print("Configuration loaded.")

    if command == "train":
        train(config)
    elif command == "test":
        test(config, model_path=model_path)
    else:
        logging.error(f"Unknown command: {command}")
        print(f"Unknown command: {command}")
        sys.exit(-1)
