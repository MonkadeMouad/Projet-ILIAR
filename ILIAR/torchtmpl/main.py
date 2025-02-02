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
from torchtmpl.data import initialize_global_dataset, get_dataloaders
from torchtmpl.models import build_model
from torchtmpl.optim import get_loss, get_optimizer
from torchtmpl.utils import train_one_epoch, validate, ModelCheckpoint, generate_unique_logpath
from torchtmpl.models.resnet_18 import ResNet18

def train(config):
    # Initialize W&B
    wandb.init(
        project="YourProjectName",
        config=config,
        name=config["logging"].get("run_name", "experiment"),
    )

    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize dataset
    logging.info("= Initializing the dataset")
    data_config = config["data"]
    datapath = os.path.expanduser(data_config["trainpath"])
    valid_ratio = data_config.get("valid_ratio", 0.2)
    initialize_global_dataset(datapath, valid_ratio)

    # Build dataloaders
    logging.info("= Building dataloaders")
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 0)
    train_loader, valid_loader = get_dataloaders(batch_size, num_workers, use_cuda)

    # Build the model
    logging.info("= Building the model")
    model_config = config["model"]
    input_size = (batch_size, 3, 224, 224)  # Adapt for your input
    num_classes = model_config["num_classes"]
    model = build_model(model_config, input_size=input_size, num_classes=num_classes)
    model.to(device)

    # Configure the loss function
    logging.info("= Configuring the loss function")
    criterion = torch.nn.MSELoss()

    # Configure the optimizer
    logging.info("= Configuring the optimizer")
    optim_config = config["optim"]
    optimizer = get_optimizer(optim_config, model.parameters())

    # Logging directory
    logging_config = config["logging"]
    logdir = generate_unique_logpath(logging_config["logdir"], model_config["class"])
    os.makedirs(logdir, exist_ok=True)
    logging.info(f"Logs will be saved in {logdir}")

    # Save configuration
    with open(pathlib.Path(logdir) / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Watch model with W&B
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Set up the checkpoint
    checkpoint = ModelCheckpoint(model, savepath=pathlib.Path(logdir) / "best_model.pth")

    # Training loop
    n_epochs = config["training"]["epochs"]
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        valid_loss = validate(model, valid_loader, criterion, device)

        # Save the best model
        is_best = checkpoint.update(valid_loss)
        logging.info(f"Validation Loss: {valid_loss:.4f} {'[BEST]' if is_best else ''}")

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })

    logging.info("Training complete.")
    wandb.finish()






def test(config):
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialisation des données de test
    logging.info("= Initializing the dataset for testing")
    data_config = config["data"]
    datapath = os.path.expanduser(data_config["testpath"])  # Ajoutez 'testpath' dans config.yaml
    initialize_global_dataset(datapath)

    # Création du DataLoader de test
    logging.info("= Building test dataloaders")
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 4)
    test_loader = get_dataloaders(batch_size, num_workers, use_cuda, mode="test")  # mode="test"

    # Chargement du meilleur modèle
    logging.info("= Loading the best model from the checkpoint")
    model_config = config["model"]
    input_size = (batch_size, 3, 224, 224)  # Ajusté en fonction de votre taille d'entrée
    num_classes = model_config["num_classes"]

    model = build_model(model_config, input_size=input_size, num_classes=num_classes)
    model.to(device)

    # Chargement du modèle sauvegardé
    checkpoint_path = pathlib.Path(config["logging"]["logdir"]) / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        logging.info(f"Loaded model from {checkpoint_path}")
    else:
        logging.error("No model checkpoint found!")
        return

    # Passage en mode évaluation
    model.eval()

    # Évaluation
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.MSELoss()  # Utilisation de MSE pour la régression

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Pass forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calcul de l'erreur absolue moyenne (MAE)
            abs_error = torch.abs(outputs - targets)
            correct += (abs_error < 0.5).sum().item()  # Exemple d'évaluation simple
            total += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    mae = correct / total  # Vous pouvez calculer d'autres métriques aussi

    logging.info(f"Test Loss: {avg_loss:.4f}")
    logging.info(f"Test MAE: {mae:.4f}")

    # Log des métriques si wandb est utilisé
    if wandb.run:
        wandb.log({"test_loss": avg_loss, "test_mae": mae})

    logging.info("Testing complete.")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage: {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    config_path = sys.argv[1]
    command = sys.argv[2]

    logging.info(f"Loading configuration from {config_path}")
    config = yaml.safe_load(open(config_path, "r"))

    if command not in ["train", "test"]:
        logging.error(f"Unknown command: {command}")
        sys.exit(-1)

    eval(f"{command}(config)")
