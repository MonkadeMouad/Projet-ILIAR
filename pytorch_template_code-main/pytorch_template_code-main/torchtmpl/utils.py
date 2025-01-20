# coding: utf-8

# Standard imports
import os
from sched import scheduler

# External imports
import torch
import torch.nn
import tqdm
import time
import logging

import sys
import os
import pathlib
import yaml
from torchinfo import summary
import wandb
from torchtmpl.data import initialize_global_dataset, get_dataloaders
from torchtmpl.models import build_model
from torchtmpl.optim import get_optimizer

def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False

def train(config):
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
    num_workers = data_config.get("num_workers", 4)
    train_loader, valid_loader = get_dataloaders(batch_size, num_workers, use_cuda)

    # Build the model
    logging.info("= Building the model")
    model_config = config["model"]
    input_size = (batch_size, 3, 224, 224)  # Adjusted for the input
    num_classes = model_config["num_classes"]
    model = build_model(model_config, input_size=input_size, num_classes=num_classes)
    model.to(device)

    # Configure loss function
    logging.info("= Configuring the loss function")
    criterion = torch.nn.MSELoss()  # For regression, MSELoss

    # Configure optimizer
    logging.info("= Configuring the optimizer")
    optim_config = config["optim"]
    optimizer = get_optimizer(optim_config, model.parameters())

    # Configure scheduler if provided
    scheduler = None
    if "scheduler" in config:
        scheduler_config = config["scheduler"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)

    # Set up logging and directories
    logging_config = config["logging"]
    logdir = generate_unique_logpath(logging_config["logdir"], model_config["class"])
    os.makedirs(logdir, exist_ok=True)
    logging.info(f"Logs will be saved in {logdir}")

    # Save the config to the log directory
    with open(pathlib.Path(logdir) / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Save model summary
    model_summary = summary(model, input_size=input_size, device=device, verbose=0)
    with open(pathlib.Path(logdir) / "summary.txt", "w") as f:
        f.write(str(model_summary))

    # Initialize checkpoint for saving the best model
    checkpoint = ModelCheckpoint(model, savepath=pathlib.Path(logdir) / "best_model.pth")

    # Training loop
    n_epochs = config["training"]["epochs"]
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)

        # Validate
        valid_loss = validate(model, valid_loader, criterion, device)

        # Save the best model
        is_best = checkpoint.update(valid_loss)
        logging.info(f"Validation Loss: {valid_loss:.4f} {'[BEST]' if is_best else ''}")

        # Log metrics to wandb
        if wandb.run:
            wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})

    logging.info("Training complete.")


def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

    return total_loss / num_samples


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Ensure the target has the correct shape
        targets = targets.view(-1, 1)  # Reshapes to [batch_size, 1] if needed

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy or other metrics (if applicable)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):  # Classification loss
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(dataloader)

    # If classification, you might want to log accuracy
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        accuracy = correct / total
        logging.info(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        logging.info(f"Training Loss: {avg_loss:.4f}")

    epoch_time = time.time() - start_time
    logging.info(f"Epoch completed in {epoch_time:.2f} seconds")

    return avg_loss



def validate(model, dataloader, criterion, device):
    """
    Validate the model for one epoch, iterating over the dataloader
    and using the criterion to compute the loss.
    Arguments:
    model     -- A torch.nn.Module object
    dataloader -- A torch.utils.data.DataLoader
    criterion -- The loss function, e.g., nn.MSELoss()
    device    -- A torch.device
    Returns:
    The averaged validation loss computed over all the samples.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test(model, loader, criterion, device):
    """
    Test a model over the loader using the criterion as the metric.
    Can be used to compute loss and any other metrics like MAE or accuracy.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    criterion -- The loss function (e.g., nn.MSELoss() or nn.CrossEntropyLoss())
    device    -- A torch.device (either 'cuda' or 'cpu')
    Returns :
    avg_loss  -- Average loss over the entire test set
    mae       -- For regression, Mean Absolute Error (MAE)
    accuracy  -- For classification, accuracy
    """

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    mae = 0.0

    with torch.no_grad():  # Disable gradient calculation during testing
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # For classification, calculate accuracy
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                _, predicted = outputs.max(1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
            else:
                # For regression, calculate Mean Absolute Error (MAE)
                mae += torch.abs(outputs - targets).sum().item()

    avg_loss = running_loss / len(loader)

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        accuracy = correct / total
        logging.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
    else:
        mae = mae / total
        logging.info(f"Test Loss: {avg_loss:.4f}, MAE: {mae:.4f}")
        return avg_loss, mae


# Ensure that logging is set up
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")

if __name__ == "__main__":
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
