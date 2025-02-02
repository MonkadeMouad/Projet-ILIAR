# coding: utf-8

# Standard imports
import os
import sys
import pathlib
import time
import logging

# External imports
import yaml
import torch
from torchinfo import summary
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
# Local imports
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
        run_name = f"{raw_run_name}_{i}"
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i += 1

class ModelCheckpoint:
    """
    Model checkpointing utility to save the best model based on validation loss.
    """

    def __init__(self, model: torch.nn.Module, savepath: str, min_is_best: bool = True) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        self.min_is_best = min_is_best
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

def log_gpu_stats():
    """
    Log the current GPU memory usage.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = reserved_memory - allocated_memory
        logging.debug(
            f"GPU Memory - Total: {total_memory / (1024 ** 3):.2f} GB, "
            f"Reserved: {reserved_memory / (1024 ** 3):.2f} GB, "
            f"Allocated: {allocated_memory / (1024 ** 3):.2f} GB, "
            f"Free: {free_memory / (1024 ** 3):.2f} GB"
        )

def train(config):
    """
    Train the machine learning model based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing all necessary parameters.
    """
    try:
        # Device setup
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using device: {device}")

        if use_cuda:
            torch.backends.cudnn.benchmark = True
            logging.debug("Enabled cuDNN benchmarking for optimized performance")

        # Initialize dataset
        logging.info("= Initializing the dataset")
        data_config = config["data"]
        datapath = os.path.expanduser(data_config["trainpath"])
        valid_ratio = data_config.get("valid_ratio", 0.2)
        initialize_global_dataset(datapath, valid_ratio)
        logging.debug(f"Datapath: {datapath}, Validation Ratio: {valid_ratio}")

        # Build dataloaders
        logging.info("= Building dataloaders")
        batch_size = data_config["batch_size"]
        num_workers = data_config.get("num_workers", 4)
        train_loader, valid_loader = get_dataloaders(batch_size, num_workers, use_cuda)
        logging.info(f"Train Loader: {len(train_loader)} batches")
        logging.info(f"Validation Loader: {len(valid_loader)} batches")

        # Build the model
        logging.info("= Building the model")
        model_config = config["model"]
        input_size = (batch_size, 3, 224, 224)  # Adjusted for the input
        num_classes = model_config["num_classes"]
        model = build_model(model_config, input_size=input_size, num_classes=num_classes)
        model.to(device)
        logging.info("Model successfully moved to device")

        # Save model summary
        model_summary = summary(model, input_size=input_size, device=device, verbose=0)
        with open(pathlib.Path(logdir) / "summary.txt", "w") as f:
            f.write(str(model_summary))
        logging.debug("Model summary saved.")

        # Configure loss function
        logging.info("= Configuring the loss function")
        criterion = torch.nn.MSELoss()  # For regression, MSELoss

        # Configure optimizer
        logging.info("= Configuring the optimizer")
        optim_config = config["optim"]
        optimizer = get_optimizer(optim_config, model.parameters())
        logging.debug(f"Optimizer Configuration: {optim_config}")

        # Configure scheduler if provided
        scheduler = None
        if "scheduler" in config:
            scheduler_config = config["scheduler"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
            logging.info(f"Scheduler configured: {scheduler_config}")

        # Set up logging and directories
        logging_config = config["logging"]
        logdir = generate_unique_logpath(logging_config["logdir"], model_config["class"])
        os.makedirs(logdir, exist_ok=True)
        logging.info(f"Logs will be saved in {logdir}")

        # Save the config to the log directory
        with open(pathlib.Path(logdir) / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logging.debug("Configuration file saved.")

        # Initialize checkpoint for saving the best model
        checkpoint = ModelCheckpoint(model, savepath=str(pathlib.Path(logdir) / "best_model.pth"))
        logging.debug(f"Checkpoint Path: {checkpoint.savepath}")

        # Training loop
        n_epochs = config["training"]["epochs"]
        logging.info(f"Starting training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            logging.info(f"Epoch {epoch}/{n_epochs}")
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            valid_loss = validate(model, valid_loader, criterion, device)

            # Save the best model
            is_best = checkpoint.update(valid_loss)
            logging.info(f"Validation Loss: {valid_loss:.4f} {'[BEST]' if is_best else ''}")

            # Step the scheduler if it exists
            if scheduler:
                scheduler.step()
                logging.debug(f"Scheduler stepped. Current LR: {optimizer.param_groups[0]['lr']}")

            # Log GPU stats
            log_gpu_stats()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            logging.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

        logging.info("Training complete.")

    except Exception as e:
        logging.exception(f"An error occurred during training: {e}")

def test(config):
    """
    Test the machine learning model based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing all necessary parameters.
    """
    try:
        # Device setup
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using device: {device}")

        if use_cuda:
            torch.backends.cudnn.benchmark = True
            logging.debug("Enabled cuDNN benchmarking for optimized performance")

        # Initialize dataset for testing
        logging.info("= Initializing the dataset for testing")
        data_config = config["data"]
        datapath = os.path.expanduser(data_config["testpath"])  # Ensure 'testpath' is defined in config.yaml
        initialize_global_dataset(datapath)
        logging.debug(f"Datapath: {datapath}")

        # Build test dataloader
        logging.info("= Building test dataloaders")
        batch_size = data_config["batch_size"]
        num_workers = data_config.get("num_workers", 4)
        test_loader = get_dataloaders(batch_size, num_workers, use_cuda, mode="test")  # Ensure your get_dataloaders supports 'mode'
        logging.info(f"Test Loader: {len(test_loader)} batches")

        # Build the model
        logging.info("= Building the model for testing")
        model_config = config["model"]
        input_size = (batch_size, 3, 224, 224)  # Adjust based on your input size
        num_classes = model_config["num_classes"]
        model = build_model(model_config, input_size=input_size, num_classes=num_classes)
        model.to(device)
        logging.info("Model successfully moved to device")

        # Load the saved model checkpoint
        checkpoint_path = pathlib.Path(config["logging"]["logdir"]) / "best_model.pth"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"Loaded model from {checkpoint_path}")
        else:
            logging.error("No model checkpoint found!")
            return

        # Set model to evaluation mode
        model.eval()

        # Configure loss function
        criterion = torch.nn.MSELoss()  # Using MSE for regression

        # Evaluation
        logging.info("= Starting evaluation on the test dataset")
        avg_loss, mae = test_model(model, test_loader, criterion, device)

        logging.info(f"Test Loss (MSE): {avg_loss:.4f}")
        logging.info(f"Test MAE (Proportion < 0.5): {mae:.4f}")

        logging.info("Testing complete.")

    except Exception as e:
        logging.exception(f"An error occurred during testing: {e}")

def test_model(model, loader, criterion, device):
    """
    Test the model over the loader using the criterion as the metric.
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: (average_loss, mae)
    """
    running_loss = 0.0
    correct = 0
    total = 0
    mae_total = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(loader, 1), total=len(loader), desc="Testing", leave=False)
        for batch_idx, (inputs, targets) in progress_bar:
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update loss
                running_loss += loss.item()

                # For regression, calculate MAE
                mae = torch.abs(outputs - targets)
                mae_total += mae.sum().item()

                # Example: Counting how many predictions are within 0.5 of the target
                correct += (mae < 0.5).sum().item()
                total += targets.size(0)

                # Log batch metrics every 10 batches
                if batch_idx % 10 == 0 or batch_idx == len(loader):
                    logging.debug(f"Batch {batch_idx}/{len(loader)} - Test Loss: {loss.item():.4f}")

            except Exception as e:
                logging.exception(f"Error in test batch {batch_idx}: {e}")
                raise

    avg_loss = running_loss / len(loader)
    mae = correct / total  # Example metric

    return avg_loss, mae

import torch
import time
import logging
import numpy as np
from tqdm import tqdm
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total_batches = len(dataloader)
    total_samples = 0
    correct = 0

    progress_bar = tqdm(enumerate(dataloader, 1), total=total_batches, desc="Training", leave=False)

    for batch_idx, (left, front, right, steering_angles) in progress_bar:
        try:
            left, front, right = left.to(device), front.to(device), right.to(device)
            steering_angles = steering_angles.to(device)  # Shape: (batch_size, 3) for each steering angle

            optimizer.zero_grad()

            # Forward pass for each image separately
            left_output = model(left)      # Predict for left camera
            front_output = model(front)    # Predict for front camera
            right_output = model(right)    # Predict for right camera

            # Compute loss for each view
            loss_left = criterion(left_output, steering_angles[:, [0]])  # Left steering
            loss_front = criterion(front_output, steering_angles[:, [1]])  # Front steering
            loss_right = criterion(right_output, steering_angles[:, [2]])  # Right steering

            # Average loss from all three views
            loss = (loss_left + loss_front + loss_right) / 3

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            total_samples += steering_angles.size(0)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        except Exception as e:
            logging.exception(f"Error in training batch {batch_idx}: {e}")
            raise

    avg_loss = running_loss / len(dataloader)
    logging.info(f"Training Loss: {avg_loss:.4f}")

    return avg_loss

def validate(model, dataloader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    total_samples = 0
    mae_total = 0.0
    rmse_total = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Validation", leave=False)

        for batch_idx, (left, front, right, steering_angles) in progress_bar:
            try:
                left, front, right = left.to(device), front.to(device), right.to(device)
                steering_angles = steering_angles.to(device)  # Shape: (batch_size, 3)

                # Forward pass separately for each image
                left_output = model(left)
                front_output = model(front)
                right_output = model(right)

                # Compute loss
                loss_left = criterion(left_output, steering_angles[:, [0]])
                loss_front = criterion(front_output, steering_angles[:, [1]])
                loss_right = criterion(right_output, steering_angles[:, [2]])

                loss = (loss_left + loss_front + loss_right) / 3  # Average loss

                running_loss += loss.item()
                total_samples += steering_angles.size(0)

                # Compute evaluation metrics using front camera
                mae_total += torch.abs(front_output - steering_angles[:, [1]]).sum().item()
                rmse_total += torch.sum((front_output - steering_angles[:, [1]]) ** 2).item()

                progress_bar.set_postfix(loss=loss.item())

            except Exception as e:
                logging.exception(f"Error in validation batch {batch_idx}: {e}")
                raise

    avg_loss = running_loss / len(dataloader)
    avg_mae = mae_total / total_samples if total_samples > 0 else 0.0
    avg_rmse = np.sqrt(rmse_total / total_samples) if total_samples > 0 else 0.0

    logging.info(f"Validation Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")
    return avg_loss
