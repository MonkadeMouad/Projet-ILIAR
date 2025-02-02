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



def validate(model, dataloader, criterion, device):
    """
    Validate the model for one epoch.
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: (average validation loss, average MAE, average MSE)
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        progress_bar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Validation", leave=False)
        
        for batch_idx, (left, front, right, targets) in progress_bar:
            try:
                # Move inputs and targets to the device
                left, front, right = left.to(device), front.to(device), right.to(device)
                targets = targets.to(device)

                # Ensure the target has the correct shape
                if targets.dim() == 1:
                    targets = targets.view(-1, 1)  # Reshapes to [batch_size, 1] if needed

                # Pass separate inputs to the model
                outputs = model(left, front, right)
                
                # Compute the loss
                loss = criterion(outputs, targets)

                # Update loss
                running_loss += loss.item()

                # Compute MAE and MSE
                mae = torch.abs(outputs - targets).mean().item()
                mse = torch.mean((outputs - targets) ** 2).item()

                total_mae += mae * targets.size(0)  # Multiply by batch size for weighted average
                total_mse += mse * targets.size(0)
                total_samples += targets.size(0)

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), mae=mae, mse=mse)

            except Exception as e:
                logging.exception(f"Error in validation batch {batch_idx}: {e}")
                raise

    # Compute average metrics
    avg_loss = running_loss / len(dataloader)
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    # Log metrics
    logging.info(f"Validation Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")

    return avg_loss, avg_mae, avg_mse



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: (average training loss, average MAE, average MSE)
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_batches = len(dataloader)
    total_samples = 0
    total_mae = 0.0
    total_mse = 0.0

    start_time = time.time()

    progress_bar = tqdm(enumerate(dataloader, 1), total=total_batches, desc="Training", leave=False)

    for batch_idx, (left, front, right, targets) in progress_bar:
        try:
            left, front, right = left.to(device), front.to(device), right.to(device)
            targets = targets.to(device)

            # Ensure the target has the correct shape
            if targets.dim() == 1:
                targets = targets.view(-1, 1)  # Reshapes to [batch_size, 1] if needed

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(left, front, right)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update loss
            running_loss += loss.item()

            # Compute MAE and MSE
            mae = torch.abs(outputs - targets).mean().item()
            mse = torch.mean((outputs - targets) ** 2).item()

            total_mae += mae * targets.size(0)  # Multiply by batch size for weighted average
            total_mse += mse * targets.size(0)
            total_samples += targets.size(0)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), mae=mae, mse=mse)

        except Exception as e:
            logging.exception(f"Error in training batch {batch_idx}: {e}")
            raise

    avg_loss = running_loss / len(dataloader)
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    epoch_time = time.time() - start_time

    logging.info(f"Training Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")
    logging.info(f"Epoch completed in {epoch_time:.2f} seconds")

    return avg_loss, avg_mae, avg_mse
