import os
import sys
import torch
import logging

sys.path.append(os.path.join(os.getcwd(), ".."))

from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from src.augmentation import global_augment, multiple_local_augments

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = os.getenv('PROJECT_ROOT')
if not project_root:
    raise ValueError("PROJECT_ROOT environment variable is not set.")

# Define the runs directory
runs_dir = os.path.join(project_root, 'training_logs/runs/dino')

# Ensure the runs directory exists
os.makedirs(runs_dir, exist_ok=True)

def train_dino(dino, data_loader, optimizer, device, num_epochs, tps, tpt, beta, m):
    """
    Train the DINO model.
    """
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(log_dir=runs_dir)
    steps = 0

    for epoch in range(num_epochs):
        # Wrap the data loader with tqdm
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # Initialize the total loss for the epoch
        total_loss = 0.0
        num_batches = 0
    
        for x, _ in progress_bar:
            x1, x2 = global_augment(x), multiple_local_augments(x)
            student_output1, student_output2 = dino.student(x1.to(device)), dino.student(x2.to(device))
    
            with torch.no_grad():
                teacher_output1, teacher_output2 = dino.teacher(x1.to(device)), dino.teacher(x2.to(device))
    
            # Compute distillation loss
            loss = (dino.distillation_loss(teacher_output1, student_output2, dino.center, tps, tpt) +
                    dino.distillation_loss(teacher_output2, student_output1, dino.center, tps, tpt)) / 2
    
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()
            num_batches += 1
    
            # Update teacher parameters and center
            dino.teacher_update(beta)
            with torch.no_grad():
                dino.center = m * dino.center + (1 - m) * torch.cat([teacher_output1, teacher_output2], dim=0).mean(dim=0)
    
            # Update the progress bar description with the current loss
            progress_bar.set_postfix(loss=loss.item())
    
            steps += 1
            
        # Compute the average loss for the epoch
        average_loss = total_loss / num_batches
    
        # Log the average loss to TensorBoard
        writer.add_scalar('Loss/train', average_loss, epoch)
    
        # Log the average loss to the logger
        logger.info(f"Average Loss for Epoch {epoch + 1}/{num_epochs}: {average_loss}")
    
    writer.close()
    logger.info("Training finished and TensorBoard logs saved.")