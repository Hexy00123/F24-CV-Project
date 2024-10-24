import os
import sys
import torch
import logging

sys.path.append(os.path.join(os.getcwd(), ".."))

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.augmentation import global_augment, multiple_local_augments


def train_dino(dino, data_loader, optimizer, device, num_epochs, tps=0.9, tpt=0.04, beta=0.9, m=0.9):
    """
    Train the DINO model.
    """
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(log_dir='runs/dino')
    steps = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch: {epoch + 1}/{num_epochs}")
    
        # Wrap the data loader with tqdm
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    
        for x, _ in progress_bar:
            x1, x2 = global_augment(x), multiple_local_augments(x)
            student_output1, student_output2 = dino.student(x1.to(device)), dino.student(x2.to(device))
    
            with torch.no_grad():
                teacher_output1, teacher_output2 = dino.teacher(x1.to(device)), dino.teacher(x2.to(device))
    
            # Compute distillation loss
            loss = (dino.distillation_loss(teacher_output1, student_output2, dino.center, tps, tpt) +
                    dino.distillation_loss(teacher_output2, student_output1, dino.center, tps, tpt)) / 2
    
            logger.info(f"Loss: {loss.item()}")
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Update teacher parameters and center
            dino.teacher_update(beta)
            with torch.no_grad():
                dino.center = m * dino.center + (1 - m) * torch.cat([teacher_output1, teacher_output2], dim=0).mean(dim=0)
    
            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), steps)
    
            # Update the progress bar description with the current loss
            progress_bar.set_postfix(loss=loss.item())
    
            steps += 1
    
    writer.close()
    logger.info("Training finished and TensorBoard logs saved.")