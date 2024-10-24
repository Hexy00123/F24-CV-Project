import os
import sys
import torch
import warnings

sys.path.append(os.path.join(os.getcwd(), ".."))

from src.read_config import read_config
from src.dino_model import DINO
from src.train import train_dino
from src.logs import setup_logging
from src.data.data_utils import get_dataloader, load_imagenet_dataset, get_dataloader_local, download_images_locally

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging()

    # Load configuration for training
    config = read_config(config_path='../configs', config_name='architecture_dino.yaml')
    logger.info(f"Loaded configuration: {config}")

    dino_train_config = read_config(config_path='../configs', config_name='dino_train_config.yaml')

    # Set device for training
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # directory for the data
    data_dir = 'data/images'
    
    num_workers = 2

    # Load images either locally or via streaming
    if dino_train_config.data.use_local:
        logger.info("Using local image data.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Downlaod images locally
        download_images_locally(train_dataset, data_dir, max_num_images=dino_train_config.data.max_images)
        # Load images into dataloader
        train_loader = get_dataloader_local(data_dir, dino_train_config.data.batch_size, num_workers)
    else:
        logger.info("Using streamed ImageNet dataset.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Get dataloader
        train_loader = get_dataloader(train_dataset, batch_size=dino_train_config.data.batch_size, num_workers=num_workers)
    
    # Initialize DINO model and optimizer
    dino = DINO(config.inputs.img_size, config.inputs.in_channels, config['params'], device).to(device)
    optimizer = torch.optim.AdamW(dino.parameters(), lr=dino_train_config.train_params.lr)


    # Train the model
    train_dino(dino,
               data_loader=train_loader,
               optimizer=optimizer,
               device=device,
               num_epochs=dino_train_config.train_params.num_epochs,
               tps=dino_train_config.train_params.tps,
               tpt=dino_train_config.train_params.tpt,
               beta=dino_train_config.train_params.beta,
               m=dino_train_config.train_params.m)