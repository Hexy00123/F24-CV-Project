import os
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), ".."))

from src.read_config import read_config
from src.dino_model import DINO
from src.train import train_dino
from src.logs import setup_logging
from src.data.data_utils import get_dataloader, load_imagenet_dataset, get_dataloader_local, download_images_locally

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging()

    # Load configuration for training
    config = read_config(config_path='../configs', config_name='architecture_dino.yaml')
    logger.info(f"Loaded configuration: {config}")

    # Set device for training
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # directory for the data
    data_dir = 'data/images'
    
    # These or some of these should probably be parsed from command line
    batch_size = 4 # Batch size in dataloader
    num_workers = 2
    use_local = True # If to use local images or images saved locally
    max_images = 25 # If use_local is true, max number of images to download (250000)

    # Load images either locally or via streaming
    if use_local:
        logger.info("Using local image data.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Downlaod images locally
        download_images_locally(train_dataset, data_dir, max_num_images=max_images)
        # Load images into dataloader
        train_loader = get_dataloader_local(data_dir, batch_size, num_workers)
    else:
        logger.info("Using streamed ImageNet dataset.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Get dataloader
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Initialize DINO model and optimizer
    dino = DINO(config.inputs.img_size, config.inputs.in_channels, config['params'], device).to(device)
    optimizer = torch.optim.Adam(dino.parameters(), lr=1e-4)


    # Train the model
    train_dino(dino,
               data_loader=train_loader,
               optimizer=optimizer,
               device=device,
               num_epochs=6,
               tps=0.9,
               tpt= 0.04,
               beta= 0.9,
               m= 0.9)