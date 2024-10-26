import os
import sys
import torch
import warnings
from dotenv import load_dotenv

sys.path.append(os.path.join(os.getcwd(), ".."))

from src.read_config import read_config
from src.dino_model import DINO
from src.train import train_dino
from src.logs import setup_logging
from src.data_utils import get_dataloader, load_imagenet_dataset, get_dataloader_local, download_images_locally

load_dotenv()

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging()

    # Load configuration for training
    model_config = read_config(config_path='../configs', config_name='architecture_dino.yaml')
    train_config = read_config(config_path='../configs', config_name='dino_train_config.yaml')
    
    logger.info(f"Loaded configuration: {model_config}")

    # Set device for training
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # directory for the data
    data_dir = os.environ['PROJECT_ROOT'] + '/data/images'
    
    num_workers = 2

    # Load images either locally or via streaming
    if train_config.data.use_local:
        logger.info("Using local image data.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Downlaod images locally
        download_images_locally(train_dataset, data_dir, num_images=train_config.data.num_images)
        # Load images into dataloader
        train_loader = get_dataloader_local(data_dir, train_config.data.batch_size, num_workers)
    else:
        logger.info("Using streamed ImageNet dataset.")
        # Load ImageNet dataset
        train_dataset = load_imagenet_dataset(split='train', streaming=True)
        # Get dataloader
        train_loader = get_dataloader(train_dataset, batch_size=train_config.data.batch_size, num_workers=num_workers)
    
    # Initialize DINO model and optimizer
    dino = DINO(model_config.inputs.img_size, model_config.inputs.in_channels, model_config['params'], device).to(device)
    optimizer = torch.optim.AdamW(dino.parameters(), lr=train_config.train_params.lr)


    # Train the model
    train_dino(dino,
               data_loader=train_loader,
               optimizer=optimizer,
               device=device,
               num_epochs=train_config.train_params.num_epochs,
               tps=train_config.train_params.tps,
               tpt=train_config.train_params.tpt,
               beta=train_config.train_params.beta,
               m=train_config.train_params.m,
               max_checkpoints=train_config.train_params.max_checkpoints)