import os
import torch
import warnings

from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.read_config import read_config

warnings.filterwarnings('ignore')

# Load configuration to get image size
model_config = read_config(config_path='../configs', config_name='architecture_dino.yaml')

# Load environment variables from .env file
load_dotenv()

# Accessing an environment variable
token = os.getenv('HF_TOKEN')

# Transformation pipeline for preprocessing images.
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((model_config.inputs.img_size, model_config.inputs.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    """Custom collate function to handle batch processing."""
    images, labels = [], []
    for item in batch:
        image = transform(item['image'])
        images.append(image)
        labels.append(item['label'])
    return torch.stack(images), torch.tensor(labels)


def get_dataloader(dataset, batch_size: int, num_workers: int):
    """Creates a DataLoader from a dataset."""
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)


def load_imagenet_dataset(split: str = 'train', streaming: bool = True):
    """Loads ImageNet dataset with specified split and streaming option."""
    return load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True, token=token)


def get_dataloader_local(data_dir: str, batch_size: int, num_workers: int):
    """Loads a DataLoader from locally stored images."""
    # Create the dataset with the defined transformations
    dataset = ImageFolder(root=data_dir, transform=transform)

    # return dataloader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def download_images_locally(dataset, data_dir: str, num_images: int):
    """Downloads a subset of images from the dataset to a local directory."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check how many images are already present
    existing_count = sum([len(files) for _, _, files in os.walk(data_dir)])
    remaining_images = num_images - existing_count

    if remaining_images <= 0:
        print(f"{num_images} images already downloaded in {data_dir}. Skipping download.")
        return

    print(f"{existing_count} images already present. Downloading {remaining_images} more images...")

    downloaded = existing_count
    for item in tqdm(dataset, total=remaining_images):
        image = item['image']
        label = item['label']
        label_dir = os.path.join(data_dir, str(label))

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        image_path = os.path.join(label_dir, f"{downloaded}.jpeg")
        if not os.path.exists(image_path):
            image.save(image_path)
            downloaded += 1

        if downloaded >= num_images:
            break

    print(f"Downloaded {downloaded - existing_count} images to {data_dir}. Total images: {downloaded}")
    