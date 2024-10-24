import os
import torch

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


# Transformation pipeline for preprocessing images.
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),
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
    return load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True, token='***REMOVED***')


def get_dataloader_local(data_dir: str, batch_size: int, num_workers: int):
    """Loads a DataLoader from locally stored images."""
    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def download_images_locally(dataset, data_dir: str, max_num_images: int):
    """Downloads a subset of images from the dataset to a local directory."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check how many images are already present
    existing_count = sum([len(files) for _, _, files in os.walk(data_dir)])
    remaining_images = max_num_images - existing_count

    if remaining_images <= 0:
        print(f"{max_num_images} images already downloaded in {data_dir}. Skipping download.")
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

        if downloaded >= max_num_images:
            break

    print(f"Downloaded {downloaded - existing_count} images to {data_dir}. Total images: {downloaded}")