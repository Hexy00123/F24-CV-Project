import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

def global_augment(images):
    """
    Apply global augmentations to the input images.
    """
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_images = [TF.to_pil_image(img) for img in images]
    return torch.stack([global_transform(img) for img in pil_images])

def multiple_local_augments(images, num_crops=6):
    """
    Apply multiple local augmentations to the input images.
    """
    size = 96  # Size of the smaller crops for local augmentations
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_images = [TF.to_pil_image(img) for img in images]
    return torch.stack([local_transform(img) for img in pil_images])
