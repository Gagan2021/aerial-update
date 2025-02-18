# utils.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import random
import numpy as np

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data_loaders(data_dir, img_size=224, batch_size=32, valid_split=0.2, num_workers=4):
    """
    Load the dataset from the given directory using ImageFolder and create training and validation loaders.
    
    Args:
        data_dir (str): Path to the dataset directory.
        img_size (int): Image size to resize to.
        batch_size (int): Batch size.
        valid_split (float): Fraction of the data to reserve for validation.
        num_workers (int): Number of worker threads for data loading.
        
    Returns:
        train_loader, val_loader, class_names: Data loaders and list of class names.
    """
    # Transform for training (with data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # Flip image horizontally with 50% probability
        transforms.RandomRotation(20),        # Rotate image randomly within ±20 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate image by up to 10%
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for validation (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load the full dataset using training transforms initially
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    class_names = full_dataset.classes
    dataset_size = len(full_dataset)
    
    # Calculate sizes for training and validation
    val_size = int(valid_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update the validation dataset to use the validation transforms (without augmentation)
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, class_names
