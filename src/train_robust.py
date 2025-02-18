# train_robust_hd.py
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# Option A: Disable the limit entirely
Image.MAX_IMAGE_PIXELS = None

# Option B: Increase the limit (example: 300 million pixels)
# Image.MAX_IMAGE_PIXELS = 300000000


# -------------------------------
# Mixup Helper Function
# -------------------------------
def mixup_data(x, y, alpha=0.2):
    """
    Perform mixup augmentation on a batch of data.
    
    Args:
        x (Tensor): Input batch.
        y (Tensor): Labels for the batch.
        alpha (float): Mixup hyperparameter.
    
    Returns:
        mixed_x, y_a, y_b, lam: Mixed inputs, original labels, shuffled labels, and mixup lambda.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# -------------------------------
# Training Function with Mixup and Label Smoothing
# -------------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=100, mixup_alpha=0.2):
    """
    Train the model using both training and validation phases.
    
    Uses mixup augmentation (only in training) and label smoothing via the loss function.
    
    Args:
        model: The neural network model.
        dataloaders: A dict with 'train' and 'val' DataLoaders.
        criterion: Loss function (with label smoothing).
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        device: Device (CPU/GPU) to run on.
        num_epochs (int): Number of epochs.
        mixup_alpha (float): Mixup parameter.
    
    Returns:
        The best model (based on validation loss).
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*30)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                # Use mixup only during training
                if phase == 'train' and mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    _, preds = torch.max(outputs, 1)
                    # For mixup, count prediction as partially correct if it matches either label
                    correct = (preds == targets_a).float() * lam + (preds == targets_b).float() * (1 - lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    correct = (preds == labels).float()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(correct)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Track best model via validation loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                scheduler.step()

        print()

    print(f"Best Validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# -------------------------------
# Main Function to Prepare Data, Model, and Start Training
# -------------------------------
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentations for HD images:
    # We'll resize to 512, then RandomCrop to 448. 
    # For inference/validation, we'll just center-crop to 448 after resizing to 512.
    # Adjust these sizes based on your GPU memory.
    train_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),  # Requires torchvision>=0.10
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # The dataset directory is assumed to have:
    # data/
    #    class1/
    #        image1.jpg, ...
    #    class2/
    #        image1.jpg, ...
    data_dir = "../output_data"  # Adjust path if needed

    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)  # 20% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Use the validation transforms for the validation subset
    val_dataset.dataset.transform = val_transforms

    # Create dataloaders
    # Decrease batch_size if you run out of GPU memory with HD images
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    }

    # Print the detected classes
    class_names = full_dataset.classes
    print("Detected classes:", class_names)

    # Initialize a bigger model if you have enough GPU resources:
    # model = models.resnet18(pretrained=True)
    model = models.resnet50(pretrained=True)
    
    # Optionally freeze earlier layers
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final layer for classification
    if hasattr(model, 'fc'):  # For ResNet
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(class_names))
    else:
        raise ValueError("This code expects a ResNet-like model with a 'fc' layer.")

    model = model.to(device)

    # Define the loss function (label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Decreased LR for high-res training 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Train the model
    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=100,
        mixup_alpha=0.2  # or 0.0 if you want to disable Mixup
    )

    # Save the best model weights
    os.makedirs("../models", exist_ok=True)
    model_save_path = "../models/aerial_activity_detector_robust_hd.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
