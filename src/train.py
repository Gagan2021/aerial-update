# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from utils import get_data_loaders, set_seed

def train_model(data_dir, model_save_path, epochs=100, batch_size=32, learning_rate=0.0001, img_size=224, num_workers=8):
    # Set seed for reproducibility
    set_seed(42)
    
    # Determine the device to run on (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and get class names
    train_loader, val_loader, class_names = get_data_loaders(data_dir, img_size, batch_size, valid_split=0.2, num_workers=num_workers)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Load pretrained ResNet18 model and replace the final layer
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total * 100
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Adjust these paths if needed. Assuming train.py is in src/ and data is in ../data and models in ../models
    data_dir = os.path.join("..", "data")
    model_save_path = os.path.join("..", "models", "aerial_activity_detector.pth")
    train_model(data_dir, model_save_path)
