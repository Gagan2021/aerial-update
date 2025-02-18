# predict.py
import os
import torch
from torchvision import models, transforms
from PIL import Image
import argparse

def load_model(model_path, num_classes, device):
    """
    Load the model architecture and state dictionary.
    
    Args:
        model_path (str): Path to the saved model weights.
        num_classes (int): Number of classes.
        device (torch.device): Device to load the model onto.
        
    Returns:
        model: The loaded PyTorch model.
    """
    # Create the model architecture
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, class_names, img_size=224, device="cpu"):
    """
    Predict the class of an image.
    
    Args:
        image_path (str): Path to the image file.
        model: The PyTorch model.
        class_names (list): List of class names.
        img_size (int): Image size for resizing.
        device (torch.device): Device to run prediction on.
        
    Returns:
        str: Predicted class name.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict aerial activity from an image")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--model", type=str, default=os.path.join("..", "models", "aerial_activity_detector.pth"),
                        help="Path to the saved model")
    parser.add_argument("--data", type=str, default=os.path.join("..", "data"),
                        help="Path to the dataset directory (used to load class names)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for prediction")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # To obtain class names, load the dataset using ImageFolder (without augmentation)
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(args.data, transform=transforms.ToTensor())
    class_names = dataset.classes
    num_classes = len(class_names)
    
    model = load_model(args.model, num_classes, device)
    
    predicted_class = predict_image(args.image, model, class_names, img_size=args.img_size, device=device)
    print(f"Predicted class: {predicted_class}")
