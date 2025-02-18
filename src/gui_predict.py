# gui_predict.py
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# GradCAM Implementation
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with the model and the target layer.
        Args:
            model: The trained neural network.
            target_layer: The convolutional layer from which we capture activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook: capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # Backward hook: capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fwd, handle_bwd])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a GradCAM heatmap for the given input tensor.
        Args:
            input_tensor: Preprocessed input image tensor with batch dimension.
            class_idx: (Optional) The target class index. If None, uses the predicted class.
        Returns:
            cam: A numpy array containing the normalized heatmap.
            output: The raw model output.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # Forward pass

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward()  # Backward pass

        # Retrieve gradients and activations from the target layer
        gradients = self.gradients[0]      # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]

        # Global average pooling on gradients: compute channel-wise weights
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: [C]
        # Compute weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), output

# -------------------------------
# Helper Function to Load Checkpoint Safely
# -------------------------------
def load_filtered_state_dict(model, checkpoint_path, device):
    """
    Load a checkpoint while skipping keys that do not match in shape.
    This helps avoid errors due to mismatched state_dicts.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping key '{k}': checkpoint shape {v.shape} != model shape {model_dict[k].shape}")
        else:
            print(f"Unexpected key '{k}' not found in the model.")
    # Update the model's state dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

# -------------------------------
# Main Functionality with GUI
# -------------------------------
def main():
    # Device configuration: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define class names (order must match training)
    class_names = [
        "parasailing",
        "sky diving"
    ]
    num_classes = len(class_names)

    # -------------------------------
    # Load the Trained Model
    # -------------------------------
    # Create a ResNet18 model instance and update the final layer
    # (Do not load pre-trained weights; we load our custom checkpoint below.)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model_path = "../models/aerial_activity_detector.pth"  # Adjust the path if necessary
    
    # Load checkpoint using the filtered loader to skip mismatches
    load_filtered_state_dict(model, model_path, device)
    model = model.to(device)
    model.eval()

    # -------------------------------
    # Select the Target Layer for GradCAM
    # -------------------------------
    # For ResNet18, choose one of the later convolutional layers
    target_layer = model.layer4[1].conv2  # Adjust as needed
    gradcam = GradCAM(model, target_layer)

    # -------------------------------
    # Open File Dialog to Choose an Image
    # -------------------------------
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    # -------------------------------
    # Preprocess the Image
    # -------------------------------
    orig_image = Image.open(file_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(orig_image).unsqueeze(0).to(device)

    # -------------------------------
    # Generate GradCAM Heatmap and Prediction
    # -------------------------------
    cam, output = gradcam.generate(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred_idx = torch.max(probs, dim=1)
    predicted_class = class_names[pred_idx.item()]
    confidence_value = confidence.item() * 100

    # -------------------------------
    # Prepare the Heatmap and Find a Bounding Box
    # -------------------------------
    # Resize the heatmap to match the original image size
    cam_resized = cv2.resize(cam, (orig_image.width, orig_image.height))
    heatmap = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert the original PIL image to an OpenCV BGR image
    orig_np = np.array(orig_image)
    orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    # Blend the heatmap with the original image to create an overlay
    overlay = cv2.addWeighted(orig_np, 0.5, heatmap_color, 0.5, 0)

    # Threshold the heatmap to create a binary mask for salient regions
    ret, mask = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)
    mask = np.uint8(mask)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Choose the largest contour as the primary region
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Draw the bounding box around the detected region
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Annotate the image with predicted class and confidence
        text = f"{predicted_class}: {confidence_value:.2f}%"
        cv2.putText(overlay, text, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # If no salient region is found, annotate on the image top-left
        text = f"{predicted_class}: {confidence_value:.2f}%"
        cv2.putText(overlay, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # -------------------------------
    # Display the Result in a 500x500 Window
    # -------------------------------
    cv2.namedWindow("Aerial Activity Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aerial Activity Prediction", 500, 500)
    cv2.imshow("Aerial Activity Prediction", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clean up GradCAM hooks
    gradcam.remove_hooks()

if __name__ == "__main__":
    main()
