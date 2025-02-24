import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# GradCAM Implementation (Keras)
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer_name):
        """
        Initializes GradCAM with a Keras model and the name of the target layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        # Create a sub-model that outputs both the target layer's activations and the final predictions.
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(target_layer_name).output, model.output]
        )

    def compute_heatmap(self, image, class_idx=None):
        """
        Computes the GradCAM heatmap for the given image.
        Args:
            image: Preprocessed input image with batch dimension.
            class_idx: (Optional) target class index; if None, uses the predicted class.
        Returns:
            heatmap: The GradCAM heatmap.
            predictions: The model predictions.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]
        # Compute gradients of the loss with respect to the convolutional outputs.
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        # Weight the channels by corresponding pooled gradients.
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        # Apply ReLU and normalize the heatmap.
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), predictions

# -------------------------------
# Build Model Architecture
# -------------------------------
def build_model(num_classes):
    """
    Builds the model architecture. This must match your training architecture.
    Here we use a ResNet50 base with global average pooling, dropout, and a Dense output layer.
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(448, 448, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model

# -------------------------------
# Load Model and Weights (cached)
# -------------------------------
@st.cache_resource
def load_model():
    class_names = ["helicopter", "paragliding", "parasailing", "sky diving", "flying_bird", "flying_rocket", "zipline"]
    num_classes = len(class_names)
    model = build_model(num_classes)
    # Load weights saved with model.save_weights(...)
    model_path = "../models/aerial_activity_detector_robust_hd_tf.weights.h5"  # Adjust as needed
    model.load_weights(model_path)
    model.trainable = False
    return model, class_names

model, class_names = load_model()

# Create GradCAM object (adjust target layer name if needed)
target_layer_name = "conv5_block3_out"  # Example for ResNet50; adjust if different.
gradcam = GradCAM(model, target_layer_name)

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    """
    Resizes and normalizes the input image.
    """
    image = image.resize((448, 448))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Aerial Activity Detector with GradCAM")
st.write("Upload an image to see the prediction and GradCAM visualization.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")
    else:
        input_tensor = preprocess_image(image)
        heatmap, predictions = gradcam.compute_heatmap(input_tensor)
        confidence = np.max(predictions)
        pred_idx = np.argmax(predictions)
        predicted_class = class_names[pred_idx]
        confidence_value = confidence * 100

        # Resize heatmap to original image dimensions.
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert original image to OpenCV BGR format.
        orig_np = np.array(image)
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        # Blend heatmap with original image.
        overlay = cv2.addWeighted(orig_np, 0.5, heatmap_color, 0.5, 0)
        
        # --- BOXING FEATURE (Draw bounding box around activity) ---
        ret, mask = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optionally, annotate near the bounding box.
            cv2.putText(overlay, f"{predicted_class}: {confidence_value:.2f}%", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, f"{predicted_class}: {confidence_value:.2f}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # ---------------------------------------------------------

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        st.image(overlay_rgb, caption="Prediction with GradCAM", use_column_width=True)
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence_value:.2f}%")
