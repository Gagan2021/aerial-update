import os
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, regularizers
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.data.experimental import AUTOTUNE

# -------------------------------
# Mixup Helper Function
# -------------------------------
def mixup_batch(x, y, alpha=0.2):
    """
    Perform mixup augmentation on a batch of data.
    Args:
        x (Tensor): Batch of images.
        y (Tensor): Batch of labels (integer labels).
        alpha (float): Mixup hyperparameter.
    Returns:
        mixed_x: Mixed images.
        y_a: Original labels.
        y_b: Shuffled labels.
        lam: Mixup lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    x_shuffled = tf.gather(x, indices)
    y_shuffled = tf.gather(y, indices)
    mixed_x = lam * x + (1 - lam) * x_shuffled
    return mixed_x, y, y_shuffled, lam

# -------------------------------
# Helper Function to Remove Alpha Channel
# -------------------------------
def remove_alpha_channel(image):
    """
    Remove the alpha channel if the image has more than 3 channels.
    Ensures the output tensor has a static shape with 3 channels.
    """
    channels = tf.shape(image)[-1]
    image = tf.cond(
        tf.greater(channels, 3),
        lambda: image[..., :3],
        lambda: image
    )
    # Force the static shape for the channel dimension to 3 if possible.
    shape = image.get_shape().as_list()
    if shape[-1] != 3:
        shape[-1] = 3
        image.set_shape(shape)
    return image

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def train_preprocess(image, label):
    # Convert image to float32
    image = tf.cast(image, tf.float32)
    # Remove alpha channel if present
    image = remove_alpha_channel(image)
    # Resize to 512x512
    image = tf.image.resize(image, [512, 512])
    # Random crop to 448x448x3
    image = tf.image.random_crop(image, size=[448, 448, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness, contrast, and saturation adjustments
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Preprocess input using ResNet50 preprocessing (scales pixel values appropriately)
    image = resnet50.preprocess_input(image)
    return image, label

def val_preprocess(image, label):
    # Convert image to float32
    image = tf.cast(image, tf.float32)
    # Remove alpha channel if present
    image = remove_alpha_channel(image)
    # Resize to 512x512 and then center-crop to 448x448
    image = tf.image.resize(image, [512, 512])
    image = tf.image.resize_with_crop_or_pad(image, 448, 448)
    image = resnet50.preprocess_input(image)
    return image, label

# -------------------------------
# Build Datasets
# -------------------------------
def build_datasets(data_dir, batch_size=16, validation_split=0.2, seed=123):
    # Load one image at a time (batch_size=None)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(512, 512),
        batch_size=None
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(512, 512),
        batch_size=None
    )
    # Map the preprocessing functions
    train_ds = train_ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(val_preprocess, num_parallel_calls=AUTOTUNE)
    # Re-batch the datasets
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# -------------------------------
# Build Model with Dropout and L2 Regularization
# -------------------------------
def build_model(num_classes):
    # Use ResNet50 pretrained on ImageNet (without the top classification layers)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(448, 448, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    # Add dropout to help with regularization
    x = layers.Dropout(0.5)(x)
    # Dense output layer with L2 regularization
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# -------------------------------
# Training Function with Mixup and Regularization
# -------------------------------
def train_model_tf(model, train_ds, val_ds, num_epochs=100, mixup_alpha=0.2):
    # Initialize loss function
    loss_fn = losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")
    
    # Determine steps per epoch and set up a cosine decay learning rate schedule
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    decay_steps = steps_per_epoch * 10  # For cosine decay over T_max epochs (e.g., 10 epochs)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    # Optionally, enable gradient clipping:
    # optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
    
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss.reset_state()
        train_accuracy.reset_state()
        # Training loop
        for images, labels in train_ds:
            # Apply mixup augmentation only during training
            if mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=mixup_alpha)
            else:
                labels_a = labels
                labels_b = labels
                lam = 1.0

            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_a = loss_fn(labels_a, logits)
                loss_b = loss_fn(labels_b, logits)
                loss = lam * loss_a + (1 - lam) * loss_b

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Compute predictions and update metrics
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            correct_a = tf.cast(tf.equal(preds, labels_a), tf.float32)
            correct_b = tf.cast(tf.equal(preds, labels_b), tf.float32)
            batch_accuracy = lam * correct_a + (1 - lam) * correct_b
            batch_accuracy = tf.reduce_mean(batch_accuracy)
            
            train_loss.update_state(loss)
            train_accuracy.update_state(batch_accuracy)

        print(f"Train Loss: {train_loss.result():.4f} Acc: {train_accuracy.result():.4f}")

        # Validation loop (without mixup)
        val_loss.reset_state()
        val_accuracy.reset_state()
        for images, labels in val_ds:
            logits = model(images, training=False)
            loss = loss_fn(labels, logits)
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
            val_loss.update_state(loss)
            val_accuracy.update_state(batch_accuracy)
        
        print(f"Val Loss: {val_loss.result():.4f} Acc: {val_accuracy.result():.4f}")
        
        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            best_weights = model.get_weights()
        print()

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    if best_weights is not None:
        model.set_weights(best_weights)
    return model

# -------------------------------
# Main Function
# -------------------------------
def main():
    data_dir = "../data"  # Adjust the path as needed
    batch_size = 16
    num_epochs = 100
    mixup_alpha = 0.2
    
    # Build the training and validation datasets
    train_ds, val_ds = build_datasets(data_dir, batch_size=batch_size)
    # Determine number of classes from the directory structure
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)
    print("Detected classes:", class_names)
    
    # Build the model
    model = build_model(num_classes)
    
    # Train the model with the updated hyperparameters and regularization
    model = train_model_tf(model, train_ds, val_ds, num_epochs=num_epochs, mixup_alpha=mixup_alpha)
    
    # Save the best model weights
    os.makedirs("../models", exist_ok=True)
    model_save_path = "../models/aerial_activity_detector_robust_hd_tf.h5"
    model.save_weights(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
