import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths
data_dir = "data"
crack_dir = os.path.join(data_dir, "crack")
no_crack_dir = os.path.join(data_dir, "no_crack")

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image: Read, convert to grayscale, resize, and blur
    """
    # 1. Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Resize to target size
    img_resized = cv2.resize(img, target_size)
    
    # 3. Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    return img_blurred

def visualize_preprocessing(original_path, processed_img, title):
    """
    Visualize original vs processed image
    """
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title(f"Original\n{title}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(original, cmap='gray')
    plt.title("Original (Grayscale)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(processed_img, cmap='gray')
    plt.title("Processed\n(Resized + Blurred)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test preprocessing on a few samples
print("=== TESTING PREPROCESSING ===")

# Test on one crack image
crack_sample = os.path.join(crack_dir, os.listdir(crack_dir)[0])
processed_crack = preprocess_image(crack_sample)
print(f"Crack image - Original: 227x227x3, Processed: {processed_crack.shape}")

# Test on one no_crack image  
no_crack_sample = os.path.join(no_crack_dir, os.listdir(no_crack_dir)[0])
processed_no_crack = preprocess_image(no_crack_sample)
print(f"No crack image - Original: 227x227x3, Processed: {processed_no_crack.shape}")

# Visualize the preprocessing steps
print("\n=== VISUALIZING PREPROCESSING ===")
visualize_preprocessing(crack_sample, processed_crack, "Crack Sample")
visualize_preprocessing(no_crack_sample, processed_no_crack, "No Crack Sample")