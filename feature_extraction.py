import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Define paths (ADD THIS SECTION)
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
    img_blurred =cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    return img_blurred

def extract_features(image_path):
    """
    Extract multiple features from a preprocessed image
    """
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    features = {}
    
    # 1. HOG Features (Primary feature)
    hog_features, hog_image = hog(
        processed_img, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True,
        block_norm='L2-Hys'
    )
    features['hog'] = hog_features
    features['hog_image'] = hog_image
    
    # 2. Canny Edge Features
    edges = cv2.Canny(processed_img, threshold1=50, threshold2=150)
    features['edge_pixels'] = np.sum(edges) / 255  # Count edge pixels
    
    # 3. Threshold Features
    _, thresh = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV)
    features['crack_pixels'] = np.sum(thresh) / 255  # Count potential crack pixels
    features['crack_percentage'] = features['crack_pixels'] / (128 * 128)
    
    return features, processed_img, edges, thresh

def visualize_features(original_path, processed_img, edges, thresh, hog_image, features_dict, title):
    """
    Visualize all the extracted features
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0,0].imshow(original_rgb)
    axes[0,0].set_title(f"Original\n{title}")
    axes[0,0].axis('off')
    
    # Processed image
    axes[0,1].imshow(processed_img, cmap='gray')
    axes[0,1].set_title("Processed\n(Grayscale + Blurred)")
    axes[0,1].axis('off')
    
    # Canny Edges
    axes[0,2].imshow(edges, cmap='gray')
    axes[0,2].set_title(f"Canny Edges\nPixels: {features_dict['edge_pixels']:.0f}")
    axes[0,2].axis('off')
    
    # Threshold
    axes[1,0].imshow(thresh, cmap='gray')
    axes[1,0].set_title(f"Threshold\nPixels: {features_dict['crack_pixels']:.0f}")
    axes[1,0].axis('off')
    
    # HOG Image
    axes[1,1].imshow(hog_image, cmap='gray')
    axes[1,1].set_title("HOG Features")
    axes[1,1].axis('off')
    
    # Feature summary
    axes[1,2].text(0.1, 0.8, f"Feature Summary:\n\n"
                   f"Edge Pixels: {features_dict['edge_pixels']:.0f}\n"
                   f"Crack Pixels: {features_dict['crack_pixels']:.0f}\n"
                   f"Crack %: {features_dict['crack_percentage']:.3f}\n"
                   f"HOG Features: {len(features_dict['hog'])}", 
                   fontsize=12)
    axes[1,2].set_title("Feature Values")
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test feature extraction on samples
print("=== TESTING FEATURE EXTRACTION ===")

# Test on one crack image
crack_sample = os.path.join(crack_dir, os.listdir(crack_dir)[0]) #os.listdir(crack_dir) - Lists all files in the "crack" directory
features_crack, proc_crack, edges_crack, thresh_crack = extract_features(crack_sample) #calls extract feature and function returns 4 value
print(f"Crack Image Features:")
print(f"  - HOG features: {len(features_crack['hog'])} dimensions")
print(f"  - Edge pixels: {features_crack['edge_pixels']:.0f}")
print(f"  - Crack pixels: {features_crack['crack_pixels']:.0f}")
print(f"  - Crack percentage: {features_crack['crack_percentage']:.3f}")

# Test on one no_crack image
no_crack_sample = os.path.join(no_crack_dir, os.listdir(no_crack_dir)[0])
features_no_crack, proc_no_crack, edges_no_crack, thresh_no_crack = extract_features(no_crack_sample)
print(f"\nNo Crack Image Features:")
print(f"  - HOG features: {len(features_no_crack['hog'])} dimensions")
print(f"  - Edge pixels: {features_no_crack['edge_pixels']:.0f}")
print(f"  - Crack pixels: {features_no_crack['crack_pixels']:.0f}")
print(f"  - Crack percentage: {features_no_crack['crack_percentage']:.3f}")

# Visualize the features
print("\n=== VISUALIZING FEATURES ===")
visualize_features(crack_sample, proc_crack, edges_crack, thresh_crack, 
                   features_crack['hog_image'], features_crack, "CRACK Sample")

#calling visualize features function
#Creates a 2x3 grid of subplots
#Plots all 6 components (original, processed, edges, threshold, HOG, feature summary)
#Uses the feature dictionary to display numerical values on the plots
#Applies the title "CRACK Sample" to identify this as a cracked surface

visualize_features(no_crack_sample, proc_no_crack, edges_no_crack, thresh_no_crack,
                   features_no_crack['hog_image'], features_no_crack, "NO CRACK Sample")




