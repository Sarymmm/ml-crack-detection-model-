import cv2
import numpy as np
from skimage.feature import hog
import joblib

# ======== CONFIGURATION ========
# PASTE YOUR IMAGE PATH HERE
IMAGE_PATH ="D:\\images (3).jpg"  # ← PASTE YOUR PATH HERE
# ===============================

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blurred

def extract_features(image_path):
    """Extract features from image"""
    processed_img = preprocess_image(image_path)
    
    # HOG features
    hog_features = hog(
        processed_img, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=False,
        block_norm='L2-Hys'
    )
    
    # Canny edges
    edges = cv2.Canny(processed_img, 50, 150)
    edge_count = np.sum(edges) / 255
    
    # Threshold
    _, thresh = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh) / 255
    crack_percentage = crack_pixels / (128 * 128)
    
    # Combine features
    features = np.concatenate([hog_features, [edge_count, crack_pixels, crack_percentage]])
    return features.reshape(1, -1)

def main():
    print(" CRACK DETECTION PREDICTION")
    print("=" * 40)
    
    # Load the trained model
    try:
        model = joblib.load('crack_detection_model.pkl')
        print(" Model loaded successfully!")
    except:
        print(" Error: Model file 'crack_detection_model.pkl' not found!")
        print("Please run train_model.py first to train and save the model.")
        return
    
    # Check if image path is set
    if IMAGE_PATH == "C:/Users/YourName/Pictures/concrete_wall.jpg":
        print(" Please update the IMAGE_PATH variable with your actual image path!")
        return
    
    print(f" Image path: {IMAGE_PATH}")
    
    try:
        # Extract features and predict
        features = extract_features(IMAGE_PATH)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Display results
        print("\n" + "" * 20)
        if prediction == 1:
            print(" RESULT: CRACK DETECTED!")
            # print(f" Confidence: {probability[1]:.2%}")
        else:
            print(" RESULT: NO CRACK")
            # print(f" Confidence: {probability[0]:.2%}")
        print("" * 20)
        
        # Show detailed probabilities
        # print(f"\n Detailed Probabilities:")
        # print(f"   No Crack: {probability[0]:.2%}")
        # print(f"   Crack:    {probability[1]:.2%}")
        
        # Show feature analysis
        processed_img = preprocess_image(IMAGE_PATH)
        edges = cv2.Canny(processed_img, 50, 150)
        _, thresh = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        edge_count = np.sum(edges) / 255
        crack_pixels = np.sum(thresh) / 255
        crack_percentage = crack_pixels / (128 * 128)
        
        # print(f"\n Feature Analysis:")
        # print(f"   Edge Pixels: {edge_count:.0f}")
        # print(f"   Crack Pixels: {crack_pixels:.0f}")
        # print(f"   Crack Coverage: {crack_percentage:.2%}")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        print("Please check if:")
        print("1. The image path is correct")
        print("2. The file is a valid image")
        print("3. The model file exists")

if __name__ == "__main__":
    main()