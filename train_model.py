import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import joblib

# Define paths
data_dir = "data"
crack_dir = os.path.join(data_dir, "crack")
no_crack_dir = os.path.join(data_dir, "no_crack")

def extract_feature_vector(image_path):
    """
    Extract a combined feature vector for ML model
    """
    # Preprocess
    processed_img = preprocess_image(image_path)
    
    # HOG features (primary)
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
    
    # Combine all features into one vector
    combined_features = np.concatenate([
        hog_features,
        [edge_count, crack_pixels, crack_percentage]  # Additional simple features
    ])
    
    return combined_features

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blurred

print("=== EXTRACTING FEATURES FROM ALL IMAGES ===")
X = []  # Features
y = []  # Labels

# Process crack images (label 1)
print("Processing crack images...")
for i, img_file in enumerate(os.listdir(crack_dir)):
    if i % 100 == 0:
        print(f"  Processed {i}/1000 crack images")
    img_path = os.path.join(crack_dir, img_file)
    features = extract_feature_vector(img_path)
    X.append(features)
    y.append(1)  # Crack = 1

# Process no_crack images (label 0)
print("Processing no_crack images...")
for i, img_file in enumerate(os.listdir(no_crack_dir)):
    if i % 100 == 0:
        print(f"  Processed {i}/1000 no_crack images")
    img_path = os.path.join(no_crack_dir, img_file)
    features = extract_feature_vector(img_path)
    X.append(features)
    y.append(0)  # No crack = 0

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\n=== DATASET SHAPES ===")
print(f"X shape: {X.shape}")  
print(f"y shape: {y.shape}")  

print("\n=== SPLITTING DATA ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print("\n=== TRAINING LOGISTIC REGRESSION MODEL ===")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print("Training completed!")

print("\n=== MODEL EVALUATION ===")
# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Crack', 'Crack']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Crack', 'Crack'], 
            yticklabels=['No Crack', 'Crack'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance (for the simple features)
feature_names = ['HOG'] * 8100 + ['Edge_Count', 'Crack_Pixels', 'Crack_Percentage']
simple_feature_importance = model.coef_[0][-3:]  # Last 3 features
simple_features = ['Edge_Count', 'Crack_Pixels', 'Crack_Percentage']

plt.figure(figsize=(10, 6))
plt.bar(simple_features, simple_feature_importance)
plt.title('Feature Importance (Simple Features)')
plt.ylabel('Coefficient Value')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.show()

print("\n=== MODEL SUMMARY ===")
print(f"Final Model Accuracy: {accuracy:.2%}")
print(f"Number of features used: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
joblib.dump(model, 'crack_detection_model.pkl')
print(" Model saved as 'crack_detection_model.pkl'")