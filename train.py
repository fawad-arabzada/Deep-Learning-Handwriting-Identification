import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Configuration ---
TRAIN_PATH = "train"      # Matches your folder structure
IMG_SIZE = 64             # Size of handwriting patches
EPOCHS = 26              # Increased for better convergence
BATCH = 16                # Balanced batch size for CPU
EXPORT_DIR = "exports"

# Create exports folder if it doesn't exist to avoid FileNotFoundError
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

X, y = [], []

print("📦 Extracting Handwriting Textures (Advanced Preprocessing)...")
# Check if directory exists
if not os.path.exists(TRAIN_PATH):
    print(f"❌ Error: Folder '{TRAIN_PATH}' not found. Please ensure it is in the same directory as this script.")
    exit()

for fname in sorted(os.listdir(TRAIN_PATH)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
    
    # Extract writer ID from first two characters [cite: 12]
    label = fname[:2] 
    
    img = cv2.imread(os.path.join(TRAIN_PATH, fname), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Advanced Preprocessing: CLAHE for better stroke definition
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Binary Inversion to focus on ink (1) vs paper (0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Slicing into overlapping 64x64 patches to multiply data points [cite: 43]
    h, w = img.shape
    for i in range(0, h-IMG_SIZE, 32): 
        for j in range(0, w-IMG_SIZE, 32):
            patch = img[i:i+IMG_SIZE, j:j+IMG_SIZE]
            # Normalize and reshape for CNN 
            X.append(patch.astype('float32') / 255.0)
            y.append(label)

X = np.expand_dims(np.array(X), -1) # Add channel dimension (Grayscale)
le = LabelEncoder()
y = le.fit_transform(y)

# Save encoder to exports folder
joblib.dump(le, os.path.join(EXPORT_DIR, "labels.pkl"))

print(f"🔥 Training Deep Neural Network on {len(X)} samples...")

# --- Advanced CNN Architecture [cite: 44, 48] ---
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(), # Reduces parameters for efficient CPU run [cite: 24, 53]
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, verbose=1, shuffle=True)

# Save the model into the exports folder [cite: 23, 37]
model.save(os.path.join(EXPORT_DIR, "model.keras"))
print(f"✅ Model and Labels saved in '{EXPORT_DIR}' folder.")