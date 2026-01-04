import os
import warnings

# --- 1. SILENCE EVERYTHING (Must be at the very top) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# --- 2. Configuration ---
TEST_PATH = "test"
MODEL_PATH = "model.keras" 
LABEL_PATH = "labels.pkl"
IMG_SIZE = 64

def preprocess_for_inference(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img

# --- 3. Run Analysis ---
def run_analysis():
    # Load Model and Labels quietly
    model = load_model(MODEL_PATH, compile=False)
    try:
        with open(LABEL_PATH, 'rb') as f:
            le = pickle.load(f)
    except:
        le = joblib.load(LABEL_PATH)

    print("🔮 Scanning Test Images...")

    filenames = [f for f in sorted(os.listdir(TEST_PATH)) 
                 if f.lower().endswith((".png", ".jpg", ".jpeg")) 
                 and not f.startswith(('.', '_'))]

    y_true, y_probs = [], []

    for fname in filenames:
        actual_label = fname[:2] 
        img = preprocess_for_inference(os.path.join(TEST_PATH, fname))
        if img is None: continue

        h, w = img.shape
        img_patches = []
        for i in range(0, h - IMG_SIZE, 32):
            for j in range(0, w - IMG_SIZE, 32):
                patch = img[i:i+IMG_SIZE, j:j+IMG_SIZE]
                img_patches.append(patch.astype('float32') / 255.0)

        patches_array = np.expand_dims(np.array(img_patches), -1)
        predictions = model.predict(patches_array, verbose=0)
        
        avg_prob = np.mean(predictions, axis=0)
        y_true.append(actual_label)
        y_probs.append(avg_prob)

    # Calculate Results
    y_pred = [le.inverse_transform([np.argmax(p)])[0] for p in y_probs]
    acc = accuracy_score(y_true, y_pred)
    
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    roc = roc_auc_score(y_true_bin, np.array(y_probs), multi_class='ovr', average='weighted')

    # --- 4. FINAL CLEAN OUTPUT ---
    print(f"\n🎯 Accuracy : {acc:.4f}")
    print(f"📊 ROC-AUC  : {roc:.4f}\n")

if __name__ == "__main__":
    run_analysis()