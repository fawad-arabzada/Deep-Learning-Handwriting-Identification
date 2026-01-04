# ForensiScript 🖋️🔍
**Deep Learning-based Writer Identification System**

## 🚀 Project Overview
ForensiScript is a handwriting identification tool designed to analyze and classify handwriting samples to specific writers. By leveraging Deep Learning, the system can detect subtle ink patterns and individual writing styles that are often invisible to the naked eye.

### Key Performance Metrics:
- **🎯 Accuracy:** 89.29%
- **📊 ROC-AUC:** 0.9901
- **🧠 Model:** Custom CNN (Convolutional Neural Network)

## 🛠️ How It Works
The system uses a unique **Patch-based Majority Voting** strategy:
1. **Preprocessing:** Images are enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) and Otsu's Thresholding to isolate ink from paper.
2. **Patching:** Each sample is broken into 64x64 micro-patches.
3. **Inference:** The CNN analyzes every patch separately.
4. **Voting:** The final writer identity is determined by averaging the confidence scores across all patches, ensuring high reliability.

## 📂 Project Structure
- `run.py`: The deployment script for analyzing test images.
- `train.py`: The training logic used to develop the model.
- `model.keras`: The pre-trained neural network (89.29% accuracy).
- `labels.pkl`: The writer identity encoder.
- `/test`: Directory containing handwriting samples for verification.

## 💻 Usage
python run.py
