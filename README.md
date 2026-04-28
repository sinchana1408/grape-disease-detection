🍇 GrapeCare: Intelligent Deep Learning-Based Grape Disease Detection
1. INTRODUCTION

GrapeCare is a deep learning–based intelligent system designed to automatically detect diseases in grape leaves using image processing and transfer learning techniques.

Traditional disease detection in agriculture relies on manual inspection, which is:

Subjective
Time-consuming
Prone to human error

This system provides an automated, accurate, and scalable solution for early disease detection in vineyards, helping farmers take timely action.

2. OBJECTIVE
Develop an automated grape disease detection system
Use image processing and deep learning models for prediction
Detect diseases at early stages to reduce crop loss
Provide a scalable and cost-effective agricultural solution
3. SOFTWARE REQUIREMENTS

Operating System: Windows / Linux / macOS
Programming Language: Python 3.x, JavaScript (React)
IDE: VS Code / Jupyter Notebook / Google Colab

Libraries Used:

Backend (Python):

TensorFlow / Keras
NumPy
OpenCV
Flask
Scikit-learn

Frontend (React):

React.js
Axios
HTML / CSS
4. SALIENT FEATURES
Non-invasive image-based disease detection
Uses CNN + Transfer Learning
Supports multiple models:
DenseNet121
EfficientNetB0
ResNet50
MobileNetV2
Model comparison with accuracy scores
Real-time prediction via web interface
Displays confidence score for predictions
Clean and user-friendly UI
5. DATASET DESCRIPTION

The dataset consists of grape leaf images categorized into four classes:

Grape Black Rot
Grape Esca (Black Measles)
Grape Leaf Blight
Healthy

Images are resized to 128×128 pixels and augmented using:

Rotation
Flipping
Zoom
Shifting

This improves model generalization and performance.

📸 Project Screenshots
🔹 Home Page

(Upload Image UI)

🔹 Prediction Output

(Shows predicted disease + confidence + model comparison)

6. COMPILATION / EXECUTION PROCEDURE
Step 1: Install Dependencies
pip install tensorflow numpy opencv-python flask scikit-learn
npm install
Step 2: Run Backend
python app.py
Step 3: Run Frontend
cd frontend
npm start
7. PROCEDURE TO RUN THE PROJECT
Load grape leaf dataset
Preprocess images (resize, normalize, augment)
Train models using transfer learning
Evaluate models using test data
Start Flask backend server
Upload a grape leaf image via frontend
System predicts disease using multiple models
Displays:
Final prediction
Confidence score
Model comparison
8. WORKING OF THE SYSTEM
Input:

Grape leaf image

Processing Steps:
Resize image to 128×128
Normalize pixel values
Apply augmentation
Extract features using pretrained CNN models
Classify using trained model
Output:
Predicted disease class
Confidence score
Model comparison results
9. LIMITATIONS
Requires clear and high-quality images
Accuracy depends on dataset size and diversity
Training deep learning models requires computational resources
Not suitable for real-time field conditions without optimization
10. FUTURE ENHANCEMENTS
Deploy as mobile application
Real-time camera-based disease detection
Integration with IoT-based smart farming systems
Add treatment suggestions for detected diseases
Expand to other crops and plant diseases
11. CONCLUSION

The GrapeCare system successfully detects grape leaf diseases using image-based deep learning techniques.

It reduces dependency on manual inspection and provides a fast, reliable, and scalable solution for farmers.

By leveraging transfer learning models, the system improves early disease detection, helping reduce crop loss and improve agricultural productivity.


College:
(Your College Name)
