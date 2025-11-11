# Lung Cancer Detection using CNN

A Deep Learning–based web application that detects lung cancer from CT scan images using a **Convolutional Neural Network (CNN)**.  
Built with **TensorFlow**, **Keras**, **OpenCV**, and **Streamlit**, this project demonstrates how computer vision can assist in medical diagnosis by classifying lung tissue images as **Normal**, **Adenocarcinoma (Benign)**, or **Squamous Cell Carcinoma (Malignant)**.

***The lung_cancer_model.keras couldnt be uploaded since the file size was more than 25MB.***

---

## Project Overview
Lung cancer is one of the leading causes of cancer-related deaths globally.  
Early detection through medical imaging can drastically improve survival rates.

This project uses a **CNN-based image classifier** to analyze CT scan images of lungs and determine whether the tissue is:
-  **Normal**
-  **Adenocarcinoma (Benign)**
-  **Squamous Cell Carcinoma (Malignant)**

---

## Features
- Upload any lung CT scan image (`.jpg`, `.jpeg`, `.png`)
- Automatic preprocessing and prediction via trained CNN model
- Simple and interactive **Streamlit** web interface
- Real-time feedback with prediction confidence
- Displays a sample **Normal Lung CT Scan** for comparison

---

## Model Details
- Architecture: **Convolutional Neural Network (CNN)**
- Dataset: Lung CT Scan Dataset (3 classes)
- Optimizer: `Adam`
- Loss Function: `categorical_crossentropy`
- Metrics: `accuracy`
- Final Accuracy: **92.57%**

---

## How to Run Locally

## Download Dataset: https://drive.google.com/drive/folders/191EMrBzqJ9qnf9NxEUa-mBXYIni2uyMc?usp=sharing

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection
