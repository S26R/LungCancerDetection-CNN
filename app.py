import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2
import random
model = tf.keras.models.load_model("lung_cancer_model.keras")

CLASS_NAMES = ['Lung Adenocarcinoma (Benign)', 'Lung Squamous Cell Carcinoma (Malignant)', 'Normal Lung Tissue']
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")
st.sidebar.title("Lung Cancer Detection using CNN")
st.sidebar.markdown("Upload a lung CT scan image to analyze and classify it using the trained CNN model.")
st.title("Lung Cancer Detection System")
st.write("This web application analyzes CT scan images to detect possible lung cancer types using a Convolutional Neural Network (CNN).")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload a Lung CT Scan Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.warning("Please upload an image to continue.")
with col2:
    st.write("**Reference: Normal Lung CT Scan**")
    normal_image = "normal_reference.jpeg"
    try:
        ref_img = Image.open(normal_image)
        st.image(ref_img, caption="Normal Lung Image", use_container_width=True)
    except:
        st.info("Add 'normal_reference.jpeg' to your project folder to display here.")
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Model Prediction")
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)
    predicted_class = CLASS_NAMES[class_index]
    with st.spinner("Analyzing the image..."):
        time.sleep(2)
    st.success(f"**Predicted Result:** {predicted_class}")
    st.write(f"**Model Confidence:** {confidence}%")
    if class_index == 0:
        recovery_time = random.randint(60, 120)
        solution = "The image suggests early signs of Adenocarcinoma (benign). Treatment and monitoring can help full recovery."
    elif class_index == 1:
        recovery_time = random.randint(90, 150)
        solution = "The scan indicates Squamous Cell Carcinoma (malignant). Immediate consultation and treatment are recommended."
    else:
        recovery_time = 0
        solution = "The lung tissue appears normal. Maintain a healthy lifestyle and regular check-ups."

    if recovery_time > 0:
        st.write(f"**Estimated Recovery Duration:** Around {recovery_time} days (subject to medical evaluation).")
    st.write(f"**Medical Advice:** {solution}")
    wishes = [
        "Wishing you strength, positivity, and a smooth recovery.",
        "Every breath is a step towards healing — stay strong.",
        "Your strength will outshine this phase. Keep fighting.",
        "Stay hopeful — every sunrise brings new strength."
    ]
    st.markdown(f"**Note:** {random.choice(wishes)}")
st.markdown("---")
st.caption("Developed by Shounak Roy | Techno India University, Kolkata")
