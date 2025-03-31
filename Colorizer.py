import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model (ensure your model file path is correct)
model = load_model('unet_model_with_perceptual_loss.h5', compile=False)

# Function to preprocess SAR image for model input
def preprocess_image(image_path, img_size=(256, 256)):
    sar_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sar_image = cv2.resize(sar_image, img_size)
    sar_image = (sar_image - np.min(sar_image)) / (np.max(sar_image) - np.min(sar_image))
    return sar_image[np.newaxis, ..., np.newaxis]

# Function to predict colorization
def predict_colorization(sar_image):
    predicted_rgb = model.predict(sar_image)[0]
    return predicted_rgb

# Streamlit Interface
st.title("SAR Image Colorization")
st.write("Upload a SAR image to see its colorized version using a deep learning model!")

uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the uploaded SAR image
    sar_image = preprocess_image(file_path)

    # Predict colorization
    predicted_rgb_image = predict_colorization(sar_image)

    # Display original SAR and colorized images
    st.subheader("Original SAR Image")
    st.image(cv2.imread(file_path), use_column_width=True, channels="GRAY")

    st.subheader("Colorized RGB Image")
    st.image(predicted_rgb_image, use_column_width=True)

    # Clean up temporary file
    import os
    os.remove(file_path)
