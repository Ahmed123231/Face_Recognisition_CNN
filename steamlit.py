import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image


# Streamlit Page Configuration
st.set_page_config(page_title="AI Face Recognition", layout="wide")
# Ensure model file exists
MODEL_PATH = "model.onnx"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please upload `model.onnx` to your repository.")
    st.stop()

# Load ONNX Model
@st.cache_resource  # Cache the model to speed up inference
def load_model():
    return ort.InferenceSession(MODEL_PATH)

model = load_model()


st.markdown(
    """
    <style>
        body {background-color: #0e0e0e; color: #00d4ff;}
        .stButton>button {background-color: #00d4ff; color: black; font-weight: bold; border-radius: 10px;}
        .stFileUploader {border: 2px dashed #00d4ff; padding: 10px;}
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #00d4ff;
            font-family: 'Courier New', Courier, monospace;
        }
    </style>
    <div class="title">TEAM 4 - Face Recognition</div>
    """, unsafe_allow_html=True
)

# Image Processing Function
def preprocess_image(image, target_size=(299, 299)):
    image = image.resize(target_size)  # Resize image
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize

    # Ensure the correct shape (batch_size, height, width, channels)
    if len(img_array.shape) == 3 and img_array.shape[-1] == 3:  # Ensure RGB format
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, H, W, C)

    return img_array  # Final shape: (1, 299, 299, 3)

# Face Recognition Function
def recognize_face(image, model, class_names=["Person1", "Person2", "Unknown"]):
    processed_img = preprocess_image(image)

    if model:
        input_name = model.get_inputs()[0].name
        output = model.run(None, {input_name: processed_img})[0]  # Get first output

        predicted_class = np.argmax(output)  # Get index of highest probability
        confidence = np.max(output)  # Get confidence level

        # Ensure predicted_class is within valid range
        if predicted_class >= len(class_names):
            return "Unknown", confidence  # Return "Unknown" if index is out of range

        return class_names[predicted_class], confidence

    return None, None

# Sidebar for Mode Selection
logo_url = "https://img.freepik.com/free-vector/face-recognition-biometric-scan-cyber-security-technology-blue-tone_53876-119532.jpg"
st.sidebar.image(logo_url, use_container_width=True)
st.sidebar.markdown(
    "<h1 style='font-size: 30px;'>Face Recognition System</h1>", 
    unsafe_allow_html=True
)
mode = st.sidebar.radio("Choose Mode:", ["Upload Image", "Live Camera"])

# Upload Image Mode
if mode == "Upload Image":
    st.title("Upload a Celebrity Image for Face Recognition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Recognize Face"):
            name, confidence = recognize_face(image, model)

            if name:
                st.success(f"✅ Recognized as: {name} ({confidence*100:.2f}% confidence)")
            else:
                st.warning("⚠️ Face not recognized.")

# Live Camera Mode (Fixed for Streamlit Cloud)
elif mode == "Live Camera":
    st.title("📷 Live Camera Face Recognition")
    
    uploaded_image = st.camera_input("Take a picture")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("Recognize Face"):
            name, confidence = recognize_face(image, model)

            if name:
                st.success(f"✅ Recognized as: {name} ({confidence*100:.2f}% confidence)")
            else:
                st.warning("⚠️ Face not recognized.")
