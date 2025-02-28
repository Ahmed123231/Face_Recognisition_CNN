import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

# ===========================
# Load Class Labels from File
# ===========================
# The model predicts numerical class indices, which correspond to real class names.
# These names were extracted from the dataset and saved as "class_labels.npy".
class_names = np.load("class_labels.npy").tolist()
print("Loaded Class Names:", class_names)

# ================================
# Streamlit Page Configuration
# ================================
# Set the page title and layout for the web interface.
st.set_page_config(page_title="AI Face Recognition", layout="wide")

# ========================================
# Ensure the ONNX Model File Exists
# ========================================
# The model file must be present in the directory. If not, the app will stop execution.
MODEL_PATH = "model.onnx"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please upload `model.onnx` to your repository.")
    st.stop()

# ==========================
# Load ONNX Model
# ==========================
# This function loads the ONNX model into memory.
# The `@st.cache_resource` decorator ensures the model is cached for performance.
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

# Load the model into memory
model = load_model()

# ======================================
# Streamlit UI Styling (Custom CSS)
# ======================================
# Apply styling to make the UI visually appealing.
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
    <div class="title">TEAM 4 - Celebrity Face Recognition</div>
    """, unsafe_allow_html=True
)

# ===============================
# Image Preprocessing Function
# ===============================
# Converts an image into a format suitable for the model:
# 1. Resize to the required dimensions (299x299).
# 2. Normalize pixel values to [0,1] by dividing by 255.
# 3. Expand dimensions to match the model's expected input shape (1, height, width, channels).
def preprocess_image(image, target_size=(299, 299)):
    image = image.resize(target_size)  # Resize to match model input
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize pixel values

    # Ensure input shape is (1, 299, 299, 3)
    if len(img_array.shape) == 3 and img_array.shape[-1] == 3:  # Ensure 3-channel RGB image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array  # Output shape: (1, 299, 299, 3)

# ============================
# Face Recognition Function
# ============================
# Runs the preprocessed image through the ONNX model and retrieves:
# - `predicted_class`: The index of the highest probability class.
# - `confidence`: The probability of the predicted class.
def recognize_face(image, model):
    processed_img = preprocess_image(image)  # Preprocess input image
    
    if model:
        # Get model input name dynamically (depends on the ONNX file)
        input_name = model.get_inputs()[0].name

        # Run inference and extract output
        output = model.run(None, {input_name: processed_img})[0]

        # Get the most confident prediction
        predicted_class = np.argmax(output)  # Index of highest probability
        confidence = np.max(output)  # Confidence score

        # Ensure valid index range to avoid out-of-bounds error
        if predicted_class >= len(class_names):
            return "Unknown", confidence  

        return class_names[predicted_class], confidence  # Return predicted name & confidence score

# =============================
# Sidebar: Mode Selection
# =============================
# Display a sidebar with a logo and a mode selection option.
logo_url = "https://img.freepik.com/free-vector/face-recognition-biometric-scan-cyber-security-technology-blue-tone_53876-119532.jpg"
st.sidebar.image(logo_url, use_container_width=True)

# Sidebar title
st.sidebar.markdown(
    "<h1 style='font-size: 30px;'>Face Recognition System</h1>", 
    unsafe_allow_html=True
)

# Mode selection: User can choose to upload an image or use the live camera
mode = st.sidebar.radio("Choose Mode:", ["Upload Image", "Live Camera"])

# =============================
# Mode 1: Upload Image
# =============================
if mode == "Upload Image":
    st.title("Upload a Celebrity Image for Face Recognition")

    # File uploader for images (only accepts JPG, PNG, JPEG)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run face recognition when button is clicked
        if st.button("Recognize Face"):
            name, confidence = recognize_face(image, model)

            # Display results
            if name:
                st.success(f"✅ Recognized as: {name} ({confidence*100:.2f}% confidence)")
            else:
                st.warning("⚠️ Face not recognized.")

# =============================
# Mode 2: Live Camera Input
# =============================
elif mode == "Live Camera":
    st.title("📷 Live Camera Face Recognition")

    # Open camera input for real-time face capture
    uploaded_image = st.camera_input("Take a picture")

    if uploaded_image:
        # Open and display captured image
        image = Image.open(uploaded_image)
        st.image(image, caption="Captured Image", use_container_width=True)

        # Run face recognition when button is clicked
        if st.button("Recognize Face"):
            name, confidence = recognize_face(image, model)

            # Display results
            if name:
                st.success(f"✅ Recognized as: {name} ({confidence*100:.2f}% confidence)")
            else:
                st.warning("⚠️ Face not recognized.")
