import os

try:
    import cv2
    import onnxruntime as ort
except ModuleNotFoundError:
    os.system("pip install opencv-python-headless")
    os.system("pip install onnxruntime")
    import cv2
    import onnxruntime as ort

import streamlit as st
import numpy as np
from PIL import Image

# Futuristic UI Design
st.set_page_config(page_title="AI Face Recognition", layout="wide")
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
    <div class="title">TEAM 4</div>
    """, unsafe_allow_html=True
)

# Load ONNX Model
def load_model(model_path):
    return ort.InferenceSession(model_path) if model_path else None

model_path = "model.onnx"
model = load_model(model_path)


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

        print("Raw Model Output:", output)  # Debugging step

        predicted_class = np.argmax(output)  # Get index of highest probability
        confidence = np.max(output)  # Get confidence level

        # Ensure predicted_class is within valid range
        if predicted_class >= len(class_names):
            return "Unknown", confidence  # Return "Unknown" if index is out of range

        return class_names[predicted_class], confidence

    return None, None




# Sidebar for Mode Selection
st.sidebar.title("⚡ Face Recognition System")
mode = st.sidebar.radio("Choose Mode:", ["Upload Image", "Live Camera"])

if mode == "Upload Image":
    st.title("🖼️ Upload an Image for Face Recognition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=False)
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Recognize Face"):
            name, confidence = recognize_face(image, model)

            if name:
                st.success(f"Recognized as: {name} ({confidence*100:.2f}% confidence)")
            else:
                st.warning("Face not recognized.")

elif mode == "Live Camera":
    st.title("📷 Live Camera Face Recognition")
    st.write("Turn on your camera and detect faces in real-time.")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Camera")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
