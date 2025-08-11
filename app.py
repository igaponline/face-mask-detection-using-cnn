import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Face Mask Detection")

model = tf.keras.models.load_model("project.keras")



# Choose input method
option = st.radio("Select input method:", ("Camera", "Upload Image"))

# Image placeholder
image = None

if option == "Camera":
    img_file_buffer = st.camera_input("Capture an image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

if image is not None:

    image_resized = image.resize((150, 150))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.write("Prediction: Without Mask 😷")
    else:
        st.write("Prediction: With Mask ✅")