# import streamlit as st
# from streamlit_option_menu import option_menu
# import torch
# from ultralytics import YOLO
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import os
# from PIL import Image
# from matplotlib import pyplot as plt

# # ----------------- Load Models -----------------
# # Load YOLOv8 model
# #yolo_model = YOLO("yolov8n.pt")
# MODEL_PATH = "runs/detect/train/weights/best.pt"  # Make sure this is the correct path to your trained model 
# try:
#     yolo_model = YOLO(MODEL_PATH)
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# # Load the U-Net model
# MODEL_PATH = "unet_model.h5"
# if os.path.exists(MODEL_PATH):
#     unet_model = load_model(MODEL_PATH)
# else:
#     st.error("Model file not found! Ensure 'unet_model.h5' is in the same directory.")
#     st.stop()

# # Load NST model from TensorFlow Hub
# nst_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


# # ----------------- Streamlit UI -----------------
# st.title("AI-Powered Image Processing App")

# # Sidebar Navigation
# with st.sidebar:
#     selected = option_menu("Menu", ["Object Detection", "Image Segmentation", "Neural Style Transfer"], 
#                            icons=["camera", "image", "brush"], menu_icon="menu", default_index=0)

# # ----------------- Object Detection Page -----------------
# if selected == "Object Detection":
#     st.header("🔍 Pothole Detection using YOLOv8")
#     uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         # Load Image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Convert image for YOLO processing
#         image_path = "temp_image.jpg"
#         image.save(image_path)

#         # Load image properly for YOLO
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Run YOLOv8 detection
#         results = yolo_model(img)  # ✅ Corrected

#         # Draw results
#         result_image = results[0].plot()
#         st.image(result_image, caption="Detected Potholes", use_column_width=True)

# # ----------------- Image Segmentation Page -----------------
# elif selected == "Image Segmentation":
#     st.header("🖼️ Image Segmentation using U-Net")
    
#     # Load the model once and store it in session state
#     if "unet_model" not in st.session_state:
#         MODEL_PATH = "unet_model.h5"
#         if os.path.exists(MODEL_PATH):
#             st.session_state.unet_model = load_model(MODEL_PATH)
#         else:
#             st.error("❌ Model file not found! Ensure 'unet_model.h5' is in the same directory.")
#             st.stop()

#     uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         # Load and preprocess image
#         image = Image.open(uploaded_file).convert("RGB")
#         image = np.array(image)
#         image = cv2.resize(image, (256, 256))  # Resize for model input
#         img_array = image.astype(np.float32) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Load model from session state
#         unet_model = st.session_state.unet_model

#         # Predict segmentation mask
#         mask = unet_model.predict(img_array)[0]  
#         mask = np.squeeze(mask)  # Remove extra dimensions
#         mask = (mask > 0.5).astype(np.uint8)  # Binarize mask

#         # Display original and segmented images
#         st.image(image, caption="📷 Uploaded Image", use_column_width=True)

#         # Display mask using matplotlib for better visualization
#         fig, ax = plt.subplots()
#         ax.imshow(mask, cmap="gray")
#         ax.axis("off")
#         st.pyplot(fig)

# # ----------------- Neural Style Transfer Page -----------------
# elif selected == "Neural Style Transfer":
#     st.header("🎨 Neural Style Transfer")
    
#     content_file = st.file_uploader("Upload the content image...", type=["jpg", "jpeg", "png"])
#     style_file = st.file_uploader("Upload the style image...", type=["jpg", "jpeg", "png"])

#     if content_file and style_file:
#         # Load images
#         def load_image(img_file):
#             img = Image.open(img_file)
#             img = img.resize((256, 256))  # Resize to fit model
#             img_array = np.array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)
#             return tf.convert_to_tensor(img_array, dtype=tf.float32)

#         content_image = load_image(content_file)
#         style_image = load_image(style_file)

#         # Generate stylized image
#         stylized_image = nst_model(tf.constant(content_image), tf.constant(style_image))[0]

#         # Convert tensor to image format
#         stylized_image = np.squeeze(stylized_image.numpy())

#         # Display images
#         st.image(content_file, caption="Content Image", use_column_width=True)
#         st.image(style_file, caption="Style Image", use_column_width=True)
#         st.image(stylized_image, caption="Stylized Image", use_column_width=True)

#---------

import streamlit as st
from streamlit_option_menu import option_menu
import torch
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from PIL import Image
import matplotlib.pyplot as plt

# ----------------- Load Models -----------------
st.sidebar.title("🛠️ Model Selection")

# Load YOLOv8 model
MODEL_PATH_YOLO = "runs/detect/train/weights/best.pt"
try:
    yolo_model = YOLO(MODEL_PATH_YOLO)
except Exception as e:
    st.sidebar.error(f"❌ Error loading YOLO model: {e}")
    st.stop()

# Load U-Net model
MODEL_PATH_UNET = "unet_model.h5"
if os.path.exists(MODEL_PATH_UNET):
    unet_model = load_model(MODEL_PATH_UNET)
else:
    st.sidebar.error("❌ U-Net model file not found! Ensure 'unet_model.h5' is available.")
    st.stop()

# Load NST model from TensorFlow Hub
nst_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# ----------------- Streamlit UI -----------------
st.title("🖥️ AI-Powered Image Processing App")
st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Menu", ["Object Detection", "Image Segmentation", "Neural Style Transfer"], 
                           icons=["camera", "image", "brush"], menu_icon="menu", default_index=0)

# ----------------- Object Detection -----------------
if selected == "Object Detection":
    st.header("🔍 Pothole Detection using YOLOv8")
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"], key="yolo")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        
        image_path = "temp_image.jpg"
        image.save(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = yolo_model(img)
        result_image = results[0].plot()
        st.image(result_image, caption="✅ Detected Potholes", use_column_width=True)
        st.success("✅ Detection Completed!")

# # ----------------- Image Segmentation -----------------
# elif selected == "Image Segmentation":
#     st.header("🖼️ Road Segmentation using U-Net")
#     uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"], key="unet")
    
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         image = np.array(image)
#         image = cv2.resize(image, (256, 256))
#         img_array = image.astype(np.float32) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         mask = unet_model.predict(img_array)[0]
#         mask = np.squeeze(mask)
#         mask = (mask > 0.5).astype(np.uint8)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="📷 Uploaded Image", use_column_width=True)
#         with col2:
#             fig, ax = plt.subplots()
#             ax.imshow(mask, cmap="gray")
#             ax.axis("off")
#             st.pyplot(fig)
        
#         st.success("✅ Segmentation Completed!")

# ----------------- Image Segmentation -----------------
elif selected == "Image Segmentation":
    st.header("🖼️ Road Segmentation using U-Net")
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"], key="unet")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)
        image = cv2.resize(image, (256, 256))
        img_array = image.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get the predicted mask from U-Net model
        mask = unet_model.predict(img_array)[0]
        mask = np.squeeze(mask)
        mask = (mask > 0.5).astype(np.uint8)  # Thresholding for binary mask

        # Apply colormap for better visualization
        colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        with col2:
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))  # Convert to RGB for proper display
            ax.axis("off")
            st.pyplot(fig)

        st.success("✅ Segmentation Completed!")

# ----------------- Neural Style Transfer -----------------
elif selected == "Neural Style Transfer":
    st.header("🎨 Neural Style Transfer")
    content_file = st.file_uploader("📤 Upload Content Image...", type=["jpg", "jpeg", "png"], key="content")
    style_file = st.file_uploader("🎭 Upload Style Image...", type=["jpg", "jpeg", "png"], key="style")
    
    if content_file and style_file:
        def load_image(img_file):
            img = Image.open(img_file).resize((256, 256))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        content_image = load_image(content_file)
        style_image = load_image(style_file)
        
        stylized_image = nst_model(tf.constant(content_image), tf.constant(style_image))[0]
        stylized_image = np.squeeze(stylized_image.numpy())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(content_file, caption="🖼️ Content Image", use_column_width=True)
        with col2:
            st.image(style_file, caption="🎭 Style Image", use_column_width=True)
        with col3:
            st.image(stylized_image, caption="✨ Stylized Image", use_column_width=True)
        
        st.success("✅ Style Transfer Completed!")
