import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Handwritten OCR",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Handwritten Text Recognition (TrOCR)")
st.markdown("Powered by Transformer AI Model 🚀")

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_model()

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better recognition
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh

# =========================
# OCR FUNCTION (TrOCR)
# =========================
def extract_text_trocr(image):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text

# =========================
# INPUT
# =========================
st.subheader("📤 Upload or Capture Image")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
camera_img = st.camera_input("📷 Take Photo")

image = None

if uploaded_file:
    image = Image.open(uploaded_file)

elif camera_img:
    image = Image.open(camera_img)

# =========================
# PROCESS
# =========================
if image is not None:

    st.image(image, caption="Input Image", use_container_width=True)

    img = np.array(image)

    # Convert RGB → BGR
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    processed = preprocess(img)

    pil_img = Image.fromarray(processed)

    with st.spinner("🧠 AI is reading handwriting..."):
        text = extract_text_trocr(pil_img)

    st.success("✅ Done!")

    # =========================
    # OUTPUT UI
    # =========================
    st.subheader("📄 Extracted Text")
    st.code(text if text else "⚠ No text detected")

    st.download_button(
        label="📥 Download Text",
        data=text,
        file_name="output.txt"
    )

else:
    st.info("👆 Upload or capture image")
