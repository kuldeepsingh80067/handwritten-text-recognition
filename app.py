import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Handwritten OCR Pro",
    page_icon="✍️",
    layout="centered"
)

# -----------------------------
# 🎯 HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>✍️ Handwritten Text Recognition</h1>
    <p style='text-align: center; color: gray;'>AI-powered OCR with enhanced accuracy & aspect-ratio preservation</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# 🤖 LOAD OCR
# -----------------------------
@st.cache_resource
def load_reader():
    # GPU is set to False by default, switch to True if you have CUDA installed
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# 📥 INPUT SECTION
# -----------------------------
st.subheader("📥 Input")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"], horizontal=True)
image = None

if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    if file:
        image = Image.open(file)
elif option == "Use Camera":
    cam = st.camera_input("Take photo")
    if cam:
        image = Image.open(cam)

# -----------------------------
# 🧠 CORE PROCESSING PIPELINE
# -----------------------------
def optimize_image_for_ocr(img_array):
    """Smart preprocessing that preserves aspect ratios and cleans noise."""
    # 1. Preserve Aspect Ratio while scaling up for better readability
    h, w = img_array.shape[:2]
    target_width = 1200
    target_height = int((target_width / w) * h)
    img_resized = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Denoise (Better than Gaussian Blur for text)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)

    # 4. Contrast boost via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    return enhanced

if image:
    st.divider()
    
    # Convert PIL to CV2 format safely
    img = np.array(image.convert('RGB')) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.subheader("🖼️ Original Image")
    st.image(image, use_column_width=True)

    # Preprocess
    base_processed = optimize_image_for_ocr(img)

    # Create intelligent variants for the OCR to try
    variants = {}
    
    # Variant 1: Enhanced Grayscale (Lets EasyOCR do its own binarization)
    variants["Enhanced Grayscale"] = base_processed
    
    # Variant 2: Adaptive Threshold (Great for uneven lighting / shadows on paper)
    variants["Adaptive Threshold"] = cv2.adaptiveThreshold(
        base_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    
    # Variant 3: Morphological Closing (Connects broken pen strokes in handwriting)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    variants["Morphological Fix"] = cv2.morphologyEx(variants["Adaptive Threshold"], cv2.MORPH_CLOSE, kernel)

    with st.expander("👀 View Image Processing Variants"):
        st.image(list(variants.values()), caption=list(variants.keys()), width=200)

    # -----------------------------
    # 🔍 OCR EXTRACTION & SCORING
    # -----------------------------
    best_text = ""
    best_avg_score = 0
    winning_variant = ""

    with st.spinner("🔍 AI is analyzing the handwriting..."):
        for name, v in variants.items():
            # mag_ratio=1.5 zooms in internally, contrast_ths helps with faint pencil marks
            result = reader.readtext(v, mag_ratio=1.5, contrast_ths=0.1, adjust_contrast=0.5)
            
            if not result:
                continue

            text_parts = [r[1] for r in result]
            confidences = [r[2] for r in result]

            # Use AVERAGE confidence, not SUM. (Summing rewards garbage characters).
            avg_score = sum(confidences) / len(confidences)
            combined_text = " ".join(text_parts)

            if avg_score > best_avg_score and len(combined_text.strip()) > 0:
                best_avg_score = avg_score
                best_text = combined_text
                winning_variant = name

    # Smart Text Cleaning: Keeps basic punctuation (.,!?'"-) unlike the previous regex
    best_text = re.sub(r'[^\w\s.,!?\'"-]', '', best_text)
    # Remove extra spaces
    best_text = re.sub(r'\s+', ' ', best_text).strip()

    # -----------------------------
    # 📊 OUTPUT
    # -----------------------------
    st.divider()
    st.subheader("📊 Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence Score", f"{round(best_avg_score * 100, 1)}%")
    with col2:
        st.metric("Text Length", len(best_text))
    with col3:
        st.metric("Best Filter Used", winning_variant if winning_variant else "None")

    if best_text:
        st.success("✅ Extracted Text")
        # 📋 COPY + DOWNLOAD
        st.code(best_text, language="text")

        st.download_button(
            label="📥 Download as .txt",
            data=best_text,
            file_name="ocr_output.txt",
            mime="text/plain"
        )
    else:
        st.error("No text detected ❌. Try adjusting the lighting or getting closer to the paper.")
