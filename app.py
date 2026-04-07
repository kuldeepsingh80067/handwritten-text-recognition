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
    <p style='text-align: center; color: gray;'>AI-powered OCR with enhanced accuracy</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# 📥 INPUT SECTION
# -----------------------------
st.subheader("📥 Input")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if file:
        image = Image.open(file)

elif option == "Use Camera":
    cam = st.camera_input("Take photo")
    if cam:
        image = Image.open(cam)

# -----------------------------
# 🤖 LOAD OCR
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# 🧠 PROCESSING
# -----------------------------
if image:

    st.divider()
    st.subheader("🖼️ Image Preview")
    st.image(image, use_column_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize for stability
    img = cv2.resize(img, (800, 600))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 CLAHE (contrast boost)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    variants = []

    # Variant 1
    _, t1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(t1)

    # Variant 2
    t2 = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(t2)

    # Variant 3
    t3 = cv2.bitwise_not(t1)
    variants.append(t3)

    # Variant 4
    kernel = np.ones((2,2), np.uint8)
    t4 = cv2.dilate(t1, kernel, iterations=1)
    variants.append(t4)

    st.subheader("🧠 Image Processing")
    st.image(variants, width=150)

    # -----------------------------
    # OCR BEST RESULT
    # -----------------------------
    best_text = ""
    best_score = 0

    with st.spinner("🔍 Extracting text..."):
        for v in variants:
            result = reader.readtext(v)

            text = " ".join([r[1] for r in result])
            score = sum([r[2] for r in result]) if result else 0

            if score > best_score:
                best_score = score
                best_text = text

    # Clean text
    best_text = re.sub(r'[^a-zA-Z0-9 ]', '', best_text)

    # -----------------------------
    # 📊 OUTPUT
    # -----------------------------
    st.divider()
    st.subheader("📊 Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Confidence Score", round(best_score, 2))

    with col2:
        st.metric("Text Length", len(best_text))

    st.success("✅ Extracted Text")
    st.write(best_text if best_text else "No text detected ❌")

    # -----------------------------
    # 📋 COPY + DOWNLOAD
    # -----------------------------
    st.code(best_text)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📥 Download",
            best_text,
            "output.txt"
        )

    with col2:
        st.button("📋 Copy (Ctrl+C from above)")

