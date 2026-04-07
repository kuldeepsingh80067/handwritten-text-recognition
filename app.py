import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

st.set_page_config(page_title="Advanced OCR", layout="centered")
st.title("🚀 Handwritten Text Recognition")

# -----------------------------
# 📷 INPUT OPTIONS
# -----------------------------
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image)

# -----------------------------
# LOAD OCR
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if image is not None:

    st.image(image, caption="Input Image", use_column_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # 🔥 MULTI PREPROCESSING
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = []

    # Variant 1
    _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(t1)

    # Variant 2
    t2 = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    variants.append(t2)

    # Variant 3
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, t3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(t3)

    st.subheader("🧠 Processing Variants")
    st.image(variants, width=200)

    # -----------------------------
    # 🤖 OCR BEST RESULT
    # -----------------------------
    best_text = ""
    best_score = 0
    best_boxes = None

    with st.spinner("Analyzing..."):
        for v in variants:
            result = reader.readtext(v)

            text = " ".join([r[1] for r in result])
            score = sum([r[2] for r in result]) if result else 0

            if score > best_score:
                best_score = score
                best_text = text
                best_boxes = result

    # Clean text
    best_text = re.sub(r'[^a-zA-Z0-9 ]', '', best_text)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success("✅ Prediction:")
    st.write(best_text if best_text else "No text detected ❌")

    st.info(f"Confidence Score: {round(best_score, 2)}")

    # -----------------------------
    # DRAW BOXES
    # -----------------------------
    if best_boxes:
        drawn = img.copy()
        for (bbox, txt, conf) in best_boxes:
            pts = np.array(bbox).astype(int)
            cv2.polylines(drawn, [pts], True, (0,255,0), 2)

        drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        st.image(drawn, caption="Detected Text Regions")

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.download_button(
        "📥 Download Text",
        data=best_text,
        file_name="output.txt",
        mime="text/plain"
    )
