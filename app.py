import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart OCR", page_icon="🧠")

st.title("🧠 Smart OCR (Improved Accuracy)")
st.markdown("Accurate • Fast • Clean UI 🚀")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# =========================
# PREPROCESSING (IMPROVED)
# =========================
def preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize (important for accuracy)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Variant 1: original gray
    v1 = gray

    # Variant 2: threshold
    _, v2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Variant 3: adaptive threshold
    v3 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return [v1, v2, v3]

# =========================
# OCR MULTI PASS
# =========================
def extract_text(img):

    variants = preprocess_variants(img)

    best_text = ""
    max_len = 0

    for v in variants:
        try:
            result = reader.readtext(v, detail=1, paragraph=True)

            texts = []
            for r in result:
                try:
                    texts.append(r[1])
                except:
                    continue

            text = " ".join(texts)
            text = re.sub(r'\s+', ' ', text)

            # Choose best result (longest = usually best)
            if len(text) > max_len:
                best_text = text
                max_len = len(text)

        except:
            continue

    return best_text

# =========================
# INPUT UI (FIXED CAMERA)
# =========================
st.subheader("📤 Input")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)

elif option == "Use Camera":
    camera = st.camera_input("Take Photo")
    if camera:
        image = Image.open(camera)

# =========================
# PROCESS
# =========================
if image is not None:

    st.image(image, caption="Input Image", use_container_width=True)

    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Extracting text..."):
        text = extract_text(img)

    st.success("✅ Done!")

    st.subheader("📄 Extracted Text")
    st.code(text if text else "⚠ No text detected")

    st.download_button("📥 Download Text", text, file_name="output.txt")

else:
    st.info("👆 Select input method to start")
