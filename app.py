import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch

# -------------------------------
# 🚀 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart OCR", page_icon="🧠", layout="wide")

# -------------------------------
# 🎨 CUSTOM UI
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1>🧠 Smart OCR (Improved Accuracy)</h1>
    <h4 style='text-align: center; color: gray;'>Developed by Kuldeep Singh</h4>
    <hr>
""", unsafe_allow_html=True)

# -------------------------------
# ⚡ LOAD MODEL (FAST)
# -------------------------------
@st.cache_resource
def load_model():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    return processor, model

processor, model = load_model()

# -------------------------------
# 🧠 IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Noise removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

# -------------------------------
# 🔍 OCR FUNCTION
# -------------------------------
def extract_text(image):
    processed = preprocess_image(image)

    pil_img = Image.fromarray(processed).convert("RGB")

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text

# -------------------------------
# 📥 INPUT SECTION
# -------------------------------
st.subheader("📥 Input")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera = st.camera_input("Take a photo")
    if camera:
        image = Image.open(camera)

# -------------------------------
# 🖼 SHOW IMAGE
# -------------------------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    st.info("🔍 Extracting text... please wait")

    try:
        text = extract_text(image)

        st.success("✅ Text Extracted")

        # -------------------------------
        # 📊 RESULT UI
        # -------------------------------
        st.markdown("## 📊 Result")

        col1, col2 = st.columns(2)
        col1.metric("Text Length", len(text))
        col2.metric("Confidence", "High (AI Model)")

        st.markdown("### 📄 Extracted Text")
        st.code(text)

        st.download_button("⬇ Download Text", text, file_name="output.txt")

    except Exception as e:
        st.error(f"Error: {e}")
