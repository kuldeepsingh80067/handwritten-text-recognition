import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

# Title
st.set_page_config(page_title="Handwritten Text Recognition", layout="centered")
st.title("✍️ Handwritten Text Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy
    img = np.array(image)

    # Convert RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # 🔥 PREPROCESSING (IMPORTANT)
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilation (improves thin handwriting)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Show processed image (optional)
    st.image(thresh, caption="Processed Image", use_column_width=True)

    # -----------------------------
    # 🤖 OCR
    # -----------------------------
    with st.spinner("Reading text..."):
        reader = easyocr.Reader(['en'], gpu=False)

        result = reader.readtext(
            thresh,
            detail=0,
            paragraph=True
        )

    # Join text
    text = " ".join(result)

    # Clean text
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # -----------------------------
    # ✅ OUTPUT
    # -----------------------------
    st.success("Prediction:")
    st.write(text if text else "No text detected ❌")

    # -----------------------------
    # 💾 DOWNLOAD BUTTON
    # -----------------------------
    st.download_button(
        label="Download Text",
        data=text,
        file_name="output.txt",
        mime="text/plain"
    )
