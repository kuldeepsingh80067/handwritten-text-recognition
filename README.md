# ✍️ Handwritten Text Recognition Web App

## 📌 Overview

This project is a web-based application that extracts handwritten text from images using Optical Character Recognition (OCR). It allows users to upload an image containing handwritten or printed text and converts it into machine-readable text in real time.

The application is built using Python and provides an interactive user interface powered by Streamlit.

---

## 🚀 Features

* 📤 Upload handwritten or printed text images
* 🧠 Extract text using EasyOCR
* 🖼️ Image preprocessing for improved accuracy
* 📋 Display extracted text instantly
* 📥 Download extracted text as a file
* 🌐 Interactive and user-friendly web interface

---

## 🛠️ Tech Stack

* Python
* Streamlit
* EasyOCR
* OpenCV
* NumPy
* Pillow

---

## ⚙️ How It Works

1. User uploads an image through the web interface
2. The image is preprocessed (resizing, grayscale conversion, thresholding)
3. EasyOCR processes the image to detect and recognize text
4. Extracted text is displayed on the screen
5. User can copy or download the output

---

## ▶️ Run Locally

### Step 1: Clone the repository

```
git clone https://github.com/your-username/handwritten-text-recognition.git
cd handwritten-text-recognition
```

### Step 2: Install dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the app

```
streamlit run app.py
```

---

## 🌐 Deployment

This project can be easily deployed using Streamlit Community Cloud by connecting your GitHub repository and selecting `app.py` as the entry point.

---

## ⚠️ Limitations

* Works best with clear and well-lit handwriting
* Accuracy may decrease with messy or low-quality images
* OCR-based approach is not as powerful as deep learning models trained specifically for handwriting

---

## 💼 Future Improvements

* Integrate deep learning-based handwritten text recognition model (CRNN)
* Support multiple languages
* Add real-time camera input
* Improve preprocessing for better accuracy

---

## 👨‍💻 Author

Kuldeep Singh

---

## ⭐ Conclusion

This project demonstrates how OCR and image processing techniques can be combined to build a functional handwritten text recognition system with a simple and interactive interface.
