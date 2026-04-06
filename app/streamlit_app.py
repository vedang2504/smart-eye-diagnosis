import streamlit as st
import numpy as np
import cv2
import os
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Smart Eye Diagnosis", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.glass {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}

.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 50px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")

model = keras.models.load_model(MODEL_PATH)

class_names = ['immature', 'mature', 'normal', 'pterygium']

disease_info = {
    "immature": "Early stage cataract. Regular monitoring recommended.",
    "mature": "Advanced cataract. Surgical consultation required.",
    "normal": "No visible abnormality detected.",
    "pterygium": "Tissue growth over eye surface. May require treatment."
}

# ================== NAVBAR ==================
menu = st.radio(
    "",
    ["Home", "Prediction", "About", "Contact"],
    horizontal=True
)

# ================== HOME ==================
if menu == "Home":
    st.markdown('<div class="title">👓 Smart Eye Diagnosis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
    <h3>AI-Powered Eye Disease Detection</h3>
    <p>
    Detect Cataract (Immature & Mature), Pterygium, and Normal Eye Conditions
    using Deep Learning.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ This is an AI screening tool, not a medical diagnosis.")

# ================== PREDICTION ==================
elif menu == "Prediction":
    st.title("🔍 Eye Disease Detection")

    uploaded_file = st.file_uploader("📤 Upload Eye Image", type=["jpg", "png", "jpeg"])

    use_camera = st.checkbox("Use Camera")

    img = None

    if use_camera:
        camera_img = st.camera_input("📷 Capture Image")

        if camera_img:
            img = cv2.imdecode(np.frombuffer(camera_img.read(), np.uint8), 1)

    img = None

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

    elif camera_img:
        img = cv2.imdecode(np.frombuffer(camera_img.read(), np.uint8), 1)

    if img is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Input Image", use_column_width=True)

        with st.spinner("🔍 Analyzing..."):
            img_resized = cv2.resize(img, (224, 224))
            img_processed = preprocess_input(img_resized)
            img_processed = np.expand_dims(img_processed, axis=0)

            preds = model.predict(img_processed)
            class_id = np.argmax(preds)
            confidence = np.max(preds)

            label = class_names[class_id]

        with col2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)

            st.markdown(f"### 🧠 Prediction: **{label.upper()}**")
            st.progress(float(confidence))

            st.write(f"Confidence: {confidence*100:.2f}%")

            if confidence < 0.7:
                st.warning("⚠️ Low confidence result")

            st.info(disease_info[label])

            st.markdown('</div>', unsafe_allow_html=True)

        # ================== CHART ==================
        st.subheader("📊 Confidence Distribution")
        st.bar_chart(preds[0])

        # ================== REPORT ==================
        if st.button("📄 Download Report"):
            report = f"""
Smart Eye Diagnosis Report
--------------------------
Date: {datetime.now()}

Prediction: {label}
Confidence: {confidence*100:.2f}%

Note:
This is an AI-based screening tool.
Consult a doctor for confirmation.
"""
            st.download_button("Download Report", report, file_name="eye_report.txt")

# ================== ABOUT ==================
elif menu == "About":
    st.title("📘 About")

    st.markdown("""
    <div class="glass">
    <p>
    This system uses:
    <ul>
    <li>EfficientNet Deep Learning Model</li>
    <li>Computer Vision (OpenCV)</li>
    <li>Streamlit UI</li>
    </ul>
    Built for real-time eye disease screening using low-cost systems.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ================== CONTACT ==================
elif menu == "Contact":
    st.title("📞 Contact")

    st.markdown("""
    <div class="glass">
    <p>
    Developer: Vedang Doley<br>
    Project: Smart Eye Diagnosis System<br>
    Email: your-email@example.com
    </p>
    </div>
    """, unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
© 2026 Smart Eye Diagnosis System | Built by Vedang Doley
</div>
""", unsafe_allow_html=True)