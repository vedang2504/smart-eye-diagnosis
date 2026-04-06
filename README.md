# 👓 Smart Eye Diagnosis System

An AI-powered system for detecting eye diseases such as Cataract (Mature & Immature) and Pterygium using Deep Learning and Computer Vision.

---

## 🚀 Features

- 🔍 Detects:
  - Immature Cataract
  - Mature Cataract
  - Pterygium
  - Normal Eye

- 📷 Image Upload + Camera Input
- 📊 Confidence Score Visualization
- 📄 Downloadable Report
- 🌐 Streamlit Web Interface
- ⚡ Fast and Lightweight Model (EfficientNet)

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy

---

## 📁 Project Structure
smart-eye-diagnosis/
│
├── app/ # Streamlit app
├── src/ # Prediction logic
├── data/ # Dataset (not uploaded)
├── models/ # Trained models
├── notebooks/ # Training notebooks
├── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/smart-eye-diagnosis.git
cd smart-eye-diagnosis

conda create -n eye_env python=3.10
conda activate eye_env

pip install -r requirements.txt

▶️ Run Application

streamlit run app/streamlit_app.py

📊 Model Details

Model: EfficientNetB0
Transfer Learning
Data Augmentation applied
Multi-class classification (4 classes)

📚 Dataset Sources

⚠️ Datasets are not included due to size and privacy concerns.

You can download datasets from:

🔗 https://www.kaggle.com/datasets/rifdana/cataract-photo-image-dataset

🔗 https://www.kaggle.com/datasets/linabennaa/eye-disease-image-dataset-mendeley

🔗 https://universe.roboflow.com/eyescareworkspace/pterygium-lqtlf

## 📸 Project Screenshots

### 🖥️ User Interface
![UI](assets/ui1.png)

![UI](assets/ui2.png)

### 📷 Real-time Detection
![Webcam](assets/webcam.png)

### 🔥 Grad-CAM Visualization
![GradCAM](assets/gradcam.png)

### 📊 Confusion Matrix
![Confusion](assets/confusion_matrix.png)

### 📈 ROC Curve
![ROC](assets/roc_curve.png)

### Precision vs Recall
![ROC](assets/precision_vs_recall.png)

### accuracy graph
![ROC](assets/accuracy_graph.png)

### loss graph
![ROC](assets/loss_graph.png)