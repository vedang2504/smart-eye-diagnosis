import gradio as gr
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = keras.models.load_model("models/final_model.h5")

class_names = ['immature', 'mature', 'normal', 'pterygium']

def predict(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    return f"{class_names[class_id]} ({confidence:.2f})"

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Eye Disease Detection System",
    description="Detects Cataract stages and Pterygium"
)

app.launch()