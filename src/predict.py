import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = keras.models.load_model("models/final_model.keras")

# Class labels
class_names = ['immature', 'mature', 'normal', 'pterygium']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    return class_names[class_id], confidence