import tensorflow as tf
import os

# ====== PATH SETUP ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")
TFLITE_PATH = os.path.join(BASE_DIR, "models", "model.tflite")

print("🔍 Loading model from:", MODEL_PATH)

# ====== LOAD MODEL ======
model = tf.keras.models.load_model(MODEL_PATH)

# ====== CONVERT ======
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔥 Optimization (IMPORTANT for Raspberry Pi)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

print("⚙️ Converting to TFLite...")

tflite_model = converter.convert()

# ====== SAVE ======
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved at:", TFLITE_PATH)