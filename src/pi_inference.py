import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['immature', 'mature', 'normal', 'pterygium']

# Load image
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))

# Normalize (IMPORTANT)
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)

# Set input
interpreter.set_tensor(input_details[0]['index'], img)

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])

pred = np.argmax(output)
confidence = np.max(output)

print("Prediction:", class_names[pred])
print("Confidence:", confidence)