import cv2
import numpy as np
import tensorflow as tf
import os

# ===== PATH =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")

# ===== LOAD MODEL =====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== CLASS NAMES =====
class_names = ['immature', 'mature', 'normal', 'pterygium']

# ===== WEBCAM =====
cap = cv2.VideoCapture(0)

print("📷 Webcam started... Press 'q' to quit")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Preprocess
    img = cv2.resize(frame, (160, 160))  # 🔥 optimized
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    confidence = np.max(output)

    label = f"{class_names[pred]} ({confidence:.2f})"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.imshow("Smart Eye Diagnosis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()