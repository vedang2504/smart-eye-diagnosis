import numpy as np
import cv2
import tensorflow as tf
from picamera2 import Picamera2

# ===== LOAD TFLITE MODEL =====
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== CLASS NAMES =====
class_names = ['immature', 'mature', 'normal', 'pterygium']

# ===== CAMERA SETUP =====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("📷 Camera started... Press 'q' to quit")

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Preprocess
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32)

    # Normalize (IMPORTANT)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # ===== INFERENCE =====
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    confidence = np.max(output)

    label = f"{class_names[pred]} ({confidence:.2f})"

    # Display result
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Smart Eye Diagnosis", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()