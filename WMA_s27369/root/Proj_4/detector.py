import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os, json

model = load_model("model_best.keras")

class_names = sorted(os.listdir("media/images"))
with open("label_map.json") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
IMG_SIZE = (100, 100)

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        # Predict
        preds = model.predict(face_input)
        class_idx = int(np.argmax(preds))
        label = inv_label_map[class_idx]
        confidence = np.max(preds)

        label = f"{class_names[class_idx]} ({confidence*100:.1f}%)"

        # Draw
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(orig, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition", orig)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
