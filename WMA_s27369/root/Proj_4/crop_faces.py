import cv2
import os

# Settings
IMG_SIZE = (100, 100)
INPUT_DIR = "media/images"
OUTPUT_DIR = "media/faces"
DEBUG = True  # Set to True to see debug windows

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a more robust cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

for class_name in os.listdir(INPUT_DIR):
    input_class_dir = os.path.join(INPUT_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for fname in os.listdir(input_class_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_class_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tweak parameters
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"[!] No face in {img_path}")
            if DEBUG:
                cv2.imshow("No face", img)
                cv2.waitKey(0)
            continue

        # Take the largest detected face
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        pad = int(0.2 * w)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])

        face = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face, IMG_SIZE)

        out_path = os.path.join(output_class_dir, fname)
        cv2.imwrite(out_path, face_resized)

        if DEBUG:
            cv2.imshow("Detected", face_resized)
            if cv2.waitKey(1) == 27:
                break

if DEBUG:
    cv2.destroyAllWindows()

print("Cropping done. Saved to", OUTPUT_DIR)
