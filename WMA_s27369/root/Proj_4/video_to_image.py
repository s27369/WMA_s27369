import cv2
import os

VIDEO_DIR = './media/video'
IMAGE_DIR = './media/images'

FRAME_INTERVAL = 5
if __name__=="__main__":
    for filename in os.listdir(VIDEO_DIR):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            person_name = os.path.splitext(filename)[0]
            video_path = os.path.join(VIDEO_DIR, filename)
            output_dir = os.path.join(IMAGE_DIR, person_name)
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_INTERVAL == 0:
                    image_path = os.path.join(output_dir, f"{person_name}_{saved_count:04d}.jpg")
                    cv2.imwrite(image_path, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"Saved {saved_count} frames for {person_name}")

    print("Done.")
