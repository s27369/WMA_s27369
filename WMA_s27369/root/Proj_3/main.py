import glob

import cv2
import numpy as np
import os

import hashlib

def get_image_hash(img):
    _, buffer = cv2.imencode('.jpg', img)
    return hashlib.md5(buffer).hexdigest()[:10]


def load_image(i):
    global image
    image = cv2.imread(os.path.join(file_path, files[i - ord('0')]))
    normalize_image_size()

def resize():
    global image
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

def normalize_image_size():
    global image
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 800:
        scale = 800 / max_dim
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

def get_hsv_mask():
    low = cv2.getTrackbarPos('low', 'obrazek')
    high = cv2.getTrackbarPos('high', 'obrazek')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (low, 100, 100), (high, 255, 255))

def apply_mask():
    mask = get_hsv_mask()
    cv2.imshow('obrazek', mask)

def bitwise_and_mask():
    mask = get_hsv_mask()
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', result)

def median_filter():
    mask = get_hsv_mask()
    k = cv2.getTrackbarPos('ksize', 'obrazek')
    if k%2!=1: k+=1
    filtered = cv2.bitwise_and(image, image, mask=mask)
    filtered = cv2.medianBlur(filtered, k)
    cv2.imshow('obrazek', filtered)

def apply_morphology(operation):
    mask = get_hsv_mask()
    k = cv2.getTrackbarPos('ksize', 'obrazek')
    kernel = np.ones((k, k), np.uint8)
    result = cv2.morphologyEx(mask, operation, kernel)
    cv2.imshow('obrazek', result)

def draw_marker():
    mask = get_hsv_mask()
    contours, _ = cv2.findContours(mask, 1, 2)
    if contours:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            marked = image.copy()
            cv2.drawMarker(marked, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, thickness=2)
            cv2.imshow('obrazek', marked)

def rotate():
    angle = cv2.getTrackbarPos('low', 'obrazek')
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imshow('obrazek', rotated)

def cut_fragment():
    global image2
    y = cv2.getTrackbarPos('low', 'obrazek') * 2
    x = cv2.getTrackbarPos('high', 'obrazek') * 2
    size = max(10, cv2.getTrackbarPos('ksize', 'obrazek') * 2)
    image2 = image[y:y + size // 2, x:x + size]
    cv2.imshow('obrazek', image2)
    filename = f'klucz_{get_image_hash(image2)}.jpg'
    cv2.imwrite(file_path+"\\"+filename, image2)
    print(f"Zapisano: {filename}")

def cut_from_mask():
    global image2
    mask = get_hsv_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        image2 = image[y:y + h, x:x + w]

        filename = f'klucz_{get_image_hash(image2)}.jpg'
        cv2.imwrite(filename, image2)
        print(f"Zapisano zdjęcie-klucz jako {filename}")
        cv2.imshow('obrazek', image2)
    else:
        print("Brak konturów – popraw HSV lub użyj większego zakresu.")

def cut_from_mask():
    global image2
    mask = get_hsv_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        image2 = image[y:y + h, x:x + w]

        filename = f'klucz_{get_image_hash(image2)}.jpg'
        cv2.imwrite(filename, image2)
        print(f"Zapisano zdjęcie-klucz jako {filename}")
        cv2.imshow('obrazek', image2)
    else:
        print("Brak konturów – popraw HSV lub użyj większego zakresu.")

def draw_sift(show_matches=False, use_blur=False):
    global image, image2
    sift = cv2.SIFT_create()
    img1 = image2.copy() if show_matches else image.copy()
    img2 = image.copy()
    gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if use_blur:
        k = cv2.getTrackbarPos('ksize', 'obrazek')
        if k % 2 == 0:
            k += 1
        gimg1 = cv2.medianBlur(gimg1, k)
        gimg2 = cv2.medianBlur(gimg2, k)

    kp1, des1 = sift.detectAndCompute(gimg1, None)
    kp2, des2 = sift.detectAndCompute(gimg2, None)
    if show_matches:
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        matched = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        cv2.imshow('obrazek', matched)
    else:
        out = cv2.drawKeypoints(img1, kp1, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('obrazek', out)

def draw_orb():
    global image, image2
    orb = cv2.ORB_create()
    gimg1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gimg1, None)
    kp2, des2 = orb.detectAndCompute(gimg2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    matched = cv2.drawMatches(image2, kp1, image, kp2, matches, image, flags=2)
    cv2.imshow('obrazek', matched)

def process_video():
    print("Przetwarzam wideo...")
    video_path = os.path.join(file_path, 'video.mp4')
    cap = cv2.VideoCapture(video_path)

    # załaduj wszystkie klucze
    key_images = []
    for file in glob.glob('klucz_*.jpg'):
        img = cv2.imread(file)
        if img is not None:
            key_images.append((file, img))

    if not key_images:
        print("Brak zdjęć-kluczy! Zrób je (klawisz 'v').")
        return

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    frame_num = 0
    match_stats = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(g_frame, None)

        best_match_count = -1
        best_result = frame.copy()
        best_key_name = None

        for name, key_img in key_images:
            g_key = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(g_key, None)
            if des1 is None or des2 is None:
                continue
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > best_match_count:
                best_match_count = len(matches)
                best_key_name = name
                best_result = cv2.drawMatches(key_img, kp1, frame, kp2, matches[:20], None, flags=2)

        match_stats[frame_num] = (best_key_name, best_match_count)
        frame_num += 1

        cv2.imshow('obrazek', best_result)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    print("\\n📊 Statystyki dopasowań (pierwsze 10 klatek):")
    for i in range(min(10, len(match_stats))):
        print(f"Klatka {i}: {match_stats[i][0]} ({match_stats[i][1]} dopasowań)")

def change_callback(x):
    if fun is not None:
        fun()

image = None
image2 = None
fun = None
files = None
file_path = r'./media'
# photo_path = file_path+'/photos'
# video_path = file_path+'/video'

def main():
    global image, fun, files
    files = os.listdir(file_path)
    load_image(ord('0'))
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_callback)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_callback)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_callback)

    while True:
        key = cv2.waitKey()
        if ord('0') <= key <= ord('9'):
            load_image(key)
        elif key == ord('-'):
            resize()
        elif key == ord('='):
            cv2.imshow('obrazek', image)
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        elif key == ord('e'):
            fun = apply_mask
            apply_mask()
        elif key == ord('r'):
            fun = bitwise_and_mask
            bitwise_and_mask()
        elif key == ord('t'):
            fun = median_filter
            median_filter()
        elif key == ord('f'):
            fun = lambda: apply_morphology(cv2.MORPH_OPEN)
            fun()
        elif key == ord('g'):
            fun = lambda: apply_morphology(cv2.MORPH_CLOSE)
            fun()
        elif key == ord('h'):
            fun = draw_marker
            draw_marker()
        elif key == ord('o'):
            fun = rotate
            rotate()
        elif key == ord('v'):
            # fun = cut_fragment
            # cut_fragment()
            fun = cut_from_mask
            cut_from_mask()
        elif key == ord('b'):
            fun = lambda: draw_sift(False)
            fun()
        elif key == ord('n'):
            fun = lambda: draw_sift(True)
            fun()
        elif key == ord('m'):
            fun = lambda: draw_sift(True, use_blur=True)
            fun()
        elif key == ord('l'):
            fun = draw_orb
            draw_orb()
        elif key == ord('y'):
            process_video()
        elif key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
