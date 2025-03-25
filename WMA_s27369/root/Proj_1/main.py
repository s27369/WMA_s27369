# from functions import *
import cv2
import os
import numpy as np

file_path = r"E:\PYTHON\\WMA_s27369\WMA_s27369\root\Proj_1\media"

def upload(i, path, files):
    image_path = os.path.join(path, files[i])
    image = cv2.imread(image_path)
    print("loading")
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    else:
        print("success ")
    return norm_size(image)


def resize():
    global image
    h, w = image.shape[:2]
    h = h + int(h * (-0.1))
    w = w + int(w * (-0.1))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)
    return image


def norm_size(image):
    h, w = image.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800 / h)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        if w > 800:
            s = (1 - (800 / w)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)
    return image


def read_trackbars(img: str) -> (int, int):
    low_color = cv2.getTrackbarPos('low', img)
    high_color = cv2.getTrackbarPos('high', img)
    return (low_color, high_color)


def read_ksize_trackbars(img: str) -> (int, int, int):
    low_color, high_color = read_trackbars(img)
    ksize = cv2.getTrackbarPos('ksize', img)
    return (low_color, high_color, ksize)


def get_lower_upper(low_color, high_color):
    return (np.array([low_color, 100, 100]), np.array([high_color, 255, 255]))


def get_mask() -> np.uint8:
    global image
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_color, high_color, ksize = read_ksize_trackbars('obrazek')
    lower, upper = get_lower_upper(low_color, high_color)
    return cv2.inRange(hsv_frame, lower, upper), ksize


def hsv_range():
    global image
    mask, _ = get_mask()
    cv2.imshow('obrazek', mask)

def hsv_bitwise():
    global image
    mask = get_mask()
    mask = np.asarray(mask[0], dtype=np.uint8)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)

def hsv_median():
    global image
    _, __, ksize = read_ksize_trackbars('obrazek')
    mask = get_mask()
    mask = np.asarray(mask[0], dtype=np.uint8)
    res = cv2.bitwise_and(image, image, mask=mask)
    #bo sie crashowalo
    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1
    print(ksize)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)

def change_h(x):
    global fun
    if fun is not None:
        fun()

def morphology():  # open
    global image
    mask, ksize = get_mask()
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)


def morphology2():  # close
    global image
    mask, ksize = get_mask()
    kernel = np.ones((ksize, ksize), np.uint8)
    # mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)

def marker():
    global image
    mask, _ = get_mask()
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    print(contours)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / (M['m00'] if M['m00']!=0 else 1))
    cy = int(M['m01'] / (M['m00'] if M['m00']!=0 else 1))
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)

#----------------------------------------------------------------------------------------------------------
def track_ball():
    global image
    # Konwersja do HSV
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Pobierz wartości z suwaków
    low_color, high_color, ksize = read_ksize_trackbars('obrazek')

    # Utworzenie maski
    lower, upper = get_lower_upper(low_color, high_color)
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Operacje morfologiczne - otwarcie aby usunąć szum
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Znajdź kontury i oblicz środek
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / (M['m00'] if M['m00'] != 0 else 1))
        cy = int(M['m01'] / (M['m00'] if M['m00'] != 0 else 1))

        # Dodaj marker do oryginalnego obrazu
        image_marker = image.copy()
        cv2.drawMarker(image_marker, (cx, cy), color=(0, 255, 0),
                       markerType=cv2.MARKER_CROSS, thickness=2)

        cv2.imshow('obrazek', image_marker)
#----------------------------------------------------------------------------------------------------------


image = None
fun = None
files = None

def main():
    global image, fun, files
    files = os.listdir(file_path)
    if not files:
        print("No files found in the directory.")
        return
    image = upload(0, file_path, files)
    print(image)
    nimg = image.copy()
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
    # -----------wybor obrazka----------------
        if key >= ord('0') and key <= ord('9'):
            upload(key, file_path, image)
            nimg = image.copy()
    # ----------------zmiana rozmiaru---------------
        elif key == ord('-'):
            resize()
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            cv2.imshow('obrazek', image)
            nimg = image.copy()
    # ----------------kolory------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        elif key == ord('r'):
            hsv_bitwise()
            fun = hsv_bitwise
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('z'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
    # ----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(image, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == ord('f'):
            morphology()
            fun = morphology
        elif key == ord('g'):
            morphology2()
            fun = morphology2
        elif key == ord('h'):
            marker()
            fun = marker

        elif key == ord('b'):
            track_ball()
            fun = track_ball

        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
