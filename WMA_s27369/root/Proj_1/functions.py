import cv2
import os
import numpy as np
def upload(i, path, files):
    image_path = os.path.join(path, files[i])
    image = cv2.imread(image_path)
    print("loading")
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    else:
        print("success "+str(image))
    return norm_size(image)


def resize(image):
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


def get_mask(image) -> np.uint8:
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_color, high_color, ksize = read_ksize_trackbars('obrazek')
    lower, upper = get_lower_upper(low_color, high_color)
    return cv2.inRange(hsv_frame, lower, upper), ksize


def hsv_range(image):
    mask, _ = get_mask(image)
    cv2.imshow('obrazek', mask)

def hsv_bitwise(image):
    mask = get_mask(image)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)

def hsv_median(image):
    _, __, ksize = read_ksize_trackbars('obrazek')
    mask = get_mask(image)
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)

# def morphology(image):  # open
#     mask, ksize = get_mask(image)
#     kernel = np.ones((ksize, ksize), np.uint8)
#     mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('obrazek', mask_without_noise)
#
#
# def morphology2(image):  # close
#     mask, ksize = get_mask(image)
#     kernel = np.ones((ksize, ksize), np.uint8)
#     # mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
#     mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     cv2.imshow('obrazek', mask_closed)


def marker(image):
    mask, _ = get_mask(image)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    print(contours)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / (M['m00'] if M['m00']!=0 else 1))
    cy = int(M['m01'] / (M['m00'] if M['m00']!=0 else 1))
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)