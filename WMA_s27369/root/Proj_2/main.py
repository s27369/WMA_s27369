# from functions import *
import cv2
import os
import numpy as np

file_path = r".\media"

def upload(i, path):
    global image, files
    print(files)
    print(i, type(i))
    if (i>10):
        i -= ord('0')
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

def connect_mask():
    # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    # Konwersja obrazu na przestrzeń kolorów HSV
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Utworzenie maski dla pierwszego zakresu kolorów
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Nałożenie maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('mask 1', res)

    # Utworzenie maski dla drugiego zakresu kolorów
    lower = np.array([0, 100, 100])
    upper = np.array([ksize, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower, upper)

    # Nałożenie drugiej maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=mask2)
    cv2.imshow('mask 2', res)

    # Połączenie dwóch masek za pomocą operacji bitowej OR
    b_mask = cv2.bitwise_or(mask, mask2)

    # Nałożenie połączonej maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=b_mask)
    cv2.imshow('obrazek', res)

# key j
def find_circle():
    # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
    low_color, high_color, ksize = read_ksize_trackbars('obrazek')

    # Utwórz kopię obrazu, aby nie modyfikować oryginału
    c_img = image.copy()

    # Konwersja obrazu na skalę szarości
    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)

    # Zastosowanie rozmycia na obrazie w skali szarości
    bimg = cv2.blur(gimg, (ksize, ksize))

    # Wykrywanie okręgów za pomocą transformacji Hougha
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, high_color, low_color)
    print(circles)  # Wyświetlenie wykrytych okręgów (surowe dane)

    # Zaokrąglenie współrzędnych wykrytych okręgów do liczb całkowitych
    circles = np.uint16(np.around(circles))
    print(circles)  # Wyświetlenie zaokrąglonych współrzędnych okręgów

    # Iteracja po wykrytych okręgach i rysowanie ich na obrazie
    for i in circles[0, :]:
        # Rysowanie okręgu na obrazie (środek: (i[0], i[1]), promień: i[2])
        cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Wyświetlenie obrazu z narysowanymi okręgami
    cv2.imshow('obrazek', c_img)

# key k
def line():
    global image
    # Pobierz wartości progów dolnego i górnego z trackbarów
    low_color, high_color = read_trackbars('obrazek')

    # Konwersja obrazu na skalę szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wykrywanie krawędzi za pomocą algorytmu Canny'ego
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)

    # Wykrywanie linii za pomocą transformacji Hougha
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=100, maxLineGap=5)

    # Utworzenie kopii obrazu, aby narysować linie
    image_l = image.copy()

    # Iteracja po wykrytych liniach i rysowanie ich na obrazie
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Wyświetlenie obrazu z narysowanymi liniami
    cv2.imshow("obrazek", image_l)

# key o
def rotate():
    global image
    # Pobierz wartość kąta obrotu z trackbara o nazwie 'low'
    rot, _ = read_trackbars('obrazek')

    # Pobierz wymiary obrazu
    height, width = image.shape[:2]

    # Oblicz środek obrazu
    center_x, center_y = (width / 2, height / 2)

    # Utwórz macierz transformacji dla obrotu obrazu
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)

    # Zastosuj macierz transformacji, aby obrócić obraz
    rotated_image = cv2.warpAffine(image, M, (width, height))

    # Wyświetl obrócony obraz w oknie o nazwie 'obrazek'
    cv2.imshow('obrazek', rotated_image)
#----------------------------------------------------------------------------zad
def detect_circles():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
                               param1=51, param2=38, minRadius=10, maxRadius=0)
    return circles


def detect_and_classify_coins():
    global image
    circles = detect_circles()
    output = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        radii = []

        for i in circles[0, :]:
            radius = i[2]
            radii.append(radius)
            cv2.circle(output, (i[0], i[1]), radius, (0, 255, 0), 2)

        if radii:
            median_radius = np.median(radii)
            for i, r in zip(circles[0, :], radii):
                label = "5zl" if r > median_radius else "5gr"
                cv2.putText(output, label, (i[0] - 20, i[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("obrazek", output)

def detect_tray():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tray = None
    max_area = 0
    output = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            tray = contour

    if tray is not None:
        rect = cv2.minAreaRect(tray)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (0, 255, 255), 2)
        cv2.putText(output, "Tray", (box[0][0], box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(output, "No tray found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("obrazek", output)



image = None
fun = None
files = None

def main():
    global image, fun, files
    files = os.listdir(file_path)
    if not files:
        print("No files found in the directory.")
        return
    image = upload(0, file_path)
    nimg = image.copy()
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
    # -----------wybor obrazka----------------
        if key >= ord('0') and key <= ord('9'):
            upload(key, file_path)
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
        elif key == ord('p'):
            connect_mask()
            fun = connect_mask
        elif key == ord('o'):
            rotate()
            fun = rotate
    # --------------------krztałty
        elif key == ord('j'):
            find_circle()
            fun = find_circle
        elif key == ord('k'):
            line()
            fun = line
        elif key == ord('v'):
            detect_and_classify_coins()
            fun = detect_and_classify_coins
        elif key == ord('b'):
            detect_tray()
            fun = detect_tray

        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
