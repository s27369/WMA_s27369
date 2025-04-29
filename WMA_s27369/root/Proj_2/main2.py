
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
    
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)

def hsv_median():
    global image
    _, __, ksize = read_ksize_trackbars('obrazek')
    mask = get_mask()
    mask = np.asarray(mask[0], dtype=np.uint8)
    res = cv2.bitwise_and(image, image, mask=mask)
    
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

def morphology():  
    global image
    mask, ksize = get_mask()
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)

def morphology2():  
    global image
    mask, ksize = get_mask()
    kernel = np.ones((ksize, ksize), np.uint8)
    
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
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('mask 1', res)
    lower = np.array([0, 100, 100])
    upper = np.array([ksize, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask2)
    cv2.imshow('mask 2', res)
    b_mask = cv2.bitwise_or(mask, mask2)
    res = cv2.bitwise_and(image, image, mask=b_mask)
    cv2.imshow('obrazek', res)

def find_circle():
    low_color, high_color, ksize = read_ksize_trackbars('obrazek')
    c_img = image.copy()
    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (ksize, ksize))
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, high_color, low_color)
    print(circles)
    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0, :]:
        cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.imshow('obrazek', c_img)

def line():
    global image
    low_color, high_color = read_trackbars('obrazek')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=100, maxLineGap=5)
    image_l = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("obrazek", image_l)

def rotate():
    global image
    rot, _ = read_trackbars('obrazek')
    height, width = image.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    cv2.imshow('obrazek', rotated_image)

def detect_circles():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
                               param1=51, param2=38, minRadius=10, maxRadius=0)
    return circles

def detect_tray():
    global image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    tray_contour = None
    if contours:
        tray_contour = max(contours, key=cv2.contourArea)
        tray_area = cv2.contourArea(tray_contour)
        cv2.drawContours(result, [tray_contour], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(tray_contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, f"Tray Area: {tray_area:.0f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Tray Detection', result)
    return tray_contour


def detect_and_classify_coins():
    global image
    result = image.copy()
    tray_contour = detect_tray()
    if tray_contour is None:
        print("Tray not detected, cannot classify coins")
        return
    tray_area = cv2.contourArea(tray_contour)
    circles = detect_circles()
    if circles is None:
        print("No coins detected")
        return
    circles = np.uint16(np.around(circles))
    threshold_ratio = 0.02
    for i in circles[0, :]:
        
        coin_area = np.pi * (i[2] ** 2)

        
        area_ratio = coin_area / tray_area

        
        if area_ratio < threshold_ratio:
            
            cv2.circle(result, (i[0], i[1]), i[2], (0, 165, 255), 2)  
            cv2.putText(result, "5gr", (i[0] - 20, i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            
            cv2.circle(result, (i[0], i[1]), i[2], (255, 0, 0), 2)  
            cv2.putText(result, "5zl", (i[0] - 20, i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        
        cv2.putText(result, f"{area_ratio:.4f}", (i[0] - 20, i[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('Coin Classification', result)


def analyze_coins():
    global image
    result = image.copy()
    tray_contour = detect_tray()
    if tray_contour is None:
        print("Tray not detected, cannot analyze coins")
        return

    tray_area = cv2.contourArea(tray_contour)
    circles = detect_circles()

    if circles is None:
        print("No coins detected")
        return

    circles = np.uint16(np.around(circles))
    threshold_ratio = 0.02
    total_coin_area = 0
    count_5gr = 0
    count_5zl = 0
    count_5gr_in_tray = 0
    count_5zl_in_tray = 0
    total_5zl_area = 0

    tray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(tray_mask, [tray_contour], 0, 255, -1)

    for i in circles[0, :]:
        coin_area = np.pi * (i[2] ** 2)
        total_coin_area += coin_area
        area_ratio = coin_area / tray_area
        is_in_tray = tray_mask[i[1], i[0]] > 0
        if area_ratio < threshold_ratio:
            count_5gr += 1
            if is_in_tray:
                count_5gr_in_tray += 1
            cv2.circle(result, (i[0], i[1]), i[2], (0, 165, 255), 2)
            cv2.putText(result, "5gr", (i[0] - 20, i[1]),
                        cv2.QT_FONT_NORMAL, 0.6, (0, 165, 255), 2)
        else:
            count_5zl += 1
            total_5zl_area += coin_area
            if is_in_tray:
                count_5zl_in_tray += 1
            cv2.circle(result, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.putText(result, "5zl", (i[0] - 20, i[1]),
                        cv2.QT_FONT_NORMAL, 0.6, (255, 0, 0), 2)

    money_in_tray = count_5gr_in_tray * 0.05 + count_5zl_in_tray * 5.0
    money_outside_tray = (count_5gr - count_5gr_in_tray) * 0.05 + (count_5zl - count_5zl_in_tray) * 5.0

    ratio_5zl_to_tray = total_5zl_area / tray_area if tray_area > 0 else 0

    y_pos = 30
    cv2.putText(result, f"Total Coin Area: {total_coin_area:.2f}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(result, f"5gr coins: {count_5gr}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    y_pos += 30
    cv2.putText(result, f"5zl coins: {count_5zl}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    y_pos += 30
    cv2.putText(result, f"5zl/Tray ratio: {ratio_5zl_to_tray:.4f}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_pos += 30
    cv2.putText(result, f"Money in tray: {money_in_tray:.2f} zl", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_pos += 30
    cv2.putText(result, f"Money outside: {money_outside_tray:.2f} zl", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow('Coin Analysis', result)

    return {
        'total_coin_area': total_coin_area,
        'count_5gr': count_5gr,
        'count_5zl': count_5zl,
        'ratio_5zl_to_tray': ratio_5zl_to_tray,
        'money_in_tray': money_in_tray,
        'money_outside_tray': money_outside_tray
    }


def calculate_total_coin_area():
    results = analyze_coins()
    if results:
        return results['total_coin_area']
    return 0


def count_coins_by_type():
    results = analyze_coins()
    if results:
        return results['count_5gr'], results['count_5zl']
    return 0, 0


def calculate_5zl_to_tray_ratio():
    results = analyze_coins()
    if results:
        return results['ratio_5zl_to_tray']
    return 0


def calculate_money_distribution():
    results = analyze_coins()
    if results:
        return results['money_in_tray'], results['money_outside_tray']
    return 0, 0



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
    
        if key >= ord('0') and key <= ord('9'):
            upload(key, file_path)
            nimg = image.copy()
    
        elif key == ord('-'):
            resize()
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            cv2.imshow('obrazek', image)
            nimg = image.copy()
    
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
            
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            
            cv2.imshow('obrazek', nimg[:, :, 2])
    
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
        elif key == ord('n'):
            analyze_coins()
            fun = analyze_coins
        elif key == ord('m'):
            in_tray, outside_tray = calculate_money_distribution()
            print(f"Money in tray: {in_tray:.2f} zl, Money outside tray: {outside_tray:.2f} zl")

        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
