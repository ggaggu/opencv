import numpy as np
import cv2

step = 0
mouse_is_pressing = False


def distanceBetweenTwoPoints(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def mouse_callback(event, x, y, flags, param):
    global mouse_is_pressing, points

    if step != 1:
        return

    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_is_pressing == True:

            for i, point in enumerate(points):
                if distanceBetweenTwoPoints((x, y), point) < 15:
                    points[i][0] = x
                    points[i][1] = y
                    break

    elif event == cv2.EVENT_LBUTTONDOWN:

        for point in points:
            if distanceBetweenTwoPoints((x, y), point) < 10:
                mouse_is_pressing = True
                break

    elif event == cv2.EVENT_LBUTTONUP:

        mouse_is_pressing = False


def angle_between(A, B):
    x1 = A[0]
    y1 = A[1]
    x2 = B[0]
    y2 = B[1]

    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = np.arctan2(det, dot) * 180 / np.pi

    return angle


def sort_points(points):
    points = points.astype(np.float32)

    new_points = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    min_index = np.argmin(s)
    new_points[0] = points[min_index]
    points = np.delete(points, min_index, axis=0)

    s = points.sum(axis=1)
    max_index = np.argmax(s)
    new_points[2] = points[max_index]
    points = np.delete(points, max_index, axis=0)

    v0 = points[0] - new_points[0]
    v1 = points[1] - new_points[0]

    angle = angle_between(v0, v1)

    if angle < 0:
        new_points[1] = points[1]
        new_points[3] = points[0]
    else:
        new_points[1] = points[0]
        new_points[3] = points[1]

    return new_points


def transform(img_input, points):
    points = sort_points(points)
    topLeft, topRight, bottomRight, bottomLeft = points
    print(topLeft, topRight, bottomRight, bottomLeft)
    print(topLeft[0] + topLeft[1], topRight[0] + topRight[1],
          bottomRight[0] + bottomRight[1], bottomLeft[0] + bottomLeft[1])

    topWidth = distanceBetweenTwoPoints(bottomLeft, bottomRight)
    bottomWidth = distanceBetweenTwoPoints(topLeft, topRight)
    maxWidth = max(int(topWidth), int(bottomWidth))

    leftHeight = distanceBetweenTwoPoints(topLeft, bottomLeft)
    rightHeight = distanceBetweenTwoPoints(topRight, bottomRight)
    maxHeight = max(int(leftHeight), int(rightHeight))

    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    H = cv2.getPerspectiveTransform(points, dst)
    img_warped = cv2.warpPerspective(img_input, H, (maxWidth, maxHeight))

    return img_warped


def findMaxArea(contours):
    max_area = -1
    max_index = -1

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_index = i

    if max_area < 10000:
        max_index = -1

    return max_index


def process(img_input, debug):
    points = []
    height, width = img_input.shape[:2]

    img_mask = np.zeros(img_input.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 10, width - 30, height - 30)
    cv2.grabCut(img_input, img_mask, rect, bgdModel, fgdModel,
                3, cv2.GC_INIT_WITH_RECT)

    img_mask = np.where((img_mask == 2) | (img_mask == 0), 0, 1).astype('uint8')
    img_grabcut = img_input * img_mask[:, :, np.newaxis]

    if debug:
        cv2.imshow('input0', img_input)
        cv2.imshow('grabCut', img_grabcut)

    img_gray = cv2.cvtColor(img_grabcut, cv2.COLOR_BGR2GRAY);
    img_canny = cv2.Canny(img_gray, 30, 90);

    if debug:
        cv2.imshow('Canny', img_canny)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, 1)

    if debug:
        cv2.imshow('morphology', img_canny)

    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    max_index = findMaxArea(contours)
    if max_index < 0:
        return points

    max_contour = contours[max_index]

    if debug:
        img_contour = img_input.copy()
        cv2.drawContours(img_contour, [max_contour], 0, (0, 0, 255), 3)
        cv2.imshow('Contour', img_contour)

    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    if debug:
        img_convexhull = img_input.copy()
        cv2.drawContours(img_convexhull, [hull], 0, (255, 255, 0), 5)
        cv2.imshow('convexHull', img_convexhull)

    size = len(max_contour)

    if size == 4:
        for c in hull:
            points.append(c[0])
        points = np.array(points)

    else:

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        points = np.int0(box.tolist())

    found = False
    for p in points:
        if p[0] < 0 or p[0] > width - 1 or p[1] < 0 or p[1] > height - 1:
            found = True
            break

    if found:
        points = np.array([[10, 10], [width - 11, 10],
                           [width - 11, height - 11], [10, height - 11]])

    return points


img_input = cv2.imread('book.jpg') ##############################
height, width = img_input.shape[:2]

points = process(img_input, debug=False)

size = len(points)

if size == 4:

    cv2.namedWindow('input')
    cv2.setMouseCallback("input", mouse_callback, 0);

    step = 1

    while True:

        img_result = img_input.copy()
        for point in points:
            cv2.circle(img_result, tuple(point), 10, (255, 0, 0), 3)
        cv2.imshow('input', img_result)

        key = cv2.waitKey(1)
        if key == 32:
            break

    img_final = transform(img_input, points)

    cv2.imshow('input', img_result)
    cv2.imshow('result', img_final)

else:
    cv2.imshow('input', img_input)

cv2.waitKey(0)
cv2.destroyAllWindows()