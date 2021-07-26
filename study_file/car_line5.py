import cv2
import numpy as np

trap_bottom_width = 0.85
trap_top_width = 0.075
trap_height = 0.43

def region_of_interest(img_src):
    img_mask = np.zeros_like(img_src)

    if img_src.ndim > 2:  # color영상이면
        channel_count = img_src.shape[2]
        ignore_mask_color = (255, 255, 255)
    else:
        ignore_mask_color = 255

    imshape = img_src.shape
    vertices = np.array([[ \
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]), \
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]] \
        , dtype=np.int32)

    cv2.fillPoly(img_mask, vertices, ignore_mask_color)

    img_src = cv2.bitwise_and(img_src, img_mask)
    return img_src


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    # In case of error, don't draw the line(s)
    if lines is None:
        return
    if len(lines) == 0:
        return
    draw_right = True
    draw_left = True

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # x coordinate of center of image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]

        right_lines_x.append(x1)
        right_lines_x.append(x2)

        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]

        left_lines_x.append(x1)
        left_lines_x.append(x2)

        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)


if __name__ == '__main__':
    #img_src = cv2.imread("lane_test.png", cv2.IMREAD_COLOR)

    capture = cv2.VideoCapture("runcar.mp4")  # 비디오캡쳐(0)을 하면 카메라 1개 연결되어있다는 뜻

    rho = 2
    theta = 1 * np.pi / 180
    threshold = 15
    min_line_len = 10
    max_line_gap = 20

    while cv2.waitKey(33) < 0:  # 33/1000 초
        if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):  # 프레임을 얻어옴
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 실제 비디오를 받아들일때 읽어들이다가 프레임 카운트와 같아지면 프레임 위치를 0로 돌림
        ret, frame = capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        img_canny = cv2.Canny(img_gray, 95, 200)
        img_canny = region_of_interest(img_canny)

        lines = cv2.HoughLinesP(img_canny, rho, theta, threshold, np.array([]), \
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        # for i, line in enumerate(lines):
        #      cv2.line(frame, (line[0][0],line[0][1]),(line[0][2],line[0][3]), \
        #               (0,255,0), 2)
        img_lines = np.zeros_like(frame)
        draw_lines(img_lines, lines)
        img_dst = cv2.addWeighted(frame, 0.8, img_lines, 1.0, 0.0)

        cv2.imshow("dd",img_dst)


    capture.release()
    cv2.destroyAllWindows()