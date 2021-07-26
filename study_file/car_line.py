import cv2
import numpy as np

# img_src = cv2.imread("bike.jpg", cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
# #
# #노이즈 제거
# img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

# # 소벨 필터 적용
# img_sobel1 = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, 3)
# img_sobel2 = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, 3)
# #소벨필터 취하면 +,- 나올수있다.
#
# abs_grad_x = cv2.convertScaleAbs(img_sobel1)#소벨필터적용한것에 절댓값 적용
# abs_grad_y = cv2.convertScaleAbs(img_sobel2)
#
# img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#
# cv2.imshow('src',img_gray)
# cv2.imshow('src', img_sobel)
# cv2.waitKey()
# cv2.destroyAllWindows()

###############################################

# # https://docs.opencv.org/4.5.2/d5/db5/tutorial_laplace_operator.html
#
# #라플라시안 필터
# img_laplacian= cv2.Laplacian(img_gray,cv2.CV_8U,ksize=3)
# img_laplacian=cv2.convertScaleAbs(img_laplacian)
#

################################
#
# img_canny=cv2.Canny(img_gray,100,255)
#
#
# cv2.imshow('src',img_canny)
# cv2.waitKey()
# cv2.destroyAllWindows()

#####################################

# capture=cv2.VideoCapture("runcar.mp4")#비디오캡쳐(0)을 하면 카메라 1개 연결되어있다는 뜻
#
#
# while cv2.waitKey(33)<0: # 33/1000 초
#     if(capture.get(cv2.CAP_PROP_POS_FRAMES)==capture.get(cv2.CAP_PROP_FRAME_COUNT)):# 프레임을 얻어옴
#         capture.set(cv2.CAP_PROP_POS_FRAMES,0) #실제 비디오를 받아들일때 읽어들이다가 프레임 카운트와 같아지면 프레임 위치를 0로 돌림
#     ret,frame=capture.read()
#     #cv2.imshow("Video",frame)
#     # frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     #
#     # frame_gray=cv2.GaussianBlur(frame_gray,(5,5),0)
#     # frame_lap=cv2.Laplacian(frame_gray,cv2.CV_8U,ksize=3)
#     # frame_lap = cv2.convertScaleAbs(frame_lap)
#     #frame_canny=cv2.Canny(frame_gray,50,150)
#
#     # white 뽑아내기
#     white_thr = 200
#     lower_white = np.array([white_thr, white_thr, white_thr])
#     upper_white = np.array([255, 255, 255])
#     white_mask = cv2.inRange(frame, lower_white, upper_white)
#     white_image = cv2.bitwise_and(frame, frame, mask=white_mask)
#
#     # yellow 뽑아내기
#     img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([40, 255, 255])
#     yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
#     yellow_image = cv2.bitwise_and(img_hsv, img_hsv, mask=yellow_mask)
#
#     mixed_img = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0)
#     frame_gray=cv2.cvtColor(mixed_img,cv2.COLOR_BGR2GRAY)
#     frame_gray=cv2.GaussianBlur(frame_gray,(5,5),0)
#     frame_canny=cv2.Canny(frame_gray,50,150)
#     frame_canny2 = cv2.Canny(frame_gray, 10, 100)
#
#     #cv2.imshow("Video",mixed_img)
#     cv2.imshow("convert_canny", frame_canny)
#     cv2.imshow("convert_canny2",frame_canny2)
#
# capture.release()
# cv2.destroyAllWindows()

#################################################


capture=cv2.VideoCapture("runcar.mp4")#비디오캡쳐(0)을 하면 카메라 1개 연결되어있다는 뜻
trap_bottom_width=0.9
trap_top_width=0.2
trap_height=0.3


while cv2.waitKey(33)<0: # 33/1000 초
    if(capture.get(cv2.CAP_PROP_POS_FRAMES)==capture.get(cv2.CAP_PROP_FRAME_COUNT)):# 프레임을 얻어옴
        capture.set(cv2.CAP_PROP_POS_FRAMES,0) #실제 비디오를 받아들일때 읽어들이다가 프레임 카운트와 같아지면 프레임 위치를 0로 돌림
    ret,frame=capture.read()

    # white 뽑아내기
    white_thr = 200
    lower_white = np.array([white_thr, white_thr, white_thr])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(frame, lower_white, upper_white)
    white_image = cv2.bitwise_and(frame, frame, mask=white_mask)

    # yellow 뽑아내기
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(img_hsv, img_hsv, mask=yellow_mask)

    mixed_img = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0)
    frame_gray=cv2.cvtColor(mixed_img,cv2.COLOR_BGR2GRAY)
    frame_gray=cv2.GaussianBlur(frame_gray,(5,5),0)
    frame_canny=cv2.Canny(frame_gray,50,150)


    img_mask = np.zeros_like(frame_canny)  # src만큼의 마스크 만들기

    if frame.ndim > 2:  # color 영상이면
        channel_count = frame.shape[2]
        ignore_mask_color = (255,255,255)
    else:
        ignore_mask_color = 255

    imshape = frame.shape
    vertices = np.array([[ \
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]), \
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]] \
        , dtype=np.int32)

    cv2.fillPoly(img_mask, vertices, ignore_mask_color)

    img_load = cv2.bitwise_and(frame_canny, img_mask)
    #img_onload=cv2.addWeighted(frame,1.0,img_load,1.0,0)

    #cv2.imshow("Video",mixed_img)
    cv2.imshow("convert_canny", img_load)
    #cv2.imshow("convert_canny2",frame_canny2)

capture.release()
cv2.destroyAllWindows()

#####################################################