import cv2
import numpy as np
img_src=cv2.imread("lane_test.png",cv2.IMREAD_COLOR)
height,width=img_src.shape[:2]

#white 뽑아내기
white_thr=200
lower_white=np.array([white_thr,white_thr,white_thr])
upper_white=np.array([255,255,255])
white_mask=cv2.inRange(img_src,lower_white,upper_white)
white_image=cv2.bitwise_and(img_src,img_src,mask=white_mask)

#yellow 뽑아내기
img_hsv=cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
lower_yellow=np.array([20,100,100])
upper_yellow=np.array([40,255,255])
yellow_mask=cv2.inRange(img_hsv,lower_yellow,upper_yellow)
yellow_image=cv2.bitwise_and(img_hsv,img_hsv,mask=yellow_mask)

mixed_img=cv2.addWeighted(white_image,1.0,yellow_image,1.0,0.0)


cv2.imshow('src',white_image)
cv2.imshow('yellow',yellow_image)
cv2.imshow('mix',mixed_img)
cv2.waitKey()
cv2.destroyAllWindows()