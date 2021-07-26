
import cv2
import numpy as np

trap_bottom_width=0.7
trap_top_width=0.1
trap_height=0.3

img_src=cv2.imread("lane_test.png",cv2.IMREAD_COLOR)

img_mask=np.zeros_like(img_src) #src만큼의 마스크 만들기

if img_src.ndim>2: #color 영상이면
    channel_count=img_src.shape[2]
    ignore_mask_color=(255,255,255)
else:
    ignore_mask_color=255

imshape = img_src.shape
vertices = np.array([[\
      ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
      ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
      (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
      (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
      , dtype=np.int32)

cv2.fillPoly(img_mask,vertices,ignore_mask_color)

img_src=cv2.bitwise_and(img_src,img_mask)


cv2.imshow('src',img_src)
cv2.waitKey()
cv2.destroyAllWindows()