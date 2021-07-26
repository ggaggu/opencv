import cv2
import numpy as np
from matplotlib import pyplot as plt

img_src=cv2.imread("puppy.jpg",cv2.IMREAD_COLOR)
height,width=img_src.shape[:2]


# 2) 이미지 블러
img_dst1 = cv2.blur(img_src, (3,3), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)
img_dst1 = cv2.cvtColor(img_dst1, cv2.COLOR_BGR2RGB)

img_dst2 = cv2.blur(img_src, (5,5), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)
img_dst2 = cv2.cvtColor(img_dst2, cv2.COLOR_BGR2RGB)

img_dst3 = cv2.blur(img_src, (7,7), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)
img_dst3 = cv2.cvtColor(img_dst3, cv2.COLOR_BGR2RGB)

img_dst4 = cv2.blur(img_src, (9,9), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)
img_dst4 = cv2.cvtColor(img_dst4, cv2.COLOR_BGR2RGB)

img_dst5 = cv2.blur(img_src, (11,11), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)
img_dst5 = cv2.cvtColor(img_dst5, cv2.COLOR_BGR2RGB)

img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

titles =['Original','3x3','5x5','7x7','9x9','11x11']
images = [img_src,img_dst1,img_dst2,img_dst3,img_dst4,img_dst5]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()