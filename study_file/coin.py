import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1) 이미지 로드
img_src = cv2.imread("coin.png", cv2.IMREAD_COLOR)
height, width = img_src.shape[:2]

# # 2) 이미지 블러(blur) : 노이즈 제거
img_blur = cv2.blur(img_src, (7,7), anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT)

# 3) 이미지를 그래이로 변환( 3CH BGR -> 1CH GRAY)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# 4) 그래이 이미지에 임계값 설정
#ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, img_binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

# 5) 모폴로지 연산
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

#6) Contour(컨투어)를 찾기 : findContour
#cv2.findContours(이진영상, 검색방법, 근사화방법)
contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 4) 컨투어 값을 원본영상에 그림
count=0
list_coin=["500","100","50","10","500","100","50","10"]
for i, contour in enumerate(contours):
    if (cv2.contourArea(contour) > 700)and(cv2.contourArea(contour) < 19000):

        # 모멘트 찾기
        M = cv2.moments(contour)
        cX = int(M['m10'] / (M['m00']+1e-5))
        cY = int(M['m01'] / (M['m00']+1e-5))
        cv2.circle(img_src, (cX,cY), 3, (255, 0, 0), -1)

        # Contour 그리기
        # cv2.drawContours(img_src, [contour], 0, (0,255,0), 2)
        cv2.drawContours(img_src, [contours[i]], 0, (0, 255, 0), 2)
        # cv2.putText(img_src, str(i), tuple(contours[i][0][0]), \
        #             cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)

        # 가장 바깥 외곽선 그리기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_src, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.putText(img_src, str(cv2.contourArea(contour)), \
        #             (x,y-10), cv2.FONT_HERSHEY_COMPLEX, \
        #             0.6, (0,0, 255), 1)
        print(x)
        if cv2.contourArea(contour)>18000:
            cv2.putText(img_src, str("500W"), \
                        (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_COMPLEX, \
                        0.6, (0, 0, 255), 1)
        elif cv2.contourArea(contour)>13900:
            cv2.putText(img_src, str("100W"), \
                        (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_COMPLEX, \
                        0.6, (0, 0, 255), 1)
        elif cv2.contourArea(contour)>13000:
            cv2.putText(img_src, str("10W"), \
                        (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_COMPLEX, \
                        0.6, (0, 0, 255), 1)
        else :
            cv2.putText(img_src, str("50W"), \
                        (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_COMPLEX, \
                        0.6, (0, 0, 255), 1)

cv2.imshow("src", img_src)
cv2.imshow("gray",img_gray)
cv2.imshow("binary",img_binary)
cv2.waitKey()
cv2.destroyAllWindows()