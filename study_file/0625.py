import cv2

# img_src=cv2.imread("puppy.jpg",cv2.IMREAD_COLOR)
#
# height, width = img_src.shape[:2]
#
# img_gray=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# #이미지 저장 cv2.imwrite("경로/파일이름",이미지이름)
# #cv2.imwrite('puppy_gray.jpg',img_gray)
#
# ret, img_binary1=cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
# ret, img_binary2=cv2.threshold(img_gray,150,255,cv2.THRESH_TRUNC)
# cv2.imshow("src",img_src)
# cv2.imshow("bin",img_binary1)
# cv2.imshow("bin2",img_binary2)
# cv2.waitKey()
# cv2.destroyAllWindows()

#
#
# img_src=cv2.imread("sdq.png",cv2.IMREAD_COLOR)
#
# height, width = img_src.shape[:2]
#
# img_gray=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# #이미지 저장 cv2.imwrite("경로/파일이름",이미지이름)
# #cv2.imwrite('puppy_gray.jpg',img_gray)
#
# ret, img_binary1=cv2.threshold(img_gray,100,255,cv2.THRESH_OTSU)
# ret, img_binary2=cv2.threshold(img_gray,150,255,cv2.THRESH_TRIANGLE)
# cv2.imshow("src",img_src)
# cv2.imshow("bin",img_binary1)
# cv2.imshow("bin2",img_binary2)
# cv2.waitKey()
# cv2.destroyAllWindows()

##################################
# img_src=cv2.imread("puppy.jpg",cv2.IMREAD_COLOR)
#
# height, width = img_src.shape[:2]
#
# img_blur=cv2.blur(img_src,(3,3),anchor=(-1,-1),borderType=cv2.BORDER_DEFAULT)
#
# cv2.imshow("bin2",img_blur)
# cv2.waitKey()
# cv2.destroyAllWindows()
###############################

#
#
# img_src=cv2.imread("puppy.jpg",cv2.IMREAD_COLOR)
#
# height, width = img_src.shape[:2]
# print(height,width)
#
# img_hsv=cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
# img_h,img_s,img_v=cv2.split(img_hsv)
#
# mask=cv2.inRange(img_h,13,30)
# img_yellow=cv2.bitwise_and(img_hsv,img_hsv,mask=mask)
# img_yellow=cv2.cvtColor(img_yellow,cv2.COLOR_HSV2BGR)
#
# cv2.imshow("img_yellow",img_yellow)
# cv2.imshow("img_src",img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

########################################################

img_src=cv2.imread("kimsh.jpg",cv2.IMREAD_COLOR)

height, width = img_src.shape[:2]
print(height,width)

img_hsv=cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)

hsv_mask1=cv2.inRange(img_hsv,(0,100,100),(6,255,255))
hsv_mask2=cv2.inRange(img_hsv,(165,100,100),(180,255,255))
hsv_mask_add=cv2.addWeighted(hsv_mask1,1.0,hsv_mask2,1.0,0.0)
img_red=cv2.bitwise_and(img_hsv,img_hsv,mask=hsv_mask_add)
img_red=cv2.cvtColor(img_red,cv2.COLOR_HSV2BGR)

cv2.imshow("img_src",img_red)
cv2.waitKey()
cv2.destroyAllWindows()