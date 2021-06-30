from PyQt5 import QtWidgets, uic, QtGui

import sys
import cv2
import numpy as np

#1. qt를 사용하여 GUI 프로그램 환경 구축
class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('image_load.ui', self)

        self.loadBtn = self.findChild(QtWidgets.QPushButton, 'loadBtn')
        self.loadBtn.clicked.connect(self.loadBtnClicked)
        self.procRun = self.findChild(QtWidgets.QPushButton, 'procRun')
        self.procRun.clicked.connect(self.procRunClicked) 
        self.photo = self.findChild(QtWidgets.QLabel, 'photo')     
        self.photo.setPixmap(QtGui.QPixmap("visionImage/21L90_OK.bmp"))
        self.photo.setScaledContents(True)
        self.result = self.findChild(QtWidgets.QLabel, 'result')     
        self.fnameEdit = self.findChild(QtWidgets.QLineEdit,'fnameEdit')
        self.fnameEdit.clear()
        self.show()

    def processingImage(self, img_gray, img_src):
        # 여기에 이미지 프로세싱을 진행하고 output으로 리턴하면 오른쪽에 결과 영상 출력됨
        # output = img_src.copy() #원본영상 그대로 리턴
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 5000:
                cv2.drawContours(img_src, [contour], -1, (0, 255, 0), 2)
                cv2.putText(img_src, str(int(cv2.contourArea(contour))), \
                            (contours[i][0][0][0] - 30, contours[i][0][0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, \
                            1.5, (0, 255, 0), 2)
        return img_src


    def displayOutputImage(self, img_dst, mode):
        img_info = img_dst.shape
        if img_dst.ndim == 2 :
            qImg = QtGui.QImage(img_dst, img_info[1], img_info[0], img_info[1]*1, QtGui.QImage.Format_Grayscale8)
        else :
            img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)
            qImg = QtGui.QImage(img_dst, img_info[1], img_info[0], img_info[1]*img_info[2], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        if mode == 0 :
            self.photo.setPixmap(pixmap)
            self.photo.setScaledContents(True)
        else :
            self.result.setPixmap(pixmap)
            self.result.setScaledContents(True)

    #cv2.imread가 한글 지원하지 않으므로 새로운 방식으로 파일 조합
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
        try: 
            n = np.fromfile(filename, dtype) 
            img = cv2.imdecode(n, flags) # img : bgr
            return img
        except Exception as e: 
            print(e) 
            return None

    def procRunClicked(self):
        img_gray = cv2.cvtColor(self.img_src, cv2.COLOR_BGR2GRAY)
        img_dst = self.processingImage(img_gray, self.img_src)
        self.displayOutputImage(img_dst,1) # displayOutputImage(출력영상 , 디스플레이위치 (0: 왼쪽, 1:오른쪽) )
        

    def loadBtnClicked(self):
        path = 'visionImage'
        filter = "All Images(*.jpg; *.png; *.bmp);;JPG (*.jpg);;PNG(*.png);;BMP(*.bmp)"
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "파일로드", path, filter)
        filename = str(fname[0])
        self.fnameEdit.setText(filename)
        self.img_src = self.imread(filename) #cv2.imread가 한글경로를 지원하지 않음
        self.displayOutputImage(self.img_src,0) # displayOutputImage(출력영상 , 디스플레이위치 (0: 왼쪽, 1:오른쪽) )


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()