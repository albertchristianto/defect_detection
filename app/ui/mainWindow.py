from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from loguru import logger
import cv2
import numpy as np

from .template import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.InputImage.clicked.connect(self.openFileNameDialog)
        self.setFixedSize(640, 640)

    def openFileNameDialog(self):
        file , check = QFileDialog.getOpenFileName(None, 
            "QFileDialog.getOpenFileName()", "", "All Files (*);;JPG Files (*.jpg)")
        if check:
            logger.trace(file)
            image = cv2.imread(file)
            self.InputImage.setPixmap(QPixmap(file))
            self.InputImage.setScaledContents(True)

    def convert_nparray_to_QPixmap(self,img):
        w,h,ch = img.shape
        # Convert resulting image to pixmap
        if img.ndim == 1:
            img =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        qimg = QImage(img.data, h, w, 3*h, QImage.Format_RGB888) 
        qpixmap = QPixmap(qimg)

        return qpixmap