from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from loguru import logger
import cv2
import numpy as np
import json

from .template import Ui_MainWindow
from ..backend.py.ImgClassifier import ImgClassifier

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(640, 640)
        f = open('cfgs/app.cfg')
        self.SystemParam = json.load(f)
        self.image = None
        self.DlEngine = None
        if self.SystemParam['mode'] == "ImgClassifier":
            self.DlEngine = ImgClassifier(self.SystemParam['image_classifier_path'])
        self.InputImage.clicked.connect(self.open_file_dialog)
        self.DetectButton.clicked.connect(self.detect)

    def open_file_dialog(self):
        file , check = QFileDialog.getOpenFileName(None, 
            "QFileDialog.getOpenFileName()", "", "All Files (*);;JPG Files (*.jpg)")
        if check:
            logger.trace(file)
            self.image = cv2.imread(file)
            self.disp = self.prepare_disp_img(self.image)
            self.InputImage.setPixmap(self.convert_nparray_to_QPixmap(self.disp))

    def prepare_disp_img(self, img):
        disp = img.copy()
        if disp.ndim == 1:
            disp =  cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        w, h = self.InputImage.width(), self.InputImage.height()
        disp = cv2.resize(disp, (w, h), interpolation = cv2.INTER_AREA)
        return disp

    def convert_nparray_to_QPixmap(self, disp):
        h, w, c = disp.shape
        qimg = QImage(disp.data, w, h, (3 * w), QImage.Format_RGB888) 
        qpixmap = QPixmap(qimg)
        return qpixmap

    def img_classifier_post_process(self, out):
        h, w, c = self.disp.shape
        if out == "Defect":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        tmp = cv2.rectangle(self.disp, (0,0), (w, h), color, 15)
        self.InputImage.setPixmap(self.convert_nparray_to_QPixmap(tmp))

    def detect(self):
        logger.trace("Detecting")
        if self.DlEngine is None:
            logger.warning("The engine has not spawned yet!")
            return
        if self.image is None:
            logger.warning("Select an image first!")
            return
        tmp = self.image.copy()
        output = self.DlEngine.forward(tmp)
        if self.SystemParam['mode'] == "ImgClassifier":
            self.img_classifier_post_process(output)