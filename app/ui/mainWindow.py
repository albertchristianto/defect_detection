from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from .template import Ui_MainWindow
from loguru import logger

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
            self.InputImage.setPixmap(QPixmap(file))
            self.InputImage.setScaledContents(True)
