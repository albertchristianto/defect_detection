from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .template import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
