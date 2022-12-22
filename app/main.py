import sys

from PyQt5.QtWidgets import QApplication

from dd.ui.mainWindow import MainWindow
from loguru import logger

LOG_LEVEL = 'TRACE'

logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())