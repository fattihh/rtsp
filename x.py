import os

# OpenCV içindeki (cv2/qt/plugins) yolunu devre dışı bırak
os.environ.pop("QT_PLUGIN_PATH", None)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel
# Artık güvenle cv2 import edebiliriz
import cv2


RTSP_URL = "rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101"


class VideoThread(QThread):
    frame_received = pyqtSignal(object)     # NumPy ndarray gönderir

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("RTSP akışı açılamadı.")
            return

        while self._run_flag:
            ok, frame = cap.read()
            if ok:
                self.frame_received.emit(frame)
            else:
                print("Kare alınamadı.")
                break

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt RTSP Görüntüleyici")
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black")

        self.btn_quit = QPushButton("Çıkış")
        self.btn_quit.clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.btn_quit)

        # Video iş parçacığını başlat
        self.thread = VideoThread()
        self.thread.frame_received.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(object)
    def update_image(self, cv_img):
        """
        Gelen NumPy görüntüsünü QLabel’de göster.
        İstediğiniz boyuta resize edebilirsiniz (örnek: 500×500).
        """
        # İsteğe bağlı ölçekleme:
        cv_img = cv2.resize(cv_img, (500, 500), interpolation=cv2.INTER_AREA)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap   = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Pencere kapanırken thread’i düzgün kapat."""
        self.thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(600, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
