import cv2,os,sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui  import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton


#pyqit pozisyonu ayarla 
#arayüzlere farklı channler entegre et
#pyqit arayüzü yap


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


url = "rtsp://admin:ErenEnerji@192.168.1.64:554/Streaming/Channels/101"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise IOError("RTSP akışı açılamadı. URL/kimlik bilgilerini kontrol edin.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı — akış koptu mu?")
        break

    frame_resized = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow("IP Kamera (resize)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
