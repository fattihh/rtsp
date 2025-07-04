import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import math
from vector_function import *

class UAVTracker:
    def __init__(
        self,
        video_path="ornek_3.mp4",
        model_path="best.pt",
        device=None,
        roi_div_x=3,
        roi_div_y=2,
        required_confidence=0.5,
        required_tracking_duration=4,
        bbox_size_ratio=0.06,
        altitude_padding=60,
        altitude_circle_radius=40
    ):
        """
        :param video_path: Takip edilecek video dosyasının yolu.
        :param model_path: YOLO model dosyasının yolu.
        :param device: "cuda" veya "cpu" olarak ayarlanabilir. None verilirse otomatik belirlenir.
        :param roi_div_x: X ekseninde ROI bölme sayısı (ör. 3 -> width/3).
        :param roi_div_y: Y ekseninde ROI bölme sayısı (ör. 2 -> height/2).
        :param required_confidence: Tespit edilen nesneyi UAV olarak kabul etmek için gereken minimum güven skoru.
        :param required_tracking_duration: Nesne yeterli boyut ve alanda tutulduğunda, “vuruş” saymak için gereken süre (sn).
        :param bbox_size_ratio: BBox’ın orijinal çerçeveye göre minimum boyut oranı (ör: 0.06 -> %6).
        :param altitude_padding: İrtifa daire çiziminde çerçevenin sağ alt köşesine ne kadar uzaklık bırakılacağı.
        :param altitude_circle_radius: İrtifa dairesinin yarıçapı.
        """
        # Yeni renkler (BGR formatında)
        self.shapes_color = (135, 69, 28)   # Hex #1c4587
        self.text_color   = (217, 217, 217) # Hex #d9d9d9

        # Cihaz (GPU/CPU) ayarı
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            print("CUDA destekleniyor, GPU kullanılacak!")
        else:
            print("CUDA desteklenmiyor, CPU kullanılacak.")

        # YOLO modeli yükleniyor
        self.model = YOLO(model_path)
        self.uav_class_name = "uav"

        # Video ayarları
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Video açılamadı: {self.video_path}")
            raise SystemExit

        # Orijinal genişlik/yükseklik
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Orijinal boyutlarda göstermek için
        self.frame_width = self.original_width
        self.frame_height = self.original_height

        # ROI bölümleri
        self.roi_div_x = roi_div_x
        self.roi_div_y = roi_div_y
        self.roi_width = self.frame_width // self.roi_div_x
        self.roi_height = self.frame_height // self.roi_div_y
        self.roi_positions = [
            (0, 0),
            (0, self.roi_height),
            (self.roi_width, 0),
            (self.roi_width, self.roi_height),
            (self.roi_width * 2, 0),
            (self.roi_width * 2, self.roi_height)
        ]
        self.current_roi_index = 0

        # FPS ölçümü
        self.start_time = time.time()
        self.frame_count = 0

        # Takip değişkenleri
        self.tracking = False
        self.tracker = None
        self.track_frame_count = 0
        self.uav_bbox = None
        self.tracking_start_time = None

        # Gerekli BBox boyutu
        self.required_bbox_width = int(self.frame_width * bbox_size_ratio)
        self.required_bbox_height = int(self.frame_height * bbox_size_ratio)

        # Vuruş alanı hesapları
        self.padding_x = int(self.frame_width * 0.25)
        self.padding_y = int(self.frame_height * 0.10)
        self.vurus_sayisi = 0

        # Takip süresi eşiği
        self.REQUIRED_TRACKING_DURATION = required_tracking_duration
        self.required_confidence = required_confidence

        # Zaman tutucu
        self.big_enough_start_time = None

        # Çerçeve merkez
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2

        # İrtifa-yönelme çizim
        self.altitude_padding = altitude_padding
        self.altitude_circle_radius = altitude_circle_radius
        self.altitude_circle_center = (
            self.frame_width - self.altitude_padding - self.altitude_circle_radius,
            self.frame_height - self.altitude_padding - self.altitude_circle_radius
        )

        # --- Throttle göstergesi ayarları ---
        self.target_box_area = 2000
        self.throttle_circle_center = (80, self.frame_height - 80)
        self.throttle_outer_radius = 50

    def _scan_roi_for_uav(self, frame):
        x, y = self.roi_positions[self.current_roi_index]
        roi = frame[y:y + self.roi_height, x:x + self.roi_width]
        results = self.model.predict(roi, device=self.device)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id] if hasattr(self.model, "names") else self.uav_class_name

            if conf > self.required_confidence and class_name.lower() == self.uav_class_name.lower():
                w = x2 - x1
                h = y2 - y1
                bbox = (x + x1, y + y1, w, h)
                params = cv2.TrackerCSRT_Params()
                params.use_hog               = True      # HOG özelliklerini kullan
                params.use_color_names       = True      # Color Names kanallarını dahil et
                params.use_gray              = False     # Gri seviye görüntüden vazgeç (color_names varken)
                params.use_rgb               = False     # RGB kanallarını kullanma (varsayılan)
                params.window_function       = "hann"    # Pencere fonksiyonu ("hann", "cheb", "kaiser")
                params.num_hog_channels_used = 18        # HOG kanal sayısı
                params.template_size         = 200       # Şablon boyutu (piksel)
                params.filter_lr             = 0.02      # Filtre öğrenme hızı
                params.weights_lr            = 0.4       # Kanal ağırlıklarının öğrenme hızı
                params.admm_iterations       = 2         # ADMM döngü sayısı (daha yüksek = daha yavaş ama daha doğru)
                params.histogram_bins        = 16        # Renk histogramı için bin sayısı
                params.histogram_lr          = 0.02      # Histogram güncelleme hızı
                params.number_of_scales      = 5         # Ölçek testi sayısı
                params.scale_sigma_factor    = 0.25      # Ölçek uzayının sigma faktörü

# 2) Bu parametrelerle bir CSRT izleyici oluştur
                tracker = cv2.TrackerCSRT_create(params)
                #tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)

                self.tracking = True
                self.tracker = tracker
                self.track_frame_count = 0
                self.tracking_start_time = time.time()
                self.uav_bbox = bbox
                self.big_enough_start_time = None

                print(f"UAV tespit edildi ve takip edilmeye başlandı: {bbox}")
                break

        self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_positions)
        return (x, y, x + self.roi_width, y + self.roi_width)

    def _track_uav(self, frame):
        success, bbox = self.tracker.update(frame)
        if not success:
            print("Tracker nesneyi kaybetti. Takip sonlandiriliyor.")
            self._reset_tracking()
            return

        x, y, w, h = map(int, bbox)
        box_center_x = x + w // 2
        box_center_y = y + h // 2

        # İrtifa-yönelme çizimi
        self._draw_altitude_vector(frame, box_center_x, box_center_y)

        # Hedef bbox & metin (korundu)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Takip Ediliyor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y), (box_center_x, box_center_y), (0, 0, 255), 2)

        # Throttle çizimi (yeni renkler)
        self._draw_throttle(frame, w, h)

        # Vuruş alanı (korundu)
        cv2.rectangle(frame, (self.padding_x, self.padding_y), (self.frame_width - self.padding_x, self.frame_height - self.padding_y), (0, 255, 255), 2)

        uav_within_target = (
            x >= self.padding_x and y >= self.padding_y and
            (x + w) <= (self.frame_width - self.padding_x) and
            (y + h) <= (self.frame_height - self.padding_y)
        )

        if uav_within_target:
            cv2.putText(frame, "UAV Vurus Alaninda!", (self.padding_x, self.padding_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)

            if w >= self.required_bbox_width and h >= self.required_bbox_height:
                if self.big_enough_start_time is None:
                    self.big_enough_start_time = time.time()
                duration = time.time() - self.big_enough_start_time
                cv2.putText(frame, f"Takip Suresi: {duration:.1f}s", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
                if duration >= self.REQUIRED_TRACKING_DURATION:
                    self.vurus_sayisi += 1
                    print(f"UAV {self.REQUIRED_TRACKING_DURATION} sn boyunca yeterli boyutta takip edildi. Başarılı vuruş!")
                    print(f"Toplam Vuruş Sayısı: {self.vurus_sayisi}")
                    self._reset_tracking()
            else:
                if self.big_enough_start_time is not None:
                    self.big_enough_start_time = None
                cv2.putText(frame, "Boyut Yetersiz! (Gaza Bas)", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "UAV Vuruş Alanında Değil!", (self.padding_x, self.padding_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("UAV vuruş alanından çıktı, takip sonlandırılıyor.")
            self._reset_tracking()

    def _draw_altitude_vector(self, frame, box_center_x, box_center_y):
        vx = self.frame_width // 2 - box_center_x
        vy = self.frame_height // 2 - box_center_y
        normalized_x, normalized_y = normalized_vector(vx, vy, window_width_half=self.frame_width // 2, window_height_half=self.frame_height // 2)
        magnitude = calculate_vector_magnitude(normalized_x, normalized_y)
        length = magnitude * 40
        angle = math.atan2(normalized_y, normalized_x)
        end_pt = (
            int(self.altitude_circle_center[0] + length * math.cos(angle)),
            int(self.altitude_circle_center[1] + length * math.sin(angle))
        )
        cv2.line(frame, self.altitude_circle_center, end_pt, self.shapes_color, 2)
        cv2.circle(frame, self.altitude_circle_center, self.altitude_circle_radius, self.shapes_color, 2)
        cv2.putText(frame, "Irtifa-Yonelme", (self.altitude_circle_center[0] - 60, self.altitude_circle_center[1] + 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, self.text_color, 2)

    def _draw_throttle(self, frame, w, h):
        current_box_area = w * h
        throttle_norm = 1 - (current_box_area / self.target_box_area)
        throttle_norm = np.clip(throttle_norm, 0, 1)
        throttle_pwm = int(1000 + throttle_norm * 1000)
        cv2.circle(frame, self.throttle_circle_center, self.throttle_outer_radius, self.shapes_color, 2)
        start_angle = -90
        end_angle = int(start_angle + throttle_norm * 360)
        cv2.ellipse(frame, self.throttle_circle_center, (self.throttle_outer_radius, self.throttle_outer_radius), 0, start_angle, end_angle, self.shapes_color, -1)
        text_pos = (self.throttle_circle_center[0] - 40, self.throttle_circle_center[1] - self.throttle_outer_radius - 10)
        cv2.putText(frame, f"Throttle: {throttle_pwm}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)

    def _reset_tracking(self):
        self.tracking = False
        self.tracker = None
        self.uav_bbox = None
        self.current_roi_index = 0
        self.big_enough_start_time = None

    def run(self):
        with torch.no_grad():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video sonuna gelindi veya okuyamadı.")
                    break

                # Orijinal boyutlarda geldiği için yeniden boyutlandırmaya gerek yok
                if not self.tracking:
                    roi_coords = self._scan_roi_for_uav(frame)
                    if roi_coords:
                        rx1, ry1, rx2, ry2 = roi_coords
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), self.shapes_color, 2)
                else:
                    self._track_uav(frame)

                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)
                cv2.putText(frame, f"Vurus Sayisi: {self.vurus_sayisi}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)

                cv2.imshow("Video with Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = UAVTracker(
        video_path="ornek.mp4",
        model_path="best.pt"
    )
    tracker.run()