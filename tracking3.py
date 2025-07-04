import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import math
from collections import deque
from vector_function import *

############### GPU kontrolü ##################
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("CUDA destekleniyor, GPU kullanilacak!")
else:
    print("CUDA desteklenmiyor, CPU kullanilacak.")
###############################################

# Video kaynağı
video_path = "ornek_3.mp4"  # Video dosyanızın yolu
cap = cv2.VideoCapture(video_path)

# Video genişlik ve yükseklik bilgileri
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_width = frame_width // 2           # Çözünürlüğü yarıya indirme
#frame_height = frame_height // 2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Frame merkezi (örneğin vektör çizimi için)
frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

# ROI boyutları
roi_width = frame_width // 3
roi_height = frame_height // 2

# 6 adet ROI başlangıç koordinatı
roi_positions = [
    (0, 0),                      # Alan 0
    (0, roi_height),             # Alan 1
    (roi_width, 0),              # Alan 2
    (roi_width, roi_height),     # Alan 3
    (roi_width * 2, 0),          # Alan 4
    (roi_width * 2, roi_height)  # Alan 5
]
current_roi_index = 0  # Mevcut ROI sayacı

# FPS ölçümü için
start_time = time.time()
frame_count = 0

# Takip (tracking) ile ilgili değişkenler
tracking = False
tracker = None
track_frame_count = 0
uav_bbox = None
tracking_start_time = None

# Kalman Filter ve bbox smoothing için
kf = None
bbox_history = deque(maxlen=5)

# 4 saniye takip süresi eşiği
REQUIRED_TRACKING_DURATION = 4

# Bbox genişlik ve yükseklik alt sınırları (örneğin, %6)
REQUIRED_BBOX_WİDTH = int(frame_width * 0.06)
REQUIRED_BBOX_HEIGHT = int(frame_height * 0.06)

# YOLOv8 modelini yükle
model = YOLO("best.pt")
 
# Modelde yer alan UAV sınıf adı
UAV_CLASS_NAME = "uav"

# Vuruş alanı (hedef alan) tanımları
padding_x = int(frame_width * 0.25)
padding_y = int(frame_height * 0.10)

# Başarılı vuruş sayısı
vurus_sayisi = 0

# Bbox yeterli boyutta iken zaman ölçümü için
big_enough_start_time = None

##################### IRTIFA-YONELME CONSTANT #######################
altitude_padding = 60
altitude_circle_radius = 40
altitude_circle_center = (frame_width - altitude_padding - altitude_circle_radius, frame_height - altitude_padding - altitude_circle_radius)
#####################################################################

def create_kalman_filter(bbox):
    """
    Kalman filtresini başlatır.
    Durum vektörü: [cx, cy, w, h, vx, vy, vw, vh]
    Ölçüm vektörü: [cx, cy, w, h]
    """
    kf = cv2.KalmanFilter(8, 4)
    # Ölçüm matrisi
    kf.measurementMatrix = np.zeros((4, 8), np.float32)
    for i in range(4):
        kf.measurementMatrix[i, i] = 1.0
    # Geçiş matrisi – sabit hız modeli
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i+4] = 1.0
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    # Başlangıç durumu: bbox'un merkez ve boyut bilgileri, hızlar sıfır
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
    kf.statePre = state.reshape(-1, 1)
    kf.statePost = state.reshape(-1, 1)
    return kf

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        # -------------------------------------------------------
        # ROI TARAMA MODU (tracking False iken)
        # -------------------------------------------------------
        if not tracking:
            x, y = roi_positions[current_roi_index]
            # ROI, frame sınırlarını aşmayacak şekilde ayarlanıyor
            roi = frame[y:min(y+roi_height, frame_height), x:min(x+roi_width, frame_width)]
            
            # ROI üzerinde tespit için YOLO çalıştırılıyor
            results = model(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if hasattr(model, "names") else UAV_CLASS_NAME

                if conf > 0.5 and class_name.lower() == UAV_CLASS_NAME.lower():
                    # ROI içindeki tespiti frame koordinatlarına çevir
                    w_box = x2 - x1
                    h_box = y2 - y1
                    bbox_abs = (x + x1, y + y1, w_box, h_box)
                    
                    # Bbox'un frame sınırları içinde olup olmadığı kontrol ediliyor
                    if (bbox_abs[0] < 0 or bbox_abs[1] < 0 or 
                        bbox_abs[0] + bbox_abs[2] > frame_width or 
                        bbox_abs[1] + bbox_abs[3] > frame_height):
                        continue
                    
                    # Sadece UAV vuruş alanı içindeyse takip başlatılıyor
                    if (bbox_abs[0] >= padding_x and bbox_abs[1] >= padding_y and 
                        bbox_abs[0] + bbox_abs[2] <= frame_width - padding_x and 
                        bbox_abs[1] + bbox_abs[3] <= frame_height - padding_y):

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
                        tracker.init(frame, bbox_abs)

                        # Kalman filtresi ve bbox geçmişi başlatılıyor
                        kf = create_kalman_filter(bbox_abs)
                        bbox_history.clear()
                        bbox_history.append(bbox_abs)

                        tracking = True
                        track_frame_count = 0
                        tracking_start_time = time.time()
                        uav_bbox = bbox_abs
                        big_enough_start_time = None

                        print(f"UAV tespit edildi ve takip başladı: {bbox_abs}")
                        break  # İlk geçerli UAV için takip başlatılır
                    else:
                        print("Tespit edilen UAV vuruş alanı dışında, takip başlatılmıyor. Tarama devam ediyor.")
            
            # ROI dikdörtgenini çiz (sarı)
            cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (0, 255, 255), 2)
            # Sonraki ROI'ya geç (daire şeklinde)
            current_roi_index = (current_roi_index + 1) % len(roi_positions)

        # -------------------------------------------------------
        # TAKİP MODU (tracking True iken)
        # -------------------------------------------------------
        else:
            success, bbox = tracker.update(frame)
            if success:
                x_bb, y_bb, w_bb, h_bb = map(int, bbox)
                bbox_history.append((x_bb, y_bb, w_bb, h_bb))
                avg_bbox = np.mean(bbox_history, axis=0)
                avg_bbox = tuple(avg_bbox.astype(int))

                if kf is None:
                    kf = create_kalman_filter(avg_bbox)
                meas_cx = avg_bbox[0] + avg_bbox[2] / 2
                meas_cy = avg_bbox[1] + avg_bbox[3] / 2
                measurement = np.array([[meas_cx], [meas_cy], [avg_bbox[2]], [avg_bbox[3]]], dtype=np.float32)
                kf.predict()
                kf.correct(measurement)
                final_state = kf.statePost
                pred_cx, pred_cy, pred_w, pred_h = (final_state[0, 0], final_state[1, 0],
                                                    final_state[2, 0], final_state[3, 0])
                pred_x = int(pred_cx - pred_w / 2)
                pred_y = int(pred_cy - pred_h / 2)
                pred_w = int(pred_w)
                pred_h = int(pred_h)

                tracked_center_x = pred_x + pred_w // 2
                tracked_center_y = pred_y + pred_h // 2

                ##################### IRTIFA-YONELME ÇİZİM #######################
                vector_x = frame_width // 2 - tracked_center_x
                vector_y = frame_height // 2 - tracked_center_y
                normalized_x, normalized_y = normalized_vector(
                    vector_x, vector_y,
                    window_width_half=frame_width // 2, window_height_half=frame_height // 2
                )
                magnitude = calculate_vector_magnitude(normalized_x, normalized_y)
                altitude_line_length = magnitude * 40
                angle = math.atan2(normalized_y, normalized_x)
                end_point = (int(altitude_circle_center[0] + altitude_line_length * math.cos(angle)),
                             int(altitude_circle_center[1] + altitude_line_length * math.sin(angle)))
                cv2.line(frame, altitude_circle_center, end_point, (0, 255, 255), 2)
                cv2.circle(frame, altitude_circle_center, altitude_circle_radius, (40, 150, 255), 2)
                cv2.putText(frame, "Irtifa-Yonelme", (altitude_circle_center[0] - 60, altitude_circle_center[1] + 70),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (40, 150, 255), 2)
                ###################################################################

                aileron_pwm = int(1000 + normalized_x * 1000)
                elevator_pwm = int(1000 + normalized_y * 1000)
                aileron_pwm = max(1000, min(aileron_pwm, 2000))
                elevator_pwm = max(1000, min(elevator_pwm, 2000))

                cv2.putText(frame, f"aileron pwm : {aileron_pwm}", (80, 600),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (40, 150, 255), 2)
                cv2.putText(frame, f"elevator pwm : {elevator_pwm}", (80, 630),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (40, 150, 255), 2)

                # Kalman filtreden gelen tahmini bbox (mavi) çiziliyor
                cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (255, 0, 0), 2)
                cv2.putText(frame, "Takip Ediliyor", (pred_x, pred_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.line(frame, (frame_center_x, frame_center_y),
                         (tracked_center_x, tracked_center_y), (255, 0, 0), 2)
                track_frame_count += 1
            
                # Vuruş alanı dikdörtgeni (sarı)
                cv2.rectangle(frame, (padding_x, padding_y),
                              (frame_width - padding_x, frame_height - padding_y),
                              (0, 255, 255), 2)

                # Her 20 frame'de takip edilen box'un doğrulanması
                if track_frame_count % 20 == 0:
                    # Takip edilen ROI'yi al (sınır kontrolü yapıyoruz)
                    tracked_roi = frame[max(pred_y,0):min(pred_y+pred_h, frame_height),
                                        max(pred_x,0):min(pred_x+pred_w, frame_width)]
                    if tracked_roi.size > 0:
                        verify_results = model(cv2.cvtColor(tracked_roi, cv2.COLOR_BGR2RGB))
                        uav_found = False
                        for v_box in verify_results[0].boxes:
                            v_x1, v_y1, v_x2, v_y2 = map(int, v_box.xyxy[0])
                            v_conf = float(v_box.conf[0])
                            v_class_id = int(v_box.cls[0])
                            v_class_name = model.names[v_class_id] if hasattr(model, "names") else UAV_CLASS_NAME
                            if v_conf > 0.5 and v_class_name.lower() == UAV_CLASS_NAME.lower():
                                uav_found = True
                                break
                        if not uav_found:
                            print("Takip edilen nesne UAV degil veya kayboldu, takip sonlandiriliyor.")
                            tracking = False
                            tracker = None
                            kf = None
                            uav_bbox = None
                            current_roi_index = 0
                            big_enough_start_time = None
                            continue

                # UAV vuruş alanı kontrolü (Kalman filtreden gelen tahmine göre)
                uav_within_target = (
                    pred_x >= padding_x and
                    pred_y >= padding_y and
                    (pred_x + pred_w) <= (frame_width - padding_x) and
                    (pred_y + pred_h) <= (frame_height - padding_y)
                )

                if uav_within_target:
                    cv2.putText(frame, "UAV Vurus Alaninda!", (padding_x, padding_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if pred_w >= REQUIRED_BBOX_WİDTH and pred_h >= REQUIRED_BBOX_HEIGHT:
                        if big_enough_start_time is None:
                            big_enough_start_time = time.time()
                        big_enough_duration = time.time() - big_enough_start_time
                        cv2.putText(frame, f"Takip Suresi: {big_enough_duration:.1f}s",
                                    (pred_x, pred_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        if big_enough_duration >= REQUIRED_TRACKING_DURATION:
                            vurus_sayisi += 1
                            print(f"UAV {REQUIRED_TRACKING_DURATION} sn boyunca takip edildi. Basarili vurus!")
                            print(f"Toplam Vurus Sayisi: {vurus_sayisi}")
                            tracking = False
                            tracker = None
                            kf = None
                            uav_bbox = None
                            current_roi_index = 0
                            big_enough_start_time = None
                            continue
                    else:
                        if big_enough_start_time is not None:
                            big_enough_start_time = None
                        cv2.putText(frame, "Boyut Yetersiz! (Gaza Bas)", (pred_x, pred_y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "UAV Vurus Alaninda Degil!", (padding_x, padding_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("UAV vurus alanindan cikti, takip sonlandiriliyor.")
                    tracking = False
                    tracker = None
                    kf = None
                    uav_bbox = None
                    current_roi_index = 0
                    big_enough_start_time = None
                    continue
            else:
                print("Tracker nesneyi kaybetti. Takip sonlandiriliyor.")
                tracking = False
                tracker = None
                kf = None
                uav_bbox = None
                current_roi_index = 0
                big_enough_start_time = None

        # FPS hesaplama
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Vurus Sayisi: {vurus_sayisi}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Video with Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
