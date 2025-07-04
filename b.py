import cv2
import time
import torch
import math
import numpy as np
from ultralytics import YOLO
from vector_function import *

class UAVTracker:
    def __init__(self, video_path, model_path, uav_class_name="uav"):
        # GPU kontrolü
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device.upper()} for processing")
        
        # Video kaynağı
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Video boyutları
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Frame merkezi
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # Model yükleme
        self.model = YOLO(model_path)
        self.uav_class_name = uav_class_name.lower()
        
        # ROI ayarları
        self.roi_width = self.frame_width // 3
        self.roi_height = self.frame_height // 2
        self.roi_positions = [
            (0, 0), (0, self.roi_height),
            (self.roi_width, 0), (self.roi_width, self.roi_height),
            (self.roi_width * 2, 0), (self.roi_width * 2, self.roi_height)
        ]
        self.current_roi_index = 0
        
        # Takip durumu
        self.tracking = False
        self.tracker = None
        self.uav_bbox = None
        self.tracking_start_time = None
        self.big_enough_start_time = None
        self.track_frame_count = 0
        
        # Vuruş ayarları
        self.REQUIRED_TRACKING_DURATION = 4  # saniye
        self.REQUIRED_BBOX_WIDTH = int(self.frame_width * 0.06)
        self.REQUIRED_BBOX_HEIGHT = int(self.frame_height * 0.06)
        self.padding_x = int(self.frame_width * 0.25)
        self.padding_y = int(self.frame_height * 0.10)
        self.vurus_sayisi = 0
        
        # İrtifa-yönelme göstergesi
        self.altitude_padding = 60
        self.altitude_circle_radius = 40
        self.altitude_circle_center = (
            self.frame_width - self.altitude_padding - self.altitude_circle_radius,
            self.frame_height - self.altitude_padding - self.altitude_circle_radius
        )
        
        # Performans ölçümü
        self.start_time = time.time()
        self.frame_count = 0
        
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, False
            
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        if not self.tracking:
            self._roi_scan_mode(frame)
        else:
            self._tracking_mode(frame)
            
        # FPS hesaplama
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Bilgileri ekrana yaz
        self._draw_info(frame, fps)
        
        return frame, True
    
    def _roi_scan_mode(self, frame):
        x, y = self.roi_positions[self.current_roi_index]
        roi = frame[y:y + self.roi_height, x:x + self.roi_width]
        
        results = self.model.predict(roi, device=self.device)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id] if hasattr(self.model, "names") else self.uav_class_name
            
            if conf > 0.5 and class_name.lower() == self.uav_class_name:
                w = x2 - x1
                h = y2 - y1
                bbox = (x + x1, y + y1, w, h)
                
                self.tracker = cv2.legacy.TrackerCSRT_create()
                self.tracker.init(frame, bbox)
                
                self.tracking = True
                self.track_frame_count = 0
                self.tracking_start_time = time.time()
                self.uav_bbox = bbox
                self.big_enough_start_time = None
                
                print(f"UAV detected and tracking started: {bbox}")
                break
                
        self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_positions)
        cv2.rectangle(frame, (x, y), (x + self.roi_width, y + self.roi_height), (0, 255, 255), 2)
    
    def _tracking_mode(self, frame):
        success, bbox = self.tracker.update(frame)
        if not success:
            print("Tracker lost the object. Ending tracking.")
            self._reset_tracking()
            return
            
        x, y, w, h = map(int, bbox)
        tracked_box_center = (x + w // 2, y + h // 2)
        
        # İrtifa-yönelme göstergesi
        self._draw_altitude_indicator(frame, tracked_box_center)
        
        # PWM değerleri hesapla ve göster
        aileron_pwm, elevator_pwm = self._calculate_pwm_values(tracked_box_center)
        self._display_pwm_values(frame, aileron_pwm, elevator_pwm)
        
        # Takip edilen UAV'i çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.line(frame, self.frame_center, tracked_box_center, (0, 0, 255), 2)
        
        # Vuruş alanını çiz
        cv2.rectangle(frame, 
                     (self.padding_x, self.padding_y),
                     (self.frame_width - self.padding_x, self.frame_height - self.padding_y),
                     (0, 255, 255), 2)
        
        # UAV vuruş alanında mı kontrolü
        self._check_target_zone(x, y, w, h, frame)
        
        self.track_frame_count += 1
        
    def _draw_altitude_indicator(self, frame, tracked_box_center):
        vector_x = self.frame_center[0] - tracked_box_center[0]
        vector_y = self.frame_center[1] - tracked_box_center[1]
        
        normalized_x, normalized_y = normalized_vector(
            vector_x, vector_y, 
            self.frame_width // 2, self.frame_height // 2
        )
        
        magnitude = calculate_vector_magnitude(normalized_x, normalized_y)
        altitude_line_length = magnitude * 40
        angle = math.atan2(normalized_y, normalized_x)
        
        end_point = (
            int(self.altitude_circle_center[0] + altitude_line_length * math.cos(angle)),
            int(self.altitude_circle_center[1] + altitude_line_length * math.sin(angle))
        )
        
        cv2.line(frame, self.altitude_circle_center, end_point, (0, 255, 255), 2)
        cv2.circle(frame, self.altitude_circle_center, self.altitude_circle_radius, (40, 150, 255), 2)
        cv2.putText(frame, "Altitude-Heading", 
                   (self.altitude_circle_center[0] - 60, self.altitude_circle_center[1] + 70),
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (40, 150, 255), 2)
    
    def _calculate_pwm_values(self, tracked_box_center):
        vector_x = self.frame_center[0] - tracked_box_center[0]
        vector_y = self.frame_center[1] - tracked_box_center[1]
        
        normalized_x, normalized_y = normalized_vector(
            vector_x, vector_y, 
            self.frame_width // 2, self.frame_height // 2
        )
        
        aileron_pwm = int(1000 + normalized_x * 1000)
        elevator_pwm = int(1000 + normalized_y * 1000)
        
        # PWM değerlerini sınırla
        aileron_pwm = max(1000, min(aileron_pwm, 2000))
        elevator_pwm = max(1000, min(elevator_pwm, 2000))
        
        return aileron_pwm, elevator_pwm
    
    def _display_pwm_values(self, frame, aileron_pwm, elevator_pwm):
        cv2.putText(frame, f"Aileron PWM: {aileron_pwm}", (80, 600), 
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (40, 150, 255), 2)
        cv2.putText(frame, f"Elevator PWM: {elevator_pwm}", (80, 630), 
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (40, 150, 255), 2)
    
    def _check_target_zone(self, x, y, w, h, frame):
        uav_in_target = (
            x >= self.padding_x and
            y >= self.padding_y and
            (x + w) <= (self.frame_width - self.padding_x) and
            (y + h) <= (self.frame_height - self.padding_y)
        )
        
        if uav_in_target:
            cv2.putText(frame, "UAV in Target Zone!", 
                       (self.padding_x, self.padding_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if w >= self.REQUIRED_BBOX_WIDTH and h >= self.REQUIRED_BBOX_HEIGHT:
                if self.big_enough_start_time is None:
                    self.big_enough_start_time = time.time()
                
                duration = time.time() - self.big_enough_start_time
                cv2.putText(frame, f"Tracking Time: {duration:.1f}s",
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 0), 2)
                
                if duration >= self.REQUIRED_TRACKING_DURATION:
                    self.vurus_sayisi += 1
                    print(f"UAV tracked for {self.REQUIRED_TRACKING_DURATION}s. Successful hit!")
                    print(f"Total Hits: {self.vurus_sayisi}")
                    self._reset_tracking()
            else:
                if self.big_enough_start_time is not None:
                    self.big_enough_start_time = None
                cv2.putText(frame, "Size Too Small! (Increase Throttle)", 
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 255), 2)
        else:
            cv2.putText(frame, "UAV Not in Target Zone!", 
                       (self.padding_x, self.padding_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("UAV left target zone, ending tracking.")
            self._reset_tracking()
    
    def _draw_info(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Hit Count: {self.vurus_sayisi}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    def _reset_tracking(self):
        self.tracking = False
        self.tracker = None
        self.uav_bbox = None
        self.current_roi_index = 0
        self.big_enough_start_time = None
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Kullanım örneği
if __name__ == "__main__":
    tracker = UAVTracker("ornek.mp4", "best.pt")
    
    while True:
        frame, ret = tracker.process_frame()
        if not ret:
            break
            
        cv2.imshow("UAV Tracking System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    tracker.release()