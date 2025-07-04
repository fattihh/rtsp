import cv2
from ultralytics import YOLO
import easyocr

# 1. Modelleri yükle
car_model   = YOLO('yolov8l.pt')         # Araçları tespit eden model
plate_model = YOLO('plate_detection.pt') # Plakaları tespit eden özel model

# 2. EasyOCR okuyucusunu başlat
reader = easyocr.Reader(['en','tr'])     # TR plakalarındaki karakterler için 'tr' ekledik

# 3. Video/kaynak aç
cap = cv2.VideoCapture('video.mp4')      # Kamerayla çalışmak istersen: 0 gir

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Araç tespiti
    cars = car_model(frame, conf=0.5, iou=0.5)[0]
    for box in cars.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        car_roi = frame[y1:y2, x1:x2]

        # 5. Plaka tespiti (sadece araç bölgesinde)
        plates = plate_model(car_roi, conf=0.3, iou=0.3)[0]
        for pbox in plates.boxes.xyxy.cpu().numpy():
            px1, py1, px2, py2 = map(int, pbox)
            plate_roi = car_roi[py1:py2, px1:px2]

            # 6. OCR ile plaka okuma
            ocr_res = reader.readtext(plate_roi, detail=0, paragraph=False)
            plate_text = ' '.join(ocr_res).upper()

            # 7. Sonuçları orijinal frame üzerine çiz
            # Plaka kutusunu araç-koordinatından çerçeveye çevir
            fx1, fy1 = x1 + px1, y1 + py1
            fx2, fy2 = x1 + px2, y1 + py2
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (fx1, fy1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Konsola da yazdırabilirsin
            print(f"Plaka bulundu: {plate_text}")

    # 8. Görüntüyü göster
    cv2.imshow('Araç+Plaka Tespiti & OCR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
