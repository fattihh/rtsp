import cv2
from ultralytics import YOLO
import easyocr

# ——— Ayarlar ———
IMG_PATH         = 'plaka.jpg'            # İşlenecek resim
OUTPUT_PATH      = 'output.jpg'           # Kaydedilecek sonuç
CAR_MODEL_PATH   = 'yolov8l.pt'           # Araç tespit modeli
PLATE_MODEL_PATH = 'plate_detection.pt'   # Plaka tespit modeli
OCR_LANGS        = ['tr','en']            # EasyOCR dilleri
# ————————————

# 1) Resmi yükle ve kontrol et
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Resim yüklenemedi: {IMG_PATH}")


# 2) Modelleri ve OCR okuyucusunu başlat
car_model   = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
reader      = easyocr.Reader(OCR_LANGS)

# 3) Araç tespiti
cars = car_model(img, conf=0.5, iou=0.5)[0]
for (x1, y1, x2, y2) in cars.boxes.xyxy.cpu().numpy().astype(int):
    # Araç kutusunu kırmızıyla çiz
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    car_roi = img[y1:y2, x1:x2]

    # 4) Plaka tespiti araç ROI’sinde
    plates = plate_model(car_roi, conf=0.3, iou=0.3)[0]
    for (px1, py1, px2, py2) in plates.boxes.xyxy.cpu().numpy().astype(int):
        # Plaka kutusunun koordinatlarını orijinal frame’e çevir
        fx1, fy1 = x1 + px1, y1 + py1
        fx2, fy2 = x1 + px2, y1 + py2

        # Plaka kutusunu griyle çiz
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (128,128,128), 2)
        plate_roi = car_roi[py1:py2, px1:px2]

        # 5) OCR ile plaka metnini oku
        texts = reader.readtext(plate_roi, detail=0, paragraph=False)
        plate_text = ''.join(texts).upper()
        print("OCR sonucu:", plate_text)

        # 6) Metni plaka kutusunun sol üst köşesine yaz
        #    Biraz içeri alması için +2 piksel ofset kullanıyoruz
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
        text_size, _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
        text_w, text_h = text_size

        text_x = fx1 + 2
        text_y = fy1 + text_h + 2  # kutunun içinde, üst kenarın hemen altına

        cv2.putText(img, plate_text, (text_x, text_y),
                    font, font_scale, (0,255,0), thickness)

# 7) Sonucu kaydet ve göster
cv2.imwrite(OUTPUT_PATH, img)
print(f"Çıktı kaydedildi: {OUTPUT_PATH}")


cv2.waitKey(0)
cv2.destroyAllWindows()
