import cv2

# 1) Parametreleri tanımla ve istediğin değerleri ata
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

# 3) Video kaynağını aç, ROI seç ve izlemeyi başlat
cap = cv2.VideoCapture("ornek.mp4")
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Kamera açılamadı")

# İlk karede takip edilecek nesneyi seç (manuel ROI)
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

# 4) Ana döngü: her karede güncelle ve çiz
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        # Başarıyla izleniyorsa dikdörtgen çiz
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Takip ediliyor", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.putText(frame, "Takip kaybedildi", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("CSRT Takip", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
