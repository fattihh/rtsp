import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50


def extract_feat(model, preprocess, patch):
    """
    Extracts a normalized 2048-D feature vector from an image patch using a ResNet50 backbone.
    """
    input_tensor = preprocess(patch).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    with torch.no_grad():
        feat = model(input_tensor)               # Shape: (1, 2048)
    return F.normalize(feat, dim=1)             # L2-normalized


def main(video_source=0,
         yolo_weights="/home/fatih/Masaüstü/uav/best.pt",
         similarity_thresh=0.7):
    # 1. Load YOLOv8 detector
    detector = YOLO(yolo_weights)

    # 2. Prepare ResNet50 feature extractor
    cnn = resnet50(pretrained=True)
    cnn.fc = torch.nn.Identity()  # Remove classification head
    cnn.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Open video stream
    cap = cv2.VideoCapture("/home/fatih/Masaüstü/uav/ornek_3.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from source.")
        return

    # 4. Detect object in first frame
    results = detector(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()         # Nx4 array
    scores = results.boxes.conf.cpu().numpy()        # N scores
    if len(scores) == 0:
        print("No detections in the first frame.")
        return
    best_idx = scores.argmax()
    x1, y1, x2, y2 = boxes[best_idx].astype(int)
    init_bbox = (x1, y1, x2 - x1, y2 - y1)

    # 5. Initialize CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, init_bbox)

    # 6. Extract template features
    template = frame[y1:y2, x1:x2]
    template_feat = extract_feat(cnn, preprocess, template)

    # 7. Tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = map(int, bbox)
            candidate = frame[y:y+h, x:x+w]
            cand_feat = extract_feat(cnn, preprocess, candidate)

            # Compute cosine similarity
            sim = F.cosine_similarity(template_feat, cand_feat).item()
            if sim < similarity_thresh:
                # Drift detected -> re-detect with YOLO
                results = detector(frame)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                if len(scores) > 0:
                    best_idx = scores.argmax()
                    x1, y1, x2, y2 = boxes[best_idx].astype(int)
                    init_bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, init_bbox)
                    template = frame[y1:y2, x1:x2]
                    template_feat = extract_feat(cnn, preprocess, template)
                    cv2.putText(frame, f"Re-detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                # Normal tracking display
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Sim: {sim:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 + CSRT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use 0 for webcam or replace with video file path
    main(video_source="/home/fatih/Masaüstü/uav/ornek_3.mp4",
         yolo_weights="best.pt",
         similarity_thresh=0.7)
