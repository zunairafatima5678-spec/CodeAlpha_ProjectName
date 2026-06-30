import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def main():
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30, n_init=3)

    # Colors BGR - Category Based
    CATEGORY_COLORS = {
        'person': (0, 0, 255), # Red - People
        'vehicle': (0, 255, 0), # Green - Cars, Bus, Truck etc
        'animal': (0, 165, 255), # Brown/Yellow - Cat, Dog, Horse etc
        'household': (255, 0, 255), # Magenta - Chair, TV, Laptop etc
        'sports': (255, 0, 128), # Pink - Sports ball, Racket etc
        'food': (255, 255, 0) # Sky Blue - Pizza, Apple etc
    }
    DEFAULT_COLOR = (128, 128, 128) # Gray for everything else

    # YOLO COCO 80 classes divided into category
    CATEGORIES = {
        'person': ['person'],
        'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
        'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
        'household': ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                      'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                      'hair drier', 'toothbrush', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
        'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                   'skateboard', 'surfboard', 'tennis racket'],
        'food': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']
    }

    # Webcam Setup 
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        raise FileNotFoundError("Webcam not found. Check camera permission.")

    # Webcam's width, height and FPS 10 fix 
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = 10.0
    print(f"✅ Recording FPS set to: {fps}")
    out = cv2.VideoWriter('CodeAlpha_Task4_Webcam_Recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("✅ Webcam ON. press'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, verbose=False)[0]

        # Detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_name))

        # Tracking and Drawing
        for track in tracker.update_tracks(detections, frame=frame):
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cls_name = track.det_class

            # Category check for colour
            color = DEFAULT_COLOR
            for category, classes in CATEGORIES.items():
                if cls_name in classes:
                    color = CATEGORY_COLORS[category]
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{track.track_id} {cls_name}', (x1, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame) # Recording save hoti rahegi
        cv2.imshow("Webcam Tracking - Press q to quit", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Done: CodeAlpha_Task4_Webcam_Recording.mp4 saved")

if __name__ == "__main__":
    main()
