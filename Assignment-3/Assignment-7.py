import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  

# Function to get consistent colors for each class
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        (base_colors[color_index][i] + increments[color_index][i] *
         (cls_num // len(base_colors))) % 256
        for i in range(3)
    ]
    return tuple(color)

# Start webcam
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #YOLOv8 tracking
    results = model.track(frame, stream=True)

    for result in results:
        classes_names = result.names  # Class name mapping

        for box in result.boxes:
            if box.conf[0] > 0.4:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
                cls = int(box.cls[0])  # Class ID
                class_name = classes_names[cls]
                colour = getColours(cls)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                # Draw label
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}',
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # Show the frame
    cv2.imshow("YOLOv8 Live", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
