import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"E:\MyPractice\yolov8n.pt")  # Change path if needed

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect and track only persons (Class 0)
    results = model.track(
        frame,
        persist=True,
        classes=[0],      # Detect only persons
        conf=0.5          # Confidence threshold
    )

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Count detected persons
    person_count = 0
    if results[0].boxes is not None:
        person_count = len(results[0].boxes)

    # Display customer count
    cv2.putText(
        annotated_frame,
        f"Customers: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show output
    cv2.imshow("YOLOv8 Customer Detection & Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()