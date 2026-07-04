import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('E:\MyPractice\yolov8n.pt')

# Initialize the video capture object
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("E:\MyPractice\WhatsApp Video 2026-06-29 at 4.06.17 PM.mp4")  # Replace with your video file path    
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()