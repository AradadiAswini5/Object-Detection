
import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")   # make sure best.pt is in the same folder

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    # Run YOLO detection
    results = model(frame)
    frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Fruit Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
