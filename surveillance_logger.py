import cv2
import time
import datetime
from ultralytics import YOLO

def run_surveillance():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    
    # Open log file
    with open("events.log", "a") as f:
        f.write("Timestamp,Class,Confidence,BBox\n")

    print("Surveillance Active. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        triggered = False

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Check for Person (0) or Cell phone (67) with conf > 0.65
            if cls_id in [0, 67] and conf > 0.65:
                triggered = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Log event
                with open("events.log", "a") as f:
                    f.write(f"{timestamp},{model.names[cls_id]},{conf:.2f},[{x1} {y1} {x2} {y2}]\n")
                
                # Save snapshot
                cv2.imwrite(f"alerts/{timestamp}.jpg", frame)
                
                # Draw alert on stream
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "LOGGED", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if triggered:
            cv2.putText(annotated_frame, "ALERT: TARGET DETECTED & LOGGED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Task 8: Edge-Triggered Surveillance", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_surveillance()
