import cv2
import time
from ultralytics import YOLO

def run_dashboard():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    prev_time = 0

    print("Press 'q' to quit, 's' to save a frame.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Inference & FPS
        results = model(frame, verbose=False)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Task 7: Real-Time Dashboard", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): break
        elif key == ord('s'):
            cv2.imwrite(f"snapshot_{int(curr_time)}.jpg", annotated_frame)
            print("Snapshot saved!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dashboard()
