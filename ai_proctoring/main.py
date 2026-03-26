import cv2
import time
import argparse
from detector import ProctorDetector
from utils import ViolationManager, Config
from report import generate_report

def main():
    parser = argparse.ArgumentParser(description="AI Proctoring System")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam is used.", default=None)
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Starting session with video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Starting session with webcam.")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    detector = ProctorDetector()
    violation_manager = ViolationManager()
    
    no_face_start_time = None
    looking_away_frames = 0
    
    current_status = "Normal"
    status_color = (0, 255, 0)
    
    print("Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
                
            frame = cv2.resize(frame, (640, 480))
            if args.video:
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                current_time = time.time()
            
            num_faces, head_status, gaze_status, is_blinking, frame = detector.process_frame(frame)
            
            new_status = "Normal"
            new_color = (0, 255, 0)
            
            # Rule 1: No Face
            if num_faces == 0:
                if no_face_start_time is None:
                    no_face_start_time = current_time
                
                elapsed = current_time - no_face_start_time
                if elapsed > Config.NO_FACE_THRESHOLD_SEC:
                    new_status = "Violation: No Face Detected"
                    new_color = (0, 0, 255)
                elif elapsed > Config.WARNING_THRESHOLD_SEC:
                    new_status = "Warning: No face detected"
                    new_color = (0, 165, 255)
                else:
                    # Prevent flicker
                    new_status = current_status
                    new_color = status_color
            else:
                no_face_start_time = None
                
                # Rule 2: Multiple Faces
                if num_faces > 1:
                    new_status = "Violation: Multiple People Detected"
                    new_color = (0, 0, 255)
                # Rule 3: Single Face, check movement & LIVENESS & Gaze
                elif num_faces == 1:
                    # Initialize blink tracker if needed
                    if 'last_blink_time' not in locals():
                        last_blink_time = current_time
                        
                    if is_blinking:
                        last_blink_time = current_time
                        
                    elapsed_liveness = current_time - last_blink_time
                    
                    if elapsed_liveness > Config.LIVENESS_MAX_NO_BLINK_SEC:
                        new_status = "Violation: Spoofing / Not Lively"
                        new_color = (0, 0, 255)
                    else:
                        is_looking_away = ("Looking" in head_status) or ("Gaze" in gaze_status and "Center" not in gaze_status)
                        
                        if is_looking_away:
                            looking_away_frames += 1
                            if looking_away_frames > Config.GAZE_AWAY_FRAMES_MAX:
                                new_status = "Violation: Suspicious Movement"
                                new_color = (0, 0, 255)
                            elif current_status == "Violation: Suspicious Movement":
                                # Maintain violation state while slowly recovering
                                new_status = "Violation: Suspicious Movement"
                                new_color = (0, 0, 255)
                            else:
                                new_status = "Normal" 
                                new_color = (0, 255, 0)
                        else:
                            looking_away_frames = max(0, looking_away_frames - 2)
                            if looking_away_frames > 0 and current_status == "Violation: Suspicious Movement":
                                new_status = "Violation: Suspicious Movement"
                                new_color = (0, 0, 255)
                            else:
                                new_status = "Normal"
                                new_color = (0, 255, 0)
                            
            # Detect state change correctly and log violations ONCE
            if new_status != current_status:
                current_status = new_status
                status_color = new_color
                if "Violation:" in current_status:
                    violation_type = current_status.replace("Violation: ", "")
                    violation_manager.add_violation(violation_type)

            # Display Status overlay if not headless
            if not args.headless:
                cv2.putText(frame, f"Status: {current_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.imshow("AI Proctoring", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode processing video till end, or just sleep slightly if webcam to prevent blazing loop
                # Just break if video ends
                pass
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save logs and generate report
        try:
            violation_manager.save_log()
            generate_report(violation_manager.get_violations())
        except Exception as e:
            print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()
