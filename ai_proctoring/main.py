import cv2
import time
import argparse
import threading
import numpy as np
from detector import ProctorDetector
from utils import ViolationManager, Config
from report import generate_report

try:
    import sounddevice as sd
except Exception:
    sd = None


class VoiceRadarMonitor:
    def __init__(self, threshold=0.025, calibration_seconds=2.0):
        self.threshold = threshold
        self.level = 0.0
        self.last_voice_time = 0.0
        self.stream = None
        self.lock = threading.Lock()
        self.available = sd is not None
        if not self.available:
            print("Warning: 'sounddevice' not available. Voice radar is disabled.")
            return
        try:
            self.stream = sd.InputStream(channels=1, callback=self._audio_callback)
            self.stream.start()
            self.calibrate(calibration_seconds)
            print(f"Voice radar active: microphone monitoring enabled (threshold={self.threshold:.3f}).")
        except Exception as e:
            self.available = False
            self.stream = None
            print(f"Warning: Could not start microphone stream. Voice radar disabled. ({e})")

    def _audio_callback(self, indata, frames, callback_time, status):
        if status:
            return
        rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size > 0 else 0.0
        now = time.time()
        with self.lock:
            self.level = rms
            if rms >= self.threshold:
                self.last_voice_time = now

    def snapshot(self):
        with self.lock:
            return self.level, self.last_voice_time

    def calibrate(self, seconds):
        if not self.available:
            return
        print(f"Calibrating ambient voice level for {seconds:.1f}s... stay quiet.")
        end_time = time.time() + max(0.5, seconds)
        samples = []
        while time.time() < end_time:
            level, _ = self.snapshot()
            samples.append(level)
            time.sleep(0.05)
        if not samples:
            return
        ambient = float(np.percentile(samples, 90))
        tuned = max(
            Config.VOICE_THRESHOLD_MIN,
            ambient * Config.VOICE_THRESHOLD_MARGIN
        )
        self.threshold = tuned

    def is_voice_violation(self):
        if not self.available:
            return False
        _, last_voice_time = self.snapshot()
        if last_voice_time <= 0:
            return False
        return (time.time() - last_voice_time) <= Config.VOICE_VIOLATION_HOLD_SEC

    def draw_radar(self, frame):
        if not self.available:
            return frame
        level, _ = self.snapshot()
        h, w = frame.shape[:2]
        center = (w - 90, 90)
        max_radius = 55
        base_color = (80, 80, 80)

        for radius in (20, 35, 50):
            cv2.circle(frame, center, radius, base_color, 1)
        cv2.line(frame, (center[0] - max_radius, center[1]), (center[0] + max_radius, center[1]), base_color, 1)
        cv2.line(frame, (center[0], center[1] - max_radius), (center[0], center[1] + max_radius), base_color, 1)

        normalized = min(1.0, level / max(self.threshold * 2.5, 1e-6))
        active_radius = int(10 + normalized * 45)
        active_color = (0, 0, 255) if level >= self.threshold else (0, 255, 0)
        cv2.circle(frame, center, active_radius, active_color, 2)
        cv2.circle(frame, center, 4, active_color, -1)
        cv2.putText(frame, "VOICE RADAR", (w - 170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, f"Lvl:{level:.3f}", (w - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1)
        return frame

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def main():
    parser = argparse.ArgumentParser(description="AI Proctoring System")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument(
        "--voice-threshold",
        type=float,
        default=None,
        help="Manual voice threshold (RMS). If omitted, auto calibration is used."
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    print("Starting live session with webcam.")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = ProctorDetector()
    violation_manager = ViolationManager()
    voice_monitor = VoiceRadarMonitor(
        threshold=Config.VOICE_ACTIVITY_THRESHOLD,
        calibration_seconds=Config.VOICE_CALIBRATION_SEC
    )
    if args.voice_threshold is not None and voice_monitor.available:
        voice_monitor.threshold = max(0.001, float(args.voice_threshold))
        print(f"Manual voice threshold applied: {voice_monitor.threshold:.3f}")
    
    no_face_start_time = None
    looking_away_frames = 0
    
    current_status = "Normal"
    status_color = (0, 255, 0)
    last_gaze_status = None
    last_mouth_moving = False
    last_voice_violation = False
    
    print("Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam stream.")
                break
                
            frame = cv2.resize(frame, (640, 480))
            current_time = time.time()
            
            num_faces, head_status, gaze_status, is_blinking, mouth_moving, hand_seen, frame = detector.process_frame(frame)
            if gaze_status != last_gaze_status:
                violation_manager.add_gaze_movement(gaze_status)
                last_gaze_status = gaze_status
            if mouth_moving and not last_mouth_moving:
                violation_manager.add_mouth_movement()
            last_mouth_moving = mouth_moving
            
            new_status = "Normal"
            new_color = (0, 255, 0)
            voice_violation = voice_monitor.is_voice_violation()
            if voice_violation and not last_voice_violation:
                level, _ = voice_monitor.snapshot()
                violation_manager.add_voice_event(level)
            last_voice_violation = voice_violation
            
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
                        suspicious_trigger = is_looking_away or mouth_moving or hand_seen
                        
                        if suspicious_trigger:
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

            # Rule 4: Any talking/voice is an immediate violation.
            if voice_violation:
                new_status = "Violation: Voice Detected (Talking)"
                new_color = (0, 0, 255)

            # Detect state change correctly and log violations ONCE
            if new_status != current_status:
                current_status = new_status
                status_color = new_color
                if "Violation:" in current_status:
                    violation_type = current_status.replace("Violation: ", "")
                    violation_manager.add_violation(violation_type)

            # Display Status overlay if not headless
            if not args.headless:
                frame = voice_monitor.draw_radar(frame)
                cv2.putText(frame, f"Status: {current_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.imshow("AI Proctoring", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, keep processing live webcam frames.
                pass
    finally:
        # Cleanup
        cap.release()
        voice_monitor.close()
        cv2.destroyAllWindows()
        
        # Save logs and generate report
        try:
            violation_manager.save_log()
            generate_report(violation_manager.get_violations())
        except Exception as e:
            print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()
    
