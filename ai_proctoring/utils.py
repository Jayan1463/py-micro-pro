import time
import json
import os

class Config:
    PROCESS_NTH_FRAME = 1 # Deprecated, running on every frame natively
    NO_FACE_THRESHOLD_SEC = 2.0
    WARNING_THRESHOLD_SEC = 0.5 
    LOOKING_AWAY_FRAMES_MAX = 30 # Since we are checking every frame at 30FPS, this is exactly 1 second of looking away
    LIVENESS_MAX_NO_BLINK_SEC = 20.0 # If no blink is detected for 20 seconds, flag as spoofed/not lively
    GAZE_AWAY_FRAMES_MAX = 45 # 1.5 seconds of sustained off-screen eye darting completely independently of head movement

class ViolationManager:
    def __init__(self, log_file="violations_log.json"):
        self.log_file = log_file
        self.violations = []
        self.start_time = time.time()

    def add_violation(self, violation_type):
        timestamp = int(time.time() - self.start_time)
        self.violations.append({
            "timestamp": timestamp,
            "violation": violation_type
        })
        print(f"[VIOLATION] {violation_type} at {timestamp}s")

    def save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.violations, f, indent=4)
        print(f"Violations saved to {self.log_file}")

    def get_violations(self):
        return self.violations
