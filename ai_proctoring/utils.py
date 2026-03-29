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
    VOICE_ACTIVITY_THRESHOLD = 0.025
    VOICE_VIOLATION_HOLD_SEC = 1.0
    VOICE_CALIBRATION_SEC = 2.0
    VOICE_THRESHOLD_MARGIN = 2.5
    VOICE_THRESHOLD_MIN = 0.015

class ViolationManager:
    def __init__(self, log_file="violations_log.json"):
        self.log_file = log_file
        self.violations = []
        self.gaze_movements = []
        self.mouth_movements = []
        self.voice_events = []
        self.start_time = time.time()
        self._load_log()

    def _load_log(self):
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                # Backward compatibility with older format.
                self.violations = data
            elif isinstance(data, dict):
                self.violations = data.get("violations", [])
                self.gaze_movements = data.get("gaze_movements", [])
                self.mouth_movements = data.get("mouth_movements", [])
                self.voice_events = data.get("voice_events", [])
        except Exception:
            # Ignore malformed existing logs and start clean.
            self.violations = []
            self.gaze_movements = []
            self.mouth_movements = []
            self.voice_events = []

    def add_violation(self, violation_type):
        timestamp = int(time.time() - self.start_time)
        self.violations.append({
            "timestamp": timestamp,
            "violation": violation_type
        })

    def add_gaze_movement(self, gaze_status):
        timestamp = int(time.time() - self.start_time)
        self.gaze_movements.append({
            "timestamp": timestamp,
            "gaze": gaze_status
        })

    def add_mouth_movement(self):
        timestamp = int(time.time() - self.start_time)
        self.mouth_movements.append({
            "timestamp": timestamp,
            "event": "Mouth Move"
        })

    def add_voice_event(self, level):
        timestamp = int(time.time() - self.start_time)
        self.voice_events.append({
            "timestamp": timestamp,
            "event": "Voice Activity",
            "level": float(level)
        })

    def save_log(self):
        suspicious_movement_total = sum(
            1 for v in self.violations if v.get("violation") == "Suspicious Movement"
        )
        with open(self.log_file, 'w') as f:
            json.dump(
                {
                    "violations": self.violations,
                    "total_suspicious_movements": suspicious_movement_total,
                    "total_gaze_movements": len(self.gaze_movements),
                    "total_mouth_movements": len(self.mouth_movements),
                    "total_voice_events": len(self.voice_events),
                    "total_movements": (
                        suspicious_movement_total
                        + len(self.gaze_movements)
                        + len(self.mouth_movements)
                        + len(self.voice_events)
                    )
                },
                f,
                indent=4
            )
        print(
            "Session totals saved: "
            f"suspicious={suspicious_movement_total}, "
            f"gaze={len(self.gaze_movements)}, "
            f"mouth={len(self.mouth_movements)}, "
            f"voice={len(self.voice_events)}, "
            f"overall={suspicious_movement_total + len(self.gaze_movements) + len(self.mouth_movements) + len(self.voice_events)}"
        )

    def get_violations(self):
        return self.violations

    def get_gaze_movements(self):
        return self.gaze_movements
