import time
from collections import deque

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import mss
except Exception:
    mss = None


class ScreenMonitor:
    """
    Idle-ready screen monitoring helper.

    This module is intentionally not wired into main.py yet.
    It can be integrated later to capture the display and run
    basic anomaly checks.
    """

    def __init__(self, monitor_index=1, fps=1.0):
        self.monitor_index = int(monitor_index)
        self.fps = max(0.2, float(fps))
        self.frame_interval = 1.0 / self.fps
        self.last_capture_time = 0.0
        self.started = False
        self._sct = None
        self._prev_gray = None
        self._recent_scores = deque(maxlen=30)
        self._last_tab_switch_time = 0.0

    def start(self):
        if self.started:
            return True
        if mss is None or cv2 is None or np is None:
            return False
        self._sct = mss.mss()
        self.started = True
        return True

    def stop(self):
        if self._sct is not None:
            self._sct.close()
        self._sct = None
        self._prev_gray = None
        self._recent_scores.clear()
        self._last_tab_switch_time = 0.0
        self.started = False

    def capture_frame(self):
        """
        Capture one screen frame as BGR numpy array.
        Returns None when monitor is unavailable or monitor is not started.
        """
        if not self.started or self._sct is None:
            return None
        monitors = self._sct.monitors
        if self.monitor_index >= len(monitors):
            return None
        raw = self._sct.grab(monitors[self.monitor_index])
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def poll(self):
        """
        Rate-limited capture + lightweight scene-change analysis.
        Returns:
            {
              "timestamp": <epoch_sec>,
              "scene_changed": <bool>,
              "change_score": <float>,
              "frame": <np.ndarray | None>
            }
        """
        now = time.time()
        if (now - self.last_capture_time) < self.frame_interval:
            return None

        self.last_capture_time = now
        frame = self.capture_frame()
        if frame is None or cv2 is None:
            return {
                "timestamp": now,
                "scene_changed": False,
                "change_score": 0.0,
                "frame": None,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_changed = False
        score = 0.0
        changed_ratio = 0.0
        tab_switch_detected = False
        adaptive_threshold = 18.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            score = float(np.mean(diff))
            scene_changed = score > 18.0
            changed_ratio = float(np.mean(diff > 25))

            baseline = float(np.median(self._recent_scores)) if self._recent_scores else 0.0
            adaptive_threshold = max(12.0, baseline * 2.2)
            cooldown_ok = (now - self._last_tab_switch_time) >= 2.0
            tab_switch_detected = (
                cooldown_ok
                and score > adaptive_threshold
                and changed_ratio > 0.20
            )
            if tab_switch_detected:
                self._last_tab_switch_time = now
        self._recent_scores.append(score)
        self._prev_gray = gray

        return {
            "timestamp": now,
            "scene_changed": scene_changed,
            "change_score": score,
            "changed_ratio": changed_ratio,
            "adaptive_threshold": adaptive_threshold,
            "tab_switch_detected": tab_switch_detected,
            "frame": frame,
        }
