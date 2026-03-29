# py-micro-pro Project Guide

This document is a complete learning guide for this repository.

## 1) What this project does

`py-micro-pro` is a real-time AI proctoring system.
It monitors a session using:
- webcam video (face, gaze, blink, mouth, hand)
- microphone audio (voice activity)
- optional screen capture (tab/screen switch detection)

At session end it generates:
- `violations_log.json` (structured totals + violation timeline)
- `violations_report.png` (visual summary chart)

## 2) Repository structure

- `README.md`: short project overview.
- `ai_proctoring/main.py`: entrypoint and orchestration loop.
- `ai_proctoring/detector.py`: MediaPipe-based video analysis.
- `ai_proctoring/screen_monitor.py`: screen change and tab-switch heuristics.
- `ai_proctoring/utils.py`: `Config` thresholds + `ViolationManager` logging.
- `ai_proctoring/report.py`: end-of-session report generation with Matplotlib.
- `ai_proctoring/requirements.txt`: Python dependencies.
- `ai_proctoring/face_landmarker.task`: required model file for face tracking.
- `ai_proctoring/hand_landmarker.task`: optional model file for hand detection.

## 3) Runtime flow (high level)

1. `main.py` starts webcam capture and initializes helpers.
2. For each frame:
- `ProctorDetector.process_frame(...)` returns vision signals.
- Voice monitor checks recent microphone activity.
- Optional screen monitor checks scene-change/tab-switch signals.
- Rules combine these signals into session `Status`.
3. New violations are logged once per status transition.
4. On exit, logs are saved and report image is generated.

## 4) Detection logic details

### 4.1 Face + head + gaze + liveness (`detector.py`)

`ProctorDetector` uses MediaPipe Tasks API and computes:
- `num_faces`: number of faces in frame.
- `head_status`: looking left/right/up/down or center.
- `gaze_status`: gaze left/right/up/down or center.
- `is_blinking`: eye-aspect-ratio threshold based blink event.
- `mouth_moving`: smoothed mouth dynamics with baseline + hysteresis.
- `hand_seen`: true when hand landmarks are present (if hand model exists).

Returned tuple:
```python
(num_faces, head_status, gaze_status, is_blinking, mouth_moving, hand_seen, annotated_frame)
```

### 4.2 Voice activity (`main.py` -> `VoiceRadarMonitor`)

- Uses `sounddevice` input stream.
- Computes RMS microphone level in callback.
- Auto-calibrates ambient threshold on startup.
- Keeps a short hold window (`VOICE_VIOLATION_HOLD_SEC`) so short speech bursts are caught.
- Optional UI radar overlay is drawn on video output.

### 4.3 Screen monitoring (`screen_monitor.py`)

When enabled (`SCREEN_MONITOR_ENABLED = True`):
- Captures display using `mss`.
- Compares grayscale frames to previous frame.
- Uses mean diff + changed pixel ratio + adaptive threshold.
- Flags `tab_switch_detected` with cooldown to avoid repeated spam.

## 5) Policy/rules used in `main.py`

Main status rules (simplified):
- No face for too long -> warning/violation.
- Multiple faces -> immediate violation.
- No blink for long duration -> spoofing/liveness violation.
- Sustained suspicious movement (off-center gaze/head, mouth move, hand seen) -> violation.
- Any recent voice activity -> immediate violation override.
- Tab switch signal -> logged as `Tab Switch Detected`.

Important: violations are added when status changes into a violation state, which helps avoid duplicate flood logs every frame.

## 6) Configuration and thresholds

All core thresholds are in `ai_proctoring/utils.py` inside `Config`.
Examples:
- `NO_FACE_THRESHOLD_SEC`
- `WARNING_THRESHOLD_SEC`
- `GAZE_AWAY_FRAMES_MAX`
- `LIVENESS_MAX_NO_BLINK_SEC`
- `VOICE_ACTIVITY_THRESHOLD`
- `VOICE_THRESHOLD_MARGIN`

If behavior feels too sensitive or too loose, tune these values first.

## 7) Data outputs

### 7.1 `violations_log.json`

`ViolationManager.save_log()` writes:
- `violations` (list of events with `timestamp` + `violation`)
- `total_suspicious_movements`
- `total_tab_switches`
- `total_gaze_movements`
- `total_mouth_movements`
- `total_voice_events`
- `total_movements` (sum of all movement/event categories)

### 7.2 `violations_report.png`

`report.py` creates:
- type-distribution bar chart
- timeline bar chart (violations over time)
- clean placeholder report if no violations happened

## 8) Setup from scratch

From repo root:

```bash
cd ai_proctoring
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9) Running the app

From `ai_proctoring/`:

```bash
python main.py
```

Optional flags:

```bash
python main.py --headless
python main.py --voice-threshold 0.012
```

## 10) Required files and dependencies

### Python packages
From `requirements.txt`:
- `opencv-python`
- `matplotlib`
- `numpy`
- `sounddevice`
- `mediapipe`
- `mss`

### Model assets
You need these files in `ai_proctoring/`:
- `face_landmarker.task` (required)
- `hand_landmarker.task` (optional but recommended)

If hand model is missing, app still runs and logs a warning.

## 11) Troubleshooting

### VS Code shows import errors
- Ensure interpreter is the project venv:
  - `/Users/mrithyunjayanm/Projects/py-pro/ai_proctoring/venv/bin/python`
- Reinstall dependencies:
  - `python -m pip install -r ai_proctoring/requirements.txt`

### `git push` says non-fast-forward
- Your branch is behind remote.
- Use:
  - `git pull --rebase origin main`
  - then `git push origin main`

### Webcam does not open
- Check camera permissions for Terminal/VS Code.
- Ensure no other app is exclusively using camera.

### Microphone monitor disabled
- `sounddevice` import or stream initialization failed.
- Check microphone permission and audio device availability.

### Screen monitor not working
- Ensure `mss` is installed.
- On macOS, grant Screen Recording permission to Terminal/VS Code.

## 12) Safe extension points

Best places to extend behavior:
- Add new thresholds in `Config` (`utils.py`).
- Add new event counters in `ViolationManager`.
- Add new detection outputs in `detector.py` and consume them in `main.py`.
- Add richer visualizations in `report.py`.
- Add CLI switches in `main.py` for runtime control.

## 13) Suggested learning path

1. Read `main.py` first to understand the control loop.
2. Read `detector.py` to understand signal extraction.
3. Read `utils.py` to learn how policy is tuned and logs are structured.
4. Read `report.py` to see post-session analysis output.
5. Run once with webcam and inspect generated JSON + PNG.
6. Tune one threshold and observe behavior change.

## 14) Known design tradeoffs

- Rule-based logic is easy to debug but can be sensitive to environment changes.
- Thresholds may need per-user/per-lighting calibration.
- Voice and screen heuristics can produce false positives in noisy setups.
- No dedicated automated test suite yet.

## 15) Quick command reference

```bash
# create + activate env
cd ai_proctoring
python3 -m venv venv
source venv/bin/activate

# install deps
python -m pip install -r requirements.txt

# run with UI
python main.py

# run headless
python main.py --headless

# run with custom voice threshold
python main.py --voice-threshold 0.012
```

---

If you want, next step can be a second document focused only on "how to modify this project safely" with a change checklist for each module.
