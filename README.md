# py-micro-pro

`py-micro-pro` is a real-time AI proctoring project that uses a webcam feed to monitor exam/session integrity.  
It detects suspicious behavior such as:
- no face detected
- multiple people in frame
- prolonged off-screen gaze or head movement
- repeated mouth movement
- potential spoofing/liveness issues (no blink over a long duration)

The system logs aggregated movement totals and violation counts, and can generate a session summary report at the end.

## Main Module
- `ai_proctoring/`: Core proctoring app (detector, session logic, logging, report generation)

## Run
```bash
cd ai_proctoring
python3 main.py
```
