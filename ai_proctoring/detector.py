import cv2
import mediapipe as mp
import time
import math
import os

class ProctorDetector:
    def __init__(self):
        # Configure MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        options = self.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=self.VisionRunningMode.VIDEO,
            num_faces=5
        )
        self.landmarker = self.FaceLandmarker.create_from_options(options)
        hand_model_path = 'hand_landmarker.task'
        self.hand_landmarker = None
        if os.path.exists(hand_model_path):
            hand_options = self.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=self.VisionRunningMode.VIDEO,
                num_hands=2
            )
            self.hand_landmarker = self.HandLandmarker.create_from_options(hand_options)
        else:
            print(
                "Warning: 'hand_landmarker.task' not found. "
                "Hand-sign detection is disabled until this model file is added."
            )
        self.frame_idx = 0
        self.prev_mouth_ratio = None

    def process_frame(self, frame):
        """
        Processes a BGR frame, returns:
        (num_faces, head_status, gaze_status, is_blinking, mouth_moving, hand_seen, annotated_frame)
        """
        self.frame_idx += 1
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # timestamp must be sequentially increasing in milliseconds
        timestamp_ms = self.frame_idx * 33 
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        num_faces = 0
        head_status = "Center"
        gaze_status = "Gaze Center"
        is_blinking = False
        mouth_moving = False
        hand_seen = False
        
        if results.face_landmarks:
            num_faces = len(results.face_landmarks)
            
            # Primary face
            primary_face = results.face_landmarks[0]
            
            # Pose Estimation using standard MediaPipe landmarks
            nose_x = primary_face[1].x
            left_edge_x = primary_face[234].x
            right_edge_x = primary_face[454].x
            
            dist_left = abs(nose_x - left_edge_x)
            dist_right = abs(nose_x - right_edge_x)
            
            total_dist = dist_left + dist_right
            if total_dist > 0:
                ratio = dist_left / total_dist
                # Low ratio means nose is near left edge
                if ratio < 0.25:
                    head_status = "Looking Left"
                elif ratio > 0.75:
                    head_status = "Looking Right"
                else:
                    nose_y = primary_face[1].y
                    chin_y = primary_face[152].y
                    forehead_y = primary_face[10].y
                    
                    dist_down = abs(chin_y - nose_y)
                    dist_up = abs(nose_y - forehead_y)
                    
                    if (dist_down + dist_up) > 0:
                        v_ratio = dist_up / (dist_down + dist_up)
                        # High v_ratio means nose is closer to chin, so looking down
                        if v_ratio > 0.65:
                            head_status = "Looking Down"
                        elif v_ratio < 0.35:
                            head_status = "Looking Up"

            # Gaze Estimation using Iris Tracking (horizontal + vertical)
            def get_horizontal_gaze(left_corner, right_corner, iris_center):
                dx = right_corner.x - left_corner.x
                dy = right_corner.y - left_corner.y
                length_sq = dx*dx + dy*dy
                if length_sq == 0:
                    return 0.5
                return ((iris_center.x - left_corner.x) * dx + (iris_center.y - left_corner.y) * dy) / length_sq

            def get_vertical_gaze(upper_lid, lower_lid, iris_center):
                dy = lower_lid.y - upper_lid.y
                if dy == 0:
                    return 0.5
                return (iris_center.y - upper_lid.y) / dy

            # Left Eye physically (Screen Right): Outer = 33, Inner = 133, Iris = 468
            # In screen coords: left_corner = 133, right_corner = 33
            r_eye_gaze = get_horizontal_gaze(primary_face[133], primary_face[33], primary_face[468])
            
            # Right Eye physically (Screen Left): Inner = 362, Outer = 263, Iris = 473
            # In screen coords: left_corner = 263, right_corner = 362
            l_eye_gaze = get_horizontal_gaze(primary_face[263], primary_face[362], primary_face[473])
            
            avg_gaze = (l_eye_gaze + r_eye_gaze) / 2.0
            # Vertical ratio: lower value = iris closer to upper lid (looking up)
            l_eye_v_gaze = get_vertical_gaze(primary_face[159], primary_face[145], primary_face[468])
            r_eye_v_gaze = get_vertical_gaze(primary_face[386], primary_face[374], primary_face[473])
            avg_v_gaze = (l_eye_v_gaze + r_eye_v_gaze) / 2.0

            horizontal = "Center"
            vertical = "Center"

            if avg_gaze < 0.40:
                horizontal = "Left"
            elif avg_gaze > 0.60:
                horizontal = "Right"

            if avg_v_gaze < 0.38:
                vertical = "Up"
            elif avg_v_gaze > 0.62:
                vertical = "Down"

            if horizontal != "Center" and vertical != "Center":
                gaze_status = f"Gaze {horizontal} {vertical}"
            elif horizontal != "Center":
                gaze_status = f"Gaze {horizontal}"
            elif vertical != "Center":
                gaze_status = f"Gaze {vertical}"
            
            # Liveness: Blink Detection
            def eye_aspect_ratio(p1, p4, upper, lower):
                h_dist = math.hypot(p1.x - p4.x, p1.y - p4.y)
                v_dist = math.hypot(upper.x - lower.x, upper.y - lower.y)
                return v_dist / h_dist if h_dist > 0 else 0

            # MediaPipe eye landmarks
            # Left Eye: 33 (outer), 133 (inner), 159 (upper), 145 (lower)
            # Right Eye: 362 (inner), 263 (outer), 386 (upper), 374 (lower)
            left_ear = eye_aspect_ratio(primary_face[33], primary_face[133], primary_face[159], primary_face[145])
            right_ear = eye_aspect_ratio(primary_face[362], primary_face[263], primary_face[386], primary_face[374])
            
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < 0.18:
                is_blinking = True

            # Mouth movement detection (speaking/open-close dynamics)
            upper_lip = primary_face[13]
            lower_lip = primary_face[14]
            left_mouth = primary_face[78]
            right_mouth = primary_face[308]

            mouth_height = math.hypot(upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y)
            mouth_width = math.hypot(left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y)
            mouth_ratio = (mouth_height / mouth_width) if mouth_width > 0 else 0.0

            mouth_open = mouth_ratio > 0.10
            if self.prev_mouth_ratio is None:
                mouth_changed = False
            else:
                mouth_changed = abs(mouth_ratio - self.prev_mouth_ratio) > 0.015
            self.prev_mouth_ratio = mouth_ratio
            mouth_moving = mouth_open or mouth_changed

            # Annotate Frame with primary face bounds
            x_coords = [lm.x for lm in primary_face]
            y_coords = [lm.y for lm in primary_face]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            display_text = head_status
            if gaze_status != "Gaze Center":
                display_text += f" | {gaze_status}"
            if is_blinking:
                display_text += " (Blink)"
            if mouth_moving:
                display_text += " (Mouth Move)"
            cv2.putText(frame, display_text, (x_min, max(20, y_min - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
            # Annotate extra faces (violations)
            if num_faces > 1:
                for face in results.face_landmarks[1:]:
                    fx_coords = [lm.x for lm in face]
                    fy_coords = [lm.y for lm in face]
                    x_m, x_mx = int(min(fx_coords) * w), int(max(fx_coords) * w)
                    y_m, y_mx = int(min(fy_coords) * h), int(max(fy_coords) * h)
                    cv2.rectangle(frame, (x_m, y_m), (x_mx, y_mx), (0, 0, 255), 2)

        # Hand detection for hand signs/hand presence on screen.
        if self.hand_landmarker is not None:
            hand_results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_results.hand_landmarks:
                hand_seen = True
                for hand_landmarks in hand_results.hand_landmarks:
                    hx = [int(lm.x * w) for lm in hand_landmarks]
                    hy = [int(lm.y * h) for lm in hand_landmarks]
                    if hx and hy:
                        cv2.rectangle(
                            frame,
                            (max(0, min(hx)), max(0, min(hy))),
                            (min(w - 1, max(hx)), min(h - 1, max(hy))),
                            (0, 255, 255),
                            2
                        )
                cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        return num_faces, head_status, gaze_status, is_blinking, mouth_moving, hand_seen, frame
