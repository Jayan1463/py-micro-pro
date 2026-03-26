import cv2
import mediapipe as mp
import time
import math

class ProctorDetector:
    def __init__(self):
        # Configure MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        options = self.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=self.VisionRunningMode.VIDEO,
            num_faces=5
        )
        self.landmarker = self.FaceLandmarker.create_from_options(options)
        self.frame_idx = 0

    def process_frame(self, frame):
        """
        Processes a BGR frame, returns (num_faces, head_status, gaze_status, is_blinking, annotated_frame)
        """
        self.frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # timestamp must be sequentially increasing in milliseconds
        timestamp_ms = self.frame_idx * 33 
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        num_faces = 0
        head_status = "Center"
        gaze_status = "Gaze Center"
        is_blinking = False
        
        if results.face_landmarks:
            num_faces = len(results.face_landmarks)
            
            # Primary face
            primary_face = results.face_landmarks[0]
            h, w, _ = frame.shape
            
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

            # Gaze Estimation using Iris Tracking
            def get_horizontal_gaze(left_corner, right_corner, iris_center):
                dx = right_corner.x - left_corner.x
                dy = right_corner.y - left_corner.y
                length_sq = dx*dx + dy*dy
                if length_sq == 0:
                    return 0.5
                return ((iris_center.x - left_corner.x) * dx + (iris_center.y - left_corner.y) * dy) / length_sq

            # Left Eye physically (Screen Right): Outer = 33, Inner = 133, Iris = 468
            # In screen coords: left_corner = 133, right_corner = 33
            r_eye_gaze = get_horizontal_gaze(primary_face[133], primary_face[33], primary_face[468])
            
            # Right Eye physically (Screen Left): Inner = 362, Outer = 263, Iris = 473
            # In screen coords: left_corner = 263, right_corner = 362
            l_eye_gaze = get_horizontal_gaze(primary_face[263], primary_face[362], primary_face[473])
            
            avg_gaze = (l_eye_gaze + r_eye_gaze) / 2.0
            if avg_gaze < 0.40:
                gaze_status = "Gaze Left"
            elif avg_gaze > 0.60:
                gaze_status = "Gaze Right"
            
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
                    
        return num_faces, head_status, gaze_status, is_blinking, frame
