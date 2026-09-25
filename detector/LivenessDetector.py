import time
import random
import winsound
import math
from collections import deque

import cv2
import numpy as np


class LivenessDetector:
    """
    Liveness Detection based on 5 facial landmarks from SCRFD.
    Returns:
        "STEP" = Hoàn thành 1 hành động, UI cần chụp 1 ảnh
        "DONE" = Hoàn thành cả 3 hành động, UI chụp ảnh cuối và qua bước so sánh
        False  = Thất bại (quá giờ hoặc tráo người)
        None   = Đang trong quá trình làm thử thách
    """

    def __init__(
        self,
        movement_threshold=0.032,  # Hạ từ 0.045 xuống: Dễ quay đầu hơn, không bắt buộc quay quá rộng
        vertical_threshold=0.025,  # Hạ từ 0.035 xuống: Dễ nhìn lên/xuống hơn
        confirm_frames=5,          # Giảm từ 8 xuống 5: Nhanh xác nhận trạng thái hơn
        hold_time=0.4,             # Giảm từ 0.8s xuống 0.4s: Chỉ cần giữ thế rất ngắn là ăn điểm
        min_frames=6,              # Giảm từ 10 xuống 6: Không bắt buộc chuyển động phải dài dòng
        history_size=20,
        smooth_window=5,
        max_frame_jump=0.25,       # Tăng từ 0.18 lên 0.25: Thoáng hơn, tránh bị báo lỗi "Moving too fast" vô cớ
        timeout=30.0,              # Giữ nguyên 30 giây cho thoải mái
    ):
        self.movement_threshold = movement_threshold
        self.vertical_threshold = vertical_threshold

        self.confirm_frames = confirm_frames
        self.hold_time = hold_time
        self.min_frames = min_frames

        self.history_size = history_size
        self.smooth_window = smooth_window
        self.feature_history = deque(maxlen=self.history_size)

        self.max_frame_jump = max_frame_jump
        self.timeout = timeout

        self.challenge_types = ["LEFT", "RIGHT", "UP", "DOWN", "LEFT_RIGHT"]
        self.sequence = []
        self.current_step = 0

        self.challenge_state = "WAITING"
        self.start_features = None
        self.last_position = None

        self.motion_frames = 0
        self.confirm_count = 0
        self.hold_start_time = None

        # State cho riêng bài test LEFT_RIGHT
        self.lr_stage = "LEFT_MOVING"
        self.lr_left_features = None
        self.lr_left_motion_frames = 0
        self.lr_left_confirm_count = 0
        self.lr_hold_start_time = None
        self.lr_right_motion_frames = 0
        self.lr_right_confirm_count = 0

        self.started = False
        self.finished = False
        self.passed = False

        self.frame_count = 0
        self.start_time = None
        self.status = "Not started"
        
        # --- THÔNG SỐ CHỐNG TRÁO NGƯỜI (FACE SWAP) ---
        self.face_match_threshold = 0.5
        self.anchor_embedding = None

    def generate_sequence(self):
        self.sequence = random.sample(self.challenge_types, 3)
        self.current_step = 0
        return self.sequence

    def start(self):
        self.generate_sequence()
        self.started = True
        self.finished = False
        self.passed = False
        self.frame_count = 0
        self.start_time = time.time()
        self.feature_history.clear()
        self._reset_challenge_state()
        self.status = self._get_challenge_text()
        return self.sequence

    def reset(self):
        self.sequence = []
        self.current_step = 0
        self.started = False
        self.finished = False
        self.passed = False
        self.frame_count = 0
        self.start_time = None
        self.feature_history.clear()
        self._reset_challenge_state()
        self.status = "Not started"
        self.anchor_embedding = None

    def _reset_challenge_state(self):
        self.challenge_state = "WAITING"
        self.start_features = None
        self.last_position = None
        self.motion_frames = 0
        self.confirm_count = 0
        self.hold_start_time = None
        
        self.lr_stage = "LEFT_MOVING"
        self.lr_left_features = None
        self.lr_left_motion_frames = 0
        self.lr_left_confirm_count = 0
        self.lr_right_motion_frames = 0
        self.lr_right_confirm_count = 0
        self.lr_hold_start_time = None
        self.feature_history.clear()

    def _get_current_challenge(self):
        if not self.sequence or self.current_step >= len(self.sequence):
            return None
        return self.sequence[self.current_step]

    def _get_challenge_text(self):
        challenge = self._get_current_challenge()
        if challenge == "LEFT": return "Turn face LEFT"
        if challenge == "RIGHT": return "Turn face RIGHT"
        if challenge == "UP": return "Look UP"
        if challenge == "DOWN": return "Look DOWN"
        if challenge == "LEFT_RIGHT":
            if self.lr_stage in ("LEFT_MOVING", "LEFT_HOLD"):
                return "Turn face LEFT"
            if self.lr_stage in ("RIGHT_MOVING", "RIGHT_HOLD"):
                return "Turn face RIGHT"
        return "Authenticating..."

    def extract_features(self, landmarks):
        if landmarks is None: return None
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape[0] < 5: return None
        
        try:
            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            if eye_distance < 1e-6: return None

            nose_x = (nose[0] - left_eye[0]) / eye_distance
            nose_y = (nose[1] - left_eye[1]) / eye_distance
            mouth_center = (left_mouth + right_mouth) / 2.0
            mouth_x = (mouth_center[0] - left_eye[0]) / eye_distance
            mouth_y = (mouth_center[1] - left_eye[1]) / eye_distance

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            eye_angle = math.degrees(math.atan2(dy, dx))

            return np.array([nose_x, nose_y, mouth_x, mouth_y, eye_angle], dtype=np.float32)
        except Exception:
            return None

    def _smooth_features(self, features):
        self.feature_history.append(features.copy())
        values = list(self.feature_history)[-self.smooth_window:]
        return np.mean(values, axis=0)

    def _check_frame_jump(self, current):
        if self.last_position is None:
            self.last_position = current.copy()
            return False
        jump = np.linalg.norm(current[:4] - self.last_position[:4])
        self.last_position = current.copy()
        return jump > self.max_frame_jump

    def _cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        e1 = emb1.flatten()
        e2 = emb2.flatten()
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (norm1 * norm2))

    def update(self, landmarks, face_embedding=None):
        if not self.started: return None, "Not started"
        if self.finished: return self.passed, self.status
        if self.start_time and time.time() - self.start_time > self.timeout:
            self.finished = True
            self.passed = False
            self.status = "Liveness failed: Timeout"
            return False, self.status

        features = self.extract_features(landmarks)
        if features is None:
            self.status = "No face detected"
            return None, self.status

        smooth_features = self._smooth_features(features)
        self.frame_count += 1

        if self._check_frame_jump(smooth_features):
            self.status = "Moving too fast, keep face steady"
            return None, self.status

        if self.start_features is None:
            self.start_features = smooth_features.copy()
            self.last_position = smooth_features.copy()
            
            if self.anchor_embedding is None and face_embedding is not None:
                self.anchor_embedding = face_embedding.copy()
                
            self.status = self._get_challenge_text() + " - starting..."
            return None, self.status

        if self.anchor_embedding is not None and face_embedding is not None:
            sim = self._cosine_similarity(self.anchor_embedding, face_embedding)
            if sim < self.face_match_threshold:
                self.finished = True
                self.passed = False
                self.status = "FAIL: Face swapped during challenge!"
                winsound.Beep(500, 500)
                return False, self.status

        challenge = self._get_current_challenge()
        if challenge is None:
            self.finished = True
            self.passed = True
            self.status = "Liveness PASS"
            return True, self.status

        dx = smooth_features[0] - self.start_features[0]
        dy = smooth_features[1] - self.start_features[1]

        if challenge == "LEFT":
            return self._process_direction(dx, self.movement_threshold, True, "LEFT")
        elif challenge == "RIGHT":
            return self._process_direction(dx, self.movement_threshold, False, "RIGHT")
        elif challenge == "UP":
            return self._process_direction(dy, self.vertical_threshold, True, "UP")
        elif challenge == "DOWN":
            return self._process_direction(dy, self.vertical_threshold, False, "DOWN")
        elif challenge == "LEFT_RIGHT":
            return self._update_left_right(smooth_features)

        return None, self.status

    def _process_direction(self, delta, threshold, is_negative, name):
        condition = (delta < -threshold) if is_negative else (delta > threshold)

        if self.challenge_state in ("WAITING", "MOVING"):
            if condition:
                self.challenge_state = "MOVING"
                self.motion_frames += 1
                self.status = f"Turning {name}... ({self.motion_frames}/{self.min_frames})"
                if self.motion_frames >= self.min_frames:
                    self.challenge_state = "TARGET"
                    self.confirm_count = 0
                    self.hold_start_time = None
                    self.status = f"Turned {name} - hold steady..."
                return None, self.status
            self.motion_frames = 0
            self.status = f"Turn face {name}..."
            return None, self.status

        elif self.challenge_state == "TARGET":
            if condition:
                self.confirm_count += 1
                self.status = f"Confirming {name}... ({self.confirm_count}/{self.confirm_frames})"
                if self.confirm_count >= self.confirm_frames:
                    self.challenge_state = "HOLDING"
                    self.hold_start_time = time.time()
                    self.status = f"Holding {name} position..."
                return None, self.status
            self.confirm_count = 0
            self.challenge_state = "MOVING"
            self.status = f"Turn back {name}..."
            return None, self.status

        elif self.challenge_state == "HOLDING":
            if condition:
                elapsed = time.time() - self.hold_start_time
                self.status = f"Holding {name}... {elapsed:.1f}/{self.hold_time:.1f}s"
                if elapsed >= self.hold_time:
                    return self._challenge_passed()
                return None, self.status
            self.challenge_state = "MOVING"
            self.motion_frames = 0
            self.confirm_count = 0
            self.hold_start_time = None
            self.status = f"Not held long enough - turn back {name}"
            return None, self.status

        return None, self.status

    def _update_left_right(self, features):
        if self.lr_stage == "LEFT_MOVING":
            dx = features[0] - self.start_features[0]
            if dx < -self.movement_threshold:
                self.lr_left_motion_frames += 1
                self.status = f"Turning LEFT... ({self.lr_left_motion_frames}/{self.min_frames})"
                if self.lr_left_motion_frames >= self.min_frames:
                    self.lr_stage = "LEFT_HOLD"
                    self.lr_left_confirm_count = 0
                    self.lr_hold_start_time = None
                    self.status = "Turned LEFT - hold steady..."
                return None, self.status
            self.lr_left_motion_frames = 0
            self.status = "Turn face LEFT..."
            return None, self.status

        elif self.lr_stage == "LEFT_HOLD":
            dx = features[0] - self.start_features[0]
            if dx < -self.movement_threshold:
                self.lr_left_confirm_count += 1
                self.status = f"Confirming LEFT... ({self.lr_left_confirm_count}/{self.confirm_frames})"
                if self.lr_left_confirm_count >= self.confirm_frames:
                    if self.lr_hold_start_time is None: self.lr_hold_start_time = time.time()
                    elapsed = time.time() - self.lr_hold_start_time
                    self.status = f"Holding LEFT... {elapsed:.1f}/{self.hold_time:.1f}s"
                    if elapsed >= self.hold_time:
                        self.lr_left_features = features.copy()
                        self.lr_stage = "RIGHT_MOVING"
                        self.lr_right_motion_frames = 0
                        self.lr_right_confirm_count = 0
                        self.lr_hold_start_time = None
                        self.status = "Held LEFT - now turn RIGHT"
                return None, self.status
            self.lr_left_confirm_count = 0
            self.lr_hold_start_time = None
            self.status = "Hold LEFT steady..."
            return None, self.status

        elif self.lr_stage == "RIGHT_MOVING":
            if self.lr_left_features is None: return None, "State error"
            dx = features[0] - self.lr_left_features[0]
            if dx > self.movement_threshold:
                self.lr_right_motion_frames += 1
                self.status = f"Turning RIGHT... ({self.lr_right_motion_frames}/{self.min_frames})"
                if self.lr_right_motion_frames >= self.min_frames:
                    self.lr_stage = "RIGHT_HOLD"
                    self.lr_right_confirm_count = 0
                    self.lr_hold_start_time = None
                    self.status = "Turned RIGHT - hold steady..."
                return None, self.status
            self.lr_right_motion_frames = 0
            self.status = "Now turn RIGHT..."
            return None, self.status

        elif self.lr_stage == "RIGHT_HOLD":
            if self.lr_left_features is None: return None, "State error"
            dx = features[0] - self.lr_left_features[0]
            if dx > self.movement_threshold:
                self.lr_right_confirm_count += 1
                self.status = f"Confirming RIGHT... ({self.lr_right_confirm_count}/{self.confirm_frames})"
                if self.lr_right_confirm_count >= self.confirm_frames:
                    if self.lr_hold_start_time is None: self.lr_hold_start_time = time.time()
                    elapsed = time.time() - self.lr_hold_start_time
                    self.status = f"Holding RIGHT... {elapsed:.1f}/{self.hold_time:.1f}s"
                    if elapsed >= self.hold_time:
                        return self._challenge_passed()
                return None, self.status
            self.lr_right_confirm_count = 0
            self.lr_hold_start_time = None
            self.status = "Hold RIGHT steady..."
            return None, self.status

        self.lr_stage = "LEFT_MOVING"
        self.status = "Turn face LEFT..."
        return None, self.status

    def _challenge_passed(self):
        winsound.Beep(1000, 150)
        current_challenge = self._get_current_challenge()
        self.current_step += 1

        if self.current_step >= len(self.sequence):
            self.finished = True
            self.passed = True
            self.status = "Liveness PASS"
            winsound.Beep(1200, 200)
            winsound.Beep(1500, 250)
            return "DONE", self.status

        self._reset_challenge_state()
        self.status = f"{current_challenge} PASS - Challenge {self.current_step + 1}/{len(self.sequence)}: {self._get_challenge_text()}"
        return "STEP", self.status

    def draw_status(self, frame, position=(20, 40), color=(0, 255, 0), thickness=2):
        if frame is None:
            return frame

        cv2.putText(
            frame,
            self.status,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness,
            cv2.LINE_AA,
        )

        if self.sequence:
            progress = f"Step {min(self.current_step + 1, len(self.sequence))}/{len(self.sequence)}"
            cv2.putText(
                frame,
                progress,
                (position[0], position[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame