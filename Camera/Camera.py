import cv2
import threading
import time


# =========================================================
# LỚP CAMERA ĐA LUỒNG TỰ ĐỊNH NGHĨA
# =========================================================
class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[Camera Error] Không mở được camera {self.camera_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

        return True

    def update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self, copy=True):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy() if copy else self.frame

    def capture(self, filename):
        frame = self.get_frame(copy=True)
        if frame is None:
            return False
        return cv2.imwrite(filename, frame)

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.frame = None
