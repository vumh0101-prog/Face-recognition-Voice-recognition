import os
import sys
import threading
import numpy as np
import cv2
import customtkinter as ctk
from PIL import Image
from tkinter import messagebox
import csv
from datetime import datetime
import gc
import time

# Thêm thư mục gốc dự án (thư mục 'python') vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import các module trong dự án
from Camera.Camera import Camera
from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer
from detector.AntiSpoofing import AntiSpoofing
from detector.LivenessDetector import LivenessDetector

class FaceVerifyWindow(ctk.CTkToplevel):

    def __init__(self, parent, username):
        super().__init__(parent)

        # Đồng bộ Working Directory về thư mục gốc dự án để tránh lỗi relative path
        os.chdir(BASE_DIR)

        self.result = False
        self.username = username
        self.is_running = True      # Cờ kiểm soát vòng lặp UI
        self.is_processing = False  # Cờ chống spam click nút bấm

        self.geometry("700x620")
        self.title(f"Xác thực khuôn mặt - {self.username}")
        self.resizable(False, False)

        # --------------------------------------------------
        # 1. Khởi tạo Models AI & Camera
        # --------------------------------------------------
        self.detector = FaceDetector(model_path="model/Face/det_500m.onnx", conf_threshold=0.5)
        self.recognizer = FaceRecognizer(model_path="model/Face/w600k_mbf.onnx")
        self.anti_spoof = AntiSpoofing(model_path="model/Face/best_mobilenetv3_large_balanced.onnx")
        
        self.SPOOF_THRESHOLD = 0.3  # Ngưỡng xác thực mặt thật
        self.THRESHOLD = 0.68       # Ngưỡng Cosine Similarity

        # Camera
        self.camera = Camera(camera_index=0, width=640, height=480)

        # --------------------------------------------------
        # 2. Giao diện (UI)
        # --------------------------------------------------
        self.lbl_title = ctk.CTkLabel(
            self,
            text=f"Đang xác thực cho tài khoản: {self.username}",
            font=("Arial", 16, "bold")
        )
        self.lbl_title.pack(pady=10)

        # Tối ưu CTkImage: Khởi tạo sẵn 1 đối tượng rỗng
        placeholder_img = Image.new("RGB", (640, 400), color="black")
        self.ctk_img = ctk.CTkImage(light_image=placeholder_img, dark_image=placeholder_img, size=(640, 400))

        # Frame hiển thị Video Camera
        self.lbl_video = ctk.CTkLabel(
            self,
            text="",
            image=self.ctk_img,
            width=640,
            height=400,
            fg_color="black"
        )
        self.lbl_video.pack(pady=10)

        # Nút xác thực
        self.btn_verify = ctk.CTkButton(
            self,
            text="Xác nhận khuôn mặt",
            command=self.success,
            width=220,
            height=40,
            font=("Arial", 14, "bold")
        )
        self.btn_verify.pack(pady=15)

        # --------------------------------------------------
        # 3. Sự kiện & Vòng lặp Video
        # --------------------------------------------------
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if self.camera.start():
            self.update_frame()
        else:
            self.lbl_video.configure(text="Không thể kết nối tới Webcam!", image="")

    def update_frame(self):
            """Render khung hình liên tục từ Webcam lên UI, tích hợp vẽ 5 điểm khuôn mặt"""
            if not self.is_running:
                return

            frame = self.camera.get_frame(copy=False)

            if frame is not None:
                # Phát hiện mặt ở frame hiện tại của luồng chính để vẽ landmarks tương tác real-time
                try:
                    bboxes, kpss = self.detector.detect(frame)
                    if len(bboxes) == 1:
                        # Nếu nhận diện đúng 1 mặt, vẽ 5 điểm mốc (màu xanh lá tươi)
                        frame = self.draw_landmarks(frame, kpss, color=(0, 255, 0), radius=5)
                    else:
                        # Nếu chưa có mặt hoặc có nhiều hơn 1 mặt, vẽ khung oval hướng dẫn
                        frame = self.draw_oval(frame)
                except Exception:
                    frame = self.draw_oval(frame)

                # Nếu đang chạy Liveness, vẽ thêm text hướng dẫn trạng thái
                if hasattr(self, 'liveness') and self.liveness.started and not self.liveness.finished:
                    frame = self.liveness.draw_status(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 400), interpolation=cv2.INTER_NEAREST)
                
                img = Image.fromarray(frame_resized)
                self.ctk_img.configure(light_image=img, dark_image=img)
                img.close()

            self.after(33, self.update_frame)

    def success(self):
        if self.is_processing:
            return
        self.is_processing = True
        self.btn_verify.configure(state="disabled", text="Đang chạy Liveness...")
        
        # Khởi tạo LivenessDetector
        self.liveness = LivenessDetector()
        self.liveness.start()
        
        threading.Thread(target=self._process_verification_loop, daemon=True).start()

    def _process_verification_loop(self):
            """Vòng lặp ngầm tối ưu: Chỉ dùng landmarks để check liveness, KHÔNG extract embedding liên tục"""
            try:
                embedding_db = self.load_user_embedding_from_db(self.username)
                if embedding_db is None:
                    self._show_error(f"Tài khoản '{self.username}' chưa có dữ liệu khuôn mặt!")
                    return

                captured_samples = [] # Lưu trữ các mốc hoàn thành (STEP/DONE)

                while self.is_processing and self.is_running:
                    frame = self.camera.get_frame(copy=True)
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    bboxes, kpss = self.detector.detect(frame)
                    
                    if len(bboxes) == 1:
                        bbox = bboxes[0]
                        landmarks = kpss[0]

                        # 1. Chỉ gọi Liveness update bằng landmarks (Siêu nhẹ, không tốn CPU)
                        # Không truyền face_embedding ở đây nữa để giải phóng FaceRecognizer
                        status_code, msg = self.liveness.update(landmarks)

                        # 2. Khi đạt mốc STEP hoặc DONE, lúc này mới gọi trích xuất embedding và lưu ảnh
                        if status_code in ("STEP", "DONE"):
                            print(f"📸 Đang trích xuất embedding tại bước {len(captured_samples)+1}...")
                            
                            # Trích xuất embedding duy nhất tại thời điểm chụp mốc
                            live_emb = self.recognizer.extract_embedding(frame, landmarks)
                            norm_live = np.linalg.norm(live_emb)
                            if norm_live > 0:
                                live_emb = live_emb / norm_live

                            captured_samples.append({
                                'frame': frame.copy(),
                                'bbox': bbox,
                                'embedding': live_emb
                            })
                            print(f"✅ Đã lưu mẫu bước {len(captured_samples)}/3")

                        if status_code == "DONE":
                            break
                            
                        elif status_code is False:
                            self._show_error(f"Liveness thất bại: {msg}")
                            return
                    else:
                        self.liveness.update(None)

                    # Nghỉ hợp lý để CPU thảnh thơi (giảm tải 1000% xuống mức bình thường)
                    time.sleep(0.03)

                # ==========================================================
                # SAU KHI ĐỦ 3 MẪU: KIỂM TRA SPOOFING & SO SÁNH ĐỊNH DANH
                # ==========================================================
                if len(captured_samples) == 3:
                                self.after(0, lambda: self.btn_verify.configure(text="Đang kiểm tra chống giả mạo..."))
                                
                                passed_spoof_count = 0
                                
                                for idx, sample in enumerate(captured_samples):
                                    spoof_score = self.anti_spoof.predict(sample['frame'], sample['bbox'])
                                    print(f"🛡️ Anti-Spoofing score ảnh {idx+1}: {spoof_score:.4f}")
                                    
                                    # Kiểm tra xem ảnh này có vượt qua ngưỡng mặt thật không
                                    if spoof_score >= self.SPOOF_THRESHOLD:
                                        passed_spoof_count += 1

                                print(f"🛡️ Số lượng ảnh vượt qua Anti-Spoofing: {passed_spoof_count}/3")

                                # Yêu cầu ít nhất 2/3 ảnh phải đạt chuẩn an toàn
                                if passed_spoof_count < 2:
                                    self._show_error("Phát hiện khuôn mặt giả mạo (Không đạt chuẩn an toàn qua các bước)!")
                                    return

                                # Phân tích định danh (So sánh với Database)
                                self.after(0, lambda: self.btn_verify.configure(text="Đang phân tích định danh..."))
                                
                                scores = [self._compute_matrix_similarity(sample['embedding'], embedding_db) for sample in captured_samples]
                                avg_score = sum(scores) / len(scores)

                                result = "ACCEPT" if avg_score >= self.THRESHOLD else "REJECT"
                                self.log_verification(avg_score, result)

                                if result == "ACCEPT":
                                    self.result = True
                                    self.after(0, lambda: self.btn_verify.configure(
                                        fg_color="#2ecc71", text="Xác thực thành công! ✓"
                                    ))
                                    self.after(1000, self.on_close)
                                else:
                                    self._show_error(f"Khuôn mặt không khớp!\nĐộ tương đồng: {avg_score*100:.1f}%")

            except Exception as e:
                print(f"[Error] Lỗi vòng lặp xác thực: {e}")
                self._show_error("Đã xảy ra lỗi hệ thống AI!")
                
    def _compute_matrix_similarity(self, live_emb, db_emb):
        live_emb = live_emb.flatten()

        if db_emb.ndim == 1:
            norm_db = np.linalg.norm(db_emb)
            if norm_db > 0:
                db_emb = db_emb / norm_db
            return float(np.dot(live_emb, db_emb))

        elif db_emb.ndim == 2:
            norms = np.linalg.norm(db_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            db_emb_norm = db_emb / norms

            scores = np.dot(db_emb_norm, live_emb)
            return float(np.max(scores))

        return 0.0

    def load_user_embedding_from_db(self, username):
        file_path = os.path.join("database", "face_embeddings", f"{username}.npy")
        if not os.path.exists(file_path):
            return None
        return np.load(file_path)

    def _show_error(self, message):
            """Hiển thị thông báo lỗi và mở lại nút bấm nếu cửa sổ vẫn tồn tại"""
            def _gui_update():
                # Chỉ hiển thị messagebox nếu cửa sổ vẫn đang chạy và tồn tại
                if self.is_running and self.winfo_exists():
                    try:
                        messagebox.showerror("Thông báo", message, parent=self)
                    except Exception:
                        pass
                
                self.is_processing = False
                if self.is_running and self.winfo_exists():
                    self.btn_verify.configure(state="normal", text="Xác nhận khuôn mặt")

            self.after(0, _gui_update)

    def draw_oval(self, frame):
        h, w = frame.shape[:2]
        cv2.ellipse(frame, (w // 2, h // 2), (int(w * 0.22), int(h * 0.38)), 0, 0, 360, (255, 255, 255), 2)
        return frame

    def release_resources(self):
        if hasattr(self, "camera") and self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

        if hasattr(self, "detector") and self.detector is not None:
            self.detector = None

        if hasattr(self, "recognizer") and self.recognizer is not None:
            self.recognizer = None

        if hasattr(self, "anti_spoof") and self.anti_spoof is not None:
            try:
                if hasattr(self.anti_spoof, "release"):
                    self.anti_spoof.release()
            except Exception:
                pass
            self.anti_spoof = None

        gc.collect()
        print("[FaceVerify] Đã giải phóng Face Models + AntiSpoofing + Camera")

    def on_close(self):
        if not self.is_running:
            return
        self.is_running = False
        self.release_resources()
        self.destroy()

    def log_verification(self, score, result):
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join("logs", "face_verification.csv")
        file_exists = os.path.exists(log_file)

        try:
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "username", "score", "threshold", "result"])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.username,
                    f"{score:.6f}",
                    f"{self.THRESHOLD:.6f}",
                    result
                ])
        except Exception as e:
            print(f"[Log Error] Không thể ghi log: {e}")

    def draw_landmarks(self, frame, kpss, color=(0, 255, 0), radius=4):
            """Vẽ 5 điểm đặc trưng (landmarks) lên khuôn mặt"""
            if kpss is not None and len(kpss) > 0:
                # kpss chứa mảng các điểm landmarks của khuôn mặt đầu tiên phát hiện được
                landmarks = kpss[0]
                for pt in landmarks:
                    x, y = int(pt[0]), int(pt[1])
                    # Vẽ vòng tròn đặc tại vị trí điểm mốc
                    cv2.circle(frame, (x, y), radius, color, -1)
                    # Vẽ viền trắng nhỏ quanh chấm để dễ nhìn hơn trên mọi nền
                    cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)
            return frame