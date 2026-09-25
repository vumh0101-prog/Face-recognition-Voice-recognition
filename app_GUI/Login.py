import os
import time
import threading
import psutil
import customtkinter as ctk
from tkinter import messagebox

from FaceVerifyWindow import FaceVerifyWindow
from VoiceVerifyWindow import VoiceVerifyWindow


class LoginWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Đăng nhập MFA")
        self.geometry("500x400")
        self.resizable(False, False)

        self.face_verified = False
        self.voice_verified = False

        # Khởi chạy luồng giám sát tài nguyên hệ thống (CPU & RAM) chạy ngầm
        self.monitor_thread = threading.Thread(target=self._resource_monitor_task, daemon=True)
        self.monitor_thread.start()

        # =====================
        # Username
        # =====================

        ctk.CTkLabel(
            self,
            text="Tên đăng nhập",
            font=("Arial", 18, "bold")
        ).pack(pady=(30, 10))

        self.entry_user = ctk.CTkEntry(
            self,
            width=250,
            placeholder_text="Nhập username..."
        )
        self.entry_user.pack()

        # =====================
        # Face Button
        # =====================

        self.btn_face = ctk.CTkButton(
            self,
            text="Xác thực khuôn mặt",
            command=self.verify_face
        )

        self.btn_face.pack(pady=20)

        # =====================
        # Voice Button
        # =====================

        self.btn_voice = ctk.CTkButton(
            self,
            text="Xác thực giọng nói",
            command=self.verify_voice,
            state="disabled"
        )

        self.btn_voice.pack()

        # =====================
        # Login
        # =====================

        self.btn_login = ctk.CTkButton(
            self,
            text="Đăng nhập",
            command=self.login,
            state="disabled"
        )

        self.btn_login.pack(pady=30)

    # #####################################################
    # GIÁM SÁT TÀI NGUYÊN HỆ THỐNG
    # #####################################################

    def _resource_monitor_task(self):
        """Hàm chạy ngầm đo lường RAM và CPU của tiến trình ứng dụng"""
        process = psutil.Process(os.getpid())
        while True:
            try:
                # Lấy dung lượng RAM đang sử dụng (MB)
                ram_mb = process.memory_info().rss / (1024 * 1024)
                # Lấy phần trăm CPU đang chiếm dụng (đo trong chu kỳ 1 giây)
                cpu_percent = process.cpu_percent(interval=1)
                
                print(f"📊 [Resource Monitor] RAM: {ram_mb:.2f} MB | CPU: {cpu_percent}%")
                time.sleep(5)
            except Exception:
                break

    # #####################################################

    def verify_face(self):
        username = self.entry_user.get().strip()

        if username == "":
            messagebox.showwarning(
                "Thông báo",
                "Vui lòng nhập tên đăng nhập."
            )
            return

        # Kiểm tra xem tài khoản có tồn tại trong hệ thống hay không trước khi mở camera
        face_path = f"database/face_embeddings/{username}.npy"
        voice_path = f"database/voice_embeddings/{username}.npy"

        if not os.path.exists(face_path) and not os.path.exists(voice_path):
            messagebox.showerror(
                "Lỗi",
                f"Tài khoản '{username}' không tồn tại trong hệ thống!"
            )
            return

        # Mở cửa sổ xác thực khuôn mặt
        win = FaceVerifyWindow(self, username)

        self.wait_window(win)

        if getattr(win, "result", False):
            self.face_verified = True
            self.btn_voice.configure(state="normal")
            self.entry_user.configure(state="disabled") # Khóa ô nhập tên sau khi xác thực mặt thành công

            messagebox.showinfo(
                "Thông báo",
                "Xác thực khuôn mặt thành công."
            )

    # #####################################################

    def verify_voice(self):
        username = self.entry_user.get().strip()

        win = VoiceVerifyWindow(self, username)

        self.wait_window(win)

        if getattr(win, "result", False):
            self.voice_verified = True
            self.btn_login.configure(state="normal")

            messagebox.showinfo(
                "Thông báo",
                "Xác thực giọng nói thành công."
            )

    # #####################################################

    def login(self):
        if not self.face_verified:
            messagebox.showerror(
                "Lỗi",
                "Chưa xác thực khuôn mặt."
            )
            return

        if not self.voice_verified:
            messagebox.showerror(
                "Lỗi",
                "Chưa xác thực giọng nói."
            )
            return

        messagebox.showinfo(
            "Thành công",
            "Đăng nhập thành công!"
        )

        self.destroy()


if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()