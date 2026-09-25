import os
import cv2
import numpy as np
import onnxruntime as ort


class AntiSpoofing:

    def __init__(self, model_path, input_size=(224, 224), crop_scale=2.7):
        self.input_size = input_size
        self.crop_scale = crop_scale

        # Chuẩn hóa ImageNet theo tập huấn luyện PyTorch
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Khởi tạo ONNX Runtime Session (Ưu tiên CUDA nếu có)
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img, bbox=None):
        """
        Cắt vùng mặt tỉ lệ vuông (Square Crop) an toàn và chuẩn hóa theo PyTorch ImageNet Pipeline.
        """
        crop_img = img

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox[:4])
            h, w = img.shape[:2]

            bw = x2 - x1
            bh = y2 - y1

            if bw > 0 and bh > 0:
                # Tâm khuôn mặt
                cx = x1 + bw // 2
                cy = int(y1 + bh * 0.48)

                # Tính kích thước ô vuông mở rộng theo scale
                crop_size = int(max(bw, bh) * self.crop_scale)

                # Tọa độ ô vuông crop
                nx1 = cx - crop_size // 2
                ny1 = cy - crop_size // 2
                nx2 = nx1 + crop_size
                ny2 = ny1 + crop_size

                # Tính padding khi vượt giới hạn khung hình
                pad_l = max(0, -nx1)
                pad_t = max(0, -ny1)
                pad_r = max(0, nx2 - w)
                pad_b = max(0, ny2 - h)

                # Mở rộng ảnh nếu bị tràn viền
                if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                    img_padded = cv2.copyMakeBorder(
                        img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT
                    )
                    crop_img = img_padded[
                        ny1 + pad_t : ny2 + pad_t, nx1 + pad_l : nx2 + pad_l
                    ]
                else:
                    crop_img = img[ny1:ny2, nx1:nx2]

        # Bảo vệ chống ảnh rỗng
        if crop_img is None or crop_img.size == 0:
            crop_img = img

        # 1. Resize về kích thước đầu vào
        resized = cv2.resize(crop_img, self.input_size, interpolation=cv2.INTER_LINEAR)

        # 2. Chuyển BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Chuẩn hóa [0, 1] và trừ Mean / chia Std
        blob = rgb.astype(np.float32) / 255.0
        blob = (blob - self.mean) / self.std

        # 4. Chuyển NCHW: (H, W, C) -> (1, C, H, W)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        return blob

    def predict(self, img, bbox=None):
        """
        Dự đoán độ tin cậy mặt thật (Real Score trong khoảng [0.0, 1.0]).
        """
        input_data = self.preprocess(img, bbox)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]

        logits = outputs[0]

        # Case 1: Sigmoid (1 output node)
        if logits.size == 1:
            val = float(logits.squeeze())
            real_score = 1.0 / (1.0 + np.exp(-val)) if (val < 0.0 or val > 1.0) else val

        # Case 2: Softmax (2 output nodes - [Spoof_Score, Real_Score])
        else:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            real_score = float(probs[1])

        return real_score