import time
import cv2
import numpy as np
import onnxruntime as ort

# 5 điểm mốc chuẩn cho khuôn mặt kích thước 112x112 (InsightFace Standard)
ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class FaceRecognizer:

    def __init__(self, model_path="model/Face/w600k_mbf.onnx"):
        # Khởi tạo ONNX Session chạy trên CPU
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def align_crop(self, img, landmarks):
        """Căn chỉnh (Alignment) khuôn mặt về dạng 112x112 bằng phép biến đổi Affine"""
        landmarks = np.asarray(landmarks, dtype=np.float32)
        
        # Tính toán ma trận biến đổi Similarity Transformation
        tfm, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_REF_POINTS)
        
        # FIX LỖI: Kiểm tra nếu không thể tính được ma trận biến đổi (landmarks bị lỗi/suy biến)
        if tfm is None:
            # Fallback: Crop theo bounding box đơn giản hoặc trả về resized image
            x_min, y_min = np.min(landmarks, axis=0).astype(int)
            x_max, y_max = np.max(landmarks, axis=0).astype(int)
            
            # Mở rộng bounding box một chút
            h_img, w_img = img.shape[:2]
            x_min, y_min = max(0, x_min - 10), max(0, y_min - 10)
            x_max, y_max = min(w_img, x_max + 10), min(h_img, y_max + 10)
            
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                aligned_face = cv2.resize(img, (112, 112))
            else:
                aligned_face = cv2.resize(crop, (112, 112))
            return aligned_face

        # Crop và xoay thẳng khuôn mặt về kích thước 112x112
        aligned_face = cv2.warpAffine(img, tfm, (112, 112), borderValue=0.0)
        return aligned_face

    def extract_embedding(self, img, landmarks):
        """
        Trích xuất Feature Embedding (Vector 512 chiều)
        :param img: Ảnh BGR gốc
        :param landmarks: 5 điểm mốc của khuôn mặt
        :return: Vector embedding (512,)
        """
        # Bắt đầu bấm giờ đo hiệu năng trích xuất AI
        start_time = time.time()

        # 1. Căn chỉnh ảnh mặt 112x112
        face_crop = self.align_crop(img, landmarks)

        # 2. Preprocess: Chuyển BGR -> RGB, chuẩn hóa [-1, 1] và đổi sang Shape [1, 3, 112, 112]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_blob = np.transpose(face_blob, (2, 0, 1))
        face_blob = np.expand_dims(face_blob, axis=0)

        # 3. Forward ONNX Model MobileFaceNet
        embedding = self.session.run(None, {self.input_name: face_blob})[0][0]

        # 4. Chuẩn hóa L2 Normalize cho vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Kết thúc bấm giờ và in thời gian xử lý (miliseconds)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"⏱️ [FaceRecognizer] Thời gian trích xuất embedding: {latency_ms:.2f} ms")

        return embedding

    @staticmethod
    def compute_cosine_similarity(emb1, emb2):
        """
        Tính Cosine Similarity giữa 2 vector embedding đã được L2-Normalized.
        Giá trị trả về nằm trong khoảng [-1.0, 1.0].
        """
        return float(np.dot(emb1, emb2))