import os
import time
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


class VoiceRecognizer:

    def __init__(self, model_dir="Voice"):
        """
        Khởi tạo ECAPA-TDNN từ thư mục model local.
        :param model_dir: Đường dẫn đến thư mục chứa các file .ckpt và hyperparams.yaml
        """
        self.device = "cpu"  # Mô hình chạy trên CPU

        # Kiểm tra file cấu hình bắt buộc
        yaml_path = os.path.join(model_dir, "hyperparams.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file {yaml_path}. Hãy kiểm tra lại đường dẫn!")

        # Load model từ thư mục local bằng EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source=model_dir,
            savedir=model_dir,
            run_opts={"device": self.device}
        )

        # --------------------------------------------------
        # TỐI ƯU DUNG LƯỢNG: Lượng tử hóa mô hình sang INT8 (Dynamic Quantization)
        # Giúp giảm ~50% lượng RAM tiêu thụ khi suy luận trên CPU
        # --------------------------------------------------
        try:
            # Truy cập vào module embedding cốt lõi của ECAPA-TDNN và lượng tử hóa các lớp Linear
            self.classifier.mods.embedding_model = torch.quantization.quantize_dynamic(
                self.classifier.mods.embedding_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("🚀 [VoiceRecognizer] Đã lượng tử hóa mô hình giọng nói sang INT8 thành công!")
        except Exception as e:
            print(f"⚠️ [VoiceRecognizer] Không thể lượng tử hóa mô hình (vẫn dùng chuẩn FP32 gốc): {e}")

    def extract_embedding_from_file(self, wav_path):
        """
        Trích xuất Feature Embedding từ file .wav
        :param wav_path: Đường dẫn file âm thanh (.wav)
        :return: Vector embedding đã L2-Normalized (192,)
        """
        signal, fs = torchaudio.load(wav_path)

        # Chuyển đổi về Mono nếu là Stereo (2 kênh)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resample về 16kHz chuẩn
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        return self._extract_from_tensor(signal)

    def extract_embedding_from_array(self, audio_data, sample_rate=16000):
        """
        Trích xuất Feature Embedding từ mảng numpy (Dữ liệu ghi âm trực tiếp từ Micro)
        :param audio_data: Mảng numpy float32 hoặc int16
        :param sample_rate: Tần số lấy mẫu gốc
        :return: Vector embedding đã L2-Normalized (192,)
        """
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        signal = torch.from_numpy(audio_data)

        # Chuyển thành Tensor 2D: [channels, samples]
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        # Chuyển đổi về Mono nếu có nhiều kênh
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resample về 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)

        return self._extract_from_tensor(signal)

    def _extract_from_tensor(self, signal):
        """Forward tensor qua model ECAPA-TDNN kèm đo thời gian hiệu năng"""
        signal = signal.to(self.device)

        # Bắt đầu bấm giờ đo hiệu năng suy luận AI giọng nói
        start_time = time.time()

        with torch.no_grad():
            # EncoderClassifier trả về embeddings dạng [batch_size, 1, 192]
            embeddings = self.classifier.encode_batch(signal)
            embedding = embeddings.squeeze().cpu().numpy()

        # L2 Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Kết thúc bấm giờ và in thời gian xử lý (miliseconds)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"⏱️ [VoiceRecognizer] Thời gian trích xuất giọng nói: {latency_ms:.2f} ms")

        return embedding

    @staticmethod
    def compute_cosine_similarity(emb1, emb2):
        """
        Tính Cosine Similarity giữa 2 vector embedding giọng nói.
        """
        emb1 = np.asarray(emb1, dtype=np.float32)
        emb2 = np.asarray(emb2, dtype=np.float32)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 > 0:
            emb1 = emb1 / norm1
        if norm2 > 0:
            emb2 = emb2 / norm2

        sim = np.dot(emb1, emb2)
        return float(np.clip(sim, -1.0, 1.0))