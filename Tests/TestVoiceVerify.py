import glob
import os
import sys
import time
import numpy as np

# ============================================================
# 1. Thiết lập đường dẫn về thư mục gốc dự án
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.chdir(BASE_DIR)

# ============================================================
# 2. Import Voice Recognizer & Preprocessor
# ============================================================

from PreProcess.VoicePreProcess import VoicePreprocessor
from recognizer.VoiceRecognizer import VoiceRecognizer


class VoiceVerifyTester:

    def __init__(
        self, model_dir="model/Voice", sample_rate=16000, threshold=0.60
    ):
        print("[Init] Đang tải các mô hình AI...")

        # ECAPA-TDNN
        self.recognizer = VoiceRecognizer(model_dir=model_dir)

        # Voice preprocessing
        self.preprocessor = VoicePreprocessor(sample_rate=sample_rate)

        self.sample_rate = sample_rate
        self.THRESHOLD = threshold

        # Cache cho DB Embeddings
        self._db_cache = {}

        print("[Init] Khởi tạo VoiceVerifyTester thành công.")

    # ========================================================
    # 3. Load embedding của user từ database (có Cache)
    # ========================================================

    def load_user_embedding_from_db(self, username):
        if username in self._db_cache:
            return self._db_cache[username]

        file_path = os.path.join(
            "database", "voice_embeddings", f"{username}.npy"
        )

        if not os.path.exists(file_path):
            print(
                f"[Error] Không tìm thấy embedding DB cho user '{username}' tại: {file_path}"
            )
            return None

        try:
            embedding = np.load(file_path)

            # L2 Normalize trước khi lưu vào Cache
            if embedding.ndim == 1:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            elif embedding.ndim == 2:
                norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embedding = embedding / norms

            self._db_cache[username] = embedding
            print(
                f"[DB] Loaded & Cached embedding cho '{username}': {embedding.shape}"
            )
            return embedding

        except Exception as e:
            print(f"[Error] Không thể đọc embedding '{file_path}': {e}")
            return None

    # ========================================================
    # 4. Tính Cosine Similarity
    # ========================================================

    def compute_matrix_similarity(self, live_emb, db_emb):
        """db_emb đã được L2 Normalized sẵn khi load từ DB."""
        live_emb = np.asarray(live_emb, dtype=np.float32).flatten()

        norm_live = np.linalg.norm(live_emb)
        if norm_live > 0:
            live_emb = live_emb / norm_live

        # DB lưu 1 vector (1D)
        if db_emb.ndim == 1:
            return float(
                np.clip(np.dot(live_emb, db_emb.flatten()), -1.0, 1.0)
            )

        # DB lưu nhiều vector (2D)
        elif db_emb.ndim == 2:
            scores = np.dot(db_emb, live_emb)
            return float(np.clip(np.max(scores), -1.0, 1.0))

        return 0.0

    # ========================================================
    # 5. Safe Audio Loader (Tránh crash TorchCodec / MP3 trên Windows)
    # ========================================================

    def _load_audio_safe(self, audio_path):
        """Hàm đọc audio đa định dạng an toàn tuyệt đối."""
        # Cách 1: Dùng soundfile (Rất ổn định với WAV/FLAC)
        try:
            import soundfile as sf

            audio_np, fs = sf.read(audio_path, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)
            return audio_np, fs
        except Exception:
            pass

        # Cách 2: Dùng librosa
        try:
            import librosa

            audio_np, fs = librosa.load(audio_path, sr=None, mono=True)
            return audio_np, fs
        except Exception:
            pass

        # Cách 3: Torchaudio Fallback
        import torchaudio

        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        raw_audio = signal.squeeze(0).numpy().astype(np.float32)
        return raw_audio, fs

    # ========================================================
    # 6. Verify một file Audio
    # ========================================================

    def verify_audio(
        self,
        audio_path,
        target_username,
        db_emb=None,
        threshold=None,
    ):
        if threshold is None:
            threshold = self.THRESHOLD

        filename = os.path.basename(audio_path)

        # 6.1. Đọc audio an toàn
        try:
            raw_audio, fs = self._load_audio_safe(audio_path)
        except Exception as e:
            print(f"[{filename}] -> ERROR ⚠️ Không thể đọc audio: {e}")
            return "ERROR", 0.0

        # 6.2. Resample nếu tần số lấy mẫu khác target (vd: 16kHz)
        if fs != self.sample_rate:
            try:
                import torch
                import torchaudio

                resampler = torchaudio.transforms.Resample(
                    orig_freq=fs, new_freq=self.sample_rate
                )
                tensor_audio = torch.from_numpy(raw_audio).unsqueeze(0)
                raw_audio = resampler(tensor_audio).squeeze(0).numpy()
            except Exception as e:
                print(f"[{filename}] -> ERROR ⚠️ Resample thất bại: {e}")
                return "ERROR", 0.0

        # 6.3. Preprocessing
        try:
            clean_audio = self.preprocessor.process(raw_audio)
        except Exception as e:
            print(f"[{filename}] -> ERROR ⚠️ Preprocessing thất bại: {e}")
            return "ERROR", 0.0

        # 6.4. Kiểm tra độ dài tiếng nói tối thiểu (Ví dụ: 0.8s = 12,800 samples)
        min_samples = int(self.sample_rate * 0.8)
        if len(clean_audio) < min_samples:
            print(
                f"[{filename}] -> NO VOICE ⚠️ (Không phát hiện đủ giọng nói)"
            )
            return "NO_VOICE", 0.0

        # 6.5. Trích xuất Feature Embedding bằng ECAPA-TDNN
        try:
            start_time = time.perf_counter()
            embedding_live = (
                self.recognizer.extract_embedding_from_array(
                    clean_audio, sample_rate=self.sample_rate
                )
            )
            latency = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            print(
                f"[{filename}] -> ERROR ⚠️ Không thể extract embedding: {e}"
            )
            return "ERROR", 0.0

        # 6.6. Lấy DB Embedding từ Cache
        if db_emb is None:
            db_emb = self.load_user_embedding_from_db(target_username)

        if db_emb is None:
            return "ERROR", 0.0

        # 6.7. Tính điểm Cosine Similarity
        score = self.compute_matrix_similarity(embedding_live, db_emb)

        # 6.8. Đánh giá ACCEPT / REJECT
        is_match = score >= threshold
        status_text = "ACCEPT ✅" if is_match else "REJECT ❌"
        result_code = "ACCEPT" if is_match else "REJECT"

        print(
            f"[{filename}] -> Score: {score:.4f} | Result: {status_text} | Latency: {latency:.2f} ms"
        )
        return result_code, score

    # ========================================================
    # 7. Test toàn bộ dataset
    # ========================================================

    def test_dataset(self, dataset_dir, target_username):
        # Mở rộng quét đầy đủ các định dạng audio phổ biến
        extensions = ("*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.m4a")
        audio_paths_set = set()

        for ext in extensions:
            found_files = glob.glob(
                os.path.join(dataset_dir, "**", ext), recursive=True
            )
            for file_path in found_files:
                audio_paths_set.add(os.path.normpath(file_path))

        audio_paths = sorted(list(audio_paths_set))

        print("\n" + "=" * 60)
        print(
            f"BẮT ĐẦU KÍCH HOẠT TEST BỘ DỮ LIỆU VOICE CỦA USER: '{target_username}'"
        )
        print(f"Tổng số audio tìm thấy: {len(audio_paths)}")
        print("=" * 60)

        # Pre-load DB Embedding vào RAM trước khi chạy loop
        db_emb = self.load_user_embedding_from_db(target_username)
        if db_emb is None:
            print("[Fatal Error] Không tìm thấy DB Embedding. Dừng kiểm thử.")
            return

        accept_count = 0
        reject_count = 0
        no_voice_count = 0
        error_count = 0

        total_start = time.perf_counter()

        for audio_path in audio_paths:
            result_code, score = self.verify_audio(
                audio_path, target_username, db_emb=db_emb
            )

            if result_code == "ACCEPT":
                accept_count += 1
            elif result_code == "REJECT":
                reject_count += 1
            elif result_code == "NO_VOICE":
                no_voice_count += 1
            else:
                error_count += 1

        total_time = time.perf_counter() - total_start
        total_audio = len(audio_paths)

        if total_audio > 0:
            overall_accuracy = (accept_count / total_audio) * 100
            detected_audio = accept_count + reject_count
            verification_accuracy = (
                (accept_count / detected_audio * 100)
                if detected_audio > 0
                else 0.0
            )
        else:
            overall_accuracy = 0.0
            verification_accuracy = 0.0

        print("\n" + "=" * 60)
        print(
            f"KẾT QUẢ ĐÁNH GIÁ THỐNG KÊ CHO USER '{target_username}':"
        )
        print(f"- Tổng số audio kiểm tra: {total_audio}")
        print(f"- Nhận diện đúng (ACCEPT): {accept_count}")
        print(f"- Sai khác/Bị từ chối (REJECT): {reject_count}")
        print(f"- Không phát hiện giọng nói (NO VOICE): {no_voice_count}")
        if error_count > 0:
            print(f"- Lỗi đọc file/DB (ERROR): {error_count}")
        print("-" * 60)
        print(
            f"- Độ chính xác toàn bộ tập dữ liệu: {overall_accuracy:.2f}%"
        )
        print(
            f"- Độ chính xác trên audio hợp lệ: {verification_accuracy:.2f}%"
        )
        print(f"- Tổng thời gian test: {total_time:.2f} s")
        if total_audio > 0:
            print(
                f"- Thời gian trung bình/audio: {(total_time / total_audio) * 1000:.2f} ms"
            )
        print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    tester = VoiceVerifyTester(
        model_dir="model/Voice", sample_rate=16000, threshold=0.60
    )

    TARGET_USER = "Duy123"
    DATASET_DIR = "C:/Users/admin/OneDrive/Desktop/Dataset/Voice"

    tester.test_dataset(dataset_dir=DATASET_DIR, target_username=TARGET_USER)