import numpy as np
from scipy.signal import butter, lfilter


class VoicePreprocessor:
    """
    Lớp xử lý tín hiệu âm thanh chuyên biệt cho nhận dạng giọng nói (Voice Biometrics).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def _butter_bandpass(self, lowcut: float, highcut: float, order: int = 5):
        """Tạo bộ lọc Bandpass Butterworth."""
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, audio_data: np.ndarray, lowcut: float = 80.0, highcut: float = 4000.0) -> np.ndarray:
        """
        Lọc dải tần (Bandpass Filter): Giữ lại tần số giọng nói con người (80Hz - 4000Hz),
        loại bỏ tiếng ù tần số thấp (máy lạnh, quạt) và tiếng rít tần số cao.
        """
        if len(audio_data) == 0:
            return audio_data
        b, a = self._butter_bandpass(lowcut, highcut, order=5)
        filtered = lfilter(b, a, audio_data)
        return filtered.astype(np.float32)

    def normalize_amplitude(self, audio_data: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Chuẩn hóa biên độ âm thanh (Amplitude Normalization).
        Đưa peak âm thanh về mức target_peak (mặc định 0.9) để đồng nhất âm lượng.
        """
        if len(audio_data) == 0:
            return audio_data
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = (audio_data / max_val) * target_peak
        return audio_data.astype(np.float32)

    def trim_silence(self, audio_data: np.ndarray, frame_duration_ms: int = 25, hop_duration_ms: int = 10, energy_ratio: float = 0.015) -> np.ndarray:
        """
        Cắt bỏ khoảng lặng (Silence Trimming / Basic VAD) ở đầu và cuối tín hiệu.
        - frame_duration_ms: Độ dài mỗi khung (ms)
        - hop_duration_ms: Độ nhích khung (ms)
        - energy_ratio: Ngưỡng năng lượng để coi là có tiếng nói (1.5% năng lượng đỉnh)
        """
        if len(audio_data) == 0:
            return audio_data

        frame_length = int(self.sample_rate * (frame_duration_ms / 1000.0))
        hop_length = int(self.sample_rate * (hop_duration_ms / 1000.0))

        if len(audio_data) < frame_length:
            return audio_data

        # Chia khung (Frames)
        frames = [
            audio_data[i : i + frame_length]
            for i in range(0, len(audio_data) - frame_length, hop_length)
        ]

        if not frames:
            return audio_data

        # Tính năng lượng ngắn hạn (Short-time Energy)
        energies = np.array([np.sum(f ** 2) for f in frames])
        threshold = energy_ratio * np.max(energies)

        # Lọc các khung vượt ngưỡng
        voiced_indices = np.where(energies > threshold)[0]

        if len(voiced_indices) > 0:
            start_sample = voiced_indices[0] * hop_length
            end_sample = min(len(audio_data), (voiced_indices[-1] + 1) * hop_length + frame_length)
            return audio_data[start_sample:end_sample]

        return audio_data

    def process(self, audio_data: np.ndarray, apply_filter: bool = True) -> np.ndarray:
        """
        Pipeline tiền xử lý hoàn chỉnh:
        1. Lọc Bandpass tần số giọng nói (Tùy chọn)
        2. Cắt khoảng lặng đầu/cuối
        3. Chuẩn hóa biên độ âm thanh
        """
        if audio_data is None or len(audio_data) == 0:
            return audio_data

        # Đảm bảo mảng phẳng 1D float32
        processed = audio_data.flatten().astype(np.float32)

        # Bước 1: Lọc Bandpass lọc nhiễu nền/rít
        if apply_filter:
            processed = self.apply_bandpass_filter(processed)

        # Bước 2: Trim khoảng lặng
        processed = self.trim_silence(processed)

        # Bước 3: Chuẩn hóa âm lượng
        processed = self.normalize_amplitude(processed)

        return processed