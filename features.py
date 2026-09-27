"""
Audio Feature Engineering & Signal Processing for TinyML KWS
------------------------------------------------------------
Implements:
1. 40-band Triangular Log-Mel Filterbank Spectrogram (matching Google MicroFrontend & ESP-DSP)
2. PCEN (Per-Channel Energy Normalization) for ambient noise robustness
3. SpecAugment (Time & Frequency masking for regularized training)
"""

import math
import numpy as np


class AudioFeatureExtractor:
    """
    Computes 40-channel Log-Mel Spectrogram or PCEN features:
    - Sample rate: 16,000 Hz
    - Window size: 30 ms (480 samples, padded to 512 for FFT)
    - Stride / Hop: 20 ms (320 samples)
    - Mel bands: 40 (100 Hz to 7500 Hz)
    - Target Output for 1.0s audio: (49, 40) [Time Frames, Mel Bands]
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 480,
        hop_length: int = 320,
        n_mels: int = 40,
        f_min: float = 100.0,
        f_max: float = 7500.0,
        use_pcen: bool = False
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.use_pcen = use_pcen

        # Precompute Hanning window
        self.window = np.hanning(win_length).astype(np.float32)

        # Precompute Mel Filterbank matrix (n_fft // 2 + 1, n_mels)
        self.mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self, sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        weights = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
        for i in range(n_mels):
            left = bins[i]
            center = bins[i + 1]
            right = bins[i + 2]
            if center > left:
                weights[left:center, i] = np.linspace(0, 1, center - left, endpoint=False)
            if right > center:
                weights[center:right, i] = np.linspace(1, 0, right - center, endpoint=False)
        return weights

    def compute_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Input: 1D numpy array of audio samples (e.g. 16,000 samples for 1.0s at 16kHz)
        Output: 2D numpy array of shape (49, 40)
        """
        # Ensure exact 16000 samples (1.0s)
        target_len = self.sample_rate
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        elif len(waveform) > target_len:
            waveform = waveform[:target_len]

        num_frames = (len(waveform) - self.win_length) // self.hop_length + 1
        frames = np.zeros((num_frames, self.win_length), dtype=np.float32)
        for t in range(num_frames):
            start = t * self.hop_length
            frames[t] = waveform[start:start + self.win_length] * self.window

        # Pad frames to n_fft
        padded_frames = np.pad(frames, ((0, 0), (0, self.n_fft - self.win_length)))

        # Real FFT & Power Spectrum |FFT|^2
        fft_out = np.fft.rfft(padded_frames, n=self.n_fft, axis=1)
        power_spec = np.abs(fft_out) ** 2  # (num_frames, n_fft // 2 + 1)

        # Apply 40-band Mel Filterbank
        mel_spec = np.dot(power_spec, self.mel_fb)  # (num_frames, 40)

        if self.use_pcen:
            # Per-Channel Energy Normalization
            features = self._apply_pcen(mel_spec)
        else:
            # Standard Log Compression: log(E + eps)
            features = np.log(mel_spec + 1e-5)

        # Ensure exact time steps (49 frames)
        if len(features) < 49:
            features = np.pad(features, ((0, 49 - len(features)), (0, 0)))
        elif len(features) > 49:
            features = features[:49, :]

        return features.astype(np.float32)

    def _apply_pcen(self, mel_spec: np.ndarray, s: float = 0.025, alpha: float = 0.98, delta: float = 2.0, r: float = 0.5) -> np.ndarray:
        """
        PCEN: Per-Channel Energy Normalization (Wang et al. 2017)
        Performs adaptive temporal smoothing and dynamic range compression.
        """
        m = np.zeros_like(mel_spec)
        m[0] = mel_spec[0]
        for t in range(1, len(mel_spec)):
            m[t] = (1.0 - s) * m[t - 1] + s * mel_spec[t]

        pcen = ((mel_spec / ((1e-6 + m) ** alpha)) + delta) ** r - (delta ** r)
        return pcen.astype(np.float32)


def apply_spec_augment(spec: np.ndarray, max_f: int = 4, max_t: int = 6) -> np.ndarray:
    """
    Applies SpecAugment directly to a (Time, Mel) 2D spectrogram.
    Randomly zeroes out horizontal (frequency) and vertical (time) bars.
    """
    augmented = spec.copy()
    time_len, mel_len = augmented.shape

    # Frequency masking
    f_width = np.random.randint(1, max_f + 1)
    f0 = np.random.randint(0, mel_len - f_width)
    augmented[:, f0:f0 + f_width] = 0.0

    # Time masking
    t_width = np.random.randint(1, max_t + 1)
    t0 = np.random.randint(0, time_len - t_width)
    augmented[t0:t0 + t_width, :] = 0.0

    return augmented


if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    dummy_audio = np.random.normal(0, 0.2, 16000).astype(np.float32)
    spec = extractor.compute_spectrogram(dummy_audio)
    print("AudioFeatureExtractor output shape:", spec.shape)
    assert spec.shape == (49, 40), f"Expected (49, 40), got {spec.shape}"
    aug_spec = apply_spec_augment(spec)
    print("SpecAugment successfully applied!")
