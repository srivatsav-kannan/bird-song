"""Log-mel features and spectrogram-domain augmentation for the CNN baseline."""

import numpy as np

MEL_SR = 32_000
N_MELS = 128
N_FFT = 2048
HOP = 320
FMAX = 12_000


def logmel(y_48k: np.ndarray) -> np.ndarray:
    """3 s @ 48 kHz mono -> (128, T) log-mel array in [0, 1]."""
    import librosa
    y = librosa.resample(y_48k, orig_sr=48_000, target_sr=MEL_SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=MEL_SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
    S = librosa.power_to_db(S, ref=np.max)
    return (S + 80.0) / 80.0  # [-80, 0] dB -> [0, 1]


def spec_augment(S: np.ndarray, rng: np.random.Generator,
                 n_time_masks=2, n_freq_masks=2, max_t=30, max_f=16) -> np.ndarray:
    S = S.copy()
    n_mels, T = S.shape
    for _ in range(n_time_masks):
        w = rng.integers(0, max_t + 1)
        t0 = rng.integers(0, max(1, T - w))
        S[:, t0:t0 + w] = S.mean()
    for _ in range(n_freq_masks):
        w = rng.integers(0, max_f + 1)
        f0 = rng.integers(0, max(1, n_mels - w))
        S[f0:f0 + w, :] = S.mean()
    return S


def waveform_augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gain jitter, circular time shift, and white-noise injection at random SNR."""
    y = y * rng.uniform(0.6, 1.4)
    y = np.roll(y, rng.integers(0, len(y)))
    if rng.random() < 0.5:
        snr_db = rng.uniform(8, 25)
        p_sig = np.mean(y ** 2) + 1e-12
        p_noise = p_sig / (10 ** (snr_db / 10))
        y = y + rng.normal(0, np.sqrt(p_noise), len(y)).astype(np.float32)
    return y.astype(np.float32)
