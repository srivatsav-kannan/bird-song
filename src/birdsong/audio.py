"""Audio decoding, windowing, and vocalization-activity detection."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .ff import FFMPEG

SR = 48_000            # decode rate (BirdNET-native; mels are resampled downstream)
WIN_S = 3.0            # analysis window length (s)
HOP_S = 1.5            # window hop (s)
BAND = (150, 11_000)   # Hz band considered for vocal activity


def decode_cache_dir() -> Path:
    d = Path.home() / "Datasets" / "Bird Classification" / "Bird Sound" / "Decoded48k"
    d.mkdir(parents=True, exist_ok=True)
    return d


def decode(src_path: str, recording_id: str) -> Path:
    """Decode any input to mono 48 kHz WAV, cached by recording id."""
    dest = decode_cache_dir() / f"{recording_id}.wav"
    if not dest.exists():
        subprocess.run(
            [FFMPEG, "-y", "-v", "quiet", "-i", src_path,
             "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dest)],
            check=True,
        )
    return dest


def band_energy_db(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Frame-wise energy (dB) within the bird-vocalization band."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= BAND[0]) & (freqs <= BAND[1])
    e = S[mask].sum(axis=0)
    return 10.0 * np.log10(e + 1e-10)


def extract_windows(wav_path: Path, snr_thresh_db: float = 3.0, min_keep: int = 1):
    """Slice a recording into overlapping windows and score vocal activity.

    Activity score = median in-band frame energy of the window minus the
    recording's noise floor (10th percentile of in-band frame energy).
    Windows scoring >= snr_thresh_db are kept; if none clear the bar the
    top `min_keep` windows are kept so no recording drops out of the dataset.

    Returns list of dicts: start_s, end_s, snr_db, kept.
    """
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    if len(y) < win:                       # pad ultra-short recordings to one window
        y = np.pad(y, (0, win - len(y)))

    frame_hop = 512
    e_db = band_energy_db(y, sr, hop=frame_hop)
    noise_floor = np.percentile(e_db, 10)

    out = []
    for start in range(0, len(y) - win + 1, hop):
        f0 = start // frame_hop
        f1 = (start + win) // frame_hop
        snr = float(np.median(e_db[f0:f1]) - noise_floor)
        out.append({"start_s": round(start / sr, 3),
                    "end_s": round((start + win) / sr, 3),
                    "snr_db": round(snr, 2),
                    "kept": snr >= snr_thresh_db})
    if not any(w["kept"] for w in out):
        for w in sorted(out, key=lambda w: -w["snr_db"])[:min_keep]:
            w["kept"] = True
    return out


def load_window(wav_path: Path, start_s: float, sr: int = SR, dur_s: float = WIN_S) -> np.ndarray:
    n0, n = int(start_s * sr), int(dur_s * sr)
    y, _ = sf.read(wav_path, start=n0, frames=n, dtype="float32", fill_value=0.0)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y
