"""Precompute log-mel arrays for all kept windows -> data/mels.npy (+ index)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.audio import decode_cache_dir, load_window  # noqa: E402
from birdsong.features import logmel  # noqa: E402


def main():
    windows = pd.read_csv(REPO / "data" / "windows.csv")
    windows = windows[windows.kept].reset_index(drop=True)
    cache = decode_cache_dir()

    first = logmel(load_window(cache / f"{windows.iloc[0].recording_id}.wav",
                               windows.iloc[0].start_s))
    out = np.lib.format.open_memmap(
        REPO / "data" / "mels.npy", mode="w+", dtype=np.float16,
        shape=(len(windows), *first.shape))
    ids = []
    for i, r in enumerate(tqdm(windows.itertuples(), total=len(windows))):
        y = load_window(cache / f"{r.recording_id}.wav", r.start_s)
        out[i] = logmel(y).astype(np.float16)
        ids.append(f"{r.recording_id}|{r.window_idx}")
    out.flush()
    np.save(REPO / "data" / "mels_ids.npy", np.array(ids))
    print(f"mels: {out.shape} saved")


if __name__ == "__main__":
    main()
