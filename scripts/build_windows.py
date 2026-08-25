"""Decode all manifest recordings and extract activity-scored 3 s windows.

Output: data/windows.csv — one row per window (kept + rejected, for auditability).
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.audio import decode, extract_windows  # noqa: E402


def main():
    manifest = pd.read_csv(REPO / "data" / "manifest.csv")
    rows = []
    for r in tqdm(manifest.itertuples(), total=len(manifest)):
        wav = decode(r.path, r.recording_id)
        for i, w in enumerate(extract_windows(wav)):
            rows.append({
                "recording_id": r.recording_id,
                "species": r.species,
                "window_idx": i,
                **w,
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "data" / "windows.csv", index=False)
    kept = df[df.kept]
    print(f"{len(df)} windows total, {len(kept)} kept")
    print(kept.groupby("species").agg(windows=("kept", "size"),
                                      recordings=("recording_id", "nunique")))


if __name__ == "__main__":
    main()
