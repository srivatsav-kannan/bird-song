"""Build the auxiliary domain-adaptation corpus: the OTHER 29 endemic species.

These recordings adapt the embedding space to Indian endemic avifauna
(supervised-contrastive pretraining). The 4 target species are excluded here by
construction, so no auxiliary data can leak into target-species evaluation.

Pipeline mirrors the target pipeline: dedupe (sha256) -> decode 48 kHz mono ->
3 s activity-scored windows -> BirdNET embeddings.

Output: data/aux_embeddings.npz (embeddings, species labels, recording ids)
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.audio import decode, extract_windows, load_window, decode_cache_dir  # noqa: E402

import birdnet_analyzer.config as cfg  # noqa: E402
cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
import birdnet_analyzer.model as model  # noqa: E402

AUDIO_ROOT = Path.home() / "Datasets" / "Bird Classification" / "Bird Sound" / "Dataset"
TARGET_DIRS = {
    "ForestOwlet-Atheneblewitti",
    "BanasuraLaughingthrush-Montecinclajerdoni",
    "BugunLiocichla-Liocichlabugunorum",
    "Jerdon'sCourser-Rhinoptilusbitorquatus",
}
VALID_EXT = {".wav", ".mp3", ".m4a"}
MAX_WINDOWS_PER_REC = 12   # cap so long recordings don't dominate
BATCH = 16


def rec_id(path: Path, species: str) -> str:
    m = re.search(r"XC\s?(\d+)", path.name)
    if m:
        return f"XC{m.group(1)}"
    m = re.match(r"(\d+)", path.stem)
    if m:
        return f"ML{m.group(1)}"
    return "H" + hashlib.sha1(str(path).encode()).hexdigest()[:10]


def main():
    rows, seen = [], set()
    for spdir in sorted(AUDIO_ROOT.iterdir()):
        if not spdir.is_dir() or spdir.name in TARGET_DIRS:
            continue
        species = spdir.name.split("-")[0]
        for path in sorted(spdir.rglob("*")):
            if path.suffix.lower() not in VALID_EXT or path.name.startswith("."):
                continue
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            rows.append({"recording_id": f"AUX_{rec_id(path, species)}",
                         "species": species, "path": str(path)})
    df = pd.DataFrame(rows).drop_duplicates("recording_id")
    print(f"Aux corpus: {df.species.nunique()} species, {len(df)} unique recordings")

    cache = decode_cache_dir()
    win_rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="decode+window"):
        try:
            wav = decode(r.path, r.recording_id)
        except Exception as e:
            print(f"decode failed {r.recording_id}: {e}")
            continue
        wins = [w for w in extract_windows(wav) if w["kept"]]
        wins = sorted(wins, key=lambda w: -w["snr_db"])[:MAX_WINDOWS_PER_REC]
        for w in wins:
            win_rows.append({"recording_id": r.recording_id, "species": r.species,
                             "start_s": w["start_s"]})
    wdf = pd.DataFrame(win_rows)
    print(f"{len(wdf)} aux windows")

    ids, specs, embs = [], [], []
    for start in tqdm(range(0, len(wdf), BATCH), desc="embed"):
        chunk = wdf.iloc[start:start + BATCH]
        batch = np.stack([
            load_window(cache / f"{r.recording_id}.wav", r.start_s)
            for r in chunk.itertuples()
        ])
        embs.append(model.embeddings(batch))
        ids.extend(chunk.recording_id)
        specs.extend(chunk.species)

    np.savez_compressed(
        REPO / "data" / "aux_embeddings.npz",
        ids=np.array(ids), species=np.array(specs),
        embeddings=np.concatenate(embs).astype(np.float32),
    )
    print("Saved aux embeddings")


if __name__ == "__main__":
    main()
