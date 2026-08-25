"""Extract BirdNET V2.4 embeddings (1024-d) for every kept window.

Also runs BirdNET's own classifier head on each window and stores the logits for
our species where present in BirdNET's label set (external reference baseline).

Output: data/embeddings.npz  (ids, embeddings, plus birdnet-prior scores)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.audio import decode_cache_dir, load_window  # noqa: E402

import birdnet_analyzer.config as cfg  # noqa: E402
cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
cfg.LABELS_FILE = str(Path(cfg.BIRDNET_MODEL_PATH).parent / "BirdNET_GLOBAL_6K_V2.4_Labels.txt")
import birdnet_analyzer.model as model  # noqa: E402

# BirdNET label names for our species (scientific_CommonName format)
TARGET_LABELS = {
    "ForestOwlet": "Athene blewitti_Forest Owlet",
    "BanasuraLaughingthrush": "Montecincla jerdoni_Banasura Laughingthrush",
    "BugunLiocichla": "Liocichla bugunorum_Bugun Liocichla",
    "JerdonsCourser": "Rhinoptilus bitorquatus_Jerdon's Courser",
}

BATCH = 16


def main():
    windows = pd.read_csv(REPO / "data" / "windows.csv")
    windows = windows[windows.kept].reset_index(drop=True)
    cache = decode_cache_dir()

    labels = Path(cfg.LABELS_FILE).read_text().splitlines()
    label_idx = {}
    for sp, lab in TARGET_LABELS.items():
        label_idx[sp] = labels.index(lab) if lab in labels else -1
    print("BirdNET label coverage:", {k: (v >= 0) for k, v in label_idx.items()})

    ids, embs, bn_scores = [], [], []
    for start in tqdm(range(0, len(windows), BATCH)):
        chunk = windows.iloc[start:start + BATCH]
        batch = np.stack([
            load_window(cache / f"{r.recording_id}.wav", r.start_s)
            for r in chunk.itertuples()
        ])
        embs.append(model.embeddings(batch))
        logits = model.predict(batch)
        probs = 1 / (1 + np.exp(-np.asarray(logits)))
        cols = np.stack([
            probs[:, label_idx[sp]] if label_idx[sp] >= 0 else np.zeros(len(chunk))
            for sp in TARGET_LABELS
        ], axis=1)
        bn_scores.append(cols)
        ids.extend(f"{r.recording_id}|{r.window_idx}" for r in chunk.itertuples())

    np.savez_compressed(
        REPO / "data" / "embeddings.npz",
        ids=np.array(ids),
        embeddings=np.concatenate(embs).astype(np.float32),
        birdnet_scores=np.concatenate(bn_scores).astype(np.float32),
        birdnet_score_species=np.array(list(TARGET_LABELS.keys())),
    )
    print(f"Saved {len(ids)} embeddings")


if __name__ == "__main__":
    main()
