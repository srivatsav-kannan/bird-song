"""Shared experiment data loading: windows, embeddings, splits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import SPECIES

REPO = Path(__file__).resolve().parents[2]


def load_all():
    windows = pd.read_csv(REPO / "data" / "windows.csv")
    windows = windows[windows.kept].reset_index(drop=True)
    z = np.load(REPO / "data" / "embeddings.npz", allow_pickle=False)
    emb_of = {i: e for i, e in zip(z["ids"], z["embeddings"])}
    bn = {i: s for i, s in zip(z["ids"], z["birdnet_scores"])}
    splits = json.loads((REPO / "data" / "splits.json").read_text())
    return windows, emb_of, bn, splits


def window_arrays(windows: pd.DataFrame, emb_of: dict, rec_ids: set):
    """Return X (N,1024), y (N,), rid (N,) for windows of the given recordings."""
    sub = windows[windows.recording_id.isin(rec_ids)]
    X = np.stack([emb_of[f"{r.recording_id}|{r.window_idx}"] for r in sub.itertuples()])
    y = np.array([SPECIES.index(r.species) for r in sub.itertuples()])
    rid = np.array([r.recording_id for r in sub.itertuples()])
    return X, y, rid


def bags(windows: pd.DataFrame, emb_of: dict, rec_ids: set, max_k: int = 64):
    """Return list of (recording_id, label, (K,1024) array), highest-SNR windows first."""
    out = []
    sub = windows[windows.recording_id.isin(rec_ids)]
    for rid, g in sub.groupby("recording_id"):
        g = g.sort_values("snr_db", ascending=False).head(max_k)
        X = np.stack([emb_of[f"{rid}|{i}"] for i in g.window_idx])
        out.append((rid, SPECIES.index(g.species.iloc[0]), X))
    return out
