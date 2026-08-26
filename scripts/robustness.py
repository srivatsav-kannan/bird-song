"""Noise-robustness stress test.

For each recording-scheme fold: train the MLP probe on clean train embeddings
(seed 0, same protocol as train_probe), then evaluate on test windows that have
been corrupted with additive white noise at fixed SNRs BEFORE BirdNET embedding
extraction. Reports recording-level accuracy vs SNR.

Output: results/robustness.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.audio import decode_cache_dir, load_window  # noqa: E402
from birdsong.evaluation import SPECIES, aggregate_recordings, metrics  # noqa: E402
from birdsong.expdata import load_all, window_arrays  # noqa: E402

import birdnet_analyzer.config as cfg  # noqa: E402
cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
import birdnet_analyzer.model as bn_model  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from train_probe import train_probe, predict_probe  # noqa: E402

SNRS_DB = [20, 10, 5, 0]
BATCH = 16


def noisy(y, snr_db, rng):
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(p_noise), len(y))).astype(np.float32)


def embed_noisy(windows, rec_ids, snr_db, cache, rng):
    sub = windows[windows.recording_id.isin(rec_ids)].reset_index(drop=True)
    embs = []
    for start in range(0, len(sub), BATCH):
        chunk = sub.iloc[start:start + BATCH]
        batch = np.stack([
            noisy(load_window(cache / f"{r.recording_id}.wav", r.start_s), snr_db, rng)
            for r in chunk.itertuples()])
        embs.append(bn_model.embeddings(batch))
    X = np.concatenate(embs)
    y = np.array([SPECIES.index(s) for s in sub.species])
    rid = sub.recording_id.to_numpy()
    return X, y, rid


def main():
    windows, emb_of, _, splits = load_all()
    cache = decode_cache_dir()
    y_by_rid = {r.recording_id: SPECIES.index(r.species)
                for r in windows.drop_duplicates("recording_id").itertuples()}

    results = {str(s): {"y_true": [], "y_pred": []} for s in SNRS_DB}
    results["clean"] = {"y_true": [], "y_pred": []}

    for fold, parts in tqdm(splits["recording"].items()):
        tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
        Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
        Xva, yva, rva = window_arrays(windows, emb_of, va)
        model = train_probe(Xtr, ytr, Xva, yva, rva, y_by_rid, seed=0, in_dim=1024)

        def eval_on(X, y, rid, key):
            probs = predict_probe(model, X)
            ids, agg = aggregate_recordings(probs, list(rid))
            results[key]["y_true"].extend(int(y_by_rid[r]) for r in ids)
            results[key]["y_pred"].extend(int(p) for p in agg.argmax(1))

        Xte, yte, rte = window_arrays(windows, emb_of, te)
        eval_on(Xte, yte, rte, "clean")
        rng = np.random.default_rng(42)
        for snr in SNRS_DB:
            Xn, yn, rn = embed_noisy(windows, te, snr, cache, rng)
            eval_on(Xn, yn, rn, str(snr))

    summary = {}
    for key, d in results.items():
        y_t, y_p = np.array(d["y_true"]), np.array(d["y_pred"])
        m = metrics(y_t, y_p)
        summary[key] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "n": int(len(y_t)),
                        "y_true": [int(v) for v in y_t],
                        "y_pred": [int(v) for v in y_p]}
        print(f"SNR {key:>5}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")

    with open(REPO / "results" / "robustness.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
