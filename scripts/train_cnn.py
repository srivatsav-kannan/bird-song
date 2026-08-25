"""Baseline A: fine-tuned EfficientNet-B0 on log-mel windows (the old paper's
approach done right: real features, real augmentation, leakage-proof splits,
selection on val only).

Usage: python scripts/train_cnn.py [--schemes recording event] [--seeds 0 1 2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.evaluation import SPECIES, aggregate_recordings, metrics  # noqa: E402
from birdsong.features import spec_augment  # noqa: E402
from birdsong.models import build_cnn  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_EPOCHS = 30
PATIENCE = 8
BATCH = 32


def load_data():
    windows = pd.read_csv(REPO / "data" / "windows.csv")
    windows = windows[windows.kept].reset_index(drop=True)
    mels = np.load(REPO / "data" / "mels.npy", mmap_mode="r")
    ids = list(np.load(REPO / "data" / "mels_ids.npy"))
    idx_of = {k: i for i, k in enumerate(ids)}
    splits = json.loads((REPO / "data" / "splits.json").read_text())
    return windows, mels, idx_of, splits


def subset(windows, idx_of, rec_ids):
    sub = windows[windows.recording_id.isin(rec_ids)]
    idx = np.array([idx_of[f"{r.recording_id}|{r.window_idx}"] for r in sub.itertuples()])
    y = np.array([SPECIES.index(r.species) for r in sub.itertuples()])
    rid = np.array([r.recording_id for r in sub.itertuples()])
    return idx, y, rid


def batches(idx, y, mels, rng=None, train=False, batch=BATCH):
    order = rng.permutation(len(idx)) if train else np.arange(len(idx))
    for i in range(0, len(order), batch):
        sel = order[i:i + batch]
        X = mels[idx[sel]].astype(np.float32)
        if train:
            X = np.stack([spec_augment(x, rng) for x in X])
        yield (torch.tensor(X).unsqueeze(1),
               torch.tensor(y[sel]))


def rec_macro_f1(probs, rids, y_map):
    ids, agg = aggregate_recordings(probs, list(rids))
    y_true = np.array([y_map[r] for r in ids])
    return metrics(y_true, agg.argmax(1))["macro_f1"]


def run_fold(windows, mels, idx_of, parts, seed):
    tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
    itr, ytr, _ = subset(windows, idx_of, tr)
    iva, yva, rva = subset(windows, idx_of, va)
    ite, yte, rte = subset(windows, idx_of, te)
    y_map = {r.recording_id: SPECIES.index(r.species)
             for r in windows.drop_duplicates("recording_id").itertuples()}

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_cnn(len(SPECIES)).to(DEVICE)
    counts = np.bincount(ytr, minlength=len(SPECIES))
    w_np = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = torch.tensor(w_np, dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w / w.mean(), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_f1, best_state, since = -1, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for X, yb in batches(itr, ytr, mels, rng, train=True):
            X, yb = X.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        pv = []
        with torch.no_grad():
            for X, _ in batches(iva, yva, mels):
                pv.append(torch.softmax(model(X.to(DEVICE)), 1).cpu().numpy())
        f1 = rec_macro_f1(np.concatenate(pv), rva, y_map)
        if f1 > best_f1:
            best_f1, since = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)

    model.eval()
    pt = []
    with torch.no_grad():
        for X, _ in batches(ite, yte, mels):
            pt.append(torch.softmax(model(X.to(DEVICE)), 1).cpu().numpy())
    return {"window_probs": np.concatenate(pt).tolist(),
            "window_rids": rte.tolist(), "window_y": yte.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["recording", "event"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    windows, mels, idx_of, splits = load_data()
    for scheme in args.schemes:
        for fold, parts in splits[scheme].items():
            for seed in args.seeds:
                out = run_fold(windows, mels, idx_of, parts, seed)
                dest = REPO / "results" / "preds" / scheme / "cnn"
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / f"{fold}_seed{seed}.json", "w") as f:
                    json.dump(out, f)
                print(f"cnn {scheme} {fold} seed{seed}: done", flush=True)


if __name__ == "__main__":
    main()
