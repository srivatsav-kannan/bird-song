"""Train and evaluate embedding-based models (MLP probe, attention-MIL) across
all folds of both split schemes, with multiple seeds.

Model selection (early stopping on val recording-level macro-F1) never touches
the test fold. Test predictions are written to results/preds/ for evaluate.py.

Usage:
  python scripts/train_probe.py --model probe|mil [--adapter results/adapter.pt]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.evaluation import SPECIES, aggregate_recordings, metrics  # noqa: E402
from birdsong.expdata import load_all, window_arrays, bags  # noqa: E402
from birdsong.models import MLPProbe, GatedAttentionMIL, AdapterProjection  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [0, 1, 2]
MAX_EPOCHS = 200
PATIENCE = 20


def apply_adapter(adapter, X):
    with torch.no_grad():
        return adapter(torch.tensor(X)).cpu().numpy()


def rec_macro_f1(probs, rids, y_by_rid, n_classes=len(SPECIES)):
    ids, agg = aggregate_recordings(probs, list(rids))
    y_true = np.array([y_by_rid[r] for r in ids])
    y_pred = agg.argmax(1)
    return metrics(y_true, y_pred, n_classes=n_classes)["macro_f1"]


def train_probe(Xtr, ytr, Xva, yva, rva, y_by_rid, seed, in_dim):
    n_classes = int(ytr.max()) + 1
    torch.manual_seed(seed)
    model = MLPProbe(n_classes, in_dim=in_dim).to(DEVICE)
    counts = np.bincount(ytr, minlength=n_classes)
    w_np = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = torch.tensor(w_np, dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w / w.mean())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr).to(DEVICE)
    ytr_t = torch.tensor(ytr).to(DEVICE)
    Xva_t = torch.tensor(Xva).to(DEVICE)

    best_f1, best_state, since = -1, None, 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(DEVICE)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.softmax(model(Xva_t), 1).cpu().numpy()
        f1 = rec_macro_f1(pva, rva, y_by_rid, n_classes=n_classes)
        if f1 > best_f1:
            best_f1, since = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def predict_probe(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X).to(DEVICE)), 1).cpu().numpy()


def collate_bags(bag_list, device):
    K = max(len(b[2]) for b in bag_list)
    D = bag_list[0][2].shape[1]
    X = torch.zeros(len(bag_list), K, D)
    M = torch.zeros(len(bag_list), K, dtype=torch.bool)
    y = torch.tensor([b[1] for b in bag_list])
    for i, (_, _, arr) in enumerate(bag_list):
        X[i, :len(arr)] = torch.tensor(arr)
        M[i, :len(arr)] = True
    return X.to(device), M.to(device), y.to(device)


def train_mil(bags_tr, bags_va, seed, in_dim):
    torch.manual_seed(seed)
    model = GatedAttentionMIL(len(SPECIES), in_dim=in_dim).to(DEVICE)
    counts = np.bincount([b[1] for b in bags_tr], minlength=len(SPECIES))
    w_np = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = torch.tensor(w_np, dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w / w.mean())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    rng = np.random.default_rng(seed)
    best_f1, best_state, since = -1, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(bags_tr))
        for i in range(0, len(order), 16):
            chunk = [bags_tr[j] for j in order[i:i + 16]]
            X, M, y = collate_bags(chunk, DEVICE)
            opt.zero_grad()
            logits, _ = model(X, M)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        # val: recording-level macro-F1 directly (bags are recordings)
        model.eval()
        with torch.no_grad():
            X, M, y = collate_bags(bags_va, DEVICE)
            probs = torch.softmax(model(X, M)[0], 1).cpu().numpy()
        f1 = metrics(np.array([b[1] for b in bags_va]), probs.argmax(1))["macro_f1"]
        if f1 > best_f1:
            best_f1, since = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def predict_mil(model, bag_list):
    model.eval()
    with torch.no_grad():
        X, M, _ = collate_bags(bag_list, DEVICE)
        probs, attn = model(X, M)
        return torch.softmax(probs, 1).cpu().numpy(), attn.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["probe", "mil"], required=True)
    ap.add_argument("--adapter", default=None, help="path to trained adapter .pt")
    ap.add_argument("--schemes", nargs="+", default=["recording", "event"])
    args = ap.parse_args()

    windows, emb_of, _, splits = load_all()
    y_by_rid = {r.recording_id: SPECIES.index(r.species)
                for r in windows.drop_duplicates("recording_id").itertuples()}

    in_dim = 1024
    adapter = None
    if args.adapter:
        ckpt = torch.load(args.adapter, map_location="cpu", weights_only=True)
        adapter = AdapterProjection(out_dim=ckpt["out_dim"])
        adapter.load_state_dict(ckpt["state"])
        adapter.eval()
        emb_of = {k: v for k, v in emb_of.items()}
        keys = list(emb_of.keys())
        X = np.stack([emb_of[k] for k in keys])
        Xp = apply_adapter(adapter, X)
        emb_of = {k: Xp[i] for i, k in enumerate(keys)}
        in_dim = ckpt["out_dim"]

    tag = args.model + ("_adapted" if args.adapter else "")
    for scheme in args.schemes:
        for fold, parts in splits[scheme].items():
            tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
            for seed in SEEDS:
                if args.model == "probe":
                    Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
                    Xva, yva, rva = window_arrays(windows, emb_of, va)
                    Xte, yte, rte = window_arrays(windows, emb_of, te)
                    m = train_probe(Xtr, ytr, Xva, yva, rva, y_by_rid, seed, in_dim)
                    pte = predict_probe(m, Xte)
                    out = {"window_probs": pte.tolist(), "window_rids": rte.tolist(),
                           "window_y": yte.tolist()}
                else:
                    btr = bags(windows, emb_of, tr)
                    bva = bags(windows, emb_of, va)
                    bte = bags(windows, emb_of, te)
                    m = train_mil(btr, bva, seed, in_dim)
                    probs, attn = predict_mil(m, bte)
                    out = {"rec_probs": probs.tolist(),
                           "rec_ids": [b[0] for b in bte],
                           "rec_y": [b[1] for b in bte],
                           "attn": [a[:len(b[2])].tolist() for a, b in zip(attn, bte)]}
                dest = REPO / "results" / "preds" / scheme / tag
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / f"{fold}_seed{seed}.json", "w") as f:
                    json.dump(out, f)
            print(f"{tag} {scheme} {fold}: done ({len(te)} test recordings)")


if __name__ == "__main__":
    main()
