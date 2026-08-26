"""Analyses requested by review: channel-negative test, cross-archive
near-duplicate screen, cluster bootstrap, multi-split open-set with per-species
false acceptance, event-threshold sensitivity, per-species event counts,
BirdNET-native Forest Owlet reference, and attention-activity correlation.

Output: results/review_fixes.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from birdsong.audio import decode_cache_dir, load_window  # noqa: E402
from birdsong.evaluation import SPECIES, aggregate_recordings  # noqa: E402
from birdsong.expdata import load_all, window_arrays  # noqa: E402
from train_probe import train_probe, predict_probe  # noqa: E402

import birdnet_analyzer.config as cfg  # noqa: E402
cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
import birdnet_analyzer.model as bn_model  # noqa: E402

out = {}
windows, emb_of, _, splits = load_all()
manifest = pd.read_csv(REPO / "data" / "manifest.csv")
y_by_rid = {r.recording_id: SPECIES.index(r.species)
            for r in windows.drop_duplicates("recording_id").itertuples()}
cache = decode_cache_dir()

# ---------- 1. channel-matched negatives (M4) ----------
# Rejected (below-threshold) windows of the Jerdon's Courser ML session are
# ambient audio through the same recording chain. Embed them and classify with
# fold probes whose test fold contains JC recordings.
allw = pd.read_csv(REPO / "data" / "windows.csv")
jc_amb = allw[(allw.species == "JerdonsCourser") & (~allw.kept)
              & allw.recording_id.str.startswith("ML")]
print(f"JC ambient windows: {len(jc_amb)}")
amb_X = []
BATCH = 16
rows = list(jc_amb.itertuples())
for i in range(0, len(rows), BATCH):
    chunk = rows[i:i + BATCH]
    batch = np.stack([load_window(cache / f"{r.recording_id}.wav", r.start_s)
                      for r in chunk])
    amb_X.append(bn_model.embeddings(batch))
amb_X = np.concatenate(amb_X) if amb_X else np.zeros((0, 1024))

pred_counts = Counter()
probs_jc = []
for fold, parts in splits["recording"].items():
    tr, va = set(parts["train"]), set(parts["val"])
    Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
    Xva, yva, rva = window_arrays(windows, emb_of, va)
    model = train_probe(Xtr, ytr, Xva, yva, rva, y_by_rid, seed=0, in_dim=1024)
    p = predict_probe(model, amb_X.astype(np.float32))
    pred_counts.update(SPECIES[i] for i in p.argmax(1))
    probs_jc.extend(p[:, SPECIES.index("JerdonsCourser")].tolist())
out["channel_negatives"] = {
    "n_ambient_windows": int(len(jc_amb)),
    "pred_fraction": {k: v / max(1, sum(pred_counts.values()))
                      for k, v in pred_counts.items()},
    "mean_jc_prob": float(np.mean(probs_jc)) if probs_jc else None,
    "frac_called_jc": pred_counts["JerdonsCourser"] / max(1, sum(pred_counts.values())),
}
print("channel negatives:", out["channel_negatives"])

# ---------- 2. cross-archive near-duplicate screen (M2) ----------
rec_emb = defaultdict(list)
for r in windows.itertuples():
    rec_emb[r.recording_id].append(emb_of[f"{r.recording_id}|{r.window_idx}"])
rec_mat = {k: np.stack(v) / np.linalg.norm(np.stack(v), axis=1, keepdims=True)
           for k, v in rec_emb.items()}
pairs = []
for sp, g in manifest.groupby("species"):
    xc = [r for r in g[g.source == "XC"].recording_id if r in rec_mat]
    ml = [r for r in g[g.source == "ML"].recording_id if r in rec_mat]
    for a in xc:
        for b in ml:
            s = float((rec_mat[a] @ rec_mat[b].T).max())
            pairs.append((sp, a, b, s))
sims = sorted(pairs, key=lambda x: -x[3])
out["cross_archive_screen"] = {
    "n_pairs": len(pairs),
    "max_similarity": sims[0][3] if sims else None,
    "top5": [{"species": s, "xc": a, "ml": b, "max_cos": round(v, 3)}
             for s, a, b, v in sims[:5]],
    "n_above_0.95": sum(1 for *_, v in sims if v > 0.95),
}
print("cross-archive:", out["cross_archive_screen"]["max_similarity"],
      "pairs>0.95:", out["cross_archive_screen"]["n_above_0.95"])

# ---------- 3. cluster bootstrap over recording events (M8) ----------
ev_of = splits["event_group_of"]
# seed-averaged probe predictions per recording (recording scheme)
per_seed = defaultdict(dict)
for f in sorted((REPO / "results" / "preds" / "recording" / "probe").glob("*.json")):
    seed = f.stem.split("seed")[1]
    data = json.loads(f.read_text())
    ids, agg = aggregate_recordings(np.array(data["window_probs"]), data["window_rids"])
    for rid, p in zip(ids, agg):
        per_seed[seed][rid] = p
rids = sorted(y_by_rid)
y = np.array([y_by_rid[r] for r in rids])
yp = np.stack([np.mean([per_seed[s][r] for s in per_seed], 0) for r in rids]).argmax(1)
clusters = defaultdict(list)
for i, r in enumerate(rids):
    clusters[ev_of[r]].append(i)
keys = list(clusters)
rng = np.random.default_rng(0)
from sklearn.metrics import f1_score
accs, f1s = [], []
for _ in range(2000):
    pick = rng.integers(0, len(keys), len(keys))
    idx = np.concatenate([clusters[keys[k]] for k in pick])
    accs.append((y[idx] == yp[idx]).mean())
    f1s.append(f1_score(y[idx], yp[idx], labels=range(4), average="macro",
                        zero_division=0))
out["cluster_bootstrap_probe"] = {
    "n_events": len(keys),
    "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
    "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
}
print("cluster bootstrap:", out["cluster_bootstrap_probe"])

# ---------- 4. events per species + threshold sensitivity (M3, minor 9) ----------
def build_events(gap):
    ev = {}
    for (sp, src), g in manifest.groupby(["species", "source"]):
        ids = sorted(g.recording_id, key=lambda r: int(re.sub(r"\D", "", r)))
        c, prev = 0, None
        for rid in ids:
            n = int(re.sub(r"\D", "", rid))
            if prev is not None and n - prev >= gap:
                c += 1
            ev[rid] = f"{sp}|{src}|{c}"
            prev = n
    return ev

out["events_per_species"] = dict(Counter(
    sp for (sp, src, c) in {tuple(v.split("|")) for v in ev_of.values()}))
gap_counts = {g: len(set(build_events(g).values())) for g in [200, 500, 1000, 2000, 5000]}
out["event_threshold_sensitivity"] = gap_counts
print("events/species:", out["events_per_species"], "| gap sensitivity:", gap_counts)

# ---------- 5. open-set: repeated species splits + per-species FA (M6) ----------
z = np.load(REPO / "data" / "aux_embeddings.npz")
aux_X, aux_sp, aux_rid = z["embeddings"], z["species"], z["ids"]
aux_species = sorted(set(aux_sp))
OTHER = len(SPECIES)
split_results = []
per_species_fa = defaultdict(list)
for split_seed in [11, 21, 31, 41, 51]:
    rng = np.random.default_rng(split_seed)
    order = list(aux_species)
    rng.shuffle(order)
    bg_train = set(order[:14])
    tr_mask = np.array([s in bg_train for s in aux_sp])
    ev_mask = ~tr_mask
    accs_, fas_ = [], []
    for fold, parts in splits["recording"].items():
        tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
        Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
        Xva, yva, rva = window_arrays(windows, emb_of, va)
        Xte, yte, rte = window_arrays(windows, emb_of, te)
        n_bg = int(np.bincount(ytr).mean() * 2)
        bg_idx = rng.choice(np.flatnonzero(tr_mask), size=min(n_bg, tr_mask.sum()), replace=False)
        Xtr5 = np.concatenate([Xtr, aux_X[bg_idx]])
        ytr5 = np.concatenate([ytr, np.full(len(bg_idx), OTHER)])
        bg_val = rng.choice(np.flatnonzero(tr_mask), size=max(20, len(bg_idx) // 5), replace=False)
        Xva5 = np.concatenate([Xva, aux_X[bg_val]])
        rva5 = np.concatenate([rva, aux_rid[bg_val]])
        y5 = dict(y_by_rid)
        for r in aux_rid[bg_val]:
            y5[r] = OTHER
        model = train_probe(Xtr5, ytr5, Xva5, None, rva5, y5, seed=0, in_dim=1024)
        pte = predict_probe(model, Xte)
        ids, agg = aggregate_recordings(pte, list(rte))
        yt = np.array([y_by_rid[r] for r in ids])
        accs_.append(float((agg.argmax(1) == yt).mean()))
        po = predict_probe(model, aux_X[ev_mask])
        ids_o, agg_o = aggregate_recordings(po, list(aux_rid[ev_mask]))
        acc_mask = agg_o.argmax(1) != OTHER
        fas_.append(float(acc_mask.mean()))
        sp_of_rec = {r: s for r, s in zip(aux_rid[ev_mask], aux_sp[ev_mask])}
        fa_by_sp = defaultdict(list)
        for r, a in zip(ids_o, acc_mask):
            fa_by_sp[sp_of_rec[r]].append(bool(a))
        for s, v in fa_by_sp.items():
            per_species_fa[s].append(float(np.mean(v)))
    split_results.append({"seed": split_seed,
                          "target_acc": float(np.mean(accs_)),
                          "fa": float(np.mean(fas_))})
    print(f"split {split_seed}: acc={np.mean(accs_):.3f} fa={np.mean(fas_):.3f}")

fa_mean = {s: float(np.mean(v)) for s, v in per_species_fa.items()}
out["openset_repeated"] = {
    "per_split": split_results,
    "target_acc_mean_sd": [float(np.mean([r["target_acc"] for r in split_results])),
                           float(np.std([r["target_acc"] for r in split_results]))],
    "fa_mean_sd": [float(np.mean([r["fa"] for r in split_results])),
                   float(np.std([r["fa"] for r in split_results]))],
    "per_species_fa_top": dict(sorted(fa_mean.items(), key=lambda kv: -kv[1])[:10]),
    "congeners": {s: fa_mean.get(s) for s in
                  ["PalaniLaughingthrush", "NilgiriLaughingthrush",
                   "AshambuLaughingthrush", "NilgiriSholakili",
                   "White-belliedSholakili"] if s in fa_mean},
}
print("openset repeated:", out["openset_repeated"]["fa_mean_sd"],
      "| congeners:", out["openset_repeated"]["congeners"])

# ---------- 6. BirdNET native Forest Owlet reference (M9) ----------
z2 = np.load(REPO / "data" / "embeddings.npz")
bn_sp = list(z2["birdnet_score_species"])
fo_col = bn_sp.index("ForestOwlet")
ids2 = list(z2["ids"])
sc = z2["birdnet_scores"][:, fo_col]
rec_max = defaultdict(float)
rec_true = {}
for i, wid in enumerate(ids2):
    rid = wid.split("|")[0]
    rec_max[rid] = max(rec_max[rid], float(sc[i]))
for r in windows.drop_duplicates("recording_id").itertuples():
    rec_true[r.recording_id] = r.species == "ForestOwlet"
fo = [rec_max[r] for r in rec_max if rec_true[r]]
nfo = [rec_max[r] for r in rec_max if not rec_true[r]]
out["birdnet_fo_reference"] = {
    "n_fo_recordings": len(fo),
    "recall_at_0.5": float(np.mean([v >= 0.5 for v in fo])),
    "recall_at_0.25": float(np.mean([v >= 0.25 for v in fo])),
    "recall_at_0.1": float(np.mean([v >= 0.1 for v in fo])),
    "nontarget_over_0.5": float(np.mean([v >= 0.5 for v in nfo])),
    "mean_window_conf_on_true": float(np.mean(sc[[windows.iloc[i].species == "ForestOwlet"
                                                  for i in range(len(windows))]])),
}
print("birdnet FO:", out["birdnet_fo_reference"])

# ---------- 7. attention vs activity correlation (fig 6 support) ----------
from scipy.stats import spearmanr
snr_of = {(r.recording_id, r.window_idx): r.snr_db for r in windows.itertuples()}
att_all, snr_all = [], []
for f in sorted((REPO / "results" / "preds" / "recording" / "mil").glob("*_seed0.json")):
    data = json.loads(f.read_text())
    for rid, attn in zip(data["rec_ids"], data["attn"]):
        g = windows[windows.recording_id == rid].sort_values("snr_db", ascending=False).head(len(attn))
        for (a, w) in zip(attn, g.itertuples()):
            att_all.append(a)
            snr_all.append(w.snr_db)
rho, pval = spearmanr(att_all, snr_all)
out["attention_snr_spearman"] = {"rho": float(rho), "p": float(pval), "n": len(att_all)}
print("attention-snr:", out["attention_snr_spearman"])

with open(REPO / "results" / "review_fixes.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved results/review_fixes.json")
