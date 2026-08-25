"""Open-set analysis: can the 4-class detector reject other Indian endemics?

For each recording-scheme fold: train the MLP probe on clean train embeddings
(seed 0), then measure separability between in-distribution test windows (the
4 target species) and out-of-distribution windows (the 28-species auxiliary
corpus — the most realistic hard distractors a deployed detector would hear).

Scores: maximum softmax probability (MSP) and negative energy. Reports AUROC,
plus recording-level operating points (target recall vs distractor false-accept
rate) at MSP thresholds calibrated on val data only.

Output: results/openset.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from birdsong.evaluation import SPECIES, aggregate_recordings  # noqa: E402
from birdsong.expdata import load_all, window_arrays  # noqa: E402
from train_probe import train_probe  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def scores(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(DEVICE))
        msp = torch.softmax(logits, 1).max(1).values.cpu().numpy()
        energy = torch.logsumexp(logits, 1).cpu().numpy()
    return msp, energy


def rec_scores(msp, rids):
    ids, agg = aggregate_recordings(msp[:, None], list(rids))
    return {r: float(a[0]) for r, a in zip(ids, agg)}


def main():
    windows, emb_of, _, splits = load_all()
    z = np.load(REPO / "data" / "aux_embeddings.npz")
    Xood = z["embeddings"]
    ood_rids = z["ids"]
    y_by_rid = {r.recording_id: SPECIES.index(r.species)
                for r in windows.drop_duplicates("recording_id").itertuples()}

    win_auroc_msp, win_auroc_en = [], []
    id_rec_scores, ood_rec_scores, val_rec_scores = {}, {}, []
    for fold, parts in splits["recording"].items():
        tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
        Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
        Xva, yva, rva = window_arrays(windows, emb_of, va)
        Xte, yte, rte = window_arrays(windows, emb_of, te)
        model = train_probe(Xtr, ytr, Xva, yva, rva, y_by_rid, seed=0, in_dim=1024)

        msp_id, en_id = scores(model, Xte)
        msp_ood, en_ood = scores(model, Xood)
        y_bin = np.r_[np.ones(len(msp_id)), np.zeros(len(msp_ood))]
        win_auroc_msp.append(roc_auc_score(y_bin, np.r_[msp_id, msp_ood]))
        win_auroc_en.append(roc_auc_score(y_bin, np.r_[en_id, en_ood]))

        id_rec_scores.update(rec_scores(msp_id, rte))
        # distractor recordings get a score per fold; keep the max over folds
        # (conservative: worst case for rejection)
        for r, s in rec_scores(msp_ood, ood_rids).items():
            ood_rec_scores[r] = max(s, ood_rec_scores.get(r, 0.0))
        msp_va, _ = scores(model, Xva)
        val_rec_scores.append(rec_scores(msp_va, rva))

    # threshold calibrated on val only: highest t keeping >=95% of val recordings
    all_val = np.array([s for d in val_rec_scores for s in d.values()])
    thresh = float(np.quantile(all_val, 0.05))

    id_arr = np.array(list(id_rec_scores.values()))
    ood_arr = np.array(list(ood_rec_scores.values()))
    result = {
        "window_auroc_msp": [float(np.mean(win_auroc_msp)), float(np.std(win_auroc_msp))],
        "window_auroc_energy": [float(np.mean(win_auroc_en)), float(np.std(win_auroc_en))],
        "n_id_recordings": int(len(id_arr)),
        "n_ood_recordings": int(len(ood_arr)),
        "val_calibrated_threshold": thresh,
        "id_recording_accept_rate": float((id_arr >= thresh).mean()),
        "ood_recording_false_accept_rate": float((ood_arr >= thresh).mean()),
        "rec_auroc_msp": float(roc_auc_score(
            np.r_[np.ones(len(id_arr)), np.zeros(len(ood_arr))],
            np.r_[id_arr, ood_arr])),
    }
    print(json.dumps(result, indent=1))
    with open(REPO / "results" / "openset.json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
