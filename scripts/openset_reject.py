"""Open-set rejection via an explicit background class, evaluated species-disjoint.

The 28 auxiliary species are split in half: 14 "background-train" species whose
windows form a 5th (reject) class during training, and 14 completely unseen
species used only to measure false-accept rates. Target-species evaluation uses
the same leakage-proof recording-level folds as everywhere else.

Reported per fold, then aggregated:
  - target recording-level accuracy (correct species AND not rejected)
  - target rejection rate (recordings wrongly sent to the reject class)
  - false-accept rate on unseen-species recordings (any target class won)

Output: results/openset_reject.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from birdsong.evaluation import SPECIES, aggregate_recordings  # noqa: E402
from birdsong.expdata import load_all, window_arrays  # noqa: E402
from train_probe import train_probe, predict_probe  # noqa: E402

SEED = 11


def main():
    windows, emb_of, _, splits = load_all()
    z = np.load(REPO / "data" / "aux_embeddings.npz")
    aux_X, aux_sp, aux_rid = z["embeddings"], z["species"], z["ids"]

    rng = np.random.default_rng(SEED)
    aux_species = sorted(set(aux_sp))
    rng.shuffle(aux_species)
    bg_train_species = set(aux_species[:len(aux_species) // 2])
    bg_eval_species = set(aux_species[len(aux_species) // 2:])
    tr_mask = np.array([s in bg_train_species for s in aux_sp])
    ev_mask = ~tr_mask
    print(f"background-train species: {len(bg_train_species)}, "
          f"unseen-eval species: {len(bg_eval_species)}")

    y_by_rid = {r.recording_id: SPECIES.index(r.species)
                for r in windows.drop_duplicates("recording_id").itertuples()}
    OTHER = len(SPECIES)

    acc_list, rej_list, fa_list = [], [], []
    for fold, parts in splits["recording"].items():
        tr, va, te = (set(parts[p]) for p in ["train", "val", "test"])
        Xtr, ytr, _ = window_arrays(windows, emb_of, tr)
        Xva, yva, rva = window_arrays(windows, emb_of, va)
        Xte, yte, rte = window_arrays(windows, emb_of, te)

        # add background class to train (subsampled to ~mean target-class size)
        n_bg = int(np.bincount(ytr).mean() * 2)
        bg_idx = rng.choice(np.flatnonzero(tr_mask), size=min(n_bg, tr_mask.sum()),
                            replace=False)
        Xtr5 = np.concatenate([Xtr, aux_X[bg_idx]])
        ytr5 = np.concatenate([ytr, np.full(len(bg_idx), OTHER)])

        # val gets background too (for early stopping realism): reuse a slice
        bg_val = rng.choice(np.flatnonzero(tr_mask), size=max(20, len(bg_idx)//5),
                            replace=False)
        Xva5 = np.concatenate([Xva, aux_X[bg_val]])
        rva5 = np.concatenate([rva, aux_rid[bg_val]])
        y_by_rid5 = dict(y_by_rid)
        for r in aux_rid[bg_val]:
            y_by_rid5[r] = OTHER

        global SPECIES5
        model = train_probe(Xtr5, ytr5, Xva5, None, rva5, y_by_rid5, seed=0,
                            in_dim=1024)

        # target test: recording-level argmax over 5 classes
        pte = predict_probe(model, Xte)
        ids, agg = aggregate_recordings(pte, list(rte))
        y_true = np.array([y_by_rid[r] for r in ids])
        y_pred = agg.argmax(1)
        acc_list.append(float((y_pred == y_true).mean()))
        rej_list.append(float((y_pred == OTHER).mean()))

        # unseen species: false accept if any target class wins at recording level
        pood = predict_probe(model, aux_X[ev_mask])
        ids_o, agg_o = aggregate_recordings(pood, list(aux_rid[ev_mask]))
        fa_list.append(float((agg_o.argmax(1) != OTHER).mean()))
        print(f"{fold}: target-acc={acc_list[-1]:.3f} target-rejected={rej_list[-1]:.3f} "
              f"unseen-species FA={fa_list[-1]:.3f}")

    result = {
        "n_bg_train_species": len(bg_train_species),
        "n_bg_eval_species": len(bg_eval_species),
        "target_accuracy_mean_sd": [float(np.mean(acc_list)), float(np.std(acc_list))],
        "target_reject_rate_mean_sd": [float(np.mean(rej_list)), float(np.std(rej_list))],
        "unseen_species_false_accept_mean_sd": [float(np.mean(fa_list)), float(np.std(fa_list))],
        "per_fold": {"acc": acc_list, "rej": rej_list, "fa": fa_list},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_fold"}, indent=1))
    with open(REPO / "results" / "openset_reject.json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
