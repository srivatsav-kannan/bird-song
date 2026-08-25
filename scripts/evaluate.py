"""Aggregate all fold/seed predictions into final metrics.

For each scheme (recording/event) and model tag:
  - concatenate test-fold predictions (each recording is tested exactly once
    per scheme; for the event scheme Jerdon's Courser appears only via the
    cross-source XC probe)
  - average probabilities over seeds -> final prediction per recording
  - recording-level accuracy / macro-F1 with bootstrap 95% CIs over recordings
  - per-class metrics with unique-recording support
  - per-seed spread (SD) as a stability measure
  - pairwise McNemar tests between models

Output: results/summary.json + printed report.
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.evaluation import (SPECIES, aggregate_recordings, bootstrap_ci,  # noqa: E402
                                 mcnemar_exact, metrics)


def collect(scheme: str, tag: str):
    """Return per-seed {rid: prob}, plus y_by_rid."""
    d = REPO / "results" / "preds" / scheme / tag
    if not d.exists():
        return None, None
    per_seed = defaultdict(dict)
    y_by_rid = {}
    for f in sorted(d.glob("*.json")):
        seed = f.stem.split("seed")[1]
        data = json.loads(f.read_text())
        if "rec_probs" in data:
            for rid, y, p in zip(data["rec_ids"], data["rec_y"], data["rec_probs"]):
                per_seed[seed][rid] = np.array(p)
                y_by_rid[rid] = y
        else:
            probs = np.array(data["window_probs"])
            rids = data["window_rids"]
            ids, agg = aggregate_recordings(probs, rids)
            for rid, p in zip(ids, agg):
                per_seed[seed][rid] = p
            for rid, y in zip(rids, data["window_y"]):
                y_by_rid[rid] = y
    return per_seed, y_by_rid


def evaluate(scheme: str, tag: str):
    per_seed, y_by_rid = collect(scheme, tag)
    if per_seed is None:
        return None
    rids = sorted(y_by_rid)
    y = np.array([y_by_rid[r] for r in rids])
    mean_probs = np.stack([np.mean([per_seed[s][r] for s in per_seed], axis=0) for r in rids])
    y_pred = mean_probs.argmax(1)

    acc, acc_lo, acc_hi = bootstrap_ci(y, y_pred, lambda t, p: (t == p).mean())
    from sklearn.metrics import f1_score
    f1_fn = lambda t, p: f1_score(t, p, labels=range(4), average="macro", zero_division=0)
    f1, f1_lo, f1_hi = bootstrap_ci(y, y_pred, f1_fn)

    seed_accs = []
    for s in per_seed:
        sp = np.stack([per_seed[s][r] for r in rids]).argmax(1)
        seed_accs.append((sp == y).mean())

    m = metrics(y, y_pred)
    m.update({
        "n_recordings": len(rids),
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1_ci95": [f1_lo, f1_hi],
        "per_seed_accuracy": [float(a) for a in seed_accs],
        "seed_accuracy_sd": float(np.std(seed_accs)),
    })
    return m, {r: int(p) for r, p in zip(rids, y_pred)}, y_by_rid


def evaluate_ensemble(scheme: str, members: list[str]):
    """Equal-weight soft vote over member models' seed-averaged recording probs.

    Deliberately untuned: no ensemble weight is selected on any data, so the
    ensemble involves zero additional model-selection decisions.
    """
    prob_sets, y_by_rid = [], {}
    for tag in members:
        per_seed, yb = collect(scheme, tag)
        if per_seed is None:
            return None
        rids = sorted(yb)
        probs = {r: np.mean([per_seed[s][r] for s in per_seed], axis=0) for r in rids}
        prob_sets.append(probs)
        y_by_rid.update(yb)
    rids = sorted(set.intersection(*(set(p) for p in prob_sets)))
    y = np.array([y_by_rid[r] for r in rids])
    mean_probs = np.stack([np.mean([p[r] for p in prob_sets], axis=0) for r in rids])
    y_pred = mean_probs.argmax(1)

    acc, acc_lo, acc_hi = bootstrap_ci(y, y_pred, lambda t, p: (t == p).mean())
    from sklearn.metrics import f1_score
    f1_fn = lambda t, p: f1_score(t, p, labels=range(4), average="macro", zero_division=0)
    f1, f1_lo, f1_hi = bootstrap_ci(y, y_pred, f1_fn)
    m = metrics(y, y_pred)
    m.update({"n_recordings": len(rids), "accuracy_ci95": [acc_lo, acc_hi],
              "macro_f1_ci95": [f1_lo, f1_hi], "members": members})
    return m, {r: int(p) for r, p in zip(rids, y_pred)}, y_by_rid


def main():
    tags = [d.name for d in (REPO / "results" / "preds" / "recording").glob("*") if d.is_dir()]
    summary = {}
    preds_by_tag = {}
    for scheme in ["recording", "event"]:
        summary[scheme] = {}
        for tag in sorted(tags):
            r = evaluate(scheme, tag)
            if r is None:
                continue
            m, preds, y_by_rid = r
            summary[scheme][tag] = m
            preds_by_tag[(scheme, tag)] = (preds, y_by_rid)
            print(f"[{scheme}] {tag:16s} n={m['n_recordings']:3d} "
                  f"acc={m['accuracy']:.3f} [{m['accuracy_ci95'][0]:.3f},{m['accuracy_ci95'][1]:.3f}] "
                  f"macroF1={m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.3f},{m['macro_f1_ci95'][1]:.3f}] "
                  f"seedSD={m['seed_accuracy_sd']:.3f}")
            for sp, pm in m["per_class"].items():
                print(f"    {sp:24s} P={pm['precision']:.2f} R={pm['recall']:.2f} "
                      f"F1={pm['f1']:.2f} n={pm['support']}")

    # untuned soft-vote ensembles
    for scheme in ["recording", "event"]:
        for members in [["cnn", "probe"], ["probe", "mil"], ["cnn", "probe", "mil"]]:
            r = evaluate_ensemble(scheme, members)
            if r is None:
                continue
            m, preds, y_by_rid = r
            tag = "ens_" + "+".join(members)
            summary[scheme][tag] = m
            preds_by_tag[(scheme, tag)] = (preds, y_by_rid)
            print(f"[{scheme}] {tag:16s} n={m['n_recordings']:3d} "
                  f"acc={m['accuracy']:.3f} [{m['accuracy_ci95'][0]:.3f},{m['accuracy_ci95'][1]:.3f}] "
                  f"macroF1={m['macro_f1']:.3f}")

    # McNemar between models within each scheme
    summary["mcnemar"] = {}
    for scheme in ["recording", "event"]:
        pairs = [k for k in preds_by_tag if k[0] == scheme]
        for (s1, t1), (s2, t2) in combinations(pairs, 2):
            p1, yb = preds_by_tag[(s1, t1)]
            p2, _ = preds_by_tag[(s2, t2)]
            rids = sorted(set(p1) & set(p2))
            y = np.array([yb[r] for r in rids])
            a = np.array([p1[r] for r in rids])
            b = np.array([p2[r] for r in rids])
            res = mcnemar_exact(y, a, b)
            summary["mcnemar"][f"{scheme}:{t1}_vs_{t2}"] = res
            print(f"McNemar [{scheme}] {t1} vs {t2}: {res}")

    with open(REPO / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
