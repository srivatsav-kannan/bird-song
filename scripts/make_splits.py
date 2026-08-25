"""Create leakage-proof cross-validation splits.

Two schemes, both serialized to data/splits.json:

1. "recording": stratified 5-fold CV where the unit is the unique recording.
   Every fold contains every species in train/val/test. This is the headline
   evaluation scheme (recording-level independence, the standard in bioacoustics).

2. "event": same construction but the unit is the *event group* — recordings
   whose catalog numbers on the same platform lie within GAP of each other,
   i.e. almost certainly the same upload batch / recording session. This is the
   stricter rigor check: no two recordings from one session may straddle a split.
   Jerdon's Courser has only two event groups (one 11-recording ML block + one
   XC recording), so under this scheme the ML block is pinned to train and the
   XC recording serves as a single cross-source test probe in one fold.

Within each outer fold, the non-test recordings are further split into train and
val (val used for early stopping / model selection ONLY). Test folds are used for
nothing but final reporting.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED = 20260825
N_FOLDS = 5
GAP = 1000          # max catalog-number gap within one event group
VAL_FRAC = 0.2      # of non-test units, per species


def numeric_id(rid):
    return int(re.sub(r"\D", "", rid))


def event_groups(df):
    """Cluster recordings into same-session groups by catalog-number adjacency."""
    groups = {}
    for (sp, src), g in df.groupby(["species", "source"]):
        recs = sorted(g.recording_id, key=numeric_id)
        cluster = 0
        prev = None
        for rid in recs:
            n = numeric_id(rid)
            if prev is not None and n - prev >= GAP:
                cluster += 1
            groups[rid] = f"{sp}|{src}|{cluster}"
            prev = n
    return groups


def assign_folds(units, rng):
    """Greedy balanced assignment of units (unit -> [recording_ids]) to folds."""
    folds = [[] for _ in range(N_FOLDS)]
    sizes = [0] * N_FOLDS
    ordered = sorted(units.items(), key=lambda kv: -len(kv[1]))
    # shuffle ties deterministically
    rng.shuffle(ordered)
    ordered.sort(key=lambda kv: -len(kv[1]))
    for _, recs in ordered:
        i = min(range(N_FOLDS), key=lambda k: sizes[k])
        folds[i].extend(recs)
        sizes[i] += len(recs)
    return folds


def build_scheme(df, unit_of, rng, pinned_train_units=()):
    """unit_of: recording_id -> unit key. Returns folds dict."""
    per_species_folds = defaultdict(lambda: [[] for _ in range(N_FOLDS)])
    for sp, g in df.groupby("species"):
        units = defaultdict(list)
        for rid in g.recording_id:
            units[unit_of[rid]].append(rid)
        pinned = {u: r for u, r in units.items() if u in pinned_train_units}
        for u in pinned:
            del units[u]
        folds = assign_folds(units, rng)
        for k in range(N_FOLDS):
            per_species_folds[sp][k] = folds[k]
        per_species_folds[sp + "|PINNED"] = list(pinned.values())

    scheme = {}
    for k in range(N_FOLDS):
        test, rest_units = [], defaultdict(list)
        for sp, g in df.groupby("species"):
            test.extend(per_species_folds[sp][k])
            for j in range(N_FOLDS):
                if j != k:
                    for rid in per_species_folds[sp][j]:
                        rest_units[unit_of[rid]].append(rid)
            for recs in per_species_folds.get(sp + "|PINNED", []):
                for rid in recs:
                    rest_units[unit_of[rid]].append(rid)
        # split rest into train/val at unit level, stratified per species.
        # Pinned units always go to train (that is their contract). Val fills
        # smallest-units-first and never exceeds its target, so a single large
        # unit can never swallow a species' entire training data.
        train, val = [], []
        sp_units = defaultdict(list)
        for u, recs in rest_units.items():
            if u in pinned_train_units:
                train.extend(recs)
                continue
            sp = df[df.recording_id == recs[0]].species.iloc[0]
            sp_units[sp].append((u, recs))
        for sp, items in sp_units.items():
            items = sorted(items)
            rng.shuffle(items)
            items.sort(key=lambda kv: len(kv[1]))  # stable: smallest units first
            n_val_recs = max(1, round(VAL_FRAC * sum(len(r) for _, r in items)))
            sp_val, sp_train = [], []
            for u, recs in items:
                fits = len(sp_val) + len(recs) <= n_val_recs
                if len(sp_val) < n_val_recs and (fits or not sp_val):
                    sp_val.extend(recs)
                else:
                    sp_train.extend(recs)
            if not sp_train and len(items) > 1:
                raise AssertionError(f"train starved for {sp}")
            if not sp_train:      # species has a single unit: training keeps it
                sp_train, sp_val = sp_val, []
            train.extend(sp_train)
            val.extend(sp_val)
        scheme[f"fold_{k}"] = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    return scheme


def main():
    df = pd.read_csv(REPO / "data" / "manifest.csv")
    rng = random.Random(SEED)

    rec_unit = {rid: rid for rid in df.recording_id}
    scheme_rec = build_scheme(df, rec_unit, random.Random(SEED))

    ev_unit = event_groups(df)
    # Jerdon's Courser ML block: only usable as training data under event scheme
    jc_ml_units = {u for rid, u in ev_unit.items()
                   if u.startswith("JerdonsCourser|ML")}
    scheme_ev = build_scheme(df, ev_unit, random.Random(SEED + 1),
                             pinned_train_units=jc_ml_units)

    # sanity: no recording in two partitions of one fold; all recordings covered
    for name, scheme in [("recording", scheme_rec), ("event", scheme_ev)]:
        for k, parts in scheme.items():
            tr, va, te = map(set, (parts["train"], parts["val"], parts["test"]))
            assert not (tr & va or tr & te or va & te), f"overlap in {name}/{k}"
            assert tr | va | te == set(df.recording_id), f"coverage gap in {name}/{k}"
            if name == "event":
                units_of = lambda s: {ev_unit[r] for r in s}
                assert not (units_of(tr) & units_of(te)), f"event leak {name}/{k}"
                assert not (units_of(va) & units_of(te)), f"event leak {name}/{k}"

    out = {"seed": SEED, "n_folds": N_FOLDS, "event_gap": GAP,
           "event_group_of": ev_unit,
           "recording": scheme_rec, "event": scheme_ev}
    with open(REPO / "data" / "splits.json", "w") as f:
        json.dump(out, f, indent=1)

    for name, scheme in [("recording", scheme_rec), ("event", scheme_ev)]:
        print(f"== scheme: {name}")
        for k, parts in scheme.items():
            counts = {p: len(v) for p, v in parts.items()}
            te = df[df.recording_id.isin(parts["test"])].species.value_counts().to_dict()
            print(f"  {k}: {counts}  test-by-species={te}")


if __name__ == "__main__":
    main()
