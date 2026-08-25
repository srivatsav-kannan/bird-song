# birdSong — Acoustic classification of India's endangered endemic birds

Rigorous, reproducible pipeline for classifying four endemic Indian bird species
listed as Endangered or Critically Endangered on the IUCN Red List, from field
recordings:

| Species | Scientific name | IUCN | Unique recordings |
|---|---|---|---|
| Forest Owlet | *Athene blewitti* | EN | 75 |
| Banasura Laughingthrush | *Montecincla jerdoni* | EN | 74 |
| Bugun Liocichla | *Liocichla bugunorum* | CR | 32 |
| Jerdon's Courser | *Rhinoptilus bitorquatus* | CR | 12 |

Audio: Macaulay Library + Xeno-canto (not redistributed here; the manifest lists
every catalog id so the dataset is exactly reconstructible).

## Pipeline

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. (optional) re-scrape Xeno-canto — needs a free API key
XC_API_KEY=... python scripts/scrape_xenocanto.py

# 2. provenance manifest (dedupe by catalog id + sha256)
python scripts/build_manifest.py

# 3. decode -> 3 s activity-scored windows
python scripts/build_windows.py

# 4. leakage-proof splits (recording-level and event-level 5-fold CV)
python scripts/make_splits.py

# 5. BirdNET V2.4 embeddings for all windows
python scripts/extract_embeddings.py

# 6. models
python scripts/precompute_mels.py
python scripts/train_cnn.py                     # baseline A: fine-tuned EfficientNet-B0
python scripts/train_probe.py --model probe     # baseline B: MLP on frozen embeddings
python scripts/train_probe.py --model mil       # attention-MIL over window bags
python scripts/build_aux_corpus.py              # 29-species auxiliary corpus
python scripts/train_adapter.py                 # supervised-contrastive domain adapter
python scripts/train_probe.py --model probe --adapter results/adapter.pt
python scripts/train_probe.py --model mil   --adapter results/adapter.pt

# 7. final metrics (bootstrap CIs, McNemar, per-class)
python scripts/evaluate.py
```

Design principles: no duplicated files anywhere, splits at recording level
(plus a stricter recording-*event* level scheme), early stopping and all model
selection on inner validation only, test folds used exactly once for reporting.

Manuscript: `paper/manuscript.md`. Legacy (pre-rewrite) code: `legacy/`.
