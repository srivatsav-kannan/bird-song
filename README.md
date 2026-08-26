# bird-song

Acoustic classification of India's four most threatened endemic birds from their songs and calls. This repository contains the complete pipeline, data manifest, evaluation splits, and manuscript for a study showing that the public sound archives, small as they are for these species, can support reliable automated identification.

## The species

| Species | IUCN status | Unique recordings |
|---|---|---|
| Forest Owlet (*Athene blewitti*) | Endangered | 75 |
| Banasura Laughingthrush (*Montecincla jerdoni*) | Endangered | 74 |
| Bugun Liocichla (*Liocichla bugunorum*) | Critically Endangered | 32 |
| Jerdon's Courser (*Rhinoptilus bitorquatus*) | Critically Endangered | 12 |

These are the only Indian endemics in the top IUCN threat categories with any recordings in the Macaulay Library or Xeno-canto. Only the Forest Owlet appears in BirdNET's 6,000-species label set, so the standard global classifier cannot detect the other three at all.

## What the study shows

- A small classifier on frozen BirdNET embeddings identifies the correct species for **97.4%** of recordings (95% CI 94.8 to 99.5), against 83.4% for a fine-tuned EfficientNet-B0. Every reported number comes from cross-validation test folds that played no part in training or model selection.
- Accuracy holds at **98.9%** under a stricter scheme in which recordings from the same recording session can never appear on both sides of a split, and at **92.2%** with added noise as loud as the signal.
- A four-class classifier alone wrongly accepts 93.7% of recordings of other Indian endemic species. Adding a background class trained on species held disjoint from the evaluation cuts that false positive rate to **13.4%** while keeping 89.1% of target recordings, which is the difference between a benchmark model and a usable field detector.
- An attention-based multiple instance learning model matches the probe and shows which seconds of audio drove each decision.

The full write-up is in [paper/manuscript.md](paper/manuscript.md), with figures in [paper/figures/](paper/figures/).

## Repository layout

```
paper/                  manuscript (markdown and docx) and figures
src/birdsong/           audio decoding, windowing, features, models, evaluation
scripts/                one script per pipeline step, in run order below
data/                   recording manifest, window index, CV splits
results/                metrics, open-set and robustness results
legacy/                 the pre-rewrite pipeline, kept for reference
```

## Reproducing the results

The audio itself is not redistributed, in line with archive terms. `data/manifest.csv` lists the catalogue identifier of every recording, so the exact dataset can be reassembled from the Macaulay Library and Xeno-canto.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_manifest.py        # audit and dedupe the raw collection
python scripts/build_windows.py         # 3 s activity-scored windows
python scripts/make_splits.py           # recording-level and session-level CV splits
python scripts/extract_embeddings.py    # BirdNET V2.4 embeddings
python scripts/precompute_mels.py       # log-mel arrays for the CNN baseline
python scripts/train_cnn.py             # baseline A: fine-tuned EfficientNet-B0
python scripts/train_probe.py --model probe   # baseline B: MLP on frozen embeddings
python scripts/train_probe.py --model mil     # attention-based MIL
python scripts/build_aux_corpus.py      # 28-species auxiliary corpus
python scripts/openset_reject.py        # background-class rejection experiment
python scripts/robustness.py            # noise stress test
python scripts/evaluate.py              # final metrics with bootstrap CIs
python scripts/figures.py               # regenerate all manuscript figures
```

`scripts/scrape_xenocanto.py` can refresh the Xeno-canto material and needs a free API key from your xeno-canto.org account (set `XC_API_KEY`).

Everything runs on a single laptop. The splits are serialized with their generating seeds, so reruns reproduce the published numbers.

## Data ethics

Recordings of Critically Endangered species can carry sensitive location information. This repository stores no audio and no coordinates, only catalogue identifiers that resolve through the archives' own access policies.
