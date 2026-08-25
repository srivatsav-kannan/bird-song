# BirdSong Rebuild Plan — Rigorous 4-Species Endangered Endemic Classification

Goal: rebuild the entire pipeline so the reported performance is genuine, reproducible,
and defensible against every point in the BCI editor's letter — plus a real methodological
contribution beyond vanilla transfer learning.

Target species (endemic to India, IUCN EN/CR, with available audio):
1. Forest Owlet — *Athene blewitti* (EN)
2. Banasura Laughingthrush — *Montecincla jerdoni* (EN)
3. Bugun Liocichla — *Liocichla bugunorum* (CR)
4. Jerdon's Courser — *Rhinoptilus bitorquatus* (CR)

## Phase 0 — Repo restructure
- Move legacy code to `legacy/`. New layout:
  - `src/birdsong/` — package (data, features, models, eval)
  - `scripts/` — runnable entry points (scrape, build_manifest, windows, train, evaluate)
  - `data/` — manifest CSVs, splits (audio itself stays in `~/Datasets`)
  - `results/` — metrics, figures, model checkpoints (gitignored where large)
  - `paper/` — manuscript markdown + figures
- Python: venv, PyTorch (MPS) + torchaudio + timm; TensorFlow only if needed for
  BirdNET/Perch embedding extraction. `requirements.txt` pinned.

## Phase 1 — Data acquisition & provenance
1. **Re-scrape Xeno-canto** via its public API for the 4 species (query by scientific
   name incl. synonyms, e.g. *Heteroglaux blewitti*). Protocol:
   - Respect rate limits (1 req/s), record full metadata (XC id, quality grade,
     recordist, date, location, license, length, sampling rate).
   - Exclude restricted/unavailable recordings; log everything skipped.
   - Dedupe against the on-disk collection by catalog id.
2. **Macaulay Library**: no public bulk API — keep the recordings already on disk
   (catalog ids preserved in filenames); no new scraping.
3. **Canonical manifest** `data/manifest.csv`: one row per unique recording —
   species, source (XC/ML), catalog id, quality grade, duration, sample rate,
   license, file path, sha256 (dedupe safeguard). This is the single source of truth;
   every downstream artifact derives from it. Audio dedup via content hash +
   catalog id (the old folders contain physical duplicates by construction).

## Phase 2 — Signal processing & windowing
- Decode everything to mono WAV @ 32 kHz.
- **Fixed 3 s analysis windows** (matches BirdNET's native window) with 50% overlap
  at extraction time; per-window **energy/SNR-based activity detection** to keep only
  windows likely to contain vocalizations (band-limited spectral energy vs. recording
  noise floor). Whole recordings are never squashed into one image again.
- Features: 128-bin log-mel spectrograms computed as arrays (no matplotlib PNG
  rendering, no colormaps).

## Phase 3 — Splits: leakage-proof by construction
- **Unit of splitting = unique recording** (all windows of a recording share its split).
- **Outer 5-fold grouped, stratified cross-validation** at recording level: each fold
  serves once as a **held-out test fold used for nothing but final reporting**.
- Within each outer fold, an inner train/val split (recording-level) drives early
  stopping and all model selection. Test folds are untouched by any selection —
  directly answering editor point 1. With only 12 Jerdon's Courser recordings, a
  single small test set would be statistical noise; cross-validated held-out folds +
  bootstrap CIs are the defensible design, and we say so explicitly in the paper.
- **No duplication anywhere.** Class imbalance handled by weighted sampling/loss.
- Splits serialized to `data/splits.json` with seeds; all experiments read from it.

## Phase 4 — Models
1. **Baseline A (fixed version of the old approach):** fine-tuned compact CNN
   (EfficientNet-B0 via timm) on log-mel windows, proper head, partial unfreezing,
   SpecAugment + noise/time-shift augmentation.
2. **Baseline B: frozen bioacoustic foundation embeddings** (BirdNET v2.4, 1024-d
   per 3 s window) + lightweight classifier (linear probe / small MLP).
3. **Contribution 1 — attention-based multiple-instance learning (MIL):** a recording
   is a bag of window embeddings; a gated-attention pooling head learns which windows
   carry species-diagnostic sound and yields recording-level predictions + per-window
   attention (interpretability: what the model listened to). Well-matched to weakly
   labeled field audio.
4. **Contribution 2 — domain-adaptive pretraining on India's endemic avifauna:**
   we already hold 2,117 recordings of 33 endemic species. Use the 29 non-target
   species as an auxiliary corpus to adapt the embedding space (supervised-contrastive
   / classifier pretraining), then transfer to the 4 data-poor targets. Novel, honest
   use of "more data" without touching evaluation classes.
5. Ensembling only if it wins on val; weights chosen on val, reported on test folds.
6. **External reference baseline:** BirdNET's own classifier where the species are in
   its label list (coverage check + comparison).

## Phase 5 — Evaluation that survives review
- Report per outer test fold: window-level and **recording-level** (aggregated)
  accuracy, macro-F1, per-class precision/recall with **unique-recording counts**
  stated everywhere.
- Mean ± 95% CI across folds (and bootstrap over recordings within folds).
- McNemar's test for pairwise model comparisons; confusion matrices; calibration.
- Robustness: noise-injection stress test (performance vs. SNR).
- Every number in the paper produced by `scripts/evaluate.py` → `results/` (JSON+figs).

## Phase 6 — Manuscript
- `paper/manuscript.md`: full rewrite — honest data statement, three-way separation
  of concerns (fit/select/report), reconciled ensemble story, limitations section,
  reproducibility statement (code + manifest + splits public).
- Then adapt to BCI author guidelines (separate step, after we talk).

## Order of execution
0. Restructure + env ✅ → 1. Manifest ✅ (XC re-scrape script ready; blocked on user's
free XC API key) → 2. Windowing/VAD ✅ (3,195 windows) → 3. Splits ✅ (recording +
event schemes) → 4. Models ✅ (CNN, probe, MIL, adapter=negative result, open-set
rejection added) → 5. Eval suite ✅ (bootstrap CIs, McNemar, robustness, open-set)
→ 6. Manuscript ✅ (paper/manuscript.md) → 7. BCI journal formatting (next session).
