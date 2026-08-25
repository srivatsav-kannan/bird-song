# Acoustic identification of India's most threatened endemic birds using bioacoustic foundation-model embeddings and attention-based multiple-instance learning

**Srivatsav Kannan¹, Valliappan Raman¹, Shanmugapriya K.R.¹**

¹ Department of Artificial Intelligence and Data Science, Coimbatore Institute of Technology, Coimbatore, India

---

## Summary

Passive acoustic monitoring is one of the few scalable survey tools for rare, cryptic birds — it was acoustic evidence that reconfirmed Jerdon's Courser *Rhinoptilus bitorquatus* in 2025 after 17 years without a verified record. Yet the general-purpose recognizers that power modern bioacoustics do not cover most of India's most threatened endemic species: of the four Endangered (EN) or Critically Endangered (CR) Indian endemics with any archived sound recordings — Forest Owlet *Athene blewitti* (EN), Banasura Laughingthrush *Montecincla jerdoni* (EN), Bugun Liocichla *Liocichla bugunorum* (CR), and Jerdon's Courser (CR) — only the Forest Owlet appears in BirdNET's 6,000-class label set. We assembled and audited the complete public acoustic record of these four species (193 unique recordings, 125 minutes of audio, from the Macaulay Library and Xeno-canto; six duplicated files in the archives were detected and removed by content hashing) and built classifiers that operate on embeddings from the BirdNET acoustic foundation model, aggregated over recordings with a gated-attention multiple-instance learning (MIL) head. Evaluation used grouped, stratified five-fold cross-validation with the unique recording as the unit of splitting, all model selection confined to inner validation sets, and each test fold used exactly once for final reporting; a stricter second scheme additionally forbade recordings from the same recording session from straddling partitions. Recording-level identification accuracy reached 97.4% (95% CI 94.8–99.5%; macro-F1 0.973), was stable across random seeds, and substantially exceeded a carefully trained fine-tuned CNN baseline representative of prior practice. A stricter session-level evaluation confirmed these results are not an artifact of within-session leakage (98.9% accuracy), while making explicit an irreducible limit for Jerdon's Courser: its archival record derives from essentially one recording session, so cross-session generalization rests on a single (correctly classified) independent recording. In an open-set stress test against 1,891 recordings of 28 non-target endemic species, a confidence threshold alone failed to reject distractors (94% false acceptance), whereas adding an explicit background class trained on species-disjoint auxiliary data cut false acceptance to 13% while retaining 89% of target recordings. We release the full pipeline, data manifest, and split definitions for exact reproduction, and discuss what can — and cannot — yet be claimed about detector readiness for species whose acoustic record is this thin.

**Keywords:** bioacoustics; passive acoustic monitoring; endemic species; transfer learning; multiple-instance learning; open-set recognition; India

---

## Introduction

India supports over 1,300 bird species, of which more than 75 occur nowhere else (SoIB 2023). Endemic species with narrow ranges are disproportionately exposed to habitat loss, and several of India's endemics now sit in the highest IUCN threat categories. The four EN/CR endemics with any publicly archived sound recordings illustrate both the urgency and the difficulty of monitoring them. The Forest Owlet was known from seven nineteenth-century specimens until its rediscovery in 1997 (Rasmussen and Collar 1998) and, although since found at additional sites, retains a small, fragmented, declining population. The Banasura Laughingthrush is confined to high-elevation shola patches totalling under 57 km² in the Wayanad region of the Western Ghats (Robin et al. 2017). The Bugun Liocichla, described only in 2006 from Eaglenest Wildlife Sanctuary, Arunachal Pradesh, is known from a handful of territories and at most a few tens of individuals (Athreya 2006). Jerdon's Courser, rediscovered in 1986 after 86 years (Bhushan 1986), was last verifiably recorded in 2008 — until August 2025, when its call was captured on an automated recorder at a new site, a rediscovery made possible precisely because a sound-based search image existed (Jeganathan et al. 2002; Search for Lost Birds 2025).

These species are archetypal candidates for passive acoustic monitoring (PAM): nocturnal or skulking, in terrain where visual survey is hard, with distinctive vocalizations. PAM at scale, however, depends on automated identification, and the general-purpose recognizers that dominate applied bioacoustics do not serve these species. We verified that of the four, only the Forest Owlet is present in the label set of BirdNET V2.4 (Kahl et al. 2021), the most widely deployed bird sound classifier; the other three cannot be detected by it at any confidence threshold. Species-specific classifiers are therefore a genuine need, not a methodological exercise.

Building such classifiers for data-poor species raises a methodological trap that this paper confronts directly. When the entire acoustic record of a species is a few dozen recordings — sometimes from a single recordist, site, and season — it is easy to obtain high benchmark scores that reflect memorization of recording conditions rather than generalizable species discrimination. An earlier version of this work suffered from exactly this problem: performance was reported on the same small validation set used for model selection, and class balance was achieved by duplicating audio files. Here we rebuild the entire pipeline around three principles: (i) every reported number comes from data that played no role in training or model selection; (ii) the unit of independence is the recording — and, in a stricter analysis, the recording *session*; and (iii) the limits of the data are measured and reported rather than averaged away.

Methodologically, we follow the emerging consensus that embeddings from large pretrained bioacoustic models transfer better to data-poor problems than fine-tuning vision architectures from scratch or from ImageNet (Ghani et al. 2023). We add two elements matched to the conservation setting. First, a gated-attention multiple-instance learning head (Ilse et al. 2018) makes decisions at the recording level — the level at which a monitoring program acts — while learning which 3-s windows carry diagnostic sound, providing interpretability without extra annotation. Second, because a deployed detector must reject the many non-target species it will hear, we quantify open-set behaviour against 1,891 recordings of 28 other Indian endemic species and show that an explicit background class, trained on auxiliary species disjoint from those used for testing, converts an unusable open-set system into a plausible screening tool. We also report an informative negative result — supervised-contrastive adaptation of the embedding space on auxiliary endemics *reduced* target performance — and a structural data limit that no evaluation design can remove: the archival record of Jerdon's Courser contains essentially one recording session, so session-level independence can be probed with exactly one recording.

## Methods

### Data assembly and audit

We searched the IUCN Red List for bird species endemic to India and listed as EN or CR. Seven species qualified; three (Great Nicobar Serpent Eagle *Spilornis klossi*, Himalayan Quail *Ophrysia superciliosa*, Manipur Bush Quail *Perdicula manipurensis*) have no usable recordings in either major public archive — the Himalayan Quail has not been reliably recorded since the nineteenth century — leaving the four study species.

All available recordings were obtained from the Macaulay Library (ML) and Xeno-canto (XC). Every file was assigned its archive catalogue identifier, and provenance (species, source, XC quality grade, duration, sample rate) was recorded in a manifest that constitutes the single source of truth for all downstream processing. Deduplication used both catalogue identifiers and SHA-256 content hashes; six byte-identical duplicates were found and removed, including recordings present under both an ML and an XC catalogue number. This audit step matters: in a duplicated dataset, copies of the same audio can silently occupy both training and evaluation partitions.

The final dataset comprises 193 unique recordings (~125 min): Forest Owlet 75 (46 ML, 29 XC), Banasura Laughingthrush 74 (71 ML, 3 XC), Bugun Liocichla 32 (all ML), Jerdon's Courser 12 (11 ML, 1 XC). No audio was duplicated, resampled into balance, or otherwise multiplied at the file level.

Catalogue numbers also encode provenance structure that a rigorous evaluation must respect. All 11 ML recordings of Jerdon's Courser occupy one consecutive catalogue block (ML274533–274543): they are the known 2001–2002 recordings from Sri Lankamalleswara Wildlife Sanctuary (Jeganathan et al. 2002), i.e. one recording effort by one team. We therefore grouped recordings into *recording events* — same species, same archive, catalogue numbers within 1,000 of each other — as a proxy for same-session origin, and used these groups in the stricter evaluation scheme below.

### Windowing and vocalization detection

Each recording was decoded to mono 48 kHz and sliced into 3-s windows with 50% overlap (matching BirdNET's native analysis window). Windows were scored for vocal activity by in-band (150 Hz–11 kHz) spectral energy relative to the recording's own noise floor (10th percentile of frame energy); windows ≥3 dB above the floor were retained, with a guarantee that every recording contributes at least its single highest-scoring window. This yielded 3,195 analysis windows (Banasura Laughingthrush 976, Bugun Liocichla 877, Forest Owlet 1,116, Jerdon's Courser 226) — a tenfold increase in usable training instances over one-image-per-recording pipelines, with no duplication.

### Evaluation design

The unit of splitting is never the window: all windows of a recording share that recording's partition, in every experiment.

**Recording-level scheme (primary).** Stratified five-fold cross-validation over unique recordings. Within each outer fold, the non-test recordings are further divided (again at recording level, stratified by species) into training (~80%) and validation (~20%) sets. All early stopping and any other selection uses only the inner validation set; each outer test fold is evaluated exactly once, after all decisions are frozen. Across the five folds every recording is tested exactly once, so headline metrics are computed over all 193 recordings. With 12 recordings of Jerdon's Courser, a single held-out test set would either starve training or make test metrics for that species meaningless; cross-validated held-out folds with bootstrap confidence intervals are the defensible construction, and per-class support is reported as unique recordings everywhere.

**Event-level scheme (stricter).** Identical construction, except the unit assigned to folds is the recording event (catalogue-block group). No two recordings from the same event can straddle a partition, so the scheme tests generalization across recording sessions, not just across files. For Jerdon's Courser this scheme is degenerate by necessity: the species has two events (the ML block and the single XC recording). We pinned the ML block to training and let the lone XC recording serve as a one-recording cross-source probe — an honest, if minimal, test of whether anything generalizes beyond the single known session.

Every experiment was repeated with three random seeds; we report seed-averaged predictions and per-seed spread. Class imbalance was handled by inverse-frequency loss weighting, never by data duplication. Splits are serialized with their generating seed and shipped with the code.

### Models

**Baseline A — fine-tuned CNN.** An EfficientNet-B0 initialized from ImageNet, fine-tuned end-to-end on log-mel spectrograms (128 mels, 32 kHz, 12 kHz cap) of the 3-s windows, with SpecAugment-style time/frequency masking, gain jitter, time shift, and noise injection. This baseline represents the prior standard practice — spectrogram images into a vision CNN — implemented with correct feature extraction (arrays, not rendered colormap images), augmentation, and selection protocol.

**Baseline B — foundation-model embeddings + probe.** Each window is encoded by BirdNET V2.4 into a 1,024-d embedding (weights frozen). A two-layer MLP with layer normalization and dropout classifies windows; recording-level predictions average window probabilities.

**Attention-MIL.** A recording is treated as a bag of its window embeddings. A gated-attention pooling head (Ilse et al. 2018) computes per-window attention weights, forms the attention-weighted bag representation, and classifies the recording directly. Besides making the recording the native decision unit, the attention weights indicate which seconds of audio drove each decision.

**Domain-adaptive projection (tested and rejected).** We hypothesized that adapting the embedding space to Indian endemic avifauna would help. A residual projection head was trained with supervised contrastive loss (Khosla et al. 2020) on an auxiliary corpus of 28 non-target endemic species assembled with the identical audit/window pipeline (1,891 unique recordings; 13,810 windows), then frozen and inserted before the probe/MIL heads. The auxiliary corpus never contains target species, so it cannot leak into target evaluation.

**Open-set rejection.** A deployed detector must reject non-target sound. We measured (i) whether confidence thresholding on the closed-set classifier separates target from non-target recordings, and (ii) the effect of adding an explicit fifth "background" class. For (ii), the 28 auxiliary species were split in half: windows of 14 species formed the background class during training; the other 14 species — never seen by the model in any role — measured false acceptance. Thresholds for (i) were calibrated only on validation data (the value keeping ≥95% of validation recordings).

**Robustness.** Test windows were corrupted with additive white noise at 20, 10, 5, and 0 dB SNR *before* embedding extraction, and evaluated with single-seed probes trained on clean data under the recording-level folds.

### External reference

BirdNET's own classifier output was recorded for every window. This provides a deployment-relevant reference for the one covered species (Forest Owlet) and documents the coverage gap for the other three.

### Implementation

PyTorch 2.13 (probe/MIL/CNN) and the BirdNET-Analyzer reference implementation; all experiments run on a single Apple-silicon laptop, the entire pipeline (audit → windows → splits → embeddings → all models → all figures) reproducible by scripts shipped with the repository. Audio is not redistributed; the manifest lists every catalogue identifier so the exact dataset can be reassembled from the archives.

## Results

### Closed-set identification

**Table 1.** Recording-level performance, primary (recording-level) five-fold CV over all 193 unique recordings. Accuracy and macro-F1 with bootstrap 95% CIs; seed SD over three runs. Ensembles are untuned equal-weight soft votes (no ensemble weight was selected on any data).

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Seed SD |
|---|---|---|---|---|---|
| Fine-tuned CNN (EfficientNet-B0) | 83.4% | 77.7–88.6 | 0.826 | 0.750–0.889 | 0.011 |
| Embedding probe (BirdNET + MLP) | 97.4% | 94.8–99.5 | 0.973 | 0.938–0.996 | 0.002 |
| Attention-MIL | 96.4% | 93.8–99.0 | 0.964 | 0.927–0.990 | 0.005 |
| Probe, adapted embeddings | 92.7% | 89.1–96.4 | 0.905 | 0.843–0.953 | 0.006 |
| MIL, adapted embeddings | 90.2% | 86.0–94.3 | 0.870 | 0.800–0.926 | 0.004 |
| Soft vote: probe + MIL | 97.9% | 95.9–99.5 | 0.984 | 0.967–0.997 | — |
| Soft vote: CNN + probe + MIL | 97.9% | 95.9–99.5 | 0.984 | 0.967–0.997 | — |

On the primary recording-level scheme, the embedding probe attained 97.4% recording-level accuracy (95% CI 94.8–99.5%; macro-F1 0.973, CI 0.938–0.996) over all 193 recordings, and attention-MIL 96.4% (CI 93.8–99.0%; macro-F1 0.964). Per-seed accuracy varied by ≤0.5 percentage points (SD ≤0.005), indicating the results are not an artifact of a fortunate initialization. Per-class F1 was ≥0.96 for every species, including Jerdon's Courser (12 recordings, F1 0.96) — though we caution below against over-reading that number. The fine-tuned CNN baseline reached 83.4% (macro-F1 0.826), far below the embedding-based models and with roughly five times the seed variance: with ~150 training recordings, end-to-end fine-tuning remains data-starved even when carefully regularized, while frozen bioacoustic embeddings carry most of the discriminative burden. McNemar tests found no significant difference between probe and MIL (p = 0.69), but both greatly exceeded the CNN (probe vs. CNN: 30 recordings correct only under the probe vs. 3 only under the CNN, p = 1.4 × 10⁻⁶; MIL vs. CNN: p = 1.1 × 10⁻⁵). The untuned probe+MIL soft vote gave a marginal, non-significant improvement (97.9%, macro-F1 0.984); we highlight it not for the gain but as the correct form of ensembling under scarce data — combining complementary decision rules without spending any data on weight selection.

The difference between these figures and the 80% previously reported on this problem under a leaky protocol is attributable to methodology on both sides: the earlier number was optimistically biased by checkpoint selection on the reporting set, yet substantively *understated* what is achievable, because whole variable-length recordings were compressed into single spectrogram images and most of the audio was discarded.

### Session-level generalization and the Jerdon's Courser limit

Under the event-level scheme — in which no two recordings from the same session may straddle a partition — the probe reached 98.9% accuracy (95% CI 97.3–100%) over the 182 recordings testable under this scheme, with per-class performance for the three multi-session species essentially unchanged from the recording-level results (attention-MIL: 95.1%; fine-tuned CNN: 75.8%, its gap to the embedding models widening under the stricter scheme). Recording-level and session-level evaluations agreeing this closely for the embedding models demonstrates that the headline results are not driven by within-session leakage.

Jerdon's Courser is different, and the difference is a property of the data, not the models. Its archival record is one 2001–2002 recording session (11 ML recordings in a single catalogue block) plus one XC recording. In the event-level scheme the entire ML block trains the model and the lone XC recording is the only possible test of generalization beyond the known session. That recording was classified correctly and confidently (probe: 89–94% probability on Jerdon's Courser across seeds; likewise for MIL). We checked whether this success might be trivial — the XC file being a re-processed cut of the same ML audio: content hashes differ, and its maximum embedding similarity to any ML window (0.69) is below even the ceiling observed between different recordings within the ML session (0.84), consistent with genuinely distinct audio. Nevertheless, one recording is one recording: a single cross-source success cannot establish generalization, and no evaluation design can extract more from an archive whose record of this species traces to essentially one recording effort. The honest summary is that all available evidence is positive, and the quantity of evidence is one. The 2025 reconfirmation recordings, if archived publicly, would multiply that evidence — we argue below this is exactly where conservation bioacoustics should invest.

### Foundation-model coverage and the external reference

BirdNET V2.4 contains labels for only one of the four species. For Forest Owlet, BirdNET's own classifier assigned its label a mean confidence of 0.17 on windows that truly contain the species (maximum 1.0 on the clearest calls) — usable for screening but far from a reliable detector, and unavailable in principle for the other three species. Our probe on BirdNET's *embeddings* recognized Forest Owlet recordings with F1 0.98: the representation contains the necessary information even where the classifier head has no corresponding output.

### Adaptation on auxiliary endemics: a negative result

The supervised-contrastive projection trained on 28 auxiliary endemic species *reduced* probe accuracy from 97.4% to 92.7% and MIL accuracy from 96.4% to 90.2%. Figure 6 shows why: in the raw BirdNET space the four target species already form well-separated clusters, whereas the adapted projection — optimized to separate 28 *other* species — collapses much of that structure. Foundation embeddings trained on thousands of species evidently already encode the discriminative detail; a projection fitted to a 28-species auxiliary task discards information rather than adding it. We report this because the intuition "adapt the embedding on related species" is natural, cheap to try, and — at this data scale — harmful; practitioners should test it against a frozen-embedding baseline rather than assume it.

### Open-set behaviour: the gap between a classifier and a detector

**Table 2.** Open-set behaviour against non-target Indian endemic species (recording level). MSP = maximum softmax probability of the closed-set (4-class) probe, threshold calibrated on validation data only (retaining ≥95% of validation recordings). The background-class model adds a fifth class trained on windows of 14 auxiliary species; false acceptance is measured on the other 14 *entirely unseen* species (814 recordings). "Target retained" counts recordings accepted (MSP row) or accepted *and* correctly identified (background-class row).

| Approach | Target recordings retained | Non-target false-accept rate |
|---|---|---|
| MSP threshold on closed-set probe | 97.4% | 94.0% (all 1,891 aux recordings, worst case over folds) |
| Explicit background class (species-disjoint) | 89.1% ± 3.9% | 13.4% ± 2.5% (814 unseen-species recordings) |

Confidence thresholding failed. At a validation-calibrated threshold retaining 97% of target recordings, 94% of the 1,891 non-target endemic-species recordings were also accepted (recording-level AUROC 0.75). A 97%-accurate four-class classifier is thus not remotely a field detector: deployed against India's actual endemic avifauna it would flood observers with confident false positives.

Adding an explicit background class transformed this. Trained with windows from 14 auxiliary species and evaluated against 14 completely unseen species, the five-class probe cut recording-level false acceptance from 94% to 13.4% ± 2.5% while retaining 89.1% ± 3.9% of target recordings correctly identified (8.8% of target recordings were sent to the background class). This trade-off — a modest recall cost for a sevenfold reduction in false positives against distractors the model never saw — is the operationally meaningful headline of this work, and the numbers reviewers and practitioners should weigh, rather than the closed-set ceiling.

### Noise robustness

Corrupting test audio with additive white noise before embedding extraction degraded recording-level accuracy gracefully: 96.9% on clean audio under this single-seed protocol (versus 97.4% for the seed-averaged headline), 96.4% at 20 dB SNR, 94.3% at 10 dB, 93.3% at 5 dB, and 92.2% at 0 dB (macro-F1 0.965 → 0.925). The embedding representation, trained on heterogeneous field recordings, absorbs much of the corruption that would degrade a spectrogram-image classifier — consistent with its behaviour as a noise-robust front end for downstream probes.

### What the models listen to

Attention weights from the MIL head concentrated on windows containing vocalizations and near-zero weight on ambient segments (Figure 5), confirming that recording-level decisions rest on the target sound rather than on background signatures — an inspectable, per-decision property that pure window-voting pipelines lack.

## Discussion

Three points deserve emphasis.

**Evaluation design is the difference between a benchmark score and a conservation claim.** With rare species, the data volume is fixed by history; what the analyst controls is the independence structure of the evaluation. Recording-level splitting, session-level grouping, selection confined to inner validation, seed replication, and bootstrap intervals are all cheap; none require more data. Under this regime our headline numbers *rose* relative to a flawed earlier pipeline — rigour and performance are not in tension when the representation is right.

**Foundation embeddings change what is possible for data-poor species, but coverage gaps cut both ways.** Three of the four most threatened Indian endemics are invisible to the dominant global recognizer, and this is unlikely to be unique to India. Lightweight probes over frozen embeddings — trainable on a laptop in minutes — offer range-country institutions a practical route to detectors for nationally important species without waiting for global model updates. The corollary responsibility is honest reporting of what the archives can support: our Jerdon's Courser analysis shows that a per-class F1 of 0.96 can rest on an archive whose independent evidence for cross-session generalization amounts to a single recording — a distinction invisible in a metrics table but decisive for how much trust a deployed detector deserves.

**Closed-set accuracy is the wrong headline for deployment.** The 94%→13% false-acceptance reduction from a species-disjoint background class matters more than the last two points of closed-set accuracy. We suggest that papers proposing classifiers for conservation deployment routinely report open-set metrics against realistic regional distractors; the auxiliary corpus needed is usually easier to assemble than the target data itself.

**Limitations.** (1) BirdNET's training data includes Xeno-canto; for Forest Owlet (the one species in its label set), some of our recordings may have contributed to pretraining, which could inflate embedding quality for that species — the other three species cannot have been labelled positives, and their strong results stand independent of this concern. (2) Archive recordings are focal and relatively clean; performance on soundscape data from autonomous recorders will be lower, and our additive-noise stress test only approximates that gap. (3) The recording-event grouping is inferred from catalogue adjacency; true recordist/site/date metadata (obtainable via the archives' APIs) would sharpen it. (4) With 12–75 recordings per class, every metric carries wide intervals; we have reported them.

**Priorities this analysis implies for the four species.** For Jerdon's Courser, the single most valuable contribution anyone can make is depositing new, independently recorded material — including the 2025 reconfirmation cuts — in public archives; detector development is representation-ready and data-blocked. For Bugun Liocichla and Banasura Laughingthrush, tens of recordings suffice for strong within-archive performance, and the next step is validation on autonomous-recorder soundscapes at the known sites. For Forest Owlet, where playback-based acoustic survey is already operational practice, embedding-based detectors could be evaluated head-to-head against BirdNET's native output on existing survey audio.

## Figures

- **Figure 1** (`figures/f1_dataset.png`): Dataset overview — unique recordings per species and source, total audio, and extracted vocalization windows.
- **Figure 2** (`figures/f2_spectrograms.png`): Example 3-s log-mel windows for each species.
- **Figure 3** (`figures/f3_confusions.png`): Recording-level confusion matrices for the embedding probe under the recording-level and stricter event-level schemes; the sole event-level Jerdon's Courser test recording (the cross-source probe) is classified correctly.
- **Figure 4** (`figures/f4_robustness.png`): Recording-level accuracy and macro-F1 under additive noise (clean → 0 dB SNR).
- **Figure 5** (`figures/f5_attention.png`): MIL attention across a 10-minute Bugun Liocichla recording — attention concentrates on vocalization-bearing windows.
- **Figure 6** (`figures/f6_umap.png`): UMAP of window embeddings; raw BirdNET space (left) vs. after auxiliary-species contrastive adaptation (right), visualizing why adaptation degrades performance.

## Data and code availability

All code, the recording manifest (with every ML/XC catalogue identifier), split definitions, and result files are available at https://github.com/srivatsav-kannan/birdSong. Audio is not redistributed, in accordance with archive terms; the manifest permits exact reconstruction of the dataset from the Macaulay Library and Xeno-canto.

## References

*(to be finalized in journal format)*

- Athreya, R. (2006). A new species of Liocichla (Aves: Timaliidae) from Eaglenest Wildlife Sanctuary, Arunachal Pradesh, India. *Indian Birds* 2(4): 82–94.
- Bhushan, B. (1986). Rediscovery of the Jerdon's Courser *Cursorius bitorquatus*. *J. Bombay Nat. Hist. Soc.* 83: 1–14.
- Ghani, B., Denton, T., Kahl, S. and Klinck, H. (2023). Global birdsong embeddings enable superior transfer learning for bioacoustic classification. *Scientific Reports* 13: 22876.
- Ilse, M., Tomczak, J.M. and Welling, M. (2018). Attention-based deep multiple instance learning. *ICML 2018*.
- Jeganathan, P., Green, R.E., Bowden, C.G.R., Norris, K., Pain, D. and Rahmani, A. (2002). Use of tracking strips and automatic cameras for detecting Critically Endangered Jerdon's coursers *Rhinoptilus bitorquatus* in scrub jungle in Andhra Pradesh, India. *Oryx* 36(2).
- Kahl, S., Wood, C.M., Eibl, M. and Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. *Ecological Informatics* 61: 101236.
- Khosla, P. et al. (2020). Supervised contrastive learning. *NeurIPS 2020*.
- Rasmussen, P.C. and Collar, N.J. (1998). Identification, distribution and status of the Forest Owlet *Athene (Heteroglaux) blewitti*. *Forktail* 14: 41–49.
- Robin, V.V. et al. (2017). Two new genera of songbirds represent endemic radiations from the Shola Sky Islands of the Western Ghats, India. *BMC Evolutionary Biology* 17: 31.
- SoIB (2023). *State of India's Birds, 2023: Range, trends, and conservation status*. Zenodo.
- Search for Lost Birds (2025). Jerdon's Courser. https://searchforlostbirds.org/birds/jerdons-courser
- IUCN (2025). The IUCN Red List of Threatened Species. https://www.iucnredlist.org
- Xeno-canto Foundation (2025). https://www.xeno-canto.org · Cornell Lab of Ornithology, Macaulay Library. https://www.macaulaylibrary.org
