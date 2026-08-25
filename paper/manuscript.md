# Acoustic Identification of India's Endangered Endemic Birds Using Bioacoustic Foundation Model Embeddings and Attention-Based Multiple Instance Learning

**Srivatsav Kannan¹, Valliappan Raman¹, Shanmugapriya K.R.¹**

¹ Department of Artificial Intelligence and Data Science, Coimbatore Institute of Technology, Coimbatore, India

---

## Summary

Passive acoustic monitoring is among the few survey methods that scale to rare and cryptic birds, and it was acoustic evidence that reconfirmed Jerdon's Courser *Rhinoptilus bitorquatus* in 2025 after seventeen years without a verified record. The general-purpose recognizers that support most applied bioacoustics, however, do not cover the species that need monitoring most. Of the four Endangered or Critically Endangered Indian endemics with any archived sound recordings, namely the Forest Owlet *Athene blewitti*, the Banasura Laughingthrush *Montecincla jerdoni*, the Bugun Liocichla *Liocichla bugunorum*, and Jerdon's Courser, only the Forest Owlet appears in the 6,000-class label set of BirdNET. In this study we assembled and audited the complete public acoustic record of these four species, comprising 193 unique recordings and about 125 minutes of audio from the Macaulay Library and Xeno-canto, and we trained classifiers that operate on embeddings from the BirdNET foundation model, including a gated-attention multiple instance learning head that makes decisions at the level of the whole recording. Evaluation used stratified five-fold cross-validation in which the unit of splitting is the unique recording, all model selection is confined to inner validation sets, and each test fold contributes to the reported results exactly once. A second, stricter scheme additionally prevented recordings from the same recording session from appearing on both sides of any partition. Recording-level accuracy reached 97.4% (95% CI 94.8 to 99.5, macro-F1 0.973) under the primary scheme and 98.9% under the stricter one, was stable across random seeds, and clearly exceeded a carefully trained fine-tuned CNN baseline representative of prior practice (83.4%). In an open-set stress test against 1,891 recordings of 28 non-target endemic species, a confidence threshold alone accepted 93.7% of the distractor recordings, whereas an explicit background class trained on species held disjoint from the evaluation reduced false acceptance to 13.4% while retaining 89.1% of target recordings. We release the full pipeline, the recording manifest, and the split definitions so that every reported number can be reproduced exactly, and we discuss what the available archives can and cannot support for each species.

**Keywords:** bioacoustics, passive acoustic monitoring, endemic species, transfer learning, multiple instance learning, open-set recognition, India

---

## Introduction

India supports over 1,300 bird species, of which more than 75 occur nowhere else in the world (SoIB 2023). Endemic species with narrow ranges are disproportionately exposed to habitat loss, and several of India's endemics now sit in the highest IUCN threat categories. The four Endangered or Critically Endangered endemics with any publicly archived sound recordings illustrate both the urgency and the difficulty of monitoring such species. The Forest Owlet was known from seven nineteenth-century specimens until its rediscovery in 1997 (Rasmussen and Collar 1998) and, although it has since been found at additional sites, it retains a small, fragmented, and declining population. The Banasura Laughingthrush is confined to high-elevation shola patches totalling under 57 km² in the Wayanad region of the Western Ghats (Robin et al. 2017). The Bugun Liocichla, described only in 2006 from Eaglenest Wildlife Sanctuary in Arunachal Pradesh, is known from a handful of territories holding at most a few tens of individuals (Athreya 2006). Jerdon's Courser, rediscovered in 1986 after 86 years without a record (Bhushan 1986), was last verifiably documented in 2008 until August 2025, when its call was captured by an automated recorder at a new site. That reconfirmation was possible because a sound-based search image for the species existed at all (Jeganathan et al. 2002, Search for Lost Birds 2025).

These species are natural candidates for passive acoustic monitoring. They are nocturnal or skulking, they occupy terrain where visual survey is difficult, and they have distinctive vocalizations. Passive monitoring at scale depends on automated identification, and the recognizers that dominate applied bioacoustics currently cannot provide it for these birds. We verified that only the Forest Owlet is present in the label set of BirdNET V2.4 (Kahl et al. 2021), which is the most widely deployed bird sound classifier. The other three species cannot be detected by it at any confidence threshold, so species-specific classifiers address a genuine operational need.

Building such classifiers for data-poor species raises a methodological trap that this paper confronts directly. When the entire acoustic record of a species amounts to a few dozen recordings, sometimes made by a single recordist at a single site in a single season, it is easy to obtain high benchmark scores that reflect memorization of recording conditions rather than generalizable species discrimination. An earlier version of this work suffered from exactly this problem, because performance was reported on the same small validation set that had been used for model selection, and class balance was achieved by duplicating audio files. The present study rebuilds the entire pipeline around three principles. First, every reported number comes from data that played no role in training or model selection. Second, the unit of independence is the recording, and in a stricter companion analysis it is the recording session. Third, the limits of the available data are measured and reported rather than averaged away.

Methodologically, we follow the growing body of evidence that embeddings from large pretrained bioacoustic models transfer to data-poor problems more effectively than fine-tuning vision architectures does (Ghani et al. 2023). We add two elements that are matched to the conservation setting. The first is a gated-attention multiple instance learning head (Ilse et al. 2018) that makes decisions at the level of the whole recording, which is the level at which a monitoring program acts, while learning which three-second windows carry diagnostic sound and thereby providing interpretability without additional annotation. The second is a quantitative treatment of open-set behaviour. A deployed detector must reject the many non-target species it will hear, so we measured rejection against 1,891 recordings of 28 other Indian endemic species and show that an explicit background class, trained on auxiliary species that are disjoint from those used for testing, turns an unusable open-set system into a plausible screening tool. We additionally report an informative negative result, in which supervised contrastive adaptation of the embedding space on the auxiliary endemics reduced target performance, and we characterize a structural limit of the archives themselves, since the record of Jerdon's Courser contains essentially one recording session and therefore permits only a single independent test of generalization.

## Methods

### Data collection and audit

We filtered the IUCN Red List for bird species that are endemic to India and listed as Endangered or Critically Endangered. Seven species qualified. Three of them, the Great Nicobar Serpent Eagle *Spilornis klossi*, the Himalayan Quail *Ophrysia superciliosa*, and the Manipur Bush Quail *Perdicula manipurensis*, have no usable recordings in either major public archive, and the Himalayan Quail has not been reliably recorded since the nineteenth century. The remaining four species form the study set.

All available recordings were obtained from the Macaulay Library and Xeno-canto. Every file was assigned its archive catalogue identifier, and its provenance (species, source, Xeno-canto quality grade, duration, and sample rate) was recorded in a manifest that serves as the single source of truth for all downstream processing. Deduplication used both catalogue identifiers and SHA-256 content hashes. Six byte-identical duplicates were found and removed, including recordings that were present under both a Macaulay and a Xeno-canto catalogue number. This audit step matters because copies of the same audio in a duplicated dataset can silently occupy both the training and the evaluation partition.

The final dataset comprises 193 unique recordings totalling about 125 minutes: 75 of the Forest Owlet (46 Macaulay, 29 Xeno-canto), 74 of the Banasura Laughingthrush (71 and 3), 32 of the Bugun Liocichla (all Macaulay), and 12 of Jerdon's Courser (11 and 1). No audio was duplicated or otherwise multiplied at the file level anywhere in the pipeline.

Catalogue numbers also encode provenance structure that a rigorous evaluation must respect. All 11 Macaulay recordings of Jerdon's Courser occupy one consecutive catalogue block (ML274533 to ML274543). These are the known 2001 to 2002 recordings from Sri Lankamalleswara Wildlife Sanctuary (Jeganathan et al. 2002), which means they represent one recording effort by one team. We therefore grouped recordings into recording events, defined as recordings of the same species from the same archive whose catalogue numbers lie within 1,000 of one another, as a proxy for same-session origin, and we used these groups as the unit of splitting in the stricter evaluation scheme described below.

### Windowing and vocalization detection

Each recording was decoded to mono 48 kHz audio and sliced into windows of 3 s with 50% overlap, matching the native analysis window of BirdNET. Each window was scored for vocal activity using its spectral energy in the 150 Hz to 11 kHz band relative to the noise floor of its own recording, estimated as the tenth percentile of frame energy. Windows at least 3 dB above the floor were retained, and every recording was guaranteed to contribute at least its single highest-scoring window so that no recording dropped out of the dataset. This procedure yielded 3,195 analysis windows (976 for the Banasura Laughingthrush, 877 for the Bugun Liocichla, 1,116 for the Forest Owlet, and 226 for Jerdon's Courser), which is roughly a tenfold increase in usable training instances over a pipeline that renders one image per recording, achieved without any duplication.

![Figure 1: dataset overview](figures/f1_dataset.png)

*Figure 1. Unique recordings per species and source, total audio per species, and extracted vocalization windows.*

![Figure 2: example spectrograms](figures/f2_spectrograms.png)

*Figure 2. Example 3 s log-mel windows for each species. The Jerdon's Courser example reflects the heavy tape noise of the historical source material.*

### Evaluation design

The unit of splitting is never the window. All windows of a recording share that recording's partition in every experiment.

Under the primary recording-level scheme, we performed stratified five-fold cross-validation over unique recordings. Within each outer fold, the recordings outside the test fold were further divided at the recording level, stratified by species, into a training set of roughly 80% and a validation set of roughly 20%. All early stopping and every other selection decision used only this inner validation set, and each outer test fold was evaluated exactly once after all decisions were frozen. Across the five folds every recording is tested exactly once, so the headline metrics are computed over all 193 recordings. We chose cross-validation over a single held-out test set deliberately. With only 12 recordings of Jerdon's Courser, a single test set would either starve training or make the test metrics for that species meaningless, whereas cross-validated held-out folds with bootstrap confidence intervals use every recording for evaluation exactly once while preserving independence. Per-class support is reported as unique recordings throughout.

Under the stricter event-level scheme, the construction is identical except that the unit assigned to folds is the recording event rather than the individual recording. No two recordings from the same event can then appear on opposite sides of a partition, so this scheme tests generalization across recording sessions rather than only across files. For Jerdon's Courser the scheme is degenerate by necessity, because the species has exactly two events, the Macaulay block and the single Xeno-canto recording. We pinned the Macaulay block to training and let the lone Xeno-canto recording serve as a one-recording cross-source test, which is the only test of generalization beyond the known session that the archives permit.

Every experiment was repeated with three random seeds, and we report seed-averaged predictions together with the per-seed spread. Class imbalance was handled with inverse-frequency loss weighting rather than data duplication. The splits are serialized together with their generating seed and are shipped with the code.

### Models

The first baseline is a fine-tuned CNN that represents prior standard practice implemented correctly. An EfficientNet-B0 initialized from ImageNet weights was fine-tuned end to end on log-mel spectrograms of the 3 s windows (128 mel bands, 32 kHz sample rate, 12 kHz frequency cap), with SpecAugment-style time and frequency masking, gain jitter, circular time shifting, and white noise injection applied to training data only. Features were computed as arrays rather than rendered colormap images, and model selection followed the same inner-validation protocol as every other model.

The second baseline classifies foundation model embeddings. Each window was encoded by BirdNET V2.4 into a 1,024-dimensional embedding with the BirdNET weights frozen, and a two-layer multilayer perceptron with layer normalization and dropout classified individual windows. Recording-level predictions were obtained by averaging window probabilities within each recording.

The attention-based multiple instance learning model treats a recording as a bag of its window embeddings. A gated-attention pooling head (Ilse et al. 2018) computes a weight for each window, forms the attention-weighted sum of the encoded windows, and classifies the recording directly. This design makes the recording the native decision unit and, because the attention weights are inspectable, it shows which seconds of audio drove each decision.

We also tested a domain-adaptive projection, which we report as a negative result. The hypothesis was that adapting the embedding space to Indian endemic avifauna would improve target discrimination. A residual projection head mapping the 1,024-dimensional embeddings to 256 dimensions was trained for 40 epochs with a supervised contrastive loss (Khosla et al. 2020) on an auxiliary corpus of 28 non-target endemic species that was assembled with the identical audit and windowing pipeline (1,891 unique recordings and 13,810 windows), and it was then frozen and inserted before the probe and MIL heads. The auxiliary corpus contains no target species, so it cannot leak into target evaluation.

For open-set evaluation, we measured two things. The first was whether confidence thresholding on the closed-set classifier separates target from non-target recordings, using the maximum softmax probability as the score with the threshold calibrated only on validation data (the highest threshold retaining at least 95% of validation recordings). The second was the effect of adding an explicit fifth background class. For that experiment the 28 auxiliary species were split in half, with the windows of 14 species forming the background class during training (1,077 recordings) and the other 14 species, which the model never saw in any role, measuring false acceptance (814 recordings).

For the robustness analysis, test windows were corrupted with additive white noise at 20, 10, 5, and 0 dB signal-to-noise ratio before embedding extraction, and they were evaluated with single-seed probes trained on clean data under the recording-level folds.

Finally, BirdNET's own classifier output was recorded for every window as an external reference. This provides a deployment-relevant comparison for the one species it covers and documents the coverage gap for the other three.

### Implementation

The probe, MIL, and CNN models were implemented in PyTorch 2.13, and embeddings were extracted with the BirdNET-Analyzer reference implementation. All experiments ran on a single Apple silicon laptop, and the entire pipeline, from the audit through windowing, splitting, embedding, model training, and figure generation, is reproducible from the scripts shipped with the repository. The audio itself is not redistributed, in accordance with archive terms, and the manifest lists every catalogue identifier so that the exact dataset can be reassembled from the Macaulay Library and Xeno-canto.

## Results

### Closed-set identification

Table 1 reports recording-level performance under the primary scheme for all models, and Table 2 reports the per-class metrics of the strongest single model. The embedding probe attained 97.4% accuracy (95% CI 94.8 to 99.5, macro-F1 0.973 with CI 0.938 to 0.996) over all 193 recordings, and the attention-MIL model attained 96.4% (CI 93.8 to 99.0, macro-F1 0.964). Accuracy varied by at most half a percentage point across seeds for both models, so the results do not depend on a fortunate initialization. The fine-tuned CNN baseline reached 83.4% (CI 77.7 to 88.6, macro-F1 0.826) with roughly four times the seed variance. An exact McNemar test on paired recording outcomes found the probe clearly superior to the CNN, with 30 recordings correct only under the probe against 3 correct only under the CNN (p = 1.4 × 10⁻⁶), and likewise for the MIL model (29 against 4, p = 1.1 × 10⁻⁵), while the probe and the MIL model did not differ significantly from each other (p = 0.69). We interpret the gap as a data-scale effect, since roughly 150 training recordings remain too few for end-to-end fine-tuning even with careful regularization, whereas the frozen bioacoustic embeddings already carry most of the discriminative information.

**Table 1.** Recording-level performance under the primary scheme, five-fold cross-validation over all 193 unique recordings, with bootstrap 95% confidence intervals and the standard deviation of accuracy across three seeds. The soft votes average the member models' probabilities with equal weights, so no ensemble weight was selected on any data.

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Seed SD |
|---|---|---|---|---|---|
| Fine-tuned CNN (EfficientNet-B0) | 83.4% | 77.7 to 88.6 | 0.826 | 0.750 to 0.889 | 0.011 |
| Embedding probe (BirdNET + MLP) | 97.4% | 94.8 to 99.5 | 0.973 | 0.938 to 0.996 | 0.002 |
| Attention-MIL | 96.4% | 93.8 to 99.0 | 0.964 | 0.927 to 0.990 | 0.005 |
| Probe on adapted embeddings | 92.7% | 89.1 to 96.4 | 0.905 | 0.843 to 0.953 | 0.006 |
| MIL on adapted embeddings | 90.2% | 86.0 to 94.3 | 0.870 | 0.800 to 0.926 | 0.004 |
| Soft vote, CNN + probe | 96.9% | 94.3 to 99.0 | 0.971 | 0.939 to 0.993 | n/a |
| Soft vote, probe + MIL | 97.9% | 95.9 to 99.5 | 0.984 | 0.967 to 0.997 | n/a |
| Soft vote, CNN + probe + MIL | 97.9% | 95.9 to 99.5 | 0.984 | 0.967 to 0.997 | n/a |

**Table 2.** Per-class metrics of the embedding probe under the primary scheme. Support is the number of unique recordings.

| Species | Precision | Recall | F1 | Recordings |
|---|---|---|---|---|
| Banasura Laughingthrush | 0.973 | 0.959 | 0.966 | 74 |
| Bugun Liocichla | 0.970 | 1.000 | 0.985 | 32 |
| Forest Owlet | 0.986 | 0.973 | 0.980 | 75 |
| Jerdon's Courser | 0.923 | 1.000 | 0.960 | 12 |

The untuned soft vote of the probe and the MIL model reached 97.9% accuracy with a macro-F1 of 0.984, an improvement over the probe alone that is not statistically significant (p = 0.375). We include it because equal-weight combination is the appropriate form of ensembling under scarce data, as it combines complementary decision rules without spending any data on weight selection. Adding the CNN to this vote changed nothing.

These figures differ substantially from the 80% previously reported on this problem under a flawed protocol, and the difference is attributable to methodology on both sides. The earlier number was optimistically biased because the reported checkpoint had been selected on the same validation set used for reporting, and at the same time it understated what the data support, because compressing whole variable-length recordings into single spectrogram images had discarded most of the usable audio.

![Figure 3: confusion matrices](figures/f3_confusions.png)

*Figure 3. Recording-level confusion matrices of the embedding probe under the primary recording-level scheme (left) and the stricter event-level scheme (right). The single event-level Jerdon's Courser test recording is the cross-source Xeno-canto file, which is classified correctly.*

### Session-level generalization

Under the event-level scheme, in which no two recordings from the same session may appear on opposite sides of a partition, the probe reached 98.9% accuracy (CI 97.3 to 100) over the 182 recordings testable under this scheme, the MIL model reached 95.1%, and the CNN fell to 75.8%. Per-class F1 for the probe was 0.993 for the Banasura Laughingthrush, 1.000 for the Bugun Liocichla, and 0.986 for the Forest Owlet, essentially unchanged from the primary scheme. The close agreement between the recording-level and session-level evaluations for the embedding models demonstrates that the headline results are not driven by within-session leakage, while the widening gap for the CNN suggests that part of its remaining performance did depend on session-specific cues.

Jerdon's Courser requires separate discussion because its situation is a property of the data rather than of any model. The archival record of the species consists of one 2001 to 2002 recording session (the 11 Macaulay recordings in a single catalogue block) plus one Xeno-canto recording. Under the event-level scheme the entire Macaulay block trains the model and the lone Xeno-canto recording provides the only possible test of generalization beyond the known session. That recording was classified correctly and confidently, with the probe assigning between 89% and 94% probability to Jerdon's Courser across seeds and the MIL model behaving equivalently. We checked whether this success might be trivial in the sense of the Xeno-canto file being a reprocessed cut of the same Macaulay audio. The content hashes differ, and the maximum cosine similarity between its embedding and any Macaulay window is 0.69, which is below the maximum of 0.84 observed between different recordings within the Macaulay session itself, so the evidence is consistent with genuinely distinct audio. The appropriate conclusion is nevertheless a limited one. A single cross-source recording cannot establish generalization, and no evaluation design can extract more from an archive whose record of the species traces to one recording effort. All the evidence the archives allow is positive, and there is exactly one unit of it. The 2025 reconfirmation recordings would multiply that evidence if they are deposited publicly, and we return to this point in the discussion.

### Foundation model coverage and the external reference

BirdNET V2.4 contains a label for only one of the four species. For the Forest Owlet, BirdNET's own classifier assigned its label a mean confidence of 0.17 on windows that truly contain the species, reaching 1.0 only on the clearest calls, which makes it usable for coarse screening but far from a reliable detector, and no equivalent output exists for the other three species. Our probe operating on BirdNET's embeddings identified Forest Owlet recordings with an F1 of 0.98. The comparison shows that the representation contains the necessary information even where the classifier head has no corresponding output, which is precisely the situation of most range-restricted threatened species.

### Adaptation on auxiliary endemics reduced performance

The supervised contrastive projection trained on the 28 auxiliary endemic species reduced probe accuracy from 97.4% to 92.7% (McNemar p = 0.022) and MIL accuracy from 96.4% to 90.2% (p = 0.0075). Figure 4 shows why. In the raw BirdNET space the four target species already form well-separated clusters, whereas the adapted projection, which was optimized to separate 28 other species, collapses much of that structure. Embeddings trained on thousands of species evidently already encode the discriminative detail these targets require, and a projection fitted to a 28-species auxiliary task discards information instead of adding it. We report this because the intuition that one should adapt an embedding on related species is natural and inexpensive to try, and at this data scale it is harmful, so practitioners should test any such adaptation against a frozen-embedding baseline before adopting it.

![Figure 4: embedding spaces](figures/f6_umap.png)

*Figure 4. UMAP projections of the window embeddings for the four target species in the raw BirdNET space (left) and after auxiliary-species contrastive adaptation (right).*

### Open-set behaviour

A four-class classifier, however accurate, is only useful in the field if it also declines to fire on the species it was never trained to recognize. We therefore evaluated rejection of the 1,891 auxiliary recordings of 28 other Indian endemic species, which are realistic hard distractors for any deployment in India. Confidence thresholding on the closed-set probe performed poorly. At the validation-calibrated threshold of 0.61, which retained 97.4% of target recordings, 93.7% of the distractor recordings were also accepted, taking the worst case over folds, and the recording-level area under the ROC curve for target-versus-distractor separation was only 0.746 (window-level AUROC 0.868 ± 0.045 for maximum softmax probability and 0.878 ± 0.065 for the energy score, mean and standard deviation over folds). Deployed against India's actual avifauna, the closed-set model would flood observers with confident false positives.

Adding an explicit background class changed this picture substantially. Trained with windows from 14 auxiliary species and evaluated against the 814 recordings of the 14 species it had never seen, the five-class probe reduced recording-level false acceptance from 93.7% to 13.4% ± 2.5%, while 89.1% ± 3.9% of target recordings remained correctly identified and 8.8% ± 2.1% were sent to the background class. Table 3 summarizes the comparison. The exchange of a modest amount of target recall for a sevenfold reduction in false positives on unseen species is, in our view, the operationally meaningful headline of this work, and it is the trade-off that a monitoring program would actually tune.

**Table 3.** Open-set behaviour at the recording level. MSP denotes the maximum softmax probability of the closed-set probe with the threshold calibrated on validation data only. The background-class model is trained with 14 auxiliary species and evaluated for false acceptance on 14 entirely unseen species. Target retention counts recordings accepted for the MSP row and recordings accepted and correctly identified for the background-class row.

| Approach | Target recordings retained | False acceptance of non-target recordings |
|---|---|---|
| MSP threshold on the closed-set probe | 97.4% | 93.7% (all 1,891 auxiliary recordings, worst case over folds) |
| Explicit background class, species-disjoint | 89.1% ± 3.9% | 13.4% ± 2.5% (814 unseen-species recordings) |

### Noise robustness

Corrupting the test audio with additive white noise before embedding extraction degraded performance gracefully. Recording-level accuracy under the single-seed robustness protocol was 96.9% on clean audio (compared with 97.4% for the seed-averaged headline), 96.4% at 20 dB SNR, 94.3% at 10 dB, 93.3% at 5 dB, and 92.2% at 0 dB, with macro-F1 declining from 0.965 to 0.925 over the same range (Figure 5). The embedding model, having been trained on heterogeneous field recordings, absorbs much of the corruption before it reaches the classifier.

![Figure 5: noise robustness](figures/f4_robustness.png)

*Figure 5. Recording-level accuracy and macro-F1 of the embedding probe as additive white noise increases from none to 0 dB SNR.*

### What the models attend to

The attention weights of the MIL head concentrated on windows containing vocalizations and assigned near-zero weight to ambient segments (Figure 6). Recording-level decisions therefore rest on the target sound rather than on background signatures, and each individual decision can be audited by inspecting which seconds of audio received weight, a property that plain window-voting pipelines do not offer.

![Figure 6: MIL attention](figures/f5_attention.png)

*Figure 6. MIL attention across a ten-minute Bugun Liocichla recording. The upper panel shows the spectrogram and the lower panel the attention assigned to each 3 s window.*

## Discussion

The first conclusion concerns evaluation design. With rare species the volume of data is fixed by history, and what the analyst controls is the independence structure of the evaluation. Recording-level splitting, session-level grouping, selection confined to inner validation, seed replication, and bootstrap intervals cost nothing in data, and under this regime our headline numbers rose relative to the earlier flawed pipeline because the representation improved at the same time. Rigour and performance are therefore compatible, and the widespread assumption that honest evaluation deflates results does not hold when the underlying pipeline was also wasting most of the signal.

The second conclusion concerns foundation models. Frozen bioacoustic embeddings with lightweight probes, trainable on a laptop in minutes, outperformed end-to-end fine-tuning by fourteen percentage points at this data scale, and they retained that performance under session-level splits and heavy added noise. At the same time, three of the four most threatened Indian endemics are absent from the label set of the dominant global recognizer, and India is unlikely to be unique in this respect, because the species that conservation most needs to monitor are exactly the species for which little training audio exists. Probes over frozen embeddings give range-country institutions a practical route to detectors for nationally important species without waiting for global model updates. The accompanying responsibility is honest reporting of what the archives can support. Our Jerdon's Courser analysis shows that a per-class F1 of 0.96 can rest on an archive whose independent evidence for cross-session generalization amounts to one recording, a distinction that is invisible in a metrics table and decisive for how much trust a deployed detector deserves.

The third conclusion is that closed-set accuracy is the wrong headline for deployment. The reduction of false acceptance from 93.7% to 13.4% through a species-disjoint background class matters more than the final two points of closed-set accuracy, and the auxiliary corpus required to achieve it was easier to assemble than the target data itself. We suggest that studies proposing classifiers for conservation deployment routinely report open-set metrics against realistic regional distractors.

Several limitations qualify these results. BirdNET's training data includes Xeno-canto, so for the Forest Owlet, the one target species in its label set, some of our recordings may have contributed to its pretraining and could inflate embedding quality for that species. The other three species cannot have served as labelled positives, and their results stand independent of this concern. Archive recordings are focal and comparatively clean, so performance on soundscape data from autonomous recorders will be lower than reported here, and the additive-noise stress test only approximates that gap. The recording-event grouping is inferred from catalogue adjacency, and true recordist, site, and date metadata obtainable through the archives' interfaces would sharpen it. Finally, with 12 to 75 recordings per class, every metric carries wide intervals, which we have reported throughout.

The analysis also implies concrete priorities for each species. For Jerdon's Courser, the most valuable contribution anyone can now make is the deposition of new, independently recorded material in public archives, including the 2025 reconfirmation recordings, because detector development for this species is ready on the representation side and blocked entirely on data. For the Bugun Liocichla and the Banasura Laughingthrush, the present recordings already support strong within-archive performance, and the next step is validation on autonomous-recorder soundscapes at the known sites. For the Forest Owlet, where acoustic survey is already operational practice, embedding-based detectors can be evaluated directly against BirdNET's native output on existing survey audio.

## Conclusion

This study set out to determine whether the complete public acoustic record of India's four most threatened endemic birds can support reliable automated identification, and to do so under an evaluation design in which no reported number depends on data used for training or model selection. The answer is affirmative for closed-set identification, with 97.4% recording-level accuracy that persists under session-level splits and degrades only mildly under heavy noise, and it is conditional for deployment, where an explicit background class is required before the false-acceptance rate on non-target species becomes tolerable. The pipeline, manifest, and splits are public, so every claim in this paper can be reproduced from the archives and the repository alone.

## Data and code availability

All code, the recording manifest with every Macaulay Library and Xeno-canto catalogue identifier, the split definitions, and the result files are available at https://github.com/srivatsav-kannan/birdSong. Audio is not redistributed, in accordance with archive terms, and the manifest permits exact reconstruction of the dataset from the source archives.

## References

*(to be set in journal format during submission preparation)*

- Athreya, R. (2006). A new species of Liocichla (Aves: Timaliidae) from Eaglenest Wildlife Sanctuary, Arunachal Pradesh, India. *Indian Birds* 2(4): 82–94.
- Bhushan, B. (1986). Rediscovery of the Jerdon's Courser *Cursorius bitorquatus*. *Journal of the Bombay Natural History Society* 83: 1–14.
- Ghani, B., Denton, T., Kahl, S. and Klinck, H. (2023). Global birdsong embeddings enable superior transfer learning for bioacoustic classification. *Scientific Reports* 13: 22876.
- Ilse, M., Tomczak, J.M. and Welling, M. (2018). Attention-based deep multiple instance learning. *Proceedings of the 35th International Conference on Machine Learning*.
- Jeganathan, P., Green, R.E., Bowden, C.G.R., Norris, K., Pain, D. and Rahmani, A. (2002). Use of tracking strips and automatic cameras for detecting Critically Endangered Jerdon's coursers *Rhinoptilus bitorquatus* in scrub jungle in Andhra Pradesh, India. *Oryx* 36(2).
- Kahl, S., Wood, C.M., Eibl, M. and Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. *Ecological Informatics* 61: 101236.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. and Krishnan, D. (2020). Supervised contrastive learning. *Advances in Neural Information Processing Systems* 33.
- Rasmussen, P.C. and Collar, N.J. (1998). Identification, distribution and status of the Forest Owlet *Athene (Heteroglaux) blewitti*. *Forktail* 14: 41–49.
- Robin, V.V., Vishnudas, C.K., Gupta, P., Rheindt, F.E., Hooper, D.M., Ramakrishnan, U. and Reddy, S. (2017). Two new genera of songbirds represent endemic radiations from the Shola Sky Islands of the Western Ghats, India. *BMC Evolutionary Biology* 17: 31.
- SoIB (2023). *State of India's Birds, 2023: Range, trends, and conservation status*. Zenodo.
- Search for Lost Birds (2025). Jerdon's Courser. https://searchforlostbirds.org/birds/jerdons-courser
- IUCN (2025). The IUCN Red List of Threatened Species. https://www.iucnredlist.org
- Xeno-canto Foundation (2025). https://www.xeno-canto.org and Cornell Lab of Ornithology, Macaulay Library, https://www.macaulaylibrary.org
