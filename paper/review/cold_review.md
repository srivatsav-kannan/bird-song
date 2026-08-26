# Cold peer review (independent agent, manuscript + cover letter only)

Reviewer context: given only manuscript.docx and cover_letter.docx, no access to the
repository or any project history, with permission to verify claims against public
sources. It extracted and inspected all six figures, recomputed Table 3 from the
Figure 3 confusion matrix, verified the McNemar discordant counts, checked the
BirdNET V2.4 label file, and spot-checked the references.

**Recommendation: Major revision.**

## Verified strengths

Internal consistency checks all reconciled (per-class metrics recomputed from the
confusion matrix, McNemar counts, totals). The negative result is honestly reported.
The open-set analysis is the right analysis. The motivating claim was independently
verified: the BirdNET V2.4 label file (6,522 classes) contains *Athene blewitti* but
not the other three species. All spot-checked references are accurate, and the
August 2025 Jerdon's Courser acoustic reconfirmation is corroborated by independent
coverage.

## Major concerns

**M1. The stated repository does not exist or is not public.** The Data and code
availability statement and the cover letter rest on reproducibility, and the URL
returned HTTP 404 at review time. Provide a public or reviewer-accessible
repository, ideally archived on Zenodo with a DOI.

**M2. Cross-archive duplicate leakage is not ruled out.** SHA-256 catches only
bit-identical files. The same cut uploaded to both archives at different encodings,
gains, or trims defeats it, and Forest Owlet (46 ML + 29 XC) is exactly where dual
deposits are plausible. Recording events are defined within a single archive, so a
cross-archive near-duplicate pair can straddle a partition even under the stricter
scheme. The embedding-similarity duplicate check was run only for the single
Jerdon's Courser XC file. Run it (or spectrogram cross-correlation or perceptual
hashing) for every cross-archive pair and report the results.

**M3. The recording-event proxy is weak, untested, and the paper's own results hint
it is not binding.** Catalogue numbers are assigned at upload, not at recording, so
one expedition uploaded in batches spans far more than 1,000 numbers and unrelated
recordings uploaded together fall within 1,000. Both archives expose recordist,
date, and location metadata, which are the correct grouping variables. Two
observations sharpen the concern: the stricter scheme yields higher accuracy (98.9%
vs 97.4%), which is what one expects when the grouping fails to isolate the true
dependence structure, and per-species event counts are never reported. Report
events per species in Table 1 and rebuild the strict scheme on recordist by date by
location.

**M4. Channel and recording-chain confound, visible in Figure 2.** The Jerdon's
Courser panel shows band-limited broadband noise, stationary tonal bands near 9 to
10 kHz, and a hard mid-window dropout, the signature of the 2001 to 2002 recording
chain rather than the bird. Since all Jerdon's Courser training material comes from
that session, a classifier can achieve perfect recall by recognising the channel.
Suggested tests: (a) run the trained model on channel-matched negatives such as
silent or ambient segments from the same ML session and check whether they are
called Jerdon's Courser, (b) apply band-limiting, resampling, and EQ augmentation
and report sensitivity, (c) discuss the confound explicitly. White-noise robustness
is not evidence against this confound because white noise does not disturb
narrowband channel signatures.

**M5. BirdNET pretraining contamination for Forest Owlet is acknowledged but not
quantified, and the embedding choice is not justified against alternatives.** Many
Forest Owlet recordings here were plausibly labelled training positives for the
embedding extractor itself. The clean fix is an ablation with a second foundation
model, particularly Perch, which the paper's own key citation (Ghani et al. 2023)
reports outperforms BirdNET embeddings for transfer learning. Citing Ghani et al.
as motivation while using only BirdNET embeddings is an unexplained inconsistency.

**M6. The open-set evaluation does not test the deployment-relevant distractors,
and key details are missing.** The 28 auxiliary species are never listed, nor the
selection criterion, nor how the 14/14 split was made. A single split can drive the
13.4% figure, so repeated splits with spread are needed. The distractors that
matter in the field are sympatric common species and acoustically similar
congeners: Bugun Liocichla vs *Liocichla phoenicea* and *L. ripponi*, Banasura
Laughingthrush vs *Montecincla cachinnans* and *M. fairbanki*, Forest Owlet vs
Spotted and Jungle Owlets. Report false acceptance per distractor species. If the
residual 13.4% is concentrated in congeners, that is the single most important
number in the paper and it is currently invisible. There is also no non-bird or
ambient category and no evaluation on continuous soundscape audio, so the reported
rate does not translate to a false positive rate per hour of ARU audio.

**M7. The four-class task is likely easy, and the headline accuracy should not be
the headline.** An owl, a courser, and two babblers from different mountain ranges
are acoustically far apart, and the raw-embedding UMAP shows four essentially
pre-separated clusters. The genuinely novel content is the open-set result and the
archive-structure finding. The Summary and Conclusions generalise ("reliable
classifiers can be developed... to aid in the monitoring of endangered birds")
beyond what a study with zero field audio supports. Recalibrate claims throughout.

**M8. Statistical treatment ignores clustering.** Percentile bootstrap over
recordings assumes exchangeability, which the paper's own session argument denies.
Use a cluster bootstrap over recording events, which will widen the CIs, and note
the same dependence for McNemar. At minimum report cluster-bootstrap CIs alongside.

**M9. Missing baseline: BirdNET itself on Forest Owlet.** Since BirdNET already
classifies Forest Owlet, the operational question of whether the bespoke probe
beats off-the-shelf BirdNET where both exist is unasked. Nearly free and directly
supports or undermines the advice to range-country institutions.

## Minor concerns

1. Audio decoded to 48 kHz but CNN spectrograms specified at 32 kHz. If the CNN
   branch resamples, say so.
2. Clean accuracy 96.9% in the noise section vs 97.4% in Table 2 is never
   explained.
3. Recording-level AUROC (0.746) is far below window-level (0.868 to 0.878).
   Aggregation usually improves separation. State how recording-level open-set
   scores were computed and explain why aggregation hurts.
4. Table 4 mixes reporting bases: worst-case-over-folds in one row, mean and SD in
   the other. State everywhere what the plus-minus denotes.
5. The Summary reports 93.7% without the worst-case qualifier used in Results.
6. Windows per recording are extremely uneven (one 10-minute Bugun recording is
   roughly a third of that species' audio). Note this and consider capping or
   weighting.
7. Window labels are weak labels: the energy selector keeps any sound 3 dB above
   the floor, including non-target species in focal recordings. Acknowledge and
   ideally quantify, possibly using MIL attention to estimate label noise.
8. State the Red List version and the endemism criterion. Borderline cases the
   reader will think of (Great Indian Bustard, Lesser Florican) deserve a sentence.
   Define "no usable recordings" since the Great Nicobar Serpent-eagle has some
   archive material.
9. No sensitivity analysis for the catalogue-gap threshold of 1,000.
10. With 12 Jerdon's Courser recordings, inner validation for that class rests on
    about 2 files. Acknowledge.
11. Class weighting for the 5-class model is unstated.
12. The ensembling subsection adds little. Consider cutting or moving to
    supplementary.
13. "Unrecorded from 2008": BirdLife and rediscovery coverage cite 2009 as the last
    sighting. Anchor the date to a source.
14. Give the Red List version number in the IUCN reference.
15. Spelling drifts between "vocalisation" (text) and "Vocalization" (Figure 1
    panel title).

## Figures and tables

- Figure 1: redundant with Table 1, drop one (keep the table). "ML"/"XC" undefined
  in the caption, y-axes lack labels.
- Figure 2: two of four exemplars are poor. The Bugun panel ends in zero-padding
  and the Jerdon's Courser panel is dominated by channel noise with no clearly
  visible call. Choose exemplars that show the vocalisations. These artefacts are
  also evidence for M4.
- Figure 3: clear and consistent, but the two panels have different totals (193 vs
  182) and the Jerdon's Courser row collapses from 12 to 1. The caption must say
  so.
- Figure 4: the most effective figure. Panel title "After endemic-avifauna
  adaptation" uses terminology found nowhere in the text. Add a UMAP caveat.
- Figure 5: single model, single seed, no uncertainty. Add bootstrap CIs, consider
  adding the CNN.
- Figure 6: title leaks the code identifier "BugunLiocichla". The near-zero-weight
  claim is not verifiable from the plot, bars are sparse relative to the ~390
  windows a 10-minute recording yields, and the vertical black bars are
  unexplained. One recording is anecdote; report attention mass on active vs
  inactive windows across all recordings.
- Table 1: add per-species event counts and the XC quality-grade distribution.
- Table 2: explain "n/a" for ensemble seed SD.
- Table 4: split or harmonise the mixed-meaning column, add per-distractor-species
  breakdown.

## References

All spot-checked entries are accurate. Two gaps: soften or support the superlative
"most widely used" (Pérez-Granados 2023 reviews BirdNET but does not establish the
superlative), and state the BirdNET label-file version and class count (6,522) so
the load-bearing coverage claim can be re-verified.

## Cover letter discrepancies

1. The reproducibility promise is unverifiable while the repository is
   inaccessible.
2. "Six duplicate files" appears in the letter but nowhere in the manuscript.
3. "Raises the number of genuine training instances tenfold": 3,195 windows from
   193 recordings is about 16x, and overlapping windows are not independent
   instances. This overstates in exactly the direction the previous decision
   criticised.
4. "Results hold under both schemes" glosses over the stricter scheme not testing
   11 recordings and reducing Jerdon's Courser to a single test file.

## Line-level nitpicks

1. "of which 93.7% were wrongly accepted": "of which" attaches to "species" rather
   than "recordings", and the worst-case qualifier is dropped.
2. "While rarely spotted in the wild, four such species have sound recordings":
   dangling construction, and "rarely spotted" is vague.
3. "Datasets this small cannot support training conventional deep networks":
   overstated as an absolute, the CNN reached 83.4%. Say "poorly support".
4. "These analyses effectively build... acting as a critical step": "effectively"
   is filler and "critical step" is self-assessment. Delete both.
5. "commonly used by classifier systems such as BirdNET": 3 s is BirdNET's window
   specifically.
6. "several thousand species": give the exact figure, 6,522 classes.
7. "Equal-weight combination is nevertheless the appropriate form of ensembling":
   normative claim without evidence, right after a non-significant gain.
8. "in our judgement it is the most operationally significant result": editorial
   self-ranking, argue it in the Discussion instead.
9. "absorbs much of the corruption": anthropomorphic and mechanistically
   unsupported.
10. "Window-voting pipelines do not offer an equivalent per-decision audit": false
    as stated, window probabilities localise decisions in time too.
11. "every piece of evidence the archive permits is positive": rhetorical flourish
    standing in for n = 1.
12. "97.4% ... persists under session-level splits": the 98.9% is a different,
    smaller subset with one class reduced to one test file. "Persists" papers over
    this.
13. The Summary's closing generalisation should be restricted.
14. "session-level grouping from catalogue structure costs nothing": neither does
    grouping by recordist and date metadata, which is strictly better and was not
    done.
15. 48 kHz vs 32 kHz inconsistency (as Minor 1).
16. "unrecorded from 2008": pin to a citation (as Minor 13).
17. Figure 6 title "BugunLiocichla": code-internal name in a publication figure.
18. Table 4 caption: two rows measure different quantities under one column
    heading.

## Reviewer's summary for the editor

Internally consistent and honestly reported, with a genuinely useful open-set
analysis and a verified motivating gap in BirdNET coverage. But the reproducibility
package is inaccessible despite being the submission's central warrant, leakage and
channel confounds are incompletely controlled, the open-set distractor design does
not answer the field question, and the conclusions outrun a study containing no
field audio. Major revision, with M1 to M6 mandatory.
