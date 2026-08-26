Srivatsav Kannan
Department of Artificial Intelligence and Data Science
Coimbatore Institute of Technology, Coimbatore, India
srivatsavkannan@gmail.com

26 August 2026

The Editor
Bird Conservation International

Dear Editor,

Please consider our manuscript, "Acoustic identification of India's Endangered and Critically Endangered endemic birds using bioacoustic foundation model embeddings", for publication in Bird Conservation International as an original research article.

This work follows our earlier submission BCI-2026-0101, which you declined to send for review while noting that the underlying conservation problem was worth pursuing and that a resubmission addressing the identified methodological concerns would be welcome. Rather than revising that manuscript, we rebuilt the study from the raw recordings upward, and every analysis, number, and figure in the present paper is new. We believe each of the concerns raised in your decision has been resolved.

First, on evaluation. All model selection, including early stopping, now uses inner validation sets only, and performance is reported exclusively on test folds that play no part in training or tuning, under stratified five-fold cross-validation at the level of the unique recording. A second, stricter scheme additionally prevents recordings from the same recording session from appearing on both sides of any partition, and results hold under both.

Second, on data scale and duplication. No audio file is duplicated anywhere in the pipeline. Recordings are instead segmented into three-second vocalisation windows, which raises the number of genuine training instances tenfold, and class imbalance is handled by loss weighting. A content-hash audit also removed six duplicate files present in the archives themselves.

Third, on the ensemble rationale. The models reported here are combined only through equal-weight soft voting, so no ensemble weight is selected on any data and no claim about relative model strength is required to justify one. All statements about model comparisons are supported by paired statistical tests in the text.

Fourth, on internal consistency. Every figure and value in the manuscript is generated directly by the released code from a single results file, and the complete pipeline, recording manifest, and split definitions are public, so each reported number can be reproduced from the archives alone.

Finally, the manuscript has been prepared to the journal's format and length guidance.

Beyond addressing those concerns, we believe the paper now makes a stronger contribution: it shows that only one of India's four most threatened endemic birds is covered by the most widely used global classifier, demonstrates that reliable species-specific classifiers can nevertheless be built from the small public archives, and quantifies the rejection of non-target species that separates a benchmark classifier from a usable field detector.

This manuscript is not under consideration elsewhere, all authors have approved the submission, and we declare no competing interests.

Thank you for your consideration.

Yours sincerely,

Srivatsav Kannan, on behalf of all authors
