# Final assignment and presentation audit

**Albert Kabore - PhD student in Artificial Intelligence**  
**Review date:** November 18, 2026

The report and notebook address the six substantive assignment steps using the UCI CAD example. All requested presentation changes are present. The saved main and corrected results agree with recomputation from individual predictions. The project is a single-disease research study; a general multi-disease system, clinical readiness and a clear benefit from medical priors are not established.

## Assignment coverage

The numbered line containing only "diagnoses" completes the validation sentence; it is not an additional analytical step.

| Requirement | Verified coverage | Limit |
| --- | --- | --- |
| Data collection | Four source datasets, 920 records, documented cleaning, 736/184 split and authentic raw-file hashes. | Public historical records; limited medical histories; unique patients cannot be established. |
| Prior knowledge | Published age/sex/symptom probability table, separate Beta prevalence updating and clinical coefficient priors. | One CAD outcome; prior strengths are assumptions; no demonstrated accuracy gain from medical priors. |
| Model development | PyMC Bayesian logistic models, Random Forest and TPE optimization; conditional patient predictions and labelled-data updates. | PyMC is a successor to the example PyMC3 library. Parameter updating and conditioning on symptoms are distinguished. |
| Validation | Original held-out metrics, all six bootstrap AUC intervals, paired comparisons, and corrected hospital evaluations verified. | Repeated test inspection and four historical sites limit generalisation. |
| Iteration | M0/M1/M2 refinements, companion comparison, PSIS-LOO and approximate M1 sequential updates. | The companion changes predictors and shrinkage; it is not a strict prior-only ablation. Bernoulli likelihood retained. |
| Reporting | Editable Word report, Markdown report, notebook, references and process/accuracy discussion. | Detailed grading rubric was not supplied; exact rubric compliance cannot be certified. |

## Earlier requests

| Request | Result |
| --- | --- |
| Albert Kabore, PhD student in Artificial Intelligence | Present in report, Word file and notebook author details. |
| Introduction and natural academic prose | Present; unsupported clinical and causal claims are avoided. |
| Observations after each step | Six blocks, each exactly 50 words across two bullets, before the next step begins. |
| Final insights, recommendations and conclusion last | Present; the conclusion is the final body section. Under the latest instruction, References follows the conclusion in the report and notebook. |
| Remove screenshot paragraph | Removed from notebook Step 1; its title remains. |
| Markdown subtitles for every notebook visualization | All nine saved figures are labelled; the paired comparison figures share a caption cell with two explicit subtitles. |
| Remove watermarks | The academic Word edition contains no watermark objects. Chart titles and page numbers are retained. |
| Preserve truthful results | Original numerical evidence and all 20 notebook code cells with execution outputs preserved. |

## Corrections from this audit

- Literature-only accuracy: **41.8%**, based on 77/184 correct classifications.
- M1 negative predictive value: **79.7%**, based on 63/79 negative predictions.
- Report point estimates now use the independently reproduced full-precision values, avoiding a second rounding of four-decimal source summaries.
- Added the missing `python-docx` dependency for reproducible Word exports.
- Made the single-disease scope, limited medical histories and retained Bernoulli likelihood explicit.

## Verification results

- **Raw data provenance and numerical evidence: passed.** All five raw-file hashes match the previous official-archive verification; saved numeric evidence files are unchanged.
- **Counts, split and prevalence updating: passed.** 920 records; 509 positive diagnoses; split 736/184; 102 positive test records; four Beta updates reproduced.
- **Original main metrics and all six AUC intervals: passed.** All point metrics, binned calibration errors and 2,000-resample AUC intervals recomputed from saved individual predictions.
- **Corrected main, hospital and pooled results: passed.** All stored corrected point metrics recomputed from individual predictions; four outer folds each used 50 RF trials.
- **Paired comparison interpretation: passed.** Both reported paired AUC intervals reproduced and include zero; equivalence and prior-only benefit are not claimed.
- **All earlier notebook and writing requests: passed.** Albert Kabore, PhD student in Artificial Intelligence; introduction; six 50-word observation blocks; final insights; recommendations; conclusion as the final body section followed by References; pictured paragraph removed; all nine saved figures have Markdown subtitles.
- **Notebook integrity: passed.** All 20 original code cells and their saved outputs retained; no saved exception outputs; notebook schema valid.
- **Word document, formatting and watermark objects: passed.** Current APA edition has valid Word ZIP/XML; correct author; seven tables, four figures; all six 50-word blocks; 13 references after the conclusion; black text and white tables; no watermark objects detected. The revised edition has a distinct filename because the earlier named Word document was open.
- **Rounding corrections: passed.** Literature-only accuracy is 77/184 = 41.8%; M1 NPV is 63/79 = 79.7%; percentages now formatted from full-precision results.
- **Report rebuild dependencies: passed.** python-docx added to requirements and optional reporting dependencies; Python source parses.
- **Automated model checks: passed in the preceding numerical review.** 28 tests passed in 54.26 seconds, including Bayesian fitting/updating and unseen-hospital effect checks. Model tests were not repeated for the subsequent writing and reference revision.
- **Published prior table and dataset definitions: passed.** All 32 table entries match the published Attia appendix reproduction; UCI confirms the 13-predictor, angiographic target definitions. The original 1979 full paper was not independently accessed.

## Source checks

Raw files retain the hashes recorded when they were compared byte for byte with the official [UCI archive](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease). All 32 implemented literature-table values also match the appendix in [Attia et al.](https://www.mja.com.au/journal/2004/180/9/generating-pre-test-probabilities-neglected-area-clinical-decision-making). The [PyMC development team](https://www.pymc.io/blog/v4_announcement.html) documents the successor package and rename.

## Verification limits

This final audit recomputed scores and bootstrap intervals from saved predictions and ran the complete 28-test suite, including the smaller Bayesian fit/update test. It did not repeat the full optimization searches or every full-budget MCMC fit. Original main fits had been independently reproduced earlier in the project. Authentic archive copies do not independently verify every upstream clinical measurement. Exact rubric compliance remains unverified until the detailed rubric is available.

## Submission files

- [Word report with APA references](Albert_Kabore_Medical_Diagnosis_System_Report_APA.docx)
- [Markdown report](REPORT.md)
- [Notebook](../notebooks/medical_diagnosis_system.ipynb)
- [Machine-readable audit results](FINAL_AUDIT.json)

## Academic presentation update

The current academic edition uses black text for headings, paragraphs, citations, captions, and table entries. All table cells are white, with black borders. The prose uses formal impersonal wording. Six 50-word observation blocks, figure subtitles, source citations, and the final conclusion are retained. Numerical evidence and the original 20 notebook code cells with their saved outputs are unchanged.

The earlier named Word file could not be replaced; the revised academic edition is saved separately. A [black-on-white notebook reading view](Albert_Kabore_Medical_Diagnosis_Notebook.html) is also available.

## Output-based interpretation revision

Each of the 12 observation bullets, six final insight bullets, and six recommendations identifies numerical evidence and its interpretation or proposed analytical response. Comparisons distinguish original and corrected fits. Derived quantities include a 1.4% relative CV log-loss improvement, 712/920 correct hospital predictions, and 33 false positives among 51 negative VA records. These quantities were checked against saved results and predictions. Source files and derivations are recorded in [output_analysis_trace.json](output_analysis_trace.json). All six observation blocks remain 50 words.

## APA references and academic report revision

The report now includes an abstract, introduction, six substantive methods/results sections, discussion and final insights, recommendations, and conclusion. References immediately follows the conclusion, reflecting the latest requested ordering. The Word document uses 12-point Times New Roman, double-spaced body text, one-inch margins, header page numbers, and black horizontal table rules on white backgrounds. Figure and table numbers are bold, with italic titles. References begin on a new page and use alphabetical order, double spacing and 0.5-inch hanging indents.

All 13 references are cited in the report and notebook. Their full authors, publication details and relevance were checked against journal, repository or author-maintained records linked in [reference_verification.json](reference_verification.json). The 1979 article's metadata and abstract were verified through PubMed; its numerical probability table was checked through Attia et al.'s published reproduction. No claim is made that every cited full text was independently accessed.

The current writing revision passed [structural and formatting checks](academic_presentation_check.json), including citation coverage, reference placement, Word XML colours, table fills, observation counts and preservation of all 20 code cells with their outputs. Numerical evidence hashes also match the preceding commit. Rendered pagination was not inspected, and full model fitting was not repeated for this revision.
