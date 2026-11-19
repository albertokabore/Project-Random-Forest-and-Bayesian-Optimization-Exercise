# Scientific and factual review

**Historical review of the original run.** A later corrected hospital analysis is now complete in
`reports/corrected/corrected_results.json`. The current `reports/REPORT.md` and named Word report include those
completed results and explain the limits of the companion weak-prior comparison. The issues and pending-work
statements below describe the original review stage, rather than the current implementation status.

Review date: November 18, 2026. Scope: the supplied assignment instructions and the local report, notebook,
Python source, raw data and saved results. The grading rubric was not provided.

## Verdict

The raw data are authentic copies of UCI's distributed files, and the main test results are reproducible.
There is no evidence of invented patient records or fabricated main-model performance in the material checked.
The original report nevertheless contained incorrect statements, unsupported causal explanations and overstated
validation claims. Those interpretations have been corrected in `REPORT.md` and the README.

The hospital-holdout experiment still needs code corrections and a new run before its comparisons can support
conclusions. This review does not certify the project as a clinically validated diagnosis system.

## Evidence independently verified

- Downloaded UCI's official archive and compared the four processed files and `heart-disease.names` byte for
  byte. All five match. Source URL, archive hash and individual SHA-256 hashes are in `data_verification.json`.
- Recomputed 920 records and 509 positive diagnoses: Cleveland 303/139 positive, Hungary 294/106, Switzerland
  123/115 and VA Long Beach 200/149. The split contains 736 training and 184 test records; the test labels are
  102 positive and 82 negative.
- Confirmed the zero-value cleaning counts: Swiss cholesterol 123, VA cholesterol 49 and VA resting blood
  pressure one. Treating these as missing is a documented cleaning choice, not an observed laboratory result.
- Recomputed all four Beta prevalence updates and their reported credible intervals. The numbers agree.
- Independently fitted both RF configurations and all three main Bayesian models. Every test metric agrees
  with the original rounded four-decimal value. The literature-only row also agrees.
- Recomputed the selected RF configuration's five-fold CV log loss: **0.41693899930074235**, exactly matching
  the saved best TPE score. This checks the selected configuration's score, not the entire optimization history.
- The saved test and hospital CSV tables agree with `metrics.json`. The notebook has outputs for every code
  cell and no saved exception outputs. Sampling warnings are still present in its stream outputs.
- Fast test suite: **25 passed, one slow test deselected**. Separately reran the main Bayesian fits at the original
  budget: four chains, 1,000 tuning and 1,000 retained draws per chain.

The independently recomputed test results are:

| Model | ROC-AUC (6 decimal places) | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Literature only | 0.399390 | 0.418478 | 0.459031 | 1.427193 |
| Bayesian M0 | 0.895026 | 0.836957 | 0.124063 | 0.405141 |
| Bayesian M1 | 0.875897 | 0.809783 | 0.138622 | 0.445570 |
| Bayesian M2 | 0.899450 | 0.842391 | 0.120126 | 0.396755 |
| Default RF | 0.880081 | 0.842391 | 0.131134 | 0.431559 |
| Tuned RF | 0.890005 | 0.836957 | 0.129626 | 0.421198 |

Use `review_verification.json` and `review_test_predictions.csv` as the precise machine-readable evidence.
The review's model summaries are saved separately as `review_M0_weak_summary.csv`,
`review_M1_literature_summary.csv` and `review_M2_refined_summary.csv`.

## Issues requiring changes before relying on hospital holdouts

### 1. RF hyperparameter selection includes the held-out hospital

In `scripts/build_notebook.py:237`, the search uses the pooled training set, which contains records from all
four hospitals. At line 321 the hospital-holdout loop reuses `bo.best_params`. Thus, although each final forest
is fitted on three hospitals, labels from the held-out hospital already influenced hyperparameter selection.

Required correction: tune within each outer fold using only its three training hospitals, with all preprocessing
inside the inner folds. Report the search budget and seed for every fold. Alternatively evaluate a configuration
specified independently of all four sites. Existing RF hospital scores must remain labelled exploratory.
This follows the validation principle in
[scikit-learn's nested CV documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### 2. M2 draws a different random hospital effect for each patient at the same unseen site

`src/meddx/bayes.py:287` loops over records and calls `rng.normal(0.0, post.tau_site)` separately for each
unseen-site record at line 299. Under the stated hierarchical model, all patients at one hospital should share
the same hospital effect within a posterior draw. A direct deterministic check using two identical profiles
assigned to the same unseen hospital produced different effect vectors (maximum discrepancy 2.8212 in log odds
over 100 draws).

Required correction: draw once per unique unseen hospital per posterior draw, then broadcast to all its rows.
Check identical-profile equality and prediction invariance to row permutation. Independent patient draws have
the intended single-patient marginal distribution in the infinite-sample limit, but the current implementation
loses the joint hospital dependence and introduces row-dependent Monte Carlo differences. Do not describe this
as proof that the reported marginal AUC is grossly biased; the size of the metric change requires a rerun.

### 3. Two M2 hospital fits have divergences omitted from the report

The saved notebook's hospital-holdout output records **four divergences** for M2 holding out Switzerland and
**four divergences** for M2 holding out VA Long Beach. The original blanket convergence claim referred only to
the three pooled main fits and omitted these warnings.

Required correction: investigate the two fits, consider a higher acceptance target or reparameterization, and
save R-hat, ESS, divergences and other relevant diagnostics for each fold. The independent main-model refits
have zero divergences, unrounded maximum R-hat values 1.002675 (M0), 1.003104 (M1), 1.002624 (M2), and no
pointwise PSIS Pareto-k values above 0.7. Those checks do not establish convergence of the other fits.

## Incorrect or unsupported interpretations corrected in the report

| Original interpretation | Supported interpretation |
|---|---|
| Prior knowledge improved accuracy | M1 performs worse than M0. M2 adds both feature terms and hospital structure, so a prior-only benefit is not isolated. |
| M2 is decisively best | M2 has the highest main test AUC point estimate. Paired M2 minus M0 AUC is 0.004424, with 95% bootstrap interval −0.007594 to 0.016166. |
| Overlapping AUC intervals prove differences are within sampling error | Marginal interval overlap is not a paired comparison. The review computed the paired interval directly. |
| BO improves accuracy | RF AUC and log loss improve, but threshold accuracy decreases from 155/184 to 154/184 correct records. |
| BO beats random search generally | One saved seeded run has a lower best CV loss. Repeated searches and their full histories are needed for a general comparison. |
| Every batch improves the model | Final AUC declines from 0.8763 to 0.8759 and log loss rises from 0.4428 to 0.4459. Mean coefficient uncertainty still shrinks. |
| All example risks move toward the correct diagnosis | For the 38-year-old without CAD, estimated risk rises from 0.052 to 0.121. The binary decision remains correct. |
| General-population prior explains referral bias | The historical table is not a general-population prevalence estimate. Selection is plausible, but the processed records do not establish the cause of the symptom reversal or individual referral reasons. |
| M2 measures the size of referral bias | Its chest-pain deviation is a conditional model coefficient, not an identified causal bias effect. |
| M2 has the best calibration in every sense | M2 has the lowest Brier/log loss; M0 has lower binned ECE. Individual credible-interval coverage is untested. |
| M1 is worse than M0 by 30.0 ± 7.8 ELPD | The reported 29.96 difference and 7.8 SE compare M1 with M2. M0 minus M1 is about 28.35; its paired SE was not saved. |
| LOO weight 0.59 establishes M2's probability of being best | This is an ArviZ stacking weight, not such a posterior probability. |
| The test set was used only once | It is also inspected for prevalence, selected cases and sequential performance; later hospital fits use the full dataset. The main fits/search do exclude its labels. |
| Sequential updating is available for the recommended M2 model | The supported Gaussian posterior interface excludes M2. The streaming demonstration uses M1 and a preprocessor fitted on all training covariates. |
| Posterior updating is exact | Each intermediate posterior is approximated as Gaussian. Close final predictions in one run do not prove exactness or general reliability. |
| There are 14 predictors | There are 13 predictors and one target. |
| All 920 records are independently verified unique patients | Two identical-record pairs occur in the distributed data; identifiers are absent. Neither pair crosses this saved split. |
| M2 AUC rounds to 0.900 at three decimals | Recomputed AUC is 0.8994500239120038, which rounds directly to 0.899. Rounding the saved four-decimal 0.8995 a second time gives a misleading 0.900. |

The recommendation to use these models as a clinical diagnostic engine was replaced with a recommendation for
further research evaluation. No independent clinical benefit, screening threshold or safety claim was measured.

## Medical priors and factual sources

The Diamond–Forrester paper is real: Diamond GA, Forrester JS. *N Engl J Med.* 1979;300:1350–1358.
[Publisher record](https://www.nejm.org/doi/abs/10.1056/NEJM197906143002402).
The complete original article was not accessible through the publisher in this review. All 32 encoded table
entries were checked against the table reproduced in the appendix of
[Attia et al., Med J Aust 2004;180:449–454](https://www.mja.com.au/journal/2004/180/9/generating-pre-test-probabilities-neglected-area-clinical-decision-making).
The implementation clamps 35 records outside ages 30–69 to an endpoint decade; these are extrapolation
assumptions, not directly published age-specific estimates. Exact symptom-definition equivalence also needs
checking before treating the historical table as transportable to UCI's codes.

The clinical coefficient means and SDs, and the prevalence prior's strength of 10, are modelling assumptions.
They are not fabricated observations, but must not be represented as measured or published effect estimates.
The original `fbs` rationale called a fasting-glucose threshold >120 mg/dL diabetes; these are different
variables. The original `ca` rationale described calcification although UCI documents a count of major vessels
coloured by fluoroscopy. Low peak heart rate was also overstated as a diagnosis of chronotropic incompetence
or ischaemia. Those rationale strings have been corrected without changing any prior values. Their precise
coefficient values still require sensitivity analysis rather than a claim to be literature estimates. An
unsourced “about 300x faster” Numba comment was removed, and the TPE description was made more precise.

Dataset schema and recommended attribution were checked against the
[UCI dataset page](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease) and matching local documentation.
UCI's recommended citation year is 1989; the donation date is 1988. The
[Detrano et al. paper](https://pubmed.ncbi.nlm.nih.gov/2756873/) is genuine, but its published full hospital
cohorts have different sizes from the processed files; its cohort totals should not replace the directly
verified UCI counts. Exact collection dates for every processed record were not established, so the original
unqualified “1981–1988” statement was removed.

PyMC is the renamed successor to PyMC3, as documented by its
[development team](https://www.pymc.io/blog/v4_announcement.html). This project was verified with installed
PyMC 5.28.5, not a claim about the latest release. Optuna TPE is a legitimate Bayesian optimization method, but
it models parameter densities conditional on better/worse trials, rather than directly fitting the response
surface in the way a Gaussian-process surrogate does; see the
[official TPE algorithm description](https://optuna.readthedocs.io/en/v3.0.5/reference/samplers/generated/optuna.samplers.TPESampler.html).
Hyperparameter optimization and disease-probability Bayesian inference are separate operations in this project.

## Alignment with the supplied assignment

| Requirement | Assessment |
|---|---|
| Patient data from multiple hospitals | Met using four existing UCI datasets; no new hospital collection was performed. |
| Literature-based prior probabilities | Demonstrated for one binary disease, with assumptions and population mismatch that need sensitivity analysis. |
| Bayesian model in Python | Implemented with PyMC 5; the instructions give PyMC3 as an example, not an exclusive requirement. |
| Random Forest and Bayesian Optimization | Implemented with RF and TPE, alongside a random-search comparison. Main selected-RF score reproduced. |
| Validation against actual diagnoses | Main held-out comparison reproduced. Independent hospital generalisation remains unresolved. |
| Refinement of priors and likelihood | Priors and model structure change across M0/M1/M2; all retain a Bernoulli likelihood. No alternative likelihood was tested, and the assignment does not require inventing one if unnecessary. |
| Bayesian updating with new data | Implemented for labelled M1 batches and separate Beta prevalence updates. Prospective or M2 streaming support is not established. |
| Reporting under the rubric | A report exists; detailed rubric compliance cannot be assessed because the rubric was not supplied. |

## Reproduction and limits

Run `python scripts/verify_report.py --bayesian` in the project environment. It writes review artifacts
separately from the original analysis. The installed versions and hashes of the original metrics and model
source files used by this rerun are recorded in `review_verification.json`. The full RF search, hospital fits
and sequential experiment were inspected but not independently repeated in this review. Their recorded
values are not presented as independently reproduced results.

Raw data, original numerical outputs, notebook code and execution outputs were preserved as audit evidence.
The report, README and notebook markdown were corrected; matching markdown changes were made in its generator.
Source-code comments and clinical rationale strings were corrected without changing model calculations or prior
distributions. The verification source hashes describe the files as read at refit time, before these explanatory
text edits. The new-hospital prediction fix, nested tuning and new hospital-results generation remain follow-up
implementation work. None of the original hospital numbers was replaced with an invented corrected score.
