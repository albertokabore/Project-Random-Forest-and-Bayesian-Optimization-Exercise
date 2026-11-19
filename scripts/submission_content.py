"""Shared, evidence-based prose for Albert Kabore's report and notebook.

Observation blocks contain exactly 50 whitespace-separated words, excluding
their heading and bullet markers. Results are read from completed output files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Albert Kabore"
QUALIFICATION = "PhD student in Artificial Intelligence"

from output_analysis import derive_output_analysis

OBSERVATIONS, FINAL_INSIGHTS, RECOMMENDATIONS, OUTPUT_ANALYSIS_TRACE = derive_output_analysis(ROOT)

# Original code-cell positions: captions are presentation only and do not alter
# the executed code or its saved results. The first cell displays two figures.
NOTEBOOK_VISUALIZATION_SUBTITLES = {
    5: "### Figures 1 and 2. Observed prevalence and historical probabilities\n\n"
       "- **Figure 1. Hospital-level comparison.** Observed disease prevalence and average historical probability by hospital.\n"
       "- **Figure 2. Chest-pain comparison.** Observed disease prevalence and average historical probability by symptom category.",
    6: "### Figure 3. Hospital prevalence before and after Bayesian updating",
    10: "### Figure 4. M2 posterior coefficients and clinical prior means",
    11: "### Figure 5. Random Forest optimization: TPE and random search",
    12: "### Figure 6. Tuned Random Forest feature importance",
    14: "### Figure 7. Test-set ROC curves and calibration",
    16: "### Figure 8. Original hospital-holdout ROC-AUC comparisons",
    17: "### Figure 9. Sequential M1 updating: performance and coefficient uncertainty",
}

INTRODUCTION = """This study evaluates coronary artery disease (CAD) prediction using the UCI Heart Disease datasets. The target is the presence of angiographic disease. The analysis concerns diagnostic classification and does not evaluate future cardiovascular events or multiple disease outcomes.

Random Forest classification with Bayesian hyperparameter optimization is compared with Bayesian logistic regression. The objectives are to assess predictions against recorded diagnoses, evaluate the incorporation of historical probabilities, and examine sequential updating with labelled patient batches. The data originate from a public research archive; no new patient recruitment or hospital data collection was undertaken."""

CONCLUSION = """Random Forest classification, Bayesian optimization, and Bayesian inference were applied to binary CAD prediction using the UCI Heart Disease datasets. The raw files match the source archive, and the original main results were independently reproduced. Fitted models outperform the historical probability table on this test split, although a clear contribution from medical priors has not been established. Corrected hospital comparisons and approximate sequential updating provide additional evidence of methodological limitations. Independent prospective validation is required before clinical use can be considered."""

REFERENCES = """- Diamond, G. A., and Forrester, J. S. (1979). Analysis of probability as an aid in the clinical diagnosis of coronary-artery disease. *New England Journal of Medicine, 300*, 1350–1358. [Publisher record](https://www.nejm.org/doi/abs/10.1056/NEJM197906143002402).
- Attia, J. R., Sibbritt, D. W., Ewald, B. D., et al. (2004). Generating pre-test probabilities: a neglected area in clinical decision making. *Medical Journal of Australia, 180*, 449–454. [Published article and reproduced probability table](https://www.mja.com.au/journal/2004/180/9/generating-pre-test-probabilities-neglected-area-clinical-decision-making).
- Detrano, R., Jánosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology, 64*, 304–310. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/2756873/).
- Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. [Dataset and recommended citation](https://doi.org/10.24432/C52P4X). The donation date is 1988; UCI's recommended citation year is 1989.
- PyMC development team. [Package rename and model-building API](https://www.pymc.io/blog/v4_announcement.html).
- Optuna documentation. [Tree-structured Parzen Estimator algorithm](https://optuna.readthedocs.io/en/v3.0.5/reference/samplers/generated/optuna.samplers.TPESampler.html).
- scikit-learn documentation. [Nested and non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)."""


def observation_block(step: int) -> str:
    bullets = OBSERVATIONS[step]
    assert sum(len(text.split()) for text in bullets) == 50, (step, [len(text.split()) for text in bullets])
    return "**Observations**\n\n" + "\n".join("- " + text for text in bullets)


def read_results() -> tuple[dict, dict, dict]:
    reports = ROOT / "reports"
    return tuple(json.loads((reports / name).read_text(encoding="utf-8")) for name in
                 ("metrics.json", "review_verification.json", "corrected/corrected_results.json"))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def hospital_table(corrected: dict) -> str:
    models = ["Literature only", "M1 Bayesian", "M2 Bayesian", "RF (nested BO)"]
    rows = [[site] + [f"{fold['metrics'][model]['roc_auc']:.4f}" for model in models]
            for site, fold in corrected["loho"].items()]
    rows.append(["Pooled out of hospital"] + [f"{corrected['loho_summary']['pooled_out_of_hospital'][model]['roc_auc']:.4f}" for model in models])
    return markdown_table(["Held-out hospital", "Literature", "M1", "M2", "RF with nested tuning"], rows)


def ablation_table(corrected: dict) -> str:
    names = {"M0_weak": "M0, weak priors", "M2_refined": "M2, refined", "M2_weak_ablation": "Hierarchical model without literature predictor"}
    return markdown_table(["Corrected main fit", "ROC-AUC", "Accuracy", "Brier", "Log loss"], [
        [label, f"{corrected['main']['metrics'][key]['roc_auc']:.4f}",
         f"{corrected['main']['metrics'][key]['accuracy']:.1%}",
         f"{corrected['main']['metrics'][key]['brier']:.4f}",
         f"{corrected['main']['metrics'][key]['log_loss']:.4f}"] for key, label in names.items()
    ])


def build_markdown() -> str:
    original, verification, corrected = read_results()
    site_table = markdown_table(["Hospital dataset", "Records", "CAD prevalence"], [
        [row["site"], str(int(row["patients"])), f"{row['disease_rate']:.1%}"] for row in original["sites"]
    ])
    labels = ["Literature table alone", "M0 Bayesian, weak priors", "M1 Bayesian, literature priors",
              "M2 Bayesian, refined", "Random Forest, default", "Random Forest, tuned"]
    # Use independently reproduced full-precision point estimates. Formatting
    # already-rounded metrics.json values can change displayed percentages.
    precise = {row["model"]: row for row in verification["test_metrics"]}
    test_records = [{**row, **precise[row["model"]]} for row in original["test_metrics"]]
    test_rows = [[label, f"{r['roc_auc']:.4f} ({r['auc_ci_low']:.4f}–{r['auc_ci_high']:.4f})",
                  f"{r['accuracy']:.1%}", f"{r['sensitivity']:.1%}", f"{r['specificity']:.1%}",
                  f"{r['ppv']:.1%}", f"{r['npv']:.1%}", f"{r['brier']:.3f}", f"{r['log_loss']:.3f}"]
                 for label, r in zip(labels, test_records)]
    test_table = markdown_table(["Model", "ROC-AUC (95% CI)", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier ↓", "Log loss ↓"], test_rows)
    seq_table = markdown_table(["Patients seen", "0", "184", "368", "552", "736"], [
        [label] + [f"{r[key]:.4f}" for r in original["sequential_updating"]]
        for label, key in (("Test ROC-AUC", "test_auc"), ("Test log loss", "test_log_loss"), ("Mean coefficient SD", "mean_posterior_sd"))
    ])
    pair = corrected["main"]["paired_bootstrap"]["M2_minus_M0"]
    ablation_pair = corrected["main"]["paired_bootstrap"]["M2_minus_M2weak_ablation"]
    pooled = corrected["loho_summary"]["pooled_out_of_hospital"]
    search = original["rf_search"]
    params = search["bo_best_params"]
    prevalence_table = markdown_table(["Hospital", "Prior mean", "Posterior mean (95% interval)", "Test prevalence"], [
        [row["site"], f"{row['prior_mean']:.1%}",
         f"{row['posterior_mean']:.1%} ({row['credible_interval_95'][0]:.1%}–{row['credible_interval_95'][1]:.1%})",
         f"{row['test_prevalence']:.1%}"] for row in verification["prevalence_updates"]
    ])
    text = f"""# Medical Diagnosis System: Random Forest and Bayesian Optimization

**Author:** {AUTHOR}  
**Academic status:** {QUALIFICATION}  
**Date:** November 18, 2026

## Introduction

{INTRODUCTION}

## Step 1. Data collection

The four processed UCI files contain 920 records, including 509 positive diagnoses. Each record has 13 predictors and a diagnosis label. The predictors describe age, sex, chest-pain type, resting blood pressure, cholesterol, fasting glucose, resting ECG, maximum exercise heart rate, exercise-induced angina, ST depression, ST slope, vessels coloured by fluoroscopy, and a thallium result. The binary outcome is `disease = (num > 0)`.

{site_table}

The raw files and their documentation match UCI's official archive byte for byte. This confirms their source; it does not independently audit the original hospital measurements. UCI's landing-page total of 303 describes Cleveland, whereas this project pools four processed datasets.

The processed files provide limited clinical history, mainly demographics and test measurements. This implementation addresses one CAD outcome. A system for several diseases would need additional labelled outcomes and suitable priors for each disease.

Question marks are treated as missing. Cholesterol values of zero are set to missing for all 123 Swiss records and 49 VA records; one zero blood-pressure value in the VA data is handled similarly. Continuous and binary imputation statistics are estimated from training data. Missing categorical values are represented by category contrasts and a missingness indicator.

The 80/20 split is stratified by hospital and diagnosis: 736 training records and 184 test records, of which 102 are positive and 82 negative. Two pairs of identical cleaned records occur within the source files; neither pair crosses this saved split. There are no identifiers that would establish whether those pairs represent the same people.

{observation_block(1)}

## Step 2. Prior knowledge integration

The historical Diamond–Forrester table assigns a pre-test CAD probability using age, sex and chest-pain category. All 32 encoded entries were checked against the table reproduced in Attia and colleagues' published article. The implementation maps ages outside 30–69 to an endpoint decade, affecting 35 records. That mapping is an assumption, not a published estimate for those ages.

Historical probabilities differ substantially from observed training prevalence. Among training records, the table assigns asymptomatic cases an average risk of 8.7%, while their observed disease rate is 79.2%. For typical angina, the corresponding values are 86.7% and 41.7%. This discrepancy warrants assessment of transportability; the available variables do not establish its cause.

![Historical prior probabilities and observed training prevalence by chest-pain category.](figures/prior_vs_observed_chest_pain.png)

A separate Beta–Binomial example tracks hospital prevalence. Each Beta prior is centred on the average literature probability for that hospital's training records and has a total strength of 10. Updating adds observed positive and negative diagnoses to the two Beta parameters.

{prevalence_table}

These prevalence posteriors are not fed into the patient-level logistic or Random Forest models. The test-prevalence containment is descriptive, particularly because the split was stratified on diagnosis. It is not a coverage study.

Clinical coefficient priors favour expected directions for selected predictors. Their means and standard deviations are modelling choices, not published effect estimates. The Normal distributions permit either sign. Continuous predictors are standardised; binary and category contrasts remain 0/1. In particular, `fbs` records fasting glucose above 120 mg/dL rather than diagnosed diabetes, and `ca` follows UCI's fluoroscopy vessel-count definition rather than a relabelled calcium score.

{observation_block(2)}

## Step 3. Model development

The Bayesian models use a Bernoulli likelihood with a logit link. M0 uses weak coefficient priors and ordinary predictors. M1 adds a learned coefficient on the literature logit predictor and clinical coefficient priors. M2 also adds chest-pain deviations and partially pooled hospital intercepts. Its general form is:

```
logit P(CAD_i) = intercept + gamma * logit(literature_i)
                + predictors_i * beta + hospital_effect_i
```

PyMC samples the main models with four NUTS chains, each retaining 1,000 draws after 1,000 tuning iterations. The original main fits had zero divergences and favourable saved diagnostics. The corrected analysis uses stricter acceptance settings and records diagnostics for each hospital fit. These diagnostics assess sampling behaviour; model validity requires separate predictive evaluation.

The Random Forest is fitted in a preprocessing pipeline. Optuna's Tree-structured Parzen Estimator (TPE) searches eight hyperparameters using five-fold training-set log loss. TPE uses earlier trial results to propose new settings. This optimization is separate from Bayesian inference about disease probability.

The original search used 50 trials per method. Its best CV log loss was **{search['bo_best_cv_log_loss']:.4f}** for TPE and **{search['random_best_cv_log_loss']:.4f}** for random search. This is one seeded comparison, so it does not establish a reliable advantage across repeated searches.

The selected forest has {params['n_estimators']} trees, maximum depth {params['max_depth']}, minimum leaf size {params['min_samples_leaf']}, minimum split size {params['min_samples_split']}, `max_features = {params['max_features']}`, `max_samples = {params['max_samples']:.6f}`, `class_weight = {params['class_weight']}` and `criterion = {params['criterion']}`. Its test AUC and log loss improve on the default forest, while accuracy decreases by one correctly classified record.

For one Hungarian test record, a 53-year-old man with atypical angina, the literature table assigns a probability of 58.9% and fitted M2 estimates 8.7% conditional on the recorded features; the recorded CAD diagnosis is negative. This is a feature-conditioned prediction. Parameter updating with newly confirmed diagnoses is a separate operation, evaluated in Step 5. Selected examples do not establish overall accuracy.

{observation_block(3)}

## Step 4. Validation against recorded diagnoses

The following table reports the original, independently reproduced test results at a 0.5 threshold. The AUC intervals are percentile bootstrap intervals from 2,000 resamples. They describe uncertainty conditional on these fitted models rather than all training and model-selection variability.

{test_table}

M2 correctly classifies 155 of 184 records. It misses 11 of 102 positive cases and labels 18 of 82 negative cases as positive. The default forest has the same accuracy, but different sensitivity, specificity and probability scores. M2 has the lowest original Brier score and log loss, while M0 has the lowest binned expected calibration error. These measures assess different properties of a prediction.

![ROC curves and binned calibration for four original test predictors. The table identifies these as the original run.](figures/roc_calibration.png)

The corrected analysis repeats the main fits with stricter sampling settings, so small Monte Carlo differences are expected. Its paired comparison gives M2 minus M0 AUC of **{pair['auc_diff']:.4f}**, with a 95% interval from **{pair['auc_diff_95ci'][0]:.4f} to {pair['auc_diff_95ci'][1]:.4f}**. That interval includes zero. It neither establishes a clear advantage nor proves equivalence.

The original hospital experiment reused forest settings selected with records from all four hospitals and sampled unseen-hospital effects independently by patient. A completed corrected run now tunes the forest inside each three-hospital training fold, using 50 TPE trials, and shares one unseen-hospital effect across its patients within each posterior draw. The corrected AUC results are:

{hospital_table(corrected)}

Across the 920 out-of-hospital predictions, M2 has accuracy **{pooled['M2 Bayesian']['accuracy']:.1%}**, pooled AUC **{pooled['M2 Bayesian']['roc_auc']:.4f}** and Brier score **{pooled['M2 Bayesian']['brier']:.4f}**. The nested-tuned forest has accuracy **{pooled['RF (nested BO)']['accuracy']:.1%}**, pooled AUC **{pooled['RF (nested BO)']['roc_auc']:.4f}** and Brier score **{pooled['RF (nested BO)']['brier']:.4f}**. A pooled AUC is not the average of hospital AUCs; it also includes comparisons between hospitals.

![Completed hospital-holdout results with nested Random Forest tuning and a shared unseen-hospital effect.](figures/loho_auc_corrected.png)

The final saved hospital fits have zero divergences, rounded maximum R-hat at most 1.01 and minimum bulk ESS at least 1,549. The M2 Hungary fold was refitted at acceptance target 0.99 after a divergence at 0.95. These are retrospective hospital comparisons within this project, not independent prospective validation. Switzerland has only eight negative cases, and M2 specificity is 50.0% there and 35.3% at VA Long Beach.

{observation_block(4)}

## Step 5. Iteration and Bayesian updating

The progression from M0 to M1 to M2 responds to a practical problem: the literature-based model fits this cohort less well than the weak-prior baseline. M2 allows symptom relationships and hospital differences to adjust. All variants retain the Bernoulli likelihood; the refinements change the predictors, priors and hospital structure.

A Bernoulli likelihood remains appropriate for the binary target. No alternative likelihood was tested, and no claim is made that changing the likelihood improved performance.

The corrected analysis also fits a hierarchical model without the literature predictor. These are its completed main-split results, separate from the original results above:

{ablation_table(corrected)}

The paired AUC difference between corrected M2 and this companion model is **{ablation_pair['auc_diff']:.4f}** (95% interval **{ablation_pair['auc_diff_95ci'][0]:.4f} to {ablation_pair['auc_diff_95ci'][1]:.4f}**). Their hospital structure is shared, but their feature representation and shrinkage differ: the companion model restores ordinary age and sex predictors and does not use adaptive chest-pain shrinkage. This is not a strict prior-only ablation. Similar performance therefore cannot establish that prior knowledge has no effect.

Training-data PSIS-LOO places the companion model about 3.1 ELPD above M2, with a difference standard error of about 3.0. Its stacking weight is about 0.76, which is a predictive-combination weight rather than the probability that it is the true model. Preprocessing remains fixed in these pointwise LOO calculations.

The original sequential experiment uses M1. Its posterior is approximated by a multivariate Normal distribution after each batch of 184 labelled records, then used as the next prior. The initial M1 prediction includes coefficient priors; it is not the literature table alone.

{seq_table}

![Original M1 sequential experiment: test performance and mean coefficient uncertainty.](figures/sequential_updating.png)

The final update slightly reduces AUC and worsens log loss, while mean coefficient uncertainty continues to decrease. Sequential and full-batch M1 predictions differ by at most 0.01855 and correlate at 0.9998. Agreement is observed under the Gaussian approximation; exact posterior equivalence is not established. Scaling was fitted on all training covariates before streaming, making this a retrospective evaluation. M2 does not support this Gaussian update interface.

{observation_block(5)}

## Step 6. Reporting, limitations and reproducibility

The analysis addresses the six substantive components in the supplied assignment instructions. Detailed grading criteria were unavailable; alignment could therefore be assessed only against the assignment brief.

Computational evidence comprises the executed notebook, individual prediction files, model code, and recorded software versions. The notebook retains 20 original code cells and their outputs. Initial hospital results are identified as historical; the subsequent corrected analysis is reported separately to preserve the provenance of both evaluations.

There are several practical limits. These are historical angiography cohorts, not a representative sample of contemporary primary care. Missingness can identify hospitals, and removed identifiers prevent a full patient-overlap audit. The test set has been examined repeatedly during this project. Prior strength sensitivity, prospective calibration, subgroup reliability, predictor availability at deployment and clinical benefit remain unestablished.

The main evidence files are `reports/metrics.json`, `reports/review_verification.json`, `reports/corrected/corrected_results.json` and `reports/data_verification.json`. Corrected hospital predictions are retained in `reports/corrected/checkpoint.json`. The report and notebook distinguish original results from corrected fits rather than combining their estimates silently.

The independent verification record lists Python {verification['python']}, PyMC {verification['versions']['pymc']}, scikit-learn {verification['versions']['scikit-learn']} and Optuna {verification['versions']['optuna']}, along with the remaining package versions. The dependency files specify minimum versions rather than a complete environment lock. Rebuilding the Word document also requires `python-docx`, which is included in the reporting dependencies.

Data definitions, historical probabilities, and methodological choices are documented in the following sources:

{REFERENCES}

{observation_block(6)}

## Final insights

{FINAL_INSIGHTS}

## Recommendations

{RECOMMENDATIONS}

## Conclusion

{CONCLUSION}
"""
    return text


def revise_notebook(notebook):
    """Rebuild explanatory markdown while preserving every code cell verbatim."""
    import nbformat as nbf

    code = [copy.deepcopy(cell) for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 20, "Unexpected notebook structure; inspect before revising."
    original, _, corrected = read_results()
    revised = []

    def md(text: str) -> None:
        text = text.strip()
        cell = nbf.v4.new_markdown_cell(text)
        cell["id"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        revised.append(cell)

    def keep(*indices: int) -> None:
        for index in indices:
            if index in NOTEBOOK_VISUALIZATION_SUBTITLES:
                md(NOTEBOOK_VISUALIZATION_SUBTITLES[index])
            revised.append(code[index])

    md(f"# Medical Diagnosis System: Random Forest and Bayesian Optimization\n\n**Author:** {AUTHOR}  \n**Academic status:** {QUALIFICATION}\n\n## Introduction\n\n{INTRODUCTION}\n\nInitial execution outputs are retained for provenance. Corrected hospital comparisons are reported separately using the completed analysis in `reports/corrected/corrected_results.json`. The accompanying report is `reports/REPORT.md`.")
    keep(0)
    md("## Step 1. Data collection")
    keep(1, 2, 3)
    md(observation_block(1))
    md("## Step 2. Prior knowledge integration\n\nThe Diamond–Forrester table provides historical probabilities by age, sex and chest-pain category. Its 32 entries match the reproduced published table. Clamping ages outside 30–69 affects 35 records and is an assumption. The comparison below shows a large mismatch between the literature table and observed training prevalence. Selection into angiography cohorts is plausible, but its causal role is not established by these records.")
    keep(4, 5)
    md("The Beta–Binomial example separately updates hospital prevalence. Each prior has strength 10 and is centred on the training records' average literature probability. These Beta posteriors do not enter the patient-level models. Test-prevalence containment is descriptive, particularly because the split is stratified by diagnosis.")
    keep(6)
    md("Clinical coefficient means and SD 0.5 are modelling assumptions. They favour directions without fixing the sign. Continuous features are standardised; binary and categorical contrasts are 0/1. The archived rationale table describes `fbs` as diabetes and `ca` as calcification. The source descriptions were subsequently corrected to match UCI's glucose-threshold and fluoroscopy vessel-count definitions; prior values were unchanged.")
    keep(7)
    md(observation_block(2))
    md("## Step 3. Model development\n\n### Bayesian models\n\nM0 uses weak priors. M1 adds the literature logit predictor and clinical coefficient priors. M2 adds chest-pain deviations and partially pooled hospital effects. Each uses a Bernoulli likelihood with a logit link. The saved main run uses four chains with 1,000 tuning and 1,000 retained draws per chain. The summaries below describe that original run, not the later corrected fits.")
    keep(8, 9, 10)
    md("### Random Forest and Bayesian optimization\n\nThe preprocessing pipeline is fitted inside training folds. Optuna TPE searches eight forest settings using five-fold CV log loss, with 50 trials per search method. The original best losses are 0.4169 for TPE and 0.4228 for random search. This single seeded comparison does not establish general search superiority. Hyperparameter optimization is separate from Bayesian updating of disease probabilities.")
    keep(11, 12)
    md(observation_block(3))
    md("## Step 4. Validation against recorded diagnoses\n\nThe following original test results were independently reproduced. The threshold is 0.5, with AUC intervals from 2,000 bootstrap resamples. Original M2 AUC is approximately 0.89945 and accuracy is 84.2%, tied with the default forest. M2 misses 11 positive cases and produces 18 false positives. Repeated inspection of this test set limits claims that it remained untouched throughout development.")
    keep(13, 14)
    md("### Individual examples\n\nThese selected records illustrate prediction rather than provide an additional validation sample. A 94% credible interval describes uncertainty under the fitted model; it is not a 94% guarantee that a diagnosis is correct. A case without CAD whose risk rises from 5.2% to 12.1% shows that the model does not always move toward the observed outcome.")
    keep(15)
    md("### Original hospital experiment: historical outputs\n\nThe archived hospital evaluation used forest settings influenced by all four hospitals. Its M2 predictor drew independent hospital effects for patients at the same unseen site, and two fits recorded four divergences each. These issues were addressed in a subsequent completed analysis. Historical code and outputs are retained for provenance; corrected comparisons are reported in the following subsection.")
    keep(16)
    md("### Completed corrected hospital analysis\n\nSource: `reports/corrected/corrected_results.json`. The forest is tuned with 50 TPE trials inside each outer training fold. M2 shares each posterior hospital-effect draw across that hospital's patients. All final saved fits have zero divergences; rounded maximum R-hat is at most 1.01, and minimum bulk ESS is at least 1,549. The M2 Hungary fold required a refit at acceptance target 0.99. These retrospective comparisons still require independent prospective validation.\n\n" + hospital_table(corrected) + "\n\nPooled M2 AUC is 0.8509 and accuracy is 77.4%; pooled AUC differs from the average of hospital AUCs. Switzerland contains only eight negative cases, and weak specificity at some sites remains a concern.")
    md(observation_block(4))
    md("## Step 5. Iteration and Bayesian updating\n\nA later hierarchical model removes the literature predictor and clinical coefficient means. It restores ordinary age and sex predictors and changes chest-pain shrinkage, so it is not a strict prior-only ablation. Similar performance does not prove that medical priors have no effect. These completed corrected-fit results are separate from the original output cells:\n\n" + ablation_table(corrected))
    md("### Sequential updating of M1\n\nThe following original experiment updates M1 with four labelled batches. Each posterior is approximated as multivariate Normal for the next prior. The preprocessor was fitted using all training covariates, so this is a retrospective fixed-feature simulation. Mean coefficient SD decreases, but final AUC slightly declines and log loss worsens. M2 does not support this Gaussian update interface.")
    keep(17, 18)
    md("Sequential and full-batch M1 predictions differ by at most 0.01855 and correlate at 0.9998. This is close agreement in one approximate run, not exact posterior equivalence.\n\n" + observation_block(5))
    md("## Step 6. Reporting and reproducibility\n\nThe following cell records the initial export procedure. Initial and corrected outputs are stored separately. Source-data hashes, individual predictions, model code, and completed result files document the analytical process. Detailed grading criteria were unavailable; alignment could therefore be assessed only against the supplied assignment brief.")
    keep(19)
    md("### Sources and interpretation limits\n\nThese are historical angiography records, with substantial missingness and no identifiers for a full patient-overlap audit. Prospective calibration, subgroup reliability, predictor availability and clinical benefit remain unestablished. The report distinguishes original predictions from stricter corrected fits.\n\n" + REFERENCES + "\n\n" + observation_block(6))
    md("## Final insights\n\n" + FINAL_INSIGHTS)
    md("## Recommendations\n\n" + RECOMMENDATIONS)
    md("## Conclusion\n\n" + CONCLUSION)
    notebook.cells = revised
    notebook.metadata["authors"] = [{"name": AUTHOR, "role": QUALIFICATION}]
    notebook.metadata["report_revision"] = {"date": "2026-11-18", "author": AUTHOR,
                                           "observations_words_per_step": 50, "preserved_code_cells": 20}
    assert [dict(cell) for cell in notebook.cells if cell.cell_type == "code"] == [dict(cell) for cell in code]
    return notebook
