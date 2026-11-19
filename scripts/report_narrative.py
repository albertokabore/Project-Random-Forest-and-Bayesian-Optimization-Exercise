"""Prose for the Word report. Numbers quoted here were checked against the saved
results (metrics.json, review_verification.json, corrected/corrected_results.json)."""

from __future__ import annotations

import json
from pathlib import Path

from meddx.priors import CLINICAL_COEF_PRIORS

_REPORTS = Path(__file__).resolve().parents[1] / "reports"
_M = json.loads((_REPORTS / "metrics.json").read_text())
_RF = _M["rf_search"]["bo_best_params"]

TBD = "TBD"

TEXT: dict = {}

# ----------------------------------------------------------------- summary
TEXT["executive_summary"] = [
    "This report documents a diagnostic tool that estimates the probability of coronary artery disease (CAD) for a "
    "patient from symptoms, examination and exercise-test results. It was built and validated on 920 real patients "
    "from four hospitals (UCI Heart Disease dataset). Two approaches were developed. The first is a Bayesian "
    "logistic-regression model in PyMC that starts from published pre-test probabilities and updates them with each "
    "patient's data. The second is a Random Forest whose hyperparameters were chosen by Bayesian Optimization.",
    "**Accuracy.** On 184 held-out test patients, the refined Bayesian model (M2) correctly classified **84.2%**, "
    "with ROC-AUC **0.899** (95% CI 0.85–0.94), sensitivity **89.2%** and specificity **78.0%**. The "
    "Bayesian-optimised Random Forest reached AUC **0.890** and 83.7% accuracy. Both far outperformed the published "
    "pre-test table used on its own (AUC 0.399). On hospitals the models had never seen, accuracy fell to 77% for M2 "
    "(pooled AUC 0.851), mainly because the high-prevalence Swiss and VA hospitals are harder to predict.",
    "**What helped and what did not.** Bayesian Optimization found a better Random Forest configuration than random "
    "search with the same budget (cross-validated log loss 0.417 vs. 0.423). Bayesian updating worked as intended: "
    "accuracy rose and uncertainty shrank as batches of patients arrived. The literature prior, used as published, "
    "made predictions worse, because this referral cohort contradicts it: asymptomatic patients had the highest, not "
    "the lowest, disease rate. After refinement the model recovered, but a control model without any literature "
    "knowledge performed equally well. The measurable gain came from modelling differences between hospitals, not "
    "from the medical priors.",
]
TEXT["summary_rows"] = [
    ["Test-set accuracy of the tool", "M2: 84.2% accuracy, AUC 0.899, sensitivity 89.2%, specificity 78.0% (184 patients)."],
    ["Random Forest + Bayesian Optimization", "AUC 0.890, accuracy 83.7%, sensitivity 89.2%; AUC +0.010 over the untuned forest (not statistically significant)."],
    ["Bayesian Optimization vs. random search", "Best 5-fold CV log loss 0.4169 vs. 0.4228 with 50 trials each (one seeded run)."],
    ["Does prior medical knowledge help?", "Not measurably. As published it hurt (M1); refined it was neutral; an ablation without priors matched M2."],
    ["Bayesian updating with new data", "Test AUC 0.68 → 0.86 → 0.88 over four batches; uncertainty fell at every step."],
    ["Unseen hospital (external validation)", "Pooled out-of-hospital AUC: M2 0.851, RF 0.813, M1 0.794, literature 0.376."],
    ["Are model differences significant?", "No pair of the leading models differs significantly on this test set (paired bootstrap)."],
]

# ----------------------------------------------------------------- 1
TEXT["introduction"] = [
    "A healthcare AI start-up wants a diagnostic tool that estimates the likelihood of disease from patient symptoms "
    "and medical history. The tool should improve on what clinicians already know: it should start from prior medical "
    "knowledge and update its predictions as new patient data become available. This report documents the "
    "development and validation of such a tool for one disease, coronary artery disease (CAD), using real patient "
    "records from four hospitals.",
    "Two complementary approaches were built. The first is a Bayesian logistic-regression model written in PyMC. It "
    "encodes published pre-test probabilities and clinical knowledge as prior distributions, and its posterior is "
    "updated as labelled patients arrive. The second is a Random Forest classifier whose hyperparameters were "
    "selected by Bayesian Optimization. Both were validated against the actual angiographic diagnoses of patients "
    "they had never seen, including patients from a hospital that was left out of training entirely.",
]
TEXT["research_questions"] = [
    "How accurately can the tool predict angiographically confirmed CAD from routinely recorded symptoms and test results?",
    "Does integrating prior medical knowledge (published pre-test probabilities and clinical priors) improve predictions?",
    "Does Bayesian Optimization find better Random Forest hyperparameters than random search with the same budget?",
    "Do predictions improve, and does uncertainty shrink, as new patient data are incorporated by Bayesian updating?",
    "How well does the tool transfer to a hospital whose patients it has never seen?",
]
TEXT["scope"] = (
    "The target is binary: whether any major coronary vessel has more than 50% narrowing on angiography. The tool "
    "does not predict future heart attacks, other heart diseases, or several diseases at once. Only the real UCI "
    "records were used; no synthetic or fabricated patients were added. This is an educational exercise, not a "
    "clinically validated medical device."
)

# ----------------------------------------------------------------- 2
TEXT["data_intro"] = (
    "The data are the UCI Machine Learning Repository's Heart Disease dataset (dataset 45). It pools patients referred "
    "for coronary angiography at four institutions: Cleveland Clinic Foundation (USA), the Hungarian Institute of "
    "Cardiology (Budapest), University Hospitals of Zurich and Basel (Switzerland), and the VA Medical Center (Long "
    "Beach, USA). The four processed files were downloaded from UCI and verified byte for byte against the official "
    "archive. Together they give 920 records and 509 positive diagnoses. Disease prevalence differs sharply between "
    "hospitals, from 36% in Hungary to 93% in Switzerland, so the hospital itself is an important source of variation."
)
TEXT["variables"] = [
    ["age", "Age in years"], ["sex", "1 = male, 0 = female"],
    ["cp", "Chest-pain type: typical angina, atypical angina, non-anginal pain, asymptomatic"],
    ["trestbps", "Resting systolic blood pressure (mm Hg)"], ["chol", "Serum cholesterol (mg/dL)"],
    ["fbs", "Fasting blood sugar > 120 mg/dL"], ["restecg", "Resting ECG: normal, ST-T abnormality, LV hypertrophy"],
    ["thalach", "Maximum heart rate achieved during exercise"], ["exang", "Exercise-induced angina"],
    ["oldpeak", "Exercise-induced ST depression relative to rest"],
    ["slope", "Slope of the peak-exercise ST segment: up, flat, down"],
    ["ca", "Number of major vessels (0–3) coloured by fluoroscopy"],
    ["thal", "Thallium scan: normal, fixed defect, reversible defect"],
    ["**num → disease**", "**Target:** angiographic diagnosis; disease = 1 when num > 0 (>50% narrowing)"],
]
TEXT["data_prep"] = [
    "**Missing values.** “?” entries were treated as missing. Cholesterol or blood pressure recorded as 0 is "
    "physiologically impossible and was also treated as missing: every Swiss cholesterol value (123) and 49 VA values.",
    "**No patients were dropped.** Each model receives a missing-value indicator for every feature that had missing "
    "values in training. Numeric and binary values are imputed with training-set medians and modes, learned inside the "
    "model pipeline so that no information leaks from test patients.",
    "**Encoding.** Categorical variables were one-hot encoded against a reference level; continuous variables were "
    "standardised for the Bayesian model so that prior standard deviations have a common scale.",
    "**Train/test split.** An 80/20 split stratified jointly on hospital and diagnosis gave 736 training and 184 test "
    "patients (102 with CAD, 82 without). Test labels were not used in any model fit or hyperparameter search.",
]

# ----------------------------------------------------------------- 3
TEXT["priors_intro"] = (
    "Step 2 of the assignment asks for prior probabilities taken from the medical literature. Three layers of prior "
    "knowledge were encoded, from the population level down to individual test results."
)
TEXT["df_text"] = (
    "Diamond and Forrester (1979, New England Journal of Medicine) tabulated the probability of angiographic CAD by "
    "age decade, sex and chest-pain type, and it remains the classic pre-test probability table for this question. "
    "Its three inputs are recorded for every UCI patient, so each patient receives a literature pre-test probability "
    "(Table 5). For the 35 patients younger than 30 or older than 69, the nearest age band was used; this is a "
    "modelling assumption."
)
TEXT["beta_text"] = (
    "At the hospital level, prevalence was modelled with a Beta prior centred on the average Diamond–Forrester "
    "probability of that hospital's patients. Its strength is equivalent to 10 patients, so it guides the estimate "
    "only until real data arrive. Each prior was then updated with the hospital's training diagnoses using the "
    "conjugate Beta–Binomial rule. The posterior moved strongly away from the literature prior at every hospital, and "
    "each posterior 95% interval contained the prevalence later observed in that hospital's test patients (Table 6)."
)
TEXT["clinical_priors_text"] = (
    "For test results, directional priors on the log-odds scale encode established clinical expectations: exercise ST "
    "depression, exercise-induced angina, a reversible perfusion defect and a flat ST slope all make CAD more likely. "
    "Each prior has a standard deviation of 0.5, so it favours a direction but lets the data overrule it. The signs "
    "come from clinical knowledge; the magnitudes are moderate judgements, not effect sizes measured in a specific study."
)
TEXT["clinical_prior_rows"] = [[k, f"{v.mu:+.1f}", f"{v.sigma:.1f}", v.rationale] for k, v in CLINICAL_COEF_PRIORS.items()]
TEXT["conflict_text"] = (
    "Comparing the literature prior with the data exposed a strong conflict (Figure 2). The literature expects "
    "asymptomatic patients to be the lowest-risk group (8.7% on average for this cohort); in the data they are the "
    "highest-risk group (79.2%). Conversely, typical angina is expected at 86.7% but observed at 41.7%. Used on its "
    "own as a predictor, the literature table ranks patients worse than chance (test ROC-AUC 0.399). A plausible "
    "contributor is selection: every UCI patient was referred for angiography, so an asymptomatic patient was "
    "presumably referred because of other abnormal findings. The data do not record referral reasons, so this "
    "explanation cannot be confirmed. The conflict directly motivated the refinement step described in Section 6."
)

# ----------------------------------------------------------------- 4
TEXT["bayes_intro"] = (
    "Step 3 asks for a Bayesian model that updates the probability of disease given new symptom data. The model is "
    "a Bayesian logistic regression with a Bernoulli likelihood, written in PyMC 5 (the maintained successor of PyMC3, "
    "with the same modelling interface):"
)
TEXT["bayes_terms"] = (
    "Here PreTest is the Diamond–Forrester probability. The coefficient γ has prior N(1, 0.25): with γ = 1 and every "
    "other term zero, the model reproduces the literature exactly, and the data decide how far to move away from it. "
    "β holds the effects of test results, with the clinical priors of Table 7. u is a hospital effect with partial "
    "pooling: each hospital's intercept is drawn from a shared distribution with spread τ, and a hospital never seen "
    "in training receives a draw from that distribution, so its predictive uncertainty widens accordingly."
)
TEXT["model_rows"] = [
    ["M0 weak", "None: age, sex, chest pain are ordinary features", "N(0, 1) on all", "Pooled", "Baseline without literature knowledge"],
    ["M1 literature", "Diamond–Forrester predictor, γ ~ N(1, 0.25)", "Clinical priors (Table 7)", "Pooled", "Integrate the literature as published"],
    ["**M2 refined**", "Diamond–Forrester predictor", "Clinical priors + learned chest-pain deviations (adaptive shrinkage)", "Hierarchical hospital intercepts", "Correct the prior–data conflict and hospital differences"],
    ["M2 weak (ablation)", "None", "N(0, 1) on all", "Hierarchical hospital intercepts", "Isolate the value of the literature priors"],
]
TEXT["sampling_text"] = (
    "Posteriors were sampled with the No-U-Turn Sampler (NUTS): four chains of 1,000 draws after 1,000 tuning steps. "
    "All main-split fits had maximum R-hat ≤ 1.004, bulk effective sample size above 2,600, zero divergences, and "
    "no PSIS Pareto-k value above 0.7."
)
TEXT["rf_text"] = [
    "The Random Forest pipeline applies the same leakage-safe preprocessing and then a scikit-learn "
    "RandomForestClassifier. Eight hyperparameters were tuned by Bayesian Optimization using Optuna's Tree-structured "
    "Parzen Estimator (TPE). After 10 random start-up trials, TPE models which settings tend to give good and bad "
    "scores and proposes the setting with the most favourable ratio. This uses the results of earlier trials to "
    "decide where to search next, which is what distinguishes Bayesian Optimization from grid or random search.",
    "The objective was the mean 5-fold stratified cross-validated log loss on the training data. Log loss is a "
    "proper scoring rule, so it rewards calibrated probabilities as well as correct ranking, which matters for a "
    "diagnostic tool. Random search with the same 50-trial budget and seed served as the baseline. Bayesian "
    "Optimization reached a best CV log loss of 0.4169, against 0.4228 for random search (Figure 3).",
]
TEXT["search_space_rows"] = [
    ["n_estimators", "100–800 (step 50)", str(_RF["n_estimators"])],
    ["max_depth", "2–20", str(_RF["max_depth"])],
    ["min_samples_split", "2–30", str(_RF["min_samples_split"])],
    ["min_samples_leaf", "1–20", str(_RF["min_samples_leaf"])],
    ["max_features", "sqrt, log2, or fraction 0.2–1.0", str(_RF["max_features"])],
    ["max_samples", "0.5–1.0", f"{_RF['max_samples']:.3f}"],
    ["criterion", "gini, entropy, log_loss", str(_RF["criterion"])],
    ["class_weight", "None, balanced, balanced_subsample", str(_RF["class_weight"])],
]
TEXT["validation_design"] = [
    "**Held-out test set (Step 4).** All models were scored once on the 184 test patients against their actual "
    "angiographic diagnosis, at a 0.5 decision threshold.",
    "**Metrics.** ROC-AUC (discrimination) with a 2,000-resample bootstrap 95% interval; accuracy, sensitivity, "
    "specificity, PPV and NPV (clinical error rates); Brier score and log loss (probability quality).",
    "**Paired comparisons.** Model differences were tested by resampling the same patients for both models, since "
    "overlapping individual confidence intervals do not by themselves show that two models are equivalent.",
    "**External validation.** Leave-one-hospital-out: each model was trained on three hospitals and tested on the "
    "fourth. Random Forest hyperparameters were re-tuned inside every fold, so the held-out hospital never influenced them.",
    "**Model comparison on training data.** Pareto-smoothed importance-sampling leave-one-out cross-validation (PSIS-LOO).",
]

# ----------------------------------------------------------------- 5
TEXT["test_intro"] = (
    "Table 10 reports how accurately each model predicted the actual diagnosis of the 184 test patients. The values "
    "were reproduced independently by refitting every model."
)
TEXT["test_interpret"] = (
    "Every data-driven model was far better than the literature table alone (ROC-AUC 0.876–0.899 against 0.399). "
    "The refined Bayesian model M2 had the highest AUC (0.899), the highest accuracy (84.2%, tied with the default "
    "forest), sensitivity of 89.2% and the best probability scores (Brier 0.120, log loss 0.397). In clinical terms, "
    "at the 0.5 threshold M2 missed 11 of the 102 patients with CAD and raised a false alarm for 18 of the 82 "
    "without. The Bayesian-optimised forest reached AUC 0.890 with the same sensitivity. Its probabilities were "
    "slightly less well calibrated than the Bayesian models' (expected calibration error 0.085 against 0.037–0.062)."
)
TEXT["paired_intro"] = (
    "Because the test set is small, it matters whether differences between models exceed what chance alone would "
    "produce. Table 11 shows paired bootstrap differences, computed by resampling the same patients for both models. "
    "These comparisons come from the corrected re-run, which refitted the Bayesian models with a stricter sampler "
    "setting (target acceptance 0.95); their point estimates may differ from Table 10 in the third decimal."
)
TEXT["paired_interpret"] = (
    "None of the differences between the four best models is statistically distinguishable on this test set: every "
    "95% interval for the AUC difference includes zero. Two results stand out. First, Bayesian Optimization improved "
    "the Random Forest's AUC by about 0.010 and its log loss by about 0.010, but these intervals also include zero. "
    "Second, and most informative, the ablation removes all literature knowledge from M2 and loses essentially "
    "nothing (Δ AUC +0.0004, interval −0.010 to +0.010). Adding the literature exactly as published (M1 vs. M0) "
    "lowered AUC by 0.019 and worsened log loss by 0.041, an interval that only just includes zero."
)
TEXT["patients_intro"] = (
    "For an individual patient, the tool starts from the literature pre-test probability and updates it with the "
    "patient's examination and test results, giving a post-test probability with a 94% credible interval (Table 12)."
)
TEXT["patients_interpret"] = (
    "Test results moved the probabilities a long way from the literature starting point. A 53-year-old man with "
    "atypical angina starts at 58.9% but, given his test results, ends at 8.7%; he did not have CAD. Three "
    "asymptomatic men start at 12.3% and end between 86% and 98%; all had CAD. The update is not always toward the "
    "correct answer: a 51-year-old woman without CAD rises from 8.4% to 37.4%, still below the 0.5 threshold. The "
    "credible interval tells a clinician how certain each estimate is under the model; how well those intervals "
    "cover true risk has not been validated."
)
TEXT["loho_intro"] = (
    "A tool deployed at a new hospital meets patients unlike those it was trained on. To test this, each model was "
    "trained on three hospitals and evaluated on the fourth. Two problems found in the original analysis were "
    "corrected before this run (Appendix B). First, the Random Forest hyperparameters are now re-tuned inside every "
    "fold with the same 50-trial Bayesian Optimization budget, so the held-out hospital never influences them. "
    "Second, M2 now draws one shared effect per posterior draw for the unseen hospital, as the hierarchical model "
    "requires. Bayesian fits used target acceptance 0.95, with an automatic refit at 0.99 if any divergence occurred."
)
TEXT["loho_interpret"] = (
    "All models lost accuracy when moved to a new hospital, which is the main risk for deployment. Cleveland and "
    "Hungary, with moderate prevalence, transferred well (AUC 0.86–0.91). Switzerland and VA Long Beach, where 93% "
    "and 75% of patients had CAD and most exercise-test results were missing, were much harder (AUC 0.70–0.76). The "
    "Swiss result rests on only 8 patients without CAD, so it is very uncertain. M2 had the best pooled AUC (0.851) "
    "and the lowest patient-weighted Brier score (0.161). Averaged over the four hospitals, M2 and the Random Forest "
    "were close (mean AUC 0.812 vs. 0.807). At the high-prevalence hospitals, M2's sensitivity stayed high (82–87%) "
    "but its specificity fell to 35–50%: it flagged most patients there as diseased. The corrections changed the "
    "results very little. The Random Forest's per-hospital AUC moved by at most 0.008 once tuning was nested, and "
    "M2's by at most 0.003 once the hospital effect was shared. The original conclusions therefore hold, but now rest "
    "on a valid procedure. Every final fit converged: maximum R-hat ≤ 1.01, bulk ESS ≥ 1,549 and zero divergences. "
    "One fit, M2 with Hungary held out, needed the stricter 0.99 setting after a single divergence at 0.95."
)

# ----------------------------------------------------------------- 6
TEXT["iteration_intro"] = (
    "Step 5 asks for the model to be refined by adjusting priors and likelihood functions. The models were developed "
    "in sequence, each answering what validation of the previous one had shown. M0 established a baseline with weak "
    "priors. M1 added the literature as published, and was worse. The prior–data conflict of Section 3.4 explained "
    "why. M2 therefore kept the literature predictor but let the data learn chest-pain deviations from it, shrunk "
    "toward zero unless the evidence insists, and added hospital intercepts for the large prevalence differences. "
    "Finally, an ablation (M2 weak) kept M2's hospital structure but removed all literature knowledge, to isolate "
    "what the priors themselves contribute. All models use a Bernoulli likelihood with a logit link; for a binary "
    "diagnosis this is the natural choice, and the refinement was made through the priors and the model structure."
)
TEXT["loo_interpret"] = (
    "Cross-validation on the training data tells the same story as the test set. M1, the literature as published, "
    "is clearly worst: 33.2 ELPD below the best model, about four standard errors. M2 recovers fully from M1's "
    "deficit, which shows that the refinement step worked. However, the ablation, which has M2's hospital structure "
    "but no literature knowledge, scores slightly higher than M2 (3.1 ELPD, standard error 3.0) and receives most of "
    "the stacking weight (0.76). The improvement from M1 to M2 therefore comes from modelling hospital differences "
    "and letting the data override the chest-pain prior, not from the medical priors themselves. In this cohort the "
    "literature knowledge, once corrected, is neutral: it costs nothing but adds no measurable accuracy. "
    "All Pareto-k values were below 0.43, so the LOO estimates are reliable."
)
TEXT["m2_learned"] = [
    "**γ = 0.38** (94% HDI 0.14–0.64): the literature predictor is kept, but at about a third of its published weight.",
    "**Asymptomatic deviation = +2.63** (HDI 1.25–4.01) on the log-odds scale: the data strongly correct the "
    "literature for asymptomatic patients in this cohort. This is a conditional model coefficient, not a measured "
    "causal referral effect.",
    "**Test-result effects agree in sign with the clinical priors:** reversible perfusion defect +1.27, exercise "
    "angina +0.92, flat ST slope +0.78, vessels on fluoroscopy +0.68 per SD, exercise ST depression +0.37 per SD.",
    "**Hospitals differ substantially** (τ = 1.24); Switzerland has the highest baseline risk (+1.39).",
]
TEXT["sequential_intro"] = (
    "To simulate the tool in service, model M1 started from its prior and received the 736 training patients in four "
    "batches of 184. After each batch, the posterior, approximated by a multivariate normal distribution, became the "
    "prior for the next batch. The model was scored on the fixed test set after every update (Table 17, Figure 7)."
)
TEXT["sequential_interpret"] = (
    "Test ROC-AUC rose from 0.680 with prior knowledge alone to 0.862 after the first 184 patients, then levelled off "
    "at 0.876. Posterior uncertainty fell at every step, from a mean coefficient SD of 0.72 to 0.38. The last batch "
    "did not improve every metric: AUC moved from 0.8763 to 0.8759 and log loss from 0.443 to 0.446, so more data do "
    "not guarantee a better score on a fixed test set. After all four updates, the sequential posterior's test "
    "predictions differed from a single fit on all the data by at most 0.019 (correlation 0.9998). This shows the "
    "approximate updating scheme can absorb new cases without refitting from scratch. The demonstration used M1; the "
    "update interface does not yet support M2's hierarchical structure."
)

# ----------------------------------------------------------------- 7-9
TEXT["discussion"] = [
    "**The tool predicts CAD well within the hospitals it was trained on.** All data-driven models reached ROC-AUC "
    "near 0.88–0.90 and accuracy near 81–84%, with sensitivity close to 90%. The variables that mattered most were "
    "consistent across both families of model: chest-pain type (especially being asymptomatic), ST slope, "
    "exercise-induced angina, exercise ST depression, thallium perfusion defects and vessels seen on fluoroscopy. "
    "These agree with clinical knowledge of exercise testing, which supports the face validity of the models.",
    "**Prior knowledge must match the population it is applied to.** The central lesson is the prior–data conflict. "
    "The Diamond–Forrester table describes patients evaluated for chest pain, but the UCI patients were all referred "
    "for angiography, a setting in which an asymptomatic patient is likely to have other strong evidence of disease. "
    "Applying the literature without adjustment made the model worse. Bayesian modelling made the conflict visible "
    "and correctable: the learned slope γ = 0.38 shows the data trusting the literature at about a third of its "
    "published weight, and the asymptomatic deviation (+2.63 on the log-odds scale) quantifies the correction.",
    "**Bayesian Optimization is efficient, but its gain here was modest.** It reached a lower cross-validated loss "
    "than random search and kept improving after random search stalled. On the test set the tuned forest's AUC "
    "improved by 0.010, but this is within sampling noise, and accuracy at the 0.5 threshold fell by one patient. "
    "Random Forests are fairly robust to their hyperparameters, so large gains were not expected.",
    "**Bayesian updating delivers what the start-up asked for.** The model began from prior knowledge alone "
    "(AUC 0.68), improved sharply with the first 184 patients, and its uncertainty kept shrinking as more arrived. "
    "Each update needs only the previous posterior and the new batch, so the tool can learn from new cases without "
    "retraining on the whole history.",
    "**Choosing between the models.** The Bayesian M2 and the tuned Random Forest are statistically "
    "indistinguishable on the test set. M2 has practical advantages for a clinical tool: interpretable effects, an "
    "explicit treatment of differences between hospitals, the best behaviour on unseen hospitals by Brier score, and "
    "a credible interval for every patient. The Random Forest needs fewer modelling assumptions and is a useful "
    "independent benchmark.",
]
TEXT["limitations"] = [
    "**Small test set.** 184 patients (102 with CAD) give wide confidence intervals; differences of about 0.01–0.02 "
    "in AUC cannot be resolved.",
    "**Historical, referral-based cohorts.** The records were donated to UCI in 1988 and come from patients referred "
    "for angiography. Performance in primary care or screening, and with modern tests such as CT angiography or "
    "troponin, is unknown.",
    "**Missing data concentrated by hospital.** The number of coloured vessels (ca) is 96–99% missing and the "
    "thallium result 42–90% missing outside Cleveland, so missingness partly identifies the hospital.",
    "**Prior magnitudes are judgements.** The signs of the clinical priors follow clinical knowledge, but their sizes "
    "and the Beta prior's strength of 10 patients are modelling choices; a full prior-sensitivity analysis was not performed.",
    "**Literature table transportability.** The Diamond–Forrester categories may not match the UCI chest-pain coding "
    "exactly, and ages outside 30–69 use the nearest band.",
    "**Uncertainty intervals are model-based.** The 94% credible intervals have not been checked for coverage against "
    "true patient-level risk.",
    "**Ethics and safety.** The data are de-identified public research records. The tool is not a medical device; any "
    "clinical use would require prospective validation, threshold selection with clinicians, bias assessment across "
    "sex and age groups, and regulatory review. A model like this should support, not replace, clinical judgement.",
]
TEXT["conclusion"] = (
    "A diagnostic tool combining Bayesian modelling and a Bayesian-optimised Random Forest was developed and "
    "validated on real patients from four hospitals. It predicts angiographic CAD with about 84% accuracy and "
    "AUC 0.90 on held-out patients, and about 77% accuracy (pooled AUC 0.85) at hospitals it has never seen. Bayesian "
    "updating improves the model as new patients arrive, and Bayesian Optimization finds a better forest than random "
    "search. Prior medical knowledge was integrated and tested honestly: as published it conflicted with this referral "
    "population and hurt accuracy; once refined it was neutral. The largest gains came from modelling the differences "
    "between hospitals. The main recommendations follow."
)
TEXT["recommendations"] = [
    "**Adopt M2 as the primary research model** and keep the Bayesian-optimised Random Forest as an independent "
    "benchmark; flag patients on whom the two disagree for review.",
    "**Calibrate priors to the target population** before deployment, for example with a pre-test table derived from "
    "referral patients, and run a prior-sensitivity analysis.",
    "**Recalibrate at each new hospital.** Use the hierarchical model's partial pooling with a small local sample, "
    "since baseline risk differs greatly between sites.",
    "**Choose the decision threshold with clinicians.** At 0.5 the tool misses about 1 in 9 patients with CAD; "
    "screening use would justify a lower threshold, at the cost of more false positives.",
    "**Extend sequential updating to the hierarchical model** and validate it prospectively on new patients.",
    "**Validate on contemporary data** with modern diagnostic tests before any clinical use.",
]
TEXT["references"] = [
    'Abril-Pla, O., Andreani, V., Carroll, C., et al. (2023). PyMC: A modern and comprehensive probabilistic programming framework in Python. *PeerJ Computer Science, 9*, e1516.',
    'Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD Conference*, 2623–2631.',
    'Attia, J. R., Sibbritt, D. W., Ewald, B. D., et al. (2004). Generating pre-test probabilities: A neglected area in clinical decision making. *Medical Journal of Australia, 180*, 449–454.',
    'Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems, 24*.',
    'Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.',
    'Detrano, R., Jánosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology, 64*, 304–310.',
    'Diamond, G. A., & Forrester, J. S. (1979). Analysis of probability as an aid in the clinical diagnosis of coronary-artery disease. *New England Journal of Medicine, 300*, 1350–1358. https://doi.org/10.1056/NEJM197906143002402',
    'Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X',
    'Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing, 27*, 1413–1432.',
]

# ----------------------------------------------------------------- appendices
TEXT["rubric_note"] = (
    "The detailed grading rubric was not available when this report was prepared. The report is therefore organised "
    "around the seven steps in the assignment brief, and the table below shows where each step is addressed."
)
TEXT["alignment_rows"] = [
    ["1. Data collection", "Section 2", "920 real patients from four hospitals (UCI), verified against the official archive; documented cleaning; leakage-safe imputation."],
    ["2. Prior knowledge integration", "Section 3", "Diamond–Forrester pre-test probabilities; Beta prevalence priors per hospital with conjugate updating; directional clinical priors."],
    ["3. Model development", "Section 4", "Bayesian logistic regression in PyMC (four variants); Random Forest tuned by Bayesian Optimization (Optuna TPE) vs. random search."],
    ["4. Validation", "Section 5", "Held-out test set vs. actual diagnosis; bootstrap CIs; paired comparisons; calibration; leave-one-hospital-out with nested tuning."],
    ["5. Iteration", "Section 6", "M0 → M1 → M2 driven by validation and the prior–data conflict; prior ablation; PSIS-LOO; sequential Bayesian updating."],
    ["6. Reporting", "This report", "Process, findings, accuracy, limitations and recommendations; reproducibility details in Appendix C."],
]
TEXT["corrections_intro"] = (
    "An independent review of the first version of the analysis reproduced all main test-set results, but found three "
    "problems in the hospital-holdout experiment and several over-stated interpretations. All three problems were "
    "fixed and the experiment was re-run; the results in Section 5.4 are the corrected ones."
)
TEXT["corrections_rows"] = [
    ["Random Forest hyperparameters for the hospital holdouts were tuned on data that included the held-out hospital (tuning leakage).",
     "Nested tuning: a fresh 50-trial Bayesian Optimization inside each three-hospital training fold.",
     "Per-hospital AUC changed by at most 0.008."],
    ["M2 drew a separate random hospital effect for every patient of an unseen hospital.",
     "One shared effect per posterior draw, seeded by hospital; new tests check shared effects and row-order invariance.",
     "Per-hospital AUC changed by at most 0.003."],
    ["Two M2 holdout fits had sampling divergences that were not reported.",
     "Stricter sampler (target acceptance 0.95, refit at 0.99 on any divergence); diagnostics reported for every fit.",
     "All final fits: zero divergences, R-hat ≤ 1.01."],
    ["The contribution of the medical priors had not been isolated from M2's other changes.",
     "Added an ablation model with M2's structure and no literature knowledge, plus paired bootstrap tests.",
     "Showed that the priors add no measurable accuracy."],
    ["Several claims were over-stated (e.g. a decisive best model, exact sequential updating, AUC rounded twice).",
     "Claims restated to match the evidence; unrounded values used.",
     "Conclusions are more cautious and fully supported."],
]
TEXT["reproducibility"] = [
    "**Code:** Python package src/meddx/ (data, priors, preprocessing, Bayesian models, Random Forest, evaluation, plots).",
    "**Analysis:** notebooks/medical_diagnosis_system.ipynb (original executed run); scripts/run_corrected_analysis.py "
    "(corrected hospital holdouts, ablation and paired comparisons; results in reports/corrected/).",
    "**Verification:** scripts/verify_report.py independently refits the main models; tests/ contains 28 automated "
    "tests, including checks that an unseen hospital's effect is shared by its patients and independent of row order.",
    "**Report:** scripts/build_word_report.py builds this document directly from the saved result files.",
    "**Random seed:** 42 throughout; data split stratified on hospital × diagnosis.",
]
