"""Generate notebooks/medical_diagnosis_system.ipynb from readable source.

Usage:
    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/medical_diagnosis_system.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "medical_diagnosis_system.ipynb"

cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("md", s.strip()))  # noqa: E731
code = lambda s: cells.append(("code", s.strip()))  # noqa: E731

md("""
# Medical Diagnosis System: Random Forest, Bayesian Optimization and Bayesian Updating

**Scientific review (November 18, 2026):** see `reports/REVIEW.md` and the corrected `reports/REPORT.md`.
The main test metrics were independently reproduced. Hospital holdout results below remain historical,
exploratory outputs: RF tuning includes held-out hospital labels, M2 uses independent unseen-hospital effects
per patient, and two M2 holdout fits have divergences. These issues require corrected code and a new experiment.

**Objective.** Predict the likelihood of coronary artery disease (CAD) from patient symptoms, test results and history.
This exercise evaluates prior medical knowledge and Bayesian updating; it does not establish that priors improve accuracy.

**Data.** UCI Heart Disease (dataset 45). These are 920 patient records from four hospital datasets: Cleveland Clinic, the
Hungarian Institute of Cardiology (Budapest), University Hospitals Zurich and Basel (Switzerland), and the VA Medical Center
in Long Beach. **No synthetic or fabricated records are used.**

| Assignment step | Section |
|---|---|
| 1. Data collection | §1 |
| 2. Prior knowledge integration | §2 |
| 3. Model development (PyMC Bayesian model + Random Forest with Bayesian Optimization) | §3, §4 |
| 4. Validation against actual diagnoses | §5, §6 |
| 5. Iteration: adjusting priors and likelihoods, Bayesian updating | §3 (M0 to M1 to M2), §7 |
| 6. Reporting | §8 and `reports/REPORT.md` |

> This saved analysis uses **PyMC 5**, a successor to PyMC3. The review records the installed versions;
> these dependency files do not lock an identical future environment.
""")

code("""
import json, logging, os, sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm

from meddx import plots
from meddx.config import FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, SITES, TARGET
from meddx.data import load_heart_disease, missingness_summary, train_test_split_stratified
from meddx.priors import (CLINICAL_COEF_PRIORS, clinical_prior_table, diamond_forrester_pretest,
                          diamond_forrester_table, prevalence_prior)
from meddx.bayes import BASELINE, LITERATURE, REFINED, BayesianDiagnosisModel
from meddx.rf import feature_importance, fit_random_forest, optimize_random_forest
from meddx.evaluation import diagnostic_metrics, expected_calibration_error, metrics_table
from meddx.preprocessing import DiagnosisPreprocessor

pd.set_option("display.precision", 3)
RESULTS: dict = {}
# MEDDX_QUICK=1 gives a fast smoke run; the report uses the full settings.
QUICK = os.environ.get("MEDDX_QUICK") == "1"
MCMC = dict(draws=150, tune=150, chains=2) if QUICK else dict(draws=1000, tune=1000, chains=4)
print("PyMC", pm.__version__, "| ArviZ", az.__version__)
""")

# ------------------------------------------------------------------ §1
md("""
## 1. Data collection

The four processed UCI files share 13 predictors and one target. The target is binarised using UCI's
documented presence/absence convention: `disease = (num > 0)`. These are existing angiography-cohort records.

**Cleaning rules (documented in `src/meddx/data.py`):**
* `?` becomes missing.
* `chol == 0` and `trestbps == 0` become missing. These values are physiologically impossible: every Swiss cholesterol is 0.
* Out-of-range category codes become missing.

Imputation happens later, *inside* each model pipeline, using training-set statistics only.
""")

code("""
data = load_heart_disease()
by_site = data.groupby("site", observed=True).agg(patients=(TARGET, "size"), disease_rate=(TARGET, "mean"))
by_site.loc["All"] = [len(data), data[TARGET].mean()]
RESULTS["sites"] = by_site.reset_index().to_dict(orient="records")
by_site
""")

code("""
miss = missingness_summary(data)
miss.loc[:, (miss > 0).any()]
""")

code("""
train, test = train_test_split_stratified(data)
y_test = test[TARGET].to_numpy()
print(f"train: {len(train)} patients, test: {len(test)} patients (stratified on hospital x diagnosis)")
RESULTS["split"] = {"train": len(train), "test": len(test), "test_disease_rate": float(y_test.mean())}
""")

# ------------------------------------------------------------------ §2
md("""
## 2. Prior knowledge integration

**(a) Pre-test probability from the literature.** Diamond & Forrester (1979, *NEJM* 300:1350) tabulated the prevalence
of angiographic CAD by age decade, sex and chest-pain type. Its inputs are complete here, but its population
and symptom definitions are not automatically equivalent to this cohort. Ages outside 30–69 are clamped to
the nearest table band: an extrapolation assumption affecting 35 of 920 records.
""")

code("""
dft = diamond_forrester_table()
dft.pivot_table(index=["age_band", "sex"], columns="chest_pain", values="pretest_pct")[
    ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]]
""")

code("""
train_pt = diamond_forrester_pretest(train)
cmp_site = pd.DataFrame({
    "Observed (training data)": train.groupby("site", observed=True)[TARGET].mean(),
    "Literature pre-test (mean)": pd.Series(train_pt, index=train.index).groupby(train["site"], observed=True).mean(),
})
cp_names = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal", 4: "Asymptomatic"}
cmp_cp = pd.DataFrame({
    "Observed (training data)": train.groupby("cp")[TARGET].mean(),
    "Literature pre-test (mean)": pd.Series(train_pt, index=train.index).groupby(train["cp"]).mean(),
}).rename(index=cp_names)
plots.save(plots.grouped_bars(cmp_site, "Disease prevalence by hospital: observed vs literature prior", "Disease rate"), "prior_vs_observed_site")
fig = plots.grouped_bars(cmp_cp, "Prior-data conflict: asymptomatic patients are the highest-risk group here", "Disease rate")
plots.save(fig, "prior_vs_observed_chest_pain")
RESULTS["prior_vs_observed_cp"] = cmp_cp.round(3).reset_index(names="chest_pain").to_dict(orient="records")
display(cmp_site.round(3)); display(cmp_cp.round(3))
""")

md("""
**Finding: a prior-data conflict.** The literature table expects *asymptomatic* patients to have the **lowest**
probability of disease. In this cohort they have the **highest**. Selection into angiography cohorts is a plausible
contributor, but the processed files do not establish the cause or individual referral reasons. The historical
table is not a general-population prevalence estimate. This mismatch motivates investigating model **M2** in §3.

**(b) Prevalence prior and conjugate Bayesian updating.** Each hospital gets a Beta prior. It is centred on the mean
Diamond-Forrester pre-test probability of that hospital's patients (features only, no labels) and is worth 10 patients.
It is then updated with that hospital's training diagnoses: Beta(a + cases, b + non-cases).
This is a separate prevalence demonstration; its posterior is not passed into the logistic or RF models.
The diagnosis-stratified split makes test-prevalence containment a descriptive check, not interval-coverage validation.
""")

code("""
beliefs, rows = {}, []
for site in SITES:
    tr_s, te_s = train[train.site == site], test[test.site == site]
    prior = prevalence_prior(tr_s)
    post = prior.update(int(tr_s[TARGET].sum()), len(tr_s))
    beliefs[site] = (prior, post)
    lo, hi = post.interval()
    rows.append({"site": site, "prior_mean": prior.mean, "posterior_mean": post.mean, "post_95_low": lo,
                 "post_95_high": hi, "test_prevalence": te_s[TARGET].mean(), "test_in_95_interval": lo <= te_s[TARGET].mean() <= hi})
prev = pd.DataFrame(rows).set_index("site")
plots.save(plots.prevalence_updating(beliefs), "prevalence_updating")
RESULTS["prevalence_updating"] = prev.round(3).reset_index().to_dict(orient="records")
prev
""")

md("**(c) Clinical coefficient priors** on the log-odds scale. Their means favour expected directions, but the Normal distributions permit either sign. Means and SD 0.5 are modelling assumptions, not published effect estimates. Continuous inputs are standardised; binary/category contrasts are 0/1. In the historical printed output, the rationale for `fbs` incorrectly calls a glucose threshold diabetes and `ca` is described as calcification. These strings have been corrected in the source without changing any prior values. UCI's definitions govern their interpretation.")
code("clinical_prior_table()")

# ------------------------------------------------------------------ §3
md("""
## 3. Bayesian model development (PyMC) and iteration on priors and likelihood

The likelihood is Bernoulli with a logit link. Three variants were developed in turn. Each one responds to what the
previous one's validation showed:

| Model | Prior on baseline risk | Coefficient priors | Structure |
|---|---|---|---|
| **M0_weak** | none (age, sex, cp are ordinary features) | N(0, 1) | pooled |
| **M1_literature** | Diamond-Forrester offset, slope gamma ~ N(1, 0.25) | clinical directional priors | pooled |
| **M2_refined** | Diamond-Forrester offset | clinical priors + learned **chest-pain deviations** (adaptive shrinkage) | **hierarchical hospital intercepts** |

In M1, `gamma = 1` with every other term zero reproduces the literature exactly; the data then decide how far to move.
M2 adds two refinements. Chest-pain deviation terms let the referral cohort correct the Diamond-Forrester table, but
they are shrunk toward zero (the literature) unless the data insist. Partial pooling across hospitals absorbs the large
differences in prevalence between sites.
""")

code("""
bayes_models = {}
for spec in (BASELINE, LITERATURE, REFINED):
    bayes_models[spec.name] = BayesianDiagnosisModel(spec, **MCMC).fit(train)

conv = pd.DataFrame({k: m.convergence() for k, m in bayes_models.items()}).T
RESULTS["convergence"] = conv.reset_index(names="model").to_dict(orient="records")
conv
""")

code("""
loo = az.compare({k: m.loo() for k, m in bayes_models.items()}, ic="loo")
RESULTS["loo"] = loo[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]].round(2).reset_index(names="model").to_dict(orient="records")
loo[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]]
""")

code("""
m2 = bayes_models["M2_refined"]
s2 = m2.coefficient_summary()
beta_rows = s2.loc[s2.index.str.startswith("beta[")].copy()
beta_rows.index = beta_rows.index.str.slice(5, -1)
prior_mu = {k: v.mu for k, v in CLINICAL_COEF_PRIORS.items()}
plots.save(plots.coefficient_forest(beta_rows, prior_mu, "M2: posterior effects vs clinical prior means"), "m2_coefficients")
extra = s2.loc[~s2.index.str.startswith("beta[")]
RESULTS["m2_hyperparameters"] = extra[["mean", "sd", "hdi_3%", "hdi_97%"]].round(3).reset_index(names="param").to_dict(orient="records")
RESULTS["m2_coefficients"] = beta_rows[["mean", "sd", "hdi_3%", "hdi_97%"]].round(3).reset_index(names="feature").to_dict(orient="records")
display(extra[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])
beta_rows[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]]
""")

# ------------------------------------------------------------------ §4
md("""
## 4. Random Forest with Bayesian Optimization

Hyperparameters are tuned with **Bayesian Optimization** (Optuna's Tree-structured Parzen Estimator). TPE models
parameter densities for better and worse trials and proposes settings using their density ratio. Random search
with the **same 50-trial budget** is the baseline. This is one seeded comparison, not evidence of a general search
advantage. The objective is 5-fold stratified CV
**log loss**, a proper scoring rule, so the search rewards calibrated probabilities as well as ranking.
""")

code("""
N_TRIALS = 4 if QUICK else 50
bo = optimize_random_forest(train, n_trials=N_TRIALS, sampler="tpe")
rs = optimize_random_forest(train, n_trials=N_TRIALS, sampler="random")
plots.save(plots.optimization_history({"Bayesian Optimization (TPE)": bo.history(), "Random search": rs.history()}), "rf_optimization")
RESULTS["rf_search"] = {"bo_best_cv_log_loss": bo.best_cv_log_loss, "random_best_cv_log_loss": rs.best_cv_log_loss,
                        "bo_best_params": bo.best_params, "n_trials": N_TRIALS}
for name, h in (("BO", bo.history()), ("Random", rs.history())):
    hit = h.index[h.best_so_far <= rs.best_cv_log_loss][0] + 1 if (h.best_so_far <= rs.best_cv_log_loss).any() else None
    print(f"{name}: best CV log loss {h.best_so_far.iloc[-1]:.4f}; trials to match random-search best: {hit}")
bo.best_params
""")

code("""
rf_default = fit_random_forest(train)  # sklearn defaults, the untuned baseline
rf_bo = fit_random_forest(train, bo.best_params)
imp = feature_importance(rf_bo)
plots.save(plots.horizontal_bars(imp, "Random Forest (BO-tuned): feature importance", "Mean decrease in impurity"), "rf_importance")
imp.head(10).round(3)
""")

# ------------------------------------------------------------------ §5
md("""
## 5. Validation on the held-out test set (184 patients never used for fitting or tuning)

Every model is compared with actual angiographic diagnoses. *Literature only* is the Diamond-Forrester table used
directly as a predictor, with no data. It shows what prior knowledge alone achieves. Threshold: 0.5.
""")

code("""
preds = {
    "Literature only (Diamond-Forrester)": diamond_forrester_pretest(test),
    "M0 Bayesian (weak priors)": bayes_models["M0_weak"].predict_proba(test),
    "M1 Bayesian (literature priors)": bayes_models["M1_literature"].predict_proba(test),
    "M2 Bayesian (refined)": bayes_models["M2_refined"].predict_proba(test),
    "Random Forest (default)": rf_default.predict_proba(test)[:, 1],
    "Random Forest (Bayesian-optimised)": rf_bo.predict_proba(test)[:, 1],
}
table = metrics_table(y_test, preds)
table["ece"] = [expected_calibration_error(y_test, p) for p in preds.values()]
RESULTS["test_metrics"] = table.round(4).reset_index().to_dict(orient="records")
table[["roc_auc", "auc_ci_low", "auc_ci_high", "accuracy", "sensitivity", "specificity", "ppv", "npv", "brier", "log_loss", "ece"]]
""")

code("""
plot_keys = ["Literature only (Diamond-Forrester)", "M1 Bayesian (literature priors)", "M2 Bayesian (refined)", "Random Forest (Bayesian-optimised)"]
_ = plots.save(plots.roc_and_calibration(y_test, {k: preds[k] for k in plot_keys}), "roc_calibration")
""")

md("""
**Patient-level Bayesian updating.** For individual test patients, the literature *pre-test* probability is updated
by the fitted model into a *post-test* probability with a 94% credible interval. The interval is conditional on
the model, not a guarantee that an individual's diagnosis is correct. This feature-based prediction differs from
updating model parameters with newly labelled records.
""")

code("""
ex = test.groupby(TARGET, group_keys=False).apply(lambda g: g.sample(3, random_state=RANDOM_SEED))
ci = m2.predict_interval(ex)
cases = pd.DataFrame({
    "site": ex.site.astype(str).values, "age": ex.age.values, "sex": ex.sex.map({1: "M", 0: "F"}).values,
    "chest_pain": ex.cp.map(cp_names).values, "pretest_prob": diamond_forrester_pretest(ex),
    "posttest_prob": ci.p_mean.values, "94%_low": ci.p_low.values, "94%_high": ci.p_high.values,
    "actual_diagnosis": ex[TARGET].map({1: "CAD", 0: "No CAD"}).values,
})
RESULTS["patient_examples"] = cases.round(3).to_dict(orient="records")
cases.round(3)
""")

# ------------------------------------------------------------------ §6
md("""
## 6. Exploratory hospital holdouts: corrected rerun required

Each estimator is fitted on three hospitals, but RF reuses hyperparameters selected using labels from all four.
This leaks held-out hospital information through tuning. The M2 code independently samples a hospital effect
for each unseen-site patient; the stated hierarchical model requires one shared effect per hospital per draw.
The saved output also contains four divergences in each of two M2 holdout fits. Treat the values below as
historical exploratory results, pending nested tuning, corrected prediction code and per-fold diagnostics.
""")

code("""
loho_rows = []
for held in SITES:
    tr_h, te_h = data[data.site != held].reset_index(drop=True), data[data.site == held].reset_index(drop=True)
    y_h = te_h[TARGET].to_numpy()
    cand = {
        "Literature only": diamond_forrester_pretest(te_h),
        "M1 Bayesian": BayesianDiagnosisModel(LITERATURE, **MCMC).fit(tr_h).predict_proba(te_h),
        "M2 Bayesian": BayesianDiagnosisModel(REFINED, **MCMC).fit(tr_h).predict_proba(te_h),
        "RF (BO-tuned)": fit_random_forest(tr_h, bo.best_params).predict_proba(te_h)[:, 1],
    }
    for name, p in cand.items():
        loho_rows.append({"held_out": held, "model": name, "prevalence": y_h.mean(), **diagnostic_metrics(y_h, p)})
loho = pd.DataFrame(loho_rows)
RESULTS["loho"] = loho.round(4).to_dict(orient="records")
auc_pivot = loho.pivot(index="held_out", columns="model", values="roc_auc")[["Literature only", "M1 Bayesian", "M2 Bayesian", "RF (BO-tuned)"]]
plots.save(plots.grouped_bars(auc_pivot.reindex(list(SITES)), "Leave-one-hospital-out ROC-AUC", "ROC-AUC", pct=False), "loho_auc")
display(auc_pivot.round(3))
loho.pivot(index="held_out", columns="model", values="brier").round(3)
""")

# ------------------------------------------------------------------ §7
md("""
## 7. Bayesian updating as new data become available

This simulates the tool in service. It starts from the M1 literature prior and receives the training patients in four
successive batches. After each batch the posterior becomes the prior for the next (moment-matched multivariate normal).
Before each update the model is scored on the fixed test set. Feature scaling is frozen at the start so that each
parameter keeps its meaning across updates.

The final approximate posterior is checked against a full-batch M1 fit. Gaussian projection makes this approximate
sequential inference. The preprocessor uses all training covariates, so this is a retrospective fixed-feature
simulation. M2 does not support this update interface. Test performance need not improve at every batch.
""")

code("""
rng = np.random.default_rng(RANDOM_SEED)
stream = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
batches = np.array_split(np.arange(len(stream)), 4)
frozen_prep = DiagnosisPreprocessor(exclude=LITERATURE.exclude).fit(train)  # features only; labels unused

def mean_sd(belief):
    return float(np.sqrt(np.diag(belief.cov))[2:].mean())

belief = BayesianDiagnosisModel(LITERATURE).initial_prior(frozen_prep)
prior_model = BayesianDiagnosisModel(LITERATURE).use_gaussian_belief(belief, frozen_prep)
history = [{"update": 0, "patients_seen": 0, "test_auc": diagnostic_metrics(y_test, prior_model.predict_proba(test))["roc_auc"],
            "test_log_loss": diagnostic_metrics(y_test, prior_model.predict_proba(test))["log_loss"], "mean_posterior_sd": mean_sd(belief)}]
seen = 0
for k, idx in enumerate(batches, start=1):
    model_k = BayesianDiagnosisModel(LITERATURE, **MCMC).fit(stream.iloc[idx], prior=belief, preprocessor=frozen_prep)
    belief = model_k.gaussian_posterior()
    seen += len(idx)
    m = diagnostic_metrics(y_test, model_k.predict_proba(test))
    history.append({"update": k, "patients_seen": seen, "test_auc": m["roc_auc"], "test_log_loss": m["log_loss"],
                    "mean_posterior_sd": mean_sd(belief)})
seq = pd.DataFrame(history)
plots.save(plots.sequential_updating(seq), "sequential_updating")
RESULTS["sequential_updating"] = seq.round(4).to_dict(orient="records")
seq
""")

code("""
full = bayes_models["M1_literature"]
full_mean = full._stacked_draws().mean(axis=0)
agree = pd.DataFrame({"sequential_mean": belief.mean, "full_batch_mean": full_mean,
                      "sequential_sd": np.sqrt(np.diag(belief.cov)), "full_batch_sd": full._stacked_draws().std(axis=0)},
                     index=full.parameter_names)
max_gap = float((agree.sequential_mean - agree.full_batch_mean).abs().max())
p_seq = model_k.predict_proba(test); p_full = full.predict_proba(test)
RESULTS["sequential_vs_full"] = {"max_abs_coef_mean_diff": max_gap, "max_abs_test_prob_diff": float(np.abs(p_seq - p_full).max()),
                                 "corr_test_prob": float(np.corrcoef(p_seq, p_full)[0, 1])}
print(RESULTS["sequential_vs_full"])
agree.round(3)
""")

# ------------------------------------------------------------------ §8
md("""
## 8. Save results for the report
""")

code("""
REPORTS_DIR.mkdir(exist_ok=True)
(REPORTS_DIR / "metrics.json").write_text(json.dumps(RESULTS, indent=2, default=float))
table.round(4).to_csv(REPORTS_DIR / "test_metrics.csv")
loho.round(4).to_csv(REPORTS_DIR / "loho_metrics.csv", index=False)
print("Saved:", sorted(p.name for p in FIGURES_DIR.glob("*.png")))
""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s) for kind, s in cells]
    nb.metadata["kernelspec"] = {"name": "meddx", "display_name": "Python (meddx)", "language": "python"}
    from submission_content import revise_notebook
    nb = revise_notebook(nb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
