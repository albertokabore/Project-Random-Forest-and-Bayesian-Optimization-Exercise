"""Recompute report evidence without overwriting the original analysis results.

Run with the project's Python environment. Add --bayesian to refit M0/M1/M2
with the original full sampling settings. The search trajectory and hospital
holdouts are not rerun; the selected RF configuration's CV score is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from meddx.config import RANDOM_SEED, SITES, TARGET
from meddx.data import load_heart_disease, missingness_summary, train_test_split_stratified
from meddx.evaluation import diagnostic_metrics
from meddx.priors import diamond_forrester_pretest, prevalence_prior
from meddx.rf import build_rf_pipeline, cv_log_loss, fit_random_forest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesian", action="store_true")
    args = parser.parse_args()
    reports = ROOT / "reports"
    original = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    data = load_heart_disease()
    train, test = train_test_split_stratified(data)
    y = test[TARGET].to_numpy()
    results = {
        "review_date": "2026-11-18",
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scipy", "scikit-learn", "pymc", "arviz", "optuna", "numba")},
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [reports / "metrics.json", *sorted((ROOT / "src/meddx").glob("*.py"))]},
        "records": len(data),
        "disease_positive": int(data[TARGET].sum()),
        "train_size": len(train),
        "test_size": len(test),
        "test_positive": int(y.sum()),
        "duplicate_cleaned_rows": int(data.duplicated().sum()),
        "identical_rows_across_split": len(train.merge(test.drop_duplicates(), on=list(data.columns))),
        "ages_outside_prior_table": int(((data.age < 30) | (data.age >= 70)).sum()),
        "missingness_pct": missingness_summary(data).reset_index().to_dict(orient="records"),
        "test_metrics": [],
        "prevalence_updates": [],
        "not_rerun": ["100-trial TPE/random search comparison", "hospital holdout fits", "sequential updates"],
    }
    for site in SITES:
        tr = train[train.site == site]
        prior = prevalence_prior(tr)
        posterior = prior.update(int(tr[TARGET].sum()), len(tr))
        results["prevalence_updates"].append({
            "site": site, "prior_mean": prior.mean, "posterior_mean": posterior.mean,
            "credible_interval_95": posterior.interval(),
            "test_prevalence": float(test.loc[test.site == site, TARGET].mean()),
        })

    params = original["rf_search"]["bo_best_params"]
    predictions = {
        "Literature only (Diamond-Forrester)": diamond_forrester_pretest(test),
        "Random Forest (default)": fit_random_forest(train).predict_proba(test)[:, 1],
        "Random Forest (Bayesian-optimised)": fit_random_forest(train, params).predict_proba(test)[:, 1],
    }
    results["selected_rf_cv_log_loss"] = cv_log_loss(build_rf_pipeline(params), train, train[TARGET].to_numpy())

    if args.bayesian:
        import arviz as az
        from meddx.bayes import BASELINE, LITERATURE, REFINED, BayesianDiagnosisModel

        results["bayesian_diagnostics"] = {}
        labels = ("M0 Bayesian (weak priors)", "M1 Bayesian (literature priors)", "M2 Bayesian (refined)")
        for spec, label in zip((BASELINE, LITERATURE, REFINED), labels):
            print(f"Refitting {spec.name}: 4 chains, 1000 tune + 1000 retained draws each", flush=True)
            model = BayesianDiagnosisModel(spec).fit(train)
            predictions[label] = model.predict_proba(test)
            loo = model.loo()
            rhat = az.rhat(model.idata_).to_array().values
            results["bayesian_diagnostics"][spec.name] = {
                "max_rhat_unrounded": float(np.nanmax(rhat)),
                "divergences": int(model.idata_.sample_stats.diverging.sum()),
                "elpd_loo": float(loo.elpd_loo),
                "max_pareto_k": float(loo.pareto_k.max()),
                "pareto_k_above_0_7": int((loo.pareto_k > 0.7).sum()),
            }
            model.coefficient_summary().to_csv(reports / f"review_{spec.name}_summary.csv")
            print(spec.name, results["bayesian_diagnostics"][spec.name], flush=True)
    else:
        results["not_rerun"].append("main Bayesian fits (use --bayesian)")

    for name, p in predictions.items():
        recomputed = diagnostic_metrics(y, p)
        saved = next(row for row in original["test_metrics"] if row["model"] == name)
        differences = {k: v - saved[k] for k, v in recomputed.items()}
        results["test_metrics"].append({
            "model": name, **recomputed, "minus_original_rounded": differences,
            "matches_original_to_4_decimals": all(abs(d) < 0.00005001 for d in differences.values()),
        })

    if args.bayesian:
        # Paired bootstrap of fixed-model test predictions; conditions on this
        # training split and does not estimate model-selection/training variance.
        a = predictions["M2 Bayesian (refined)"]
        b = predictions["M0 Bayesian (weak priors)"]
        rng = np.random.default_rng(RANDOM_SEED)
        differences = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size == 2:
                differences.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        results["paired_m2_minus_m0_auc"] = {
            "difference": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "percentile_ci_95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "bootstrap_samples": len(differences),
        }

    out = test[["site", "age", "sex", "cp", TARGET]].copy()
    for name, p in predictions.items():
        out[name] = p
    out.to_csv(reports / "review_test_predictions.csv", index=False)
    (reports / "review_verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in results["test_metrics"]:
        print(row["model"], "AUC", round(row["roc_auc"], 6),
              "matches original:", row["matches_original_to_4_decimals"], flush=True)
    print("Saved reports/review_verification.json and review_test_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
