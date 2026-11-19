"""Corrected analyses requested by reports/REVIEW.md.

A. Main split (same 736/184 split as the notebook)
   * Prior ablation: M2 vs M2_weak_ablation (same hospital structure, no literature knowledge).
   * Paired bootstrap differences (same resampled patients for both models).
B. Leave-one-hospital-out, corrected
   * RF hyperparameters re-tuned INSIDE each three-hospital training fold (nested; the held-out hospital's labels
     are never seen by the search).
   * M2 predicts an unseen hospital with one shared hospital effect per posterior draw (fixed in bayes.py).
   * Stricter NUTS settings; R-hat / ESS / divergences saved for every fit. A fit with divergences is refitted
     once with target_accept = 0.99.

The original notebook outputs are left untouched. Results go to reports/corrected/.
Progress is checkpointed after every step, so a re-run resumes where it stopped.

Usage:  python scripts/run_corrected_analysis.py [--trials 30]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from meddx import plots
from meddx.bayes import BASELINE, LITERATURE, REFINED, REFINED_WEAK_ABLATION, BayesianDiagnosisModel
from meddx.config import RANDOM_SEED, REPORTS_DIR, SITES, TARGET
from meddx.data import load_heart_disease, train_test_split_stratified
from meddx.evaluation import diagnostic_metrics, expected_calibration_error
from meddx.priors import diamond_forrester_pretest
from meddx.rf import fit_random_forest, optimize_random_forest

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

import os

QUICK = os.environ.get("MEDDX_QUICK") == "1"  # smoke test: tiny budgets, separate output folder
OUT = REPORTS_DIR / ("corrected_smoke" if QUICK else "corrected")
CKPT = OUT / "checkpoint.json"
MCMC = dict(draws=100, tune=100, chains=2) if QUICK else dict(draws=1000, tune=1000, chains=4)
TARGET_ACCEPT, RETRY_TARGET_ACCEPT = 0.95, 0.99
N_BOOT = 200 if QUICK else 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ckpt() -> dict:
    return json.loads(CKPT.read_text()) if CKPT.exists() else {}


def save_ckpt(state: dict) -> None:
    tmp = CKPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=float))
    tmp.replace(CKPT)


def fit_bayes(spec, train: pd.DataFrame) -> tuple[BayesianDiagnosisModel, dict]:
    """Fit with stricter NUTS; refit once at a higher target if any divergence."""
    model = BayesianDiagnosisModel(spec, target_accept=TARGET_ACCEPT, **MCMC).fit(train)
    diag = {**model.convergence(), "target_accept": TARGET_ACCEPT}
    if diag["divergences"] > 0:
        log(f"  {spec.name}: {diag['divergences']} divergences -> refit at target_accept={RETRY_TARGET_ACCEPT}")
        model = BayesianDiagnosisModel(spec, target_accept=RETRY_TARGET_ACCEPT, **MCMC).fit(train)
        diag = {**model.convergence(), "target_accept": RETRY_TARGET_ACCEPT, "refit_after_divergence": True}
    return model, diag


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """AUC(a) - AUC(b) and logloss(a) - logloss(b) on the same resampled patients."""
    rng = np.random.default_rng(RANDOM_SEED)
    d_auc, d_ll = [], []
    n = len(y)
    while len(d_auc) < N_BOOT:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        d_auc.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
        d_ll.append(log_loss(y[idx], np.clip(p_a[idx], 1e-6, 1 - 1e-6), labels=[0, 1])
                    - log_loss(y[idx], np.clip(p_b[idx], 1e-6, 1 - 1e-6), labels=[0, 1]))
    q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]  # noqa: E731
    return {"auc_diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), "auc_diff_95ci": q(d_auc),
            "logloss_diff": float(log_loss(y, np.clip(p_a, 1e-6, 1 - 1e-6)) - log_loss(y, np.clip(p_b, 1e-6, 1 - 1e-6))),
            "logloss_diff_95ci": q(d_ll), "n_boot": N_BOOT}


def part_a(state: dict, data: pd.DataFrame) -> None:
    if "main" in state:
        log("Part A already done (checkpoint)")
        return
    train, test = train_test_split_stratified(data)
    y = test[TARGET].to_numpy()
    preds, diags, models = {}, {}, {}
    for spec in (BASELINE, LITERATURE, REFINED, REFINED_WEAK_ABLATION):
        log(f"Part A: fitting {spec.name}")
        models[spec.name], diags[spec.name] = fit_bayes(spec, train)
        preds[spec.name] = models[spec.name].predict_proba(test)
    rf_params = json.loads((REPORTS_DIR / "metrics.json").read_text())["rf_search"]["bo_best_params"]
    preds["RF_default"] = fit_random_forest(train).predict_proba(test)[:, 1]
    preds["RF_tuned"] = fit_random_forest(train, rf_params).predict_proba(test)[:, 1]

    loo = az.compare({k: m.loo() for k, m in models.items()}, ic="loo")
    pareto_k_max = {k: float(np.max(m.loo().pareto_k.values)) for k, m in models.items()}
    metrics = {k: {**diagnostic_metrics(y, p), "ece": expected_calibration_error(y, p)} for k, p in preds.items()}
    pairs = {
        "M2_minus_M0": ("M2_refined", "M0_weak"),
        "M2_minus_M2weak_ablation": ("M2_refined", "M2_weak_ablation"),
        "M1_minus_M0": ("M1_literature", "M0_weak"),
        "RFtuned_minus_RFdefault": ("RF_tuned", "RF_default"),
        "M2_minus_RFtuned": ("M2_refined", "RF_tuned"),
    }
    paired = {name: paired_bootstrap(y, preds[a], preds[b]) for name, (a, b) in pairs.items()}
    ablation_summary = models["M2_weak_ablation"].coefficient_summary()
    ablation_summary.to_csv(OUT / "M2_weak_ablation_summary.csv")
    pd.DataFrame({"site": test.site.astype(str), "y": y, **preds}).to_csv(OUT / "main_test_predictions.csv", index=False)
    state["main"] = {
        "metrics": metrics, "diagnostics": diags, "pareto_k_max": pareto_k_max, "paired_bootstrap": paired,
        "loo": loo[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]].reset_index(names="model").to_dict(orient="records"),
    }
    save_ckpt(state)
    log("Part A done")


def part_b(state: dict, data: pd.DataFrame, n_trials: int) -> None:
    folds = state.setdefault("loho", {})
    for held in SITES:
        if held in folds:
            log(f"Part B: {held} already done (checkpoint)")
            continue
        tr = data[data.site != held].reset_index(drop=True)
        te = data[data.site == held].reset_index(drop=True)
        y = te[TARGET].to_numpy()
        log(f"Part B: {held} held out ({len(te)} patients); nested RF search, {n_trials} TPE trials")
        search = optimize_random_forest(tr, n_trials=n_trials, sampler="tpe")
        search.history().to_csv(OUT / f"rf_history_{held.replace(' ', '_')}.csv", index=False)
        cand = {"Literature only": diamond_forrester_pretest(te),
                "RF (nested BO)": fit_random_forest(tr, search.best_params).predict_proba(te)[:, 1]}
        diags = {}
        for label, spec in (("M1 Bayesian", LITERATURE), ("M2 Bayesian", REFINED)):
            log(f"  fitting {spec.name}")
            model, diags[label] = fit_bayes(spec, tr)
            cand[label] = model.predict_proba(te)
        folds[held] = {
            "n": int(len(te)), "prevalence": float(y.mean()),
            "metrics": {k: {**diagnostic_metrics(y, p), "ece": expected_calibration_error(y, p)} for k, p in cand.items()},
            "diagnostics": diags, "rf_best_params": search.best_params,
            "rf_best_inner_cv_log_loss": search.best_cv_log_loss, "rf_trials": n_trials,
            "predictions": {k: [float(v) for v in p] for k, p in cand.items()}, "y": [int(v) for v in y],
        }
        save_ckpt(state)
        log(f"  {held}: " + ", ".join(f"{k} AUC {v['roc_auc']:.3f}" for k, v in folds[held]["metrics"].items()))


def summarise(state: dict) -> None:
    folds = state["loho"]
    models = ["Literature only", "M1 Bayesian", "M2 Bayesian", "RF (nested BO)"]
    rows = [{"held_out": h, "model": m, "n": f["n"], "prevalence": f["prevalence"], **f["metrics"][m]}
            for h, f in folds.items() for m in models]
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "loho_corrected_metrics.csv", index=False)

    # Patient-weighted pooled metrics over all 920 out-of-hospital predictions.
    y_all = np.concatenate([folds[h]["y"] for h in SITES])
    pooled = {m: diagnostic_metrics(y_all, np.concatenate([folds[h]["predictions"][m] for h in SITES])) for m in models}
    weighted_brier = {m: float(np.average(table[table.model == m].brier, weights=table[table.model == m].n)) for m in models}
    state["loho_summary"] = {"pooled_out_of_hospital": pooled, "patient_weighted_brier": weighted_brier,
                             "mean_auc_over_hospitals": {m: float(table[table.model == m].roc_auc.mean()) for m in models}}
    auc = table.pivot(index="held_out", columns="model", values="roc_auc").reindex(list(SITES))[models]
    plots.save(plots.grouped_bars(auc, "Leave-one-hospital-out ROC-AUC (corrected: nested RF tuning, shared hospital effect)",
                                  "ROC-AUC", pct=False), "loho_auc_corrected", show=False)
    (OUT / "corrected_results.json").write_text(json.dumps(
        {k: v for k, v in state.items() if k != "loho"} | {
            "loho": {h: {k: v for k, v in f.items() if k not in ("predictions", "y")} for h, f in folds.items()}},
        indent=2, default=float))
    log("Summary written to reports/corrected/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30, help="TPE trials per outer hospital fold")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_ckpt()
    data = load_heart_disease()
    part_a(state, data)
    part_b(state, data, args.trials)
    summarise(state)


if __name__ == "__main__":
    main()
