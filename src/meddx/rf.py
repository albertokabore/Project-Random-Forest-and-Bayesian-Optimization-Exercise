"""Random Forest classifier tuned with Bayesian Optimization (Optuna TPE).

Bayesian Optimization treats cross-validated performance as an unknown,
expensive function of the hyperparameters. Optuna's Tree-structured Parzen
Estimator (TPE) models parameter densities conditional on better and worse
trial values, proposing settings through their density ratio. The same budget
spent on random search is the baseline.

Objective: mean 5-fold stratified cross-validated **log loss**. A proper
scoring rule rewards calibrated probabilities, which a diagnostic tool needs.
ROC-AUC alone would ignore calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED, TARGET
from .data import feature_frame
from .preprocessing import DiagnosisPreprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

N_SPLITS: int = 5


def build_rf_pipeline(params: dict[str, Any] | None = None, seed: int = RANDOM_SEED) -> Pipeline:
    """Preprocessing + Random Forest, fitted as one unit so that imputation
    statistics come from the training folds only."""
    params = dict(params or {})
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    return Pipeline([("prep", DiagnosisPreprocessor(scale=False)), ("rf", rf)])


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparameter search space."""
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", "frac"])
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }
    params["max_features"] = (
        trial.suggest_float("max_features_frac", 0.2, 1.0) if max_features == "frac" else max_features
    )
    return params


def params_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Rebuild RandomForestClassifier kwargs from a finished trial."""
    p = dict(trial.params)
    if p.pop("max_features") == "frac":
        p["max_features"] = p.pop("max_features_frac")
    else:
        p["max_features"] = trial.params["max_features"]
        p.pop("max_features_frac", None)
    return p


@dataclass
class OptimizationResult:
    study: optuna.Study
    best_params: dict[str, Any]
    best_cv_log_loss: float

    def history(self) -> pd.DataFrame:
        """Per-trial CV log loss and best-so-far curve."""
        values = [t.value for t in self.study.trials if t.value is not None]
        frame = pd.DataFrame({"trial": np.arange(1, len(values) + 1), "cv_log_loss": values})
        frame["best_so_far"] = frame["cv_log_loss"].cummin()
        return frame


def cv_log_loss(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int = RANDOM_SEED) -> float:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_log_loss", n_jobs=1)
    return float(-scores.mean())


def optimize_random_forest(
    train: pd.DataFrame,
    n_trials: int = 60,
    sampler: str = "tpe",
    seed: int = RANDOM_SEED,
) -> OptimizationResult:
    """Tune the Random Forest with Bayesian Optimization (``sampler="tpe"``)
    or with random search (``sampler="random"``, the baseline)."""
    if sampler == "tpe":
        opt_sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=10, multivariate=True
        )
    elif sampler == "random":
        opt_sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError("sampler must be 'tpe' or 'random'")

    X, y = feature_frame(train), train[TARGET].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        return cv_log_loss(build_rf_pipeline(suggest_params(trial), seed), X, y, seed)

    study = optuna.create_study(direction="minimize", sampler=opt_sampler, study_name=f"rf_{sampler}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    logger.info("%s best CV log loss %.4f after %d trials", sampler, best.value, n_trials)
    return OptimizationResult(study=study, best_params=params_from_trial(best), best_cv_log_loss=best.value)


def fit_random_forest(train: pd.DataFrame, params: dict[str, Any] | None = None, seed: int = RANDOM_SEED) -> Pipeline:
    return build_rf_pipeline(params, seed).fit(feature_frame(train), train[TARGET].to_numpy())


def feature_importance(pipeline: Pipeline) -> pd.Series:
    """Impurity-based importances, labelled with preprocessed feature names."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["rf"].feature_importances_, index=names).sort_values(ascending=False)
