"""Tests for evaluation metrics, the Random Forest optimiser and the Bayesian model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meddx.config import TARGET
from meddx.data import load_heart_disease, train_test_split_stratified
from meddx.evaluation import bootstrap_ci, diagnostic_metrics, expected_calibration_error
from meddx.rf import feature_importance, fit_random_forest, optimize_random_forest


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split_stratified(load_heart_disease())


# ------------------------------------------------------------ evaluation
def test_metrics_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    m = diagnostic_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == m["accuracy"] == m["sensitivity"] == m["specificity"] == 1.0
    assert m["brier"] < 0.05


def test_metrics_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        diagnostic_metrics(np.array([0, 1]), np.array([0.5]))


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    lo, hi = bootstrap_ci(y, p, lambda a, b: float(((b > 0.5) == a).mean()), n_boot=300)
    point = float(((p > 0.5) == y).mean())
    assert lo <= point <= hi


def test_ece_zero_for_calibrated_constant() -> None:
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)


# ------------------------------------------------------- random forest
def test_random_forest_beats_chance(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    model = fit_random_forest(train, {"n_estimators": 100})
    p = model.predict_proba(test)[:, 1]
    assert diagnostic_metrics(test[TARGET].to_numpy(), p)["roc_auc"] > 0.8
    assert feature_importance(model).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_optimizer_runs_and_returns_valid_params(split: tuple[pd.DataFrame, pd.DataFrame], sampler: str) -> None:
    result = optimize_random_forest(split[0], n_trials=3, sampler=sampler)
    assert result.best_cv_log_loss > 0
    assert len(result.history()) == 3
    fit_random_forest(split[0], result.best_params)  # params are valid sklearn kwargs


def test_optimizer_rejects_unknown_sampler(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    with pytest.raises(ValueError):
        optimize_random_forest(split[0], n_trials=1, sampler="grid")


# -------------------------------------------------------- Bayesian model
@pytest.mark.slow
def test_bayesian_model_fits_predicts_and_updates(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    from meddx.bayes import LITERATURE, REFINED, BayesianDiagnosisModel

    train, test = split
    model = BayesianDiagnosisModel(LITERATURE, draws=200, tune=200, chains=2).fit(train)
    p = model.predict_proba(test)
    assert p.shape == (len(test),) and np.all((p > 0) & (p < 1))
    assert diagnostic_metrics(test[TARGET].to_numpy(), p)["roc_auc"] > 0.8

    interval = model.predict_interval(test.head(5))
    assert (interval.p_low <= interval.p_mean).all() and (interval.p_mean <= interval.p_high).all()

    prior = model.gaussian_posterior()
    assert prior.cov.shape == (len(prior.names), len(prior.names))
    updated = BayesianDiagnosisModel(LITERATURE, draws=200, tune=200, chains=2).fit(
        test, prior=prior, preprocessor=model.preprocessor_
    )
    assert updated.parameter_names == model.parameter_names

    with pytest.raises(ValueError):
        BayesianDiagnosisModel(REFINED).fit(train, prior=prior)


# ------------------------------------------- unseen-hospital prediction
def _fake_posterior_model():
    """A REFINED model with a hand-built posterior (no sampling), trained on
    three hospitals, so predictions for an unseen fourth can be checked."""
    from meddx.bayes import REFINED, BayesianDiagnosisModel, Posterior
    from meddx.preprocessing import DiagnosisPreprocessor

    data = load_heart_disease()
    model = BayesianDiagnosisModel(REFINED)
    train = data[data.site != "Switzerland"]
    model.preprocessor_ = DiagnosisPreprocessor(exclude=REFINED.exclude).fit(train)
    model.train_sites_ = ["Cleveland", "Hungary", "VA Long Beach"]
    rng = np.random.default_rng(0)
    S, p = 200, len(model.preprocessor_.get_feature_names_out())
    model.posterior_ = Posterior(
        intercept=rng.normal(0, 0.1, S), beta=rng.normal(0, 0.1, (S, p)), gamma=np.full(S, 0.5),
        site_effect=rng.normal(0, 1, (S, 3)), tau_site=np.abs(rng.normal(1, 0.1, S)),
    )
    return model, data


def test_unseen_hospital_effect_is_shared_and_order_invariant() -> None:
    model, data = _fake_posterior_model()
    swiss = data[data.site == "Switzerland"].head(6).reset_index(drop=True)
    twins = pd.concat([swiss.iloc[[0]], swiss.iloc[[0]]], ignore_index=True)
    s = model.predict_samples(twins)
    np.testing.assert_array_equal(s[:, 0], s[:, 1])  # identical patients, identical draws

    effects = model._site_effects_for(swiss)
    assert np.all(effects == effects[:, [0]])  # one shared effect per posterior draw

    p = model.predict_proba(swiss)
    perm = [5, 2, 0, 4, 1, 3]
    np.testing.assert_allclose(model.predict_proba(swiss.iloc[perm].reset_index(drop=True)), p[perm])
    # Predicting a patient alone or inside a batch gives the same answer.
    np.testing.assert_allclose(model.predict_proba(swiss.iloc[[3]]), p[[3]])


def test_seen_hospital_uses_learned_effect() -> None:
    model, data = _fake_posterior_model()
    rows = data[data.site == "Hungary"].head(3)
    np.testing.assert_array_equal(model._site_effects_for(rows), np.repeat(model.posterior_.site_effect[:, [1]], 3, axis=1))
