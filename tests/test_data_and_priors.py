"""Unit tests for data loading, cleaning, priors and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meddx.config import FEATURES, SITES, TARGET
from meddx.data import clean, load_heart_disease, load_site, train_test_split_stratified
from meddx.preprocessing import DiagnosisPreprocessor
from meddx.priors import (
    BetaBelief,
    coefficient_prior_arrays,
    diamond_forrester_pretest,
    prevalence_prior,
)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return load_heart_disease()


# ----------------------------------------------------------------- data
def test_loads_all_920_real_patients(data: pd.DataFrame) -> None:
    assert len(data) == 920
    assert data["site"].value_counts().to_dict() == {
        "Cleveland": 303, "Hungary": 294, "VA Long Beach": 200, "Switzerland": 123,
    }


def test_target_is_binary_and_complete(data: pd.DataFrame) -> None:
    assert set(data[TARGET].unique()) == {0, 1}
    assert data[TARGET].isna().sum() == 0
    # Published Cleveland prevalence: 139 of 303 patients have disease.
    assert data.loc[data.site == "Cleveland", TARGET].sum() == 139


def test_zero_cholesterol_becomes_missing(data: pd.DataFrame) -> None:
    assert (data["chol"] == 0).sum() == 0
    assert data.loc[data.site == "Switzerland", "chol"].isna().all()


def test_clean_rejects_missing_diagnosis() -> None:
    frame = load_site("Cleveland").head(3).copy()
    frame.loc[0, "num"] = np.nan
    with pytest.raises(ValueError):
        clean(frame)


def test_unknown_site_raises() -> None:
    with pytest.raises(KeyError):
        load_site("Atlantis")


def test_split_is_stratified_and_disjoint(data: pd.DataFrame) -> None:
    train, test = train_test_split_stratified(data)
    assert len(train) + len(test) == len(data)
    for site in SITES:
        tr = train.loc[train.site == site, TARGET].mean()
        te = test.loc[test.site == site, TARGET].mean()
        assert abs(tr - te) < 0.05


# --------------------------------------------------------------- priors
def test_diamond_forrester_known_values() -> None:
    frame = pd.DataFrame({"age": [55, 35, 65, 25, 80], "sex": [1, 0, 0, 1, 1], "cp": [1, 4, 2, 1, 3]})
    expected = [0.920, 0.003, 0.544, 0.697, 0.281]  # ages <30 / >69 clamp to the table edges
    np.testing.assert_allclose(diamond_forrester_pretest(frame), expected, atol=1e-9)


def test_diamond_forrester_requires_complete_inputs() -> None:
    with pytest.raises(ValueError):
        diamond_forrester_pretest(pd.DataFrame({"age": [50], "sex": [np.nan], "cp": [1]}))


def test_beta_update_is_conjugate() -> None:
    prior = BetaBelief.from_mean(0.4, 10)
    post = prior.update(positives=30, n=50)
    assert (post.alpha, post.beta) == pytest.approx((34.0, 26.0))
    lo, hi = post.interval()
    assert lo < post.mean < hi


@pytest.mark.parametrize("mean,count", [(0.0, 10), (1.0, 10), (0.5, 0)])
def test_beta_rejects_invalid(mean: float, count: float) -> None:
    with pytest.raises(ValueError):
        BetaBelief.from_mean(mean, count)


def test_prevalence_prior_uses_features_only(data: pd.DataFrame) -> None:
    prior = prevalence_prior(data.drop(columns=TARGET))
    assert 0 < prior.mean < 1
    assert prior.alpha + prior.beta == pytest.approx(10.0)


def test_coefficient_priors_align() -> None:
    names = ["oldpeak", "thal_7", "chol_missing"]
    mu, sigma = coefficient_prior_arrays(names, informative=True)
    assert mu[0] > 0 and mu[1] > 0 and mu[2] == 0
    assert sigma[2] == 1.0
    mu_w, _ = coefficient_prior_arrays(names, informative=False)
    assert not mu_w.any()


# -------------------------------------------------------- preprocessing
def test_preprocessor_has_no_nans_and_stable_columns(data: pd.DataFrame) -> None:
    train, test = train_test_split_stratified(data)
    prep = DiagnosisPreprocessor().fit(train)
    Xtr, Xte = prep.transform(train), prep.transform(test)
    assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
    assert Xtr.shape[1] == Xte.shape[1] == len(prep.get_feature_names_out())
    np.testing.assert_allclose(Xtr[:, 0].mean(), 0, atol=1e-9)  # standardised age


def test_preprocessor_exclude_and_missing_category(data: pd.DataFrame) -> None:
    prep = DiagnosisPreprocessor(exclude=("age", "sex", "cp")).fit(data)
    names = list(prep.get_feature_names_out())
    assert not any(n == "age" or n == "sex" or n.startswith("cp_") for n in names)
    row = data.loc[data["thal"].isna()].head(1)
    x = dict(zip(names, prep.transform(row)[0]))
    assert x["thal_6"] == x["thal_7"] == 0 and x["thal_missing"] == 1


def test_preprocessor_validates_input(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        DiagnosisPreprocessor(exclude=("not_a_feature",)).fit(data)
    with pytest.raises(ValueError):
        DiagnosisPreprocessor().fit(data.drop(columns="age"))
    assert set(FEATURES) <= set(data.columns)
