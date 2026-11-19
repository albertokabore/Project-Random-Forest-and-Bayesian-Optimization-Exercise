"""Prior medical knowledge used by the Bayesian models.

Three kinds of prior knowledge are encoded here:

1. **Pre-test probability of coronary artery disease (CAD)** from
   Diamond GA, Forrester JS. "Analysis of probability as an aid in the clinical
   diagnosis of coronary-artery disease." *N Engl J Med* 1979;300:1350-1358.
   The table gives the prevalence of angiographic CAD by age decade, sex and
   chest-pain type. It is the classic prior for this diagnostic question and
   uses exactly the three variables (age, sex, cp) recorded in the UCI data.

2. **Prior prevalence per hospital**: a Beta distribution centred on the
   Diamond-Forrester expected prevalence of each hospital's patient mix, with
   a small pseudo-count so the observed data dominate quickly.

3. **Clinically motivated coefficient priors** (log-odds scale). Their means
   favour expected directions; the Normal distributions still permit either
   sign. Means and sigma = 0.5 are modelling judgements, not published effect
   estimates. Continuous inputs are standardised, while binary and categorical
   contrasts use a 0/1 scale. UCI's glucose threshold is not diagnosed diabetes,
   and its fluoroscopy vessel count is not relabelled as a calcium score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit

# ---------------------------------------------------------------------------
# 1. Diamond & Forrester (1979) pre-test probability table (percent).
#    Keys: (age-decade lower bound, sex [1 male / 0 female], cp code).
#    cp codes follow the UCI coding: 1 typical angina, 2 atypical angina,
#    3 non-anginal pain, 4 asymptomatic.
# ---------------------------------------------------------------------------
_DF_TABLE_PERCENT: dict[int, dict[int, dict[int, float]]] = {
    # age:  {sex: {cp: %}}
    30: {1: {1: 69.7, 2: 21.8, 3: 5.2, 4: 1.9}, 0: {1: 25.8, 2: 4.2, 3: 0.8, 4: 0.3}},
    40: {1: {1: 87.3, 2: 46.1, 3: 14.1, 4: 5.5}, 0: {1: 55.2, 2: 13.3, 3: 2.8, 4: 1.0}},
    50: {1: {1: 92.0, 2: 58.9, 3: 21.5, 4: 9.7}, 0: {1: 79.4, 2: 32.4, 3: 8.4, 4: 3.2}},
    60: {1: {1: 94.3, 2: 67.1, 3: 28.1, 4: 12.3}, 0: {1: 90.6, 2: 54.4, 3: 18.6, 4: 7.5}},
}
_DF_AGE_BANDS: tuple[int, ...] = tuple(sorted(_DF_TABLE_PERCENT))

# Probabilities are clipped away from 0/1 before taking logits.
_PROB_FLOOR: float = 1e-3


def _age_band(age: float) -> int:
    """Map an age to its Diamond-Forrester decade. Ages outside 30-69 are
    clamped to the nearest published band (the table covers 30-69 only)."""
    decade = int(age // 10 * 10)
    return min(max(decade, _DF_AGE_BANDS[0]), _DF_AGE_BANDS[-1])


def diamond_forrester_pretest(frame: pd.DataFrame) -> np.ndarray:
    """Literature pre-test probability of CAD for each patient.

    Requires non-missing ``age``, ``sex`` and ``cp``, which are complete in all
    four UCI hospitals.
    """
    required = frame[["age", "sex", "cp"]]
    if required.isna().any().any():
        raise ValueError("age, sex and cp must be non-missing to compute pre-test probability")

    probs = [
        _DF_TABLE_PERCENT[_age_band(age)][int(sex)][int(cp)] / 100.0
        for age, sex, cp in required.itertuples(index=False)
    ]
    return np.clip(np.asarray(probs, dtype=float), _PROB_FLOOR, 1 - _PROB_FLOOR)


def pretest_logit(frame: pd.DataFrame) -> np.ndarray:
    """Pre-test probability on the log-odds scale (used as a model offset)."""
    return logit(diamond_forrester_pretest(frame))


# ---------------------------------------------------------------------------
# 2. Prevalence prior + conjugate Bayesian updating.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BetaBelief:
    """Beta(alpha, beta) belief about a disease prevalence."""

    alpha: float
    beta: float

    @classmethod
    def from_mean(cls, mean: float, pseudo_count: float) -> "BetaBelief":
        if not 0 < mean < 1:
            raise ValueError("mean must lie strictly between 0 and 1")
        if pseudo_count <= 0:
            raise ValueError("pseudo_count must be positive")
        return cls(alpha=mean * pseudo_count, beta=(1 - mean) * pseudo_count)

    def update(self, positives: int, n: int) -> "BetaBelief":
        """Conjugate Beta-Binomial update with ``positives`` cases among ``n``."""
        if not 0 <= positives <= n:
            raise ValueError("require 0 <= positives <= n")
        return BetaBelief(self.alpha + positives, self.beta + n - positives)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def interval(self, mass: float = 0.95) -> tuple[float, float]:
        lo, hi = stats.beta.interval(mass, self.alpha, self.beta)
        return float(lo), float(hi)


# The literature prior is "worth" this many patients. Small on purpose, so the
# prior steers early estimates and is quickly overwhelmed by hospital data.
PREVALENCE_PSEUDO_COUNT: float = 10.0


def prevalence_prior(frame: pd.DataFrame, pseudo_count: float = PREVALENCE_PSEUDO_COUNT) -> BetaBelief:
    """Beta prior centred on the mean Diamond-Forrester pre-test probability
    of the patients in ``frame`` (uses features only, never the labels)."""
    return BetaBelief.from_mean(float(diamond_forrester_pretest(frame).mean()), pseudo_count)


# ---------------------------------------------------------------------------
# 3. Directional clinical priors for test results (log-odds per unit of the
#    preprocessed feature: per SD for continuous inputs, per category vs the
#    reference level for indicators).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoefPrior:
    mu: float
    sigma: float
    rationale: str


DEFAULT_COEF_SIGMA: float = 1.0          # weakly informative, regularising
CLINICAL_COEF_SIGMA: float = 0.5

CLINICAL_COEF_PRIORS: dict[str, CoefPrior] = {
    "trestbps": CoefPrior(0.1, CLINICAL_COEF_SIGMA, "Hypertension is a CAD risk factor (weak effect)"),
    "chol": CoefPrior(0.1, CLINICAL_COEF_SIGMA, "Hypercholesterolaemia is a CAD risk factor (weak effect)"),
    "fbs": CoefPrior(0.2, CLINICAL_COEF_SIGMA, "Assumed positive effect for fasting glucose >120 mg/dL; not diagnosed diabetes"),
    "thalach": CoefPrior(-0.3, CLINICAL_COEF_SIGMA, "Lower achieved heart rate favours CAD under this prior; not itself an ischaemia diagnosis"),
    "oldpeak": CoefPrior(0.4, CLINICAL_COEF_SIGMA, "Exercise ST depression is the main exercise-ECG sign of ischaemia"),
    "exang": CoefPrior(0.6, CLINICAL_COEF_SIGMA, "Angina provoked by exercise suggests flow-limiting stenosis"),
    "ca": CoefPrior(0.6, CLINICAL_COEF_SIGMA, "Assumed positive effect for UCI's count of major vessels coloured by fluoroscopy"),
    "restecg_1": CoefPrior(0.2, CLINICAL_COEF_SIGMA, "Resting ST-T abnormality (weak, non-specific)"),
    "restecg_2": CoefPrior(0.2, CLINICAL_COEF_SIGMA, "Left-ventricular hypertrophy (weak, non-specific)"),
    "slope_2": CoefPrior(0.5, CLINICAL_COEF_SIGMA, "Flat ST slope at peak exercise is more ischaemic than upsloping"),
    "slope_3": CoefPrior(0.5, CLINICAL_COEF_SIGMA, "Downsloping ST segment is more ischaemic than upsloping"),
    "thal_6": CoefPrior(0.7, CLINICAL_COEF_SIGMA, "A fixed perfusion defect favours CAD under this prior"),
    "thal_7": CoefPrior(1.0, CLINICAL_COEF_SIGMA, "Reversible perfusion defect (inducible ischaemia)"),
}


def coefficient_prior_arrays(
    feature_names: list[str], informative: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Prior means and SDs aligned to ``feature_names``.

    Features without a clinical prior (e.g. missing-value indicators) get a
    weakly informative N(0, 1).
    """
    mu = np.zeros(len(feature_names))
    sigma = np.full(len(feature_names), DEFAULT_COEF_SIGMA)
    if informative:
        for i, name in enumerate(feature_names):
            prior = CLINICAL_COEF_PRIORS.get(name)
            if prior is not None:
                mu[i], sigma[i] = prior.mu, prior.sigma
    return mu, sigma


def clinical_prior_table() -> pd.DataFrame:
    """Human-readable table of the clinical priors (for the report)."""
    return pd.DataFrame(
        [
            {"feature": k, "prior_mean_logodds": v.mu, "prior_sd": v.sigma,
             "prior_odds_ratio": round(float(np.exp(v.mu)), 2), "rationale": v.rationale}
            for k, v in CLINICAL_COEF_PRIORS.items()
        ]
    )


def diamond_forrester_table() -> pd.DataFrame:
    """The Diamond-Forrester table as a tidy frame (for the report)."""
    labels = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
    rows = []
    for age, by_sex in _DF_TABLE_PERCENT.items():
        for sex, by_cp in by_sex.items():
            for cp, pct in by_cp.items():
                rows.append({"age_band": f"{age}-{age + 9}", "sex": "Male" if sex else "Female",
                             "chest_pain": labels[cp], "pretest_pct": pct})
    return pd.DataFrame(rows)
