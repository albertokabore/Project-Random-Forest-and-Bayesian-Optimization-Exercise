"""Leakage-safe preprocessing shared by the Random Forest and Bayesian models.

All statistics (medians, modes, scaling, which columns need a missing-value
indicator) are learned in ``fit`` from training data only.

Output layout:
* continuous features   -> median-imputed, standardised
* binary features       -> mode-imputed (0/1)
* categorical features  -> one-hot with the first level as reference; a missing
                           category becomes all zeros (absorbed by its indicator)
* ``<feature>_missing`` -> 1 if the raw value was missing, for every feature
                           that had any missing value in the training data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS, FEATURES, NUMERIC_FEATURES


class DiagnosisPreprocessor(BaseEstimator, TransformerMixin):
    """Turn raw patient records into a numeric design matrix.

    Parameters
    ----------
    exclude:
        Raw feature names to leave out, e.g. ``("age", "sex", "cp")`` when those
        are already represented through the Diamond-Forrester offset.
    scale:
        Standardise continuous features (needed for meaningful priors; harmless
        for trees).
    """

    def __init__(self, exclude: tuple[str, ...] = (), scale: bool = True) -> None:
        self.exclude = exclude
        self.scale = scale

    # -- sklearn API ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: object = None) -> "DiagnosisPreprocessor":
        unknown = set(self.exclude) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown features in exclude: {sorted(unknown)}")
        self._validate_columns(X)

        keep = [f for f in FEATURES if f not in self.exclude]
        self.numeric_ = [f for f in NUMERIC_FEATURES if f in keep]
        self.binary_ = [f for f in BINARY_FEATURES if f in keep]
        self.categorical_ = [f for f in CATEGORICAL_FEATURES if f in keep]

        self.medians_ = X[self.numeric_].median()
        self.means_ = X[self.numeric_].fillna(self.medians_).mean()
        stds = X[self.numeric_].fillna(self.medians_).std(ddof=0)
        self.stds_ = stds.where(stds > 0, 1.0)
        self.modes_ = X[self.binary_].mode().iloc[0]
        self.indicator_cols_ = [f for f in keep if X[f].isna().any()]

        self.feature_names_out_ = (
            self.numeric_
            + self.binary_
            + [f"{c}_{lvl}" for c in self.categorical_ for lvl in CATEGORY_LEVELS[c][1:]]
            + [f"{c}_missing" for c in self.indicator_cols_]
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        self._validate_columns(X)

        blocks: list[np.ndarray] = []
        numeric = X[self.numeric_].astype(float).fillna(self.medians_)
        if self.scale:
            numeric = (numeric - self.means_) / self.stds_
        blocks.append(numeric.to_numpy())
        blocks.append(X[self.binary_].astype(float).fillna(self.modes_).to_numpy())
        for col in self.categorical_:
            values = X[col].to_numpy()
            for lvl in CATEGORY_LEVELS[col][1:]:
                blocks.append((values == lvl).astype(float)[:, None])
        if self.indicator_cols_:
            blocks.append(X[self.indicator_cols_].isna().to_numpy(dtype=float))
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DiagnosisPreprocessor expects a pandas DataFrame")
        missing = [f for f in FEATURES if f not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
