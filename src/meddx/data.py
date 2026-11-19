"""Loading, cleaning and splitting the UCI Heart Disease data (four hospitals).

Only the real UCI records are used. No rows are synthesised or imputed at load
time; missing values stay as NaN and are handled inside the model pipelines so
that imputation statistics are learned from training data only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORY_LEVELS,
    DATA_DIR,
    FEATURES,
    RANDOM_SEED,
    RAW_COLUMNS,
    SITE_FILES,
    TARGET,
    TEST_SIZE,
)

# Physiologically impossible zeros that the UCI files use as "not measured".
# Switzerland records every cholesterol as 0 and VA records 49 of them as 0.
_ZERO_MEANS_MISSING: tuple[str, ...] = ("chol", "trestbps")


class DataValidationError(ValueError):
    """Raised when a raw file does not match the documented schema."""


def load_site(site: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a single hospital's processed file and return it with a ``site`` column."""
    try:
        filename = SITE_FILES[site]
    except KeyError as exc:
        raise KeyError(f"Unknown site {site!r}; expected one of {list(SITE_FILES)}") from exc

    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` to fetch the UCI files."
        )

    frame = pd.read_csv(path, header=None, names=list(RAW_COLUMNS), na_values="?")
    if frame.shape[1] != len(RAW_COLUMNS):
        raise DataValidationError(f"{filename}: expected {len(RAW_COLUMNS)} columns")
    frame.insert(0, "site", site)
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning rules and derive the binary target.

    * zeros in ``chol`` / ``trestbps`` -> NaN (not physiologically possible);
    * category codes outside the documented levels -> NaN;
    * ``disease`` = 1 when ``num`` > 0 (any vessel with >50% narrowing), the
      standard binarisation used by Detrano et al. (1989).
    """
    out = frame.copy()
    for col in _ZERO_MEANS_MISSING:
        out.loc[out[col] == 0, col] = np.nan

    for col, levels in CATEGORY_LEVELS.items():
        out.loc[~out[col].isin(levels) & out[col].notna(), col] = np.nan

    out.loc[(out["ca"] < 0) | (out["ca"] > 3), "ca"] = np.nan

    if out["num"].isna().any():
        raise DataValidationError("Diagnosis column 'num' must not contain missing values")
    out[TARGET] = (out["num"] > 0).astype(int)
    return out.drop(columns="num")


def load_heart_disease(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and clean all four hospitals into one frame (920 patients)."""
    frames = [clean(load_site(site, data_dir)) for site in SITE_FILES]
    data = pd.concat(frames, ignore_index=True)
    data["site"] = pd.Categorical(data["site"], categories=list(SITE_FILES))
    return data


def train_test_split_stratified(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a test set stratified jointly on hospital and diagnosis.

    Joint stratification keeps each hospital's very different prevalence
    (36%-93%) represented in both halves.
    """
    strata = data["site"].astype(str) + "_" + data[TARGET].astype(str)
    train, test = train_test_split(
        data, test_size=test_size, stratify=strata, random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return the model input columns in canonical order."""
    return data.loc[:, list(FEATURES)]


def missingness_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values for each feature by hospital."""
    return (
        data.groupby("site", observed=True)[list(FEATURES)]
        .agg(lambda s: s.isna().mean() * 100)
        .round(1)
    )
