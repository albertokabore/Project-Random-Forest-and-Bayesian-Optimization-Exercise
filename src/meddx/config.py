"""Central configuration: paths, column schema and reproducibility settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

# The four "processed" files distributed by UCI (dataset id 45). Each one is a
# different hospital, which is what the assignment's "various hospitals" step
# refers to.
SITE_FILES: dict[str, str] = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
SITES: tuple[str, ...] = tuple(SITE_FILES)

# The 14 attributes used by every published experiment on this dataset,
# in file order (see data/raw/heart-disease.names, section 7).
RAW_COLUMNS: tuple[str, ...] = (
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting systolic blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl
    "restecg",   # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # thallium scan: 3 normal, 6 fixed defect, 7 reversible defect
    "num",       # angiographic diagnosis: 0 = <50% narrowing, 1-4 = >50% narrowing
)
TARGET: str = "disease"

NUMERIC_FEATURES: tuple[str, ...] = ("age", "trestbps", "chol", "thalach", "oldpeak", "ca")
BINARY_FEATURES: tuple[str, ...] = ("sex", "fbs", "exang")
CATEGORICAL_FEATURES: tuple[str, ...] = ("cp", "restecg", "slope", "thal")
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

CATEGORY_LEVELS: dict[str, tuple[int, ...]] = {
    "cp": (1, 2, 3, 4),
    "restecg": (0, 1, 2),
    "slope": (1, 2, 3),
    "thal": (3, 6, 7),
}
