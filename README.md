# Medical Diagnosis System: Random Forest + Bayesian Optimization + Bayesian Updating

**Albert Kabore — PhD student in Artificial Intelligence**

Research exercise estimating angiographic coronary artery disease from symptoms, exam and test results. It combines:

* **Prior medical knowledge**: Diamond & Forrester (1979) pre-test probabilities and directional clinical priors.
* **Bayesian models in PyMC**: logistic regression, approximate sequential updating for the pooled M0/M1 models,
  and hierarchical hospital effects in M2. M2 does not support the sequential update interface.
* **A Random Forest** whose hyperparameters are tuned by **Bayesian Optimization** (Optuna TPE), benchmarked against
  random search.

**Data:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), 920 patient records from
4 hospital datasets (Cleveland, Hungary, Switzerland, VA Long Beach). The raw files match UCI's archive exactly;
see [data verification](reports/data_verification.json). No synthetic records were added.

**Report:** [reports/REPORT.md](reports/REPORT.md) · **Notebook:** [notebooks/medical_diagnosis_system.ipynb](notebooks/medical_diagnosis_system.ipynb)

**Word report:** [Results analysis report](reports/Albert_Kabore_Medical_Diagnosis_System_Report_Results_Analysis.docx).
**Notebook reading view:** [Black text and white tables](reports/Albert_Kabore_Medical_Diagnosis_Notebook.html).
**Final assignment and presentation audit:** [FINAL_AUDIT.md](reports/FINAL_AUDIT.md).
The report and notebook include an introduction, six steps with 50-word observation blocks, final insights,
recommendations, and a conclusion placed last. The observations, insights, and recommendations analyse
specific numerical outputs. Their source files and calculated comparisons are recorded in
[output_analysis_trace.json](reports/output_analysis_trace.json).

**Scientific review:** [reports/REVIEW.md](reports/REVIEW.md) records the original review. Main test results were
checked independently. The later corrected hospital analysis is complete in
[corrected_results.json](reports/corrected/corrected_results.json): RF tuning occurs inside each hospital fold,
unseen-hospital effects are shared within each posterior draw, and final fit diagnostics are saved.
Notebook markdown distinguishes these results from its preserved original code and execution outputs.
The companion weak-prior model also changes feature encoding and shrinkage, so it does not isolate prior effects.

## Project structure

```
data/raw/                     UCI processed files for the 4 hospitals + dataset documentation
src/meddx/
  config.py                   paths, schema, seeds
  data.py                     loading, documented cleaning, stratified split
  priors.py                   Diamond-Forrester table, Beta prevalence priors, clinical coefficient priors
  preprocessing.py            leakage-safe imputation / encoding / missing indicators (sklearn transformer)
  bayes.py                    PyMC models M0 / M1 / M2, prediction with credible intervals, sequential updating
  rf.py                       Random Forest pipeline + Bayesian Optimization (TPE) vs random search
  evaluation.py               AUC, sensitivity/specificity, Brier, log loss, calibration, bootstrap CIs
  plots.py                    report figures
scripts/
  download_data.py            re-download the UCI files (validated row counts)
  build_notebook.py           generates the analysis notebook from readable source
  verify_report.py            independently recomputes evidence; --bayesian refits the main Bayesian models
  submission_content.py       shared report and notebook prose, author details, and 50-word observations
  output_analysis.py          interpretations and recommendations derived from saved numerical outputs
  revise_submission.py        updates Markdown while preserving notebook code, outputs, and numerical evidence
  export_verified_word_report.py  builds the named Word report from REPORT.md
notebooks/                    executed end-to-end analysis
reports/                      REPORT.md, metrics.json, CSV tables, figures/
tests/                        pytest suite (data, priors, preprocessing, metrics, RF, PyMC model)
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/download_data.py # only if data/raw is empty
```

The code requests PyMC's **Numba** compilation mode. The saved analysis uses **PyMC 5**, a successor to PyMC3.
Installed versions for the independent review are recorded in `reports/review_verification.json`; the dependency
files specify minimum versions and do not lock an identical future environment.

## Run

```bash
# full analysis (runtime varies substantially; writes reports/figures, metrics.json, CSVs)
python -m ipykernel install --sys-prefix --name meddx
JUPYTER_PATH=.venv/share/jupyter jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/medical_diagnosis_system.ipynb

# smoke run with tiny MCMC / search budgets
MEDDX_QUICK=1 JUPYTER_PATH=.venv/share/jupyter jupyter nbconvert --to notebook --execute \
    --output-dir /tmp notebooks/medical_diagnosis_system.ipynb
```

## Tests

```bash
pytest -m "not slow"   # fast unit tests (about 1 min)
pytest                 # include the MCMC fit/update test
```

To reproduce the review's main-model checks without replacing the original analysis outputs:

```bash
python scripts/verify_report.py --bayesian
```

This verifies the selected RF configuration and main Bayesian models; it does not repeat the complete RF search,
hospital holdouts, or sequential experiments. Results are written to `reports/review_verification.json` and
`reports/review_test_predictions.csv`.

## Using the model in code

```python
from meddx.data import load_heart_disease, train_test_split_stratified
from meddx.bayes import BayesianDiagnosisModel, REFINED

train, test = train_test_split_stratified(load_heart_disease())
model = BayesianDiagnosisModel(REFINED).fit(train)
model.predict_interval(test.head())     # posterior mean P(disease) + 94% credible interval
```

> Educational project. Not a medical device and not validated for clinical use.
