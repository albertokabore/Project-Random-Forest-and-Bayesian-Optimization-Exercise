"""Bayesian logistic-regression diagnosis models (PyMC).

The probability of disease for patient *i* is modelled as

    logit P(disease_i) = alpha
                         + gamma * logit(pretest_i)        [literature offset]
                         + x_i . beta                      [test results]
                         + u_site[i]                       [hospital effect]

* ``pretest_i`` is the Diamond-Forrester (1979) probability for the patient's
  age, sex and chest-pain type. ``gamma`` ~ N(1, 0.25) says "trust the
  literature, but let the data rescale it"; with gamma = 1 and every other
  term zero, the model *is* the literature prior.
* ``beta`` has clinically informed directional priors (see ``priors.py``).
* ``u_site`` uses partial pooling across hospitals (non-centred
  parameterisation). A hospital not seen in training is predicted by drawing
  its effect from the population distribution, N(0, tau).

Bayesian updating as new data arrive is supported by ``fit(..., prior=...)``:
the Gaussian approximation of one posterior becomes the prior for the next batch.
"""

from __future__ import annotations

import os
import warnings
import zlib
from dataclasses import dataclass, field

# PyTensor's default backend needs a C++ compiler; without one it warns at
# import. We sample through the Numba backend instead (see PYTENSOR_MODE).
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
from scipy.special import expit  # noqa: E402

from .config import RANDOM_SEED, SITES, TARGET  # noqa: E402
from .preprocessing import DiagnosisPreprocessor  # noqa: E402
from .priors import coefficient_prior_arrays, pretest_logit  # noqa: E402

# PyMC's NUTS sampler with the log-density compiled by Numba. Performance
# depends on the model and environment; no general speedup factor is asserted.
# Set MEDDX_NUTS_SAMPLER=nutpie to use the Rust sampler when available.
NUTS_SAMPLER: str = os.environ.get("MEDDX_NUTS_SAMPLER", "pymc")
PYTENSOR_MODE: str = os.environ.get("MEDDX_PYTENSOR_MODE", "NUMBA")


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one Bayesian model variant."""

    name: str
    description: str
    exclude: tuple[str, ...] = ()
    use_pretest_offset: bool = False
    informative_priors: bool = False
    offset_slope_prior: tuple[float, float] = (1.0, 0.25)
    hierarchical_site: bool = False
    # Raw categorical features whose effects get an adaptive (learned-scale)
    # shrinkage prior centred on zero: "deviations from the literature".
    adaptive_features: tuple[str, ...] = ()
    intercept_prior: tuple[float, float] = (0.0, 1.5)

    @property
    def supports_sequential_update(self) -> bool:
        return not self.hierarchical_site and not self.adaptive_features


@dataclass(frozen=True)
class GaussianPrior:
    """Multivariate normal prior on the stacked parameter vector."""

    mean: np.ndarray
    cov: np.ndarray
    names: tuple[str, ...]


@dataclass
class Posterior:
    """Posterior draws flattened over chains (S draws)."""

    intercept: np.ndarray                       # (S,)
    beta: np.ndarray                            # (S, p)
    gamma: np.ndarray | None = None             # (S,)
    site_effect: np.ndarray | None = None       # (S, K_train)
    tau_site: np.ndarray | None = None          # (S,)
    extra: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model variants used in the iteration loop (step 5 of the assignment).
# ---------------------------------------------------------------------------
BASELINE = ModelSpec(
    name="M0_weak",
    description="Weakly informative priors N(0,1) on every feature, no literature knowledge",
)
LITERATURE = ModelSpec(
    name="M1_literature",
    description="Diamond-Forrester pre-test offset + clinical directional priors",
    exclude=("age", "sex", "cp"),
    use_pretest_offset=True,
    informative_priors=True,
)
REFINED = ModelSpec(
    name="M2_refined",
    description=(
        "M1 + learned chest-pain deviations from the literature (adaptive shrinkage) "
        "+ hierarchical hospital intercepts"
    ),
    exclude=("age", "sex"),
    use_pretest_offset=True,
    informative_priors=True,
    hierarchical_site=True,
    adaptive_features=("cp",),
)
MODEL_SPECS: tuple[ModelSpec, ...] = (BASELINE, LITERATURE, REFINED)

# Ablation: M2's hospital structure with NO literature knowledge (all features,
# weak N(0,1) priors, no Diamond-Forrester predictor). Comparing it with M2
# holds hospital pooling fixed, but also changes feature encoding and shrinkage;
# it therefore does not isolate the contribution of medical priors alone.
REFINED_WEAK_ABLATION = ModelSpec(
    name="M2_weak_ablation",
    description="Hierarchical hospital intercepts + weak N(0,1) priors on all features, no literature knowledge",
    hierarchical_site=True,
)


class BayesianDiagnosisModel:
    """Fit / predict wrapper around a PyMC logistic model."""

    def __init__(
        self,
        spec: ModelSpec,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.spec = spec
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        prior: GaussianPrior | None = None,
        preprocessor: DiagnosisPreprocessor | None = None,
    ) -> "BayesianDiagnosisModel":
        """Sample the posterior given ``train``.

        ``prior`` replaces the default independent priors (sequential updating).
        ``preprocessor`` is an already-fitted preprocessor to reuse unchanged,
        so the parameter vector keeps its meaning across successive updates.
        """
        if prior is not None and not self.spec.supports_sequential_update:
            raise ValueError(f"{self.spec.name} does not support a Gaussian prior override")

        self.preprocessor_ = preprocessor or DiagnosisPreprocessor(exclude=self.spec.exclude).fit(train)
        X = self.preprocessor_.transform(train)
        y = train[TARGET].to_numpy(dtype=int)
        self.feature_names_ = list(self.preprocessor_.get_feature_names_out())

        offset = pretest_logit(train) if self.spec.use_pretest_offset else None
        site_idx = None
        if self.spec.hierarchical_site:
            self.train_sites_ = [s for s in SITES if (train["site"] == s).any()]
            site_idx = self._site_index(train, self.train_sites_)

        if prior is not None and tuple(prior.names) != tuple(self.parameter_names):
            raise ValueError("Prior parameter names do not match the model's parameters")

        self.model_ = self._build(X, y, offset, site_idx, prior)
        with self.model_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self.idata_ = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    cores=1,  # sequential chains: portable (Windows spawn) and memory-light
                    target_accept=self.target_accept,
                    random_seed=self.seed,
                    progressbar=False,
                    nuts_sampler=NUTS_SAMPLER,
                    **({"compile_kwargs": {"mode": PYTENSOR_MODE}} if NUTS_SAMPLER == "pymc" else {}),
                )
        self.posterior_ = self._extract_posterior()
        return self

    def loo(self) -> az.ELPDData:
        """PSIS leave-one-out cross-validation (log-likelihood computed lazily)."""
        if "log_likelihood" not in self.idata_.groups():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm.compute_log_likelihood(
                    self.idata_, model=self.model_, progressbar=False,
                    compile_kwargs={"mode": PYTENSOR_MODE},
                )
        return az.loo(self.idata_)

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: np.ndarray | None,
        site_idx: np.ndarray | None,
        prior: GaussianPrior | None,
    ) -> pm.Model:
        spec = self.spec
        names = self.feature_names_
        coords = {"feature": names}
        if site_idx is not None:
            coords["site"] = self.train_sites_

        adaptive_mask = np.array(
            [any(n.startswith(f"{f}_") and not n.endswith("_missing") for f in spec.adaptive_features)
             for n in names]
        )

        with pm.Model(coords=coords) as model:
            if prior is not None:
                theta = pm.MvNormal("theta", mu=prior.mean, cov=prior.cov)
                intercept = pm.Deterministic("intercept", theta[0])
                start = 1
                if spec.use_pretest_offset:
                    gamma = pm.Deterministic("gamma", theta[1])
                    start = 2
                beta = pm.Deterministic("beta", theta[start:], dims="feature")
            else:
                intercept = pm.Normal("intercept", *spec.intercept_prior)
                if spec.use_pretest_offset:
                    gamma = pm.Normal("gamma", *spec.offset_slope_prior)
                mu, sigma = coefficient_prior_arrays(names, spec.informative_priors)
                if adaptive_mask.any():
                    tau_dev = pm.HalfNormal("tau_dev", 1.0)
                    z = pm.Normal("z_beta", 0.0, 1.0, dims="feature")
                    scale = pm.math.where(adaptive_mask, tau_dev, sigma)
                    beta = pm.Deterministic("beta", mu + z * scale, dims="feature")
                else:
                    beta = pm.Normal("beta", mu=mu, sigma=sigma, dims="feature")

            eta = intercept + pm.math.dot(X, beta)
            if spec.use_pretest_offset:
                eta = eta + gamma * offset
            if site_idx is not None:
                tau_site = pm.HalfNormal("tau_site", 1.0)
                z_site = pm.Normal("z_site", 0.0, 1.0, dims="site")
                site_effect = pm.Deterministic("site_effect", z_site * tau_site, dims="site")
                eta = eta + site_effect[site_idx]

            pm.Bernoulli("y", logit_p=eta, observed=y)
        return model

    def _extract_posterior(self) -> Posterior:
        post = az.extract(self.idata_, group="posterior")
        get = lambda name: post[name].to_numpy().T if name in post else None  # noqa: E731
        beta = post["beta"].transpose("sample", "feature").to_numpy()
        site_effect = post["site_effect"].transpose("sample", "site").to_numpy() if "site_effect" in post else None
        return Posterior(
            intercept=post["intercept"].to_numpy(),
            beta=beta,
            gamma=get("gamma"),
            site_effect=site_effect,
            tau_site=get("tau_site"),
        )

    # -------------------------------------------------------------- predict
    def predict_samples(self, data: pd.DataFrame) -> np.ndarray:
        """Posterior draws of P(disease) with shape (S, n)."""
        post = self.posterior_
        X = self.preprocessor_.transform(data)
        eta = post.intercept[:, None] + post.beta @ X.T
        if self.spec.use_pretest_offset:
            eta += post.gamma[:, None] * pretest_logit(data)[None, :]
        if self.spec.hierarchical_site:
            eta += self._site_effects_for(data)
        return expit(eta)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Posterior mean probability of disease for each patient."""
        return self.predict_samples(data).mean(axis=0)

    def predict_interval(self, data: pd.DataFrame, mass: float = 0.94) -> pd.DataFrame:
        """Posterior mean and equal-tailed credible interval per patient."""
        samples = self.predict_samples(data)
        lo, hi = np.quantile(samples, [(1 - mass) / 2, 1 - (1 - mass) / 2], axis=0)
        return pd.DataFrame({"p_mean": samples.mean(axis=0), "p_low": lo, "p_high": hi})

    def _site_effects_for(self, data: pd.DataFrame) -> np.ndarray:
        """(S, n) hospital effects.

        Seen hospitals use their learned effect. An unseen hospital gets ONE
        effect per posterior draw, u_new[s] ~ N(0, tau[s]), shared by every
        patient at that hospital. That keeps the joint dependence implied by the
        hierarchical model. The draw is seeded by hospital name, so predictions
        don't depend on row order or on which other rows are in the batch.
        """
        post = self.posterior_
        S = post.intercept.shape[0]
        sites = data["site"].astype(str).to_numpy()
        out = np.empty((S, len(sites)))
        for site in np.unique(sites):
            cols = sites == site
            if site in self.train_sites_:
                effect = post.site_effect[:, self.train_sites_.index(site)]
            else:
                rng = np.random.default_rng([self.seed, zlib.crc32(site.encode("utf-8"))])
                effect = rng.standard_normal(S) * post.tau_site
            out[:, cols] = effect[:, None]
        return out

    @staticmethod
    def _site_index(data: pd.DataFrame, sites: list[str]) -> np.ndarray:
        lookup = {s: i for i, s in enumerate(sites)}
        return data["site"].astype(str).map(lookup).to_numpy(dtype=int)

    # ----------------------------------------------------- sequential update
    @property
    def parameter_names(self) -> list[str]:
        names = ["intercept"]
        if self.spec.use_pretest_offset:
            names.append("gamma")
        return names + [f"beta[{n}]" for n in self.feature_names_]

    def _stacked_draws(self) -> np.ndarray:
        post = self.posterior_
        cols = [post.intercept[:, None]]
        if self.spec.use_pretest_offset:
            cols.append(post.gamma[:, None])
        cols.append(post.beta)
        return np.hstack(cols)

    def gaussian_posterior(self, inflate: float = 1.0) -> GaussianPrior:
        """Moment-matched Gaussian approximation of the posterior, to be used
        as the prior for the next batch of data. ``inflate`` > 1 widens it
        (a "forgetting factor" for drift between batches)."""
        if not self.spec.supports_sequential_update:
            raise ValueError(f"{self.spec.name} does not support sequential updating")
        draws = self._stacked_draws()
        cov = np.cov(draws, rowvar=False) * inflate
        cov += np.eye(cov.shape[0]) * 1e-9  # numerical safety
        return GaussianPrior(mean=draws.mean(axis=0), cov=cov, names=tuple(self.parameter_names))

    def initial_prior(self, preprocessor: DiagnosisPreprocessor) -> GaussianPrior:
        """The model's own independent-normal prior as a GaussianPrior, for the
        feature layout of a fitted ``preprocessor``."""
        self.feature_names_ = list(preprocessor.get_feature_names_out())
        mu, sigma = coefficient_prior_arrays(self.feature_names_, self.spec.informative_priors)
        means = [self.spec.intercept_prior[0]]
        sds = [self.spec.intercept_prior[1]]
        if self.spec.use_pretest_offset:
            means.append(self.spec.offset_slope_prior[0])
            sds.append(self.spec.offset_slope_prior[1])
        mean = np.concatenate([means, mu])
        sd = np.concatenate([sds, sigma])
        return GaussianPrior(mean=mean, cov=np.diag(sd**2), names=tuple(self.parameter_names))

    def use_gaussian_belief(
        self, belief: GaussianPrior, preprocessor: DiagnosisPreprocessor, n_draws: int = 4000
    ) -> "BayesianDiagnosisModel":
        """Predict from a Gaussian belief without sampling (e.g. the prior
        before any patient data are seen)."""
        if not self.spec.supports_sequential_update:
            raise ValueError(f"{self.spec.name} does not support Gaussian beliefs")
        self.preprocessor_ = preprocessor
        self.feature_names_ = list(preprocessor.get_feature_names_out())
        if tuple(belief.names) != tuple(self.parameter_names):
            raise ValueError("Belief parameter names do not match the model's parameters")
        draws = np.random.default_rng(self.seed).multivariate_normal(belief.mean, belief.cov, size=n_draws)
        start = 2 if self.spec.use_pretest_offset else 1
        self.posterior_ = Posterior(
            intercept=draws[:, 0],
            gamma=draws[:, 1] if self.spec.use_pretest_offset else None,
            beta=draws[:, start:],
        )
        return self

    # ---------------------------------------------------------- diagnostics
    def coefficient_summary(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        var_names = ["intercept", "beta"]
        if self.spec.use_pretest_offset:
            var_names.append("gamma")
        if self.spec.hierarchical_site:
            var_names += ["site_effect", "tau_site"]
        if self.spec.adaptive_features:
            var_names.append("tau_dev")
        return az.summary(self.idata_, var_names=var_names, hdi_prob=hdi_prob)

    def convergence(self) -> dict[str, float]:
        summary = self.coefficient_summary()
        divergences = int(self.idata_.sample_stats["diverging"].sum())
        return {
            "max_rhat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "divergences": divergences,
        }
