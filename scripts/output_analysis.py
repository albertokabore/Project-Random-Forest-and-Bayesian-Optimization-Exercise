"""Derive interpretation text and its numerical evidence from saved outputs."""

from __future__ import annotations

import json
from pathlib import Path


def derive_output_analysis(root: Path) -> tuple[dict[int, list[str]], str, str, dict]:
    reports = root / "reports"
    original = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    verification = json.loads((reports / "review_verification.json").read_text(encoding="utf-8"))
    corrected = json.loads((reports / "corrected/corrected_results.json").read_text(encoding="utf-8"))
    points = {row["model"]: row for row in verification["test_metrics"]}
    rounded = {row["model"]: row for row in original["test_metrics"]}
    sites = {row["site"]: row for row in original["sites"]}
    cp = {row["chest_pain"]: row for row in original["prior_vs_observed_cp"]}
    prevalence = {row["site"]: row for row in verification["prevalence_updates"]}
    hyper = {row["param"]: row for row in original["m2_hyperparameters"]}
    coefficients = {row["feature"]: row for row in original["m2_coefficients"]}
    m0 = points["M0 Bayesian (weak priors)"]
    m1 = points["M1 Bayesian (literature priors)"]
    m2 = points["M2 Bayesian (refined)"]
    rf_default = points["Random Forest (default)"]
    rf_tuned = points["Random Forest (Bayesian-optimised)"]
    test_n = original["split"]["test"]
    positive_n = round(test_n * original["split"]["test_disease_rate"])
    negative_n = test_n - positive_n
    original_correct = round(m2["accuracy"] * test_n)
    forest_default_correct = round(rf_default["accuracy"] * test_n)
    forest_tuned_correct = round(rf_tuned["accuracy"] * test_n)
    pair = corrected["main"]["paired_bootstrap"]["M2_minus_M0"]
    companion_pair = corrected["main"]["paired_bootstrap"]["M2_minus_M2weak_ablation"]
    corrected_m2 = corrected["main"]["metrics"]["M2_refined"]
    companion = corrected["main"]["metrics"]["M2_weak_ablation"]
    fn = positive_n - round(corrected_m2["sensitivity"] * positive_n)
    fp = negative_n - round(corrected_m2["specificity"] * negative_n)
    pooled = corrected["loho_summary"]["pooled_out_of_hospital"]["M2 Bayesian"]
    pool_n = sum(fold["n"] for fold in corrected["loho"].values())
    pool_correct = round(pooled["accuracy"] * pool_n)
    va = corrected["loho"]["VA Long Beach"]
    va_spec = va["metrics"]["M2 Bayesian"]["specificity"]
    va_negative_n = va["n"] - round(va["prevalence"] * va["n"])
    va_fp = va_negative_n - round(va_spec * va_negative_n)
    swiss_negative_n = corrected["loho"]["Switzerland"]["n"] - round(
        corrected["loho"]["Switzerland"]["n"] * corrected["loho"]["Switzerland"]["prevalence"]
    )
    prevalence_gap = 100 * (sites["Switzerland"]["disease_rate"] - sites["Hungary"]["disease_rate"])
    observed_asym = cp["Asymptomatic"]["Observed (training data)"]
    prior_asym = cp["Asymptomatic"]["Literature pre-test (mean)"]
    asym_gap = 100 * (observed_asym - prior_asym)
    search = original["rf_search"]
    search_gain = (search["random_best_cv_log_loss"] - search["bo_best_cv_log_loss"]) / search["random_best_cv_log_loss"]
    seq = original["sequential_updating"]
    sd_reduction = 1 - seq[-1]["mean_posterior_sd"] / seq[0]["mean_posterior_sd"]
    final_loss_increase = seq[-1]["test_log_loss"] - seq[-2]["test_log_loss"]
    ece_m0 = rounded["M0 Bayesian (weak priors)"]["ece"]
    ece_m2 = rounded["M2 Bayesian (refined)"]["ece"]

    observations = {
        1: [
            f"Disease prevalence ranges from {sites['Hungary']['disease_rate']:.1%} in Hungary to {sites['Switzerland']['disease_rate']:.1%} in Switzerland. This {prevalence_gap:.1f} percentage-point difference indicates substantial hospital heterogeneity, limiting interpretation of pooled accuracy alone.",
            f"The test subset contains {positive_n} positive and {negative_n} negative records. Consequently, predicting every case as positive achieves {positive_n / test_n:.1%} accuracy, establishing a basic comparator for classification.",
        ],
        2: [
            f"Asymptomatic records show {observed_asym:.1%} observed prevalence against an {prior_asym:.1%} literature estimate, a {asym_gap:.1f} percentage-point gap. The historical table therefore substantially underestimates this training subgroup's prevalence.",
            f"Swiss prevalence updating raises the prior mean from {prevalence['Switzerland']['prior_mean']:.1%} to {prevalence['Switzerland']['posterior_mean']:.1%}, below the test prevalence of {prevalence['Switzerland']['test_prevalence']:.1%}. This remaining gap motivates evaluating alternative prior strengths.",
        ],
        3: [
            f"The literature coefficient falls from prior mean 1.000 to posterior mean {hyper['gamma']['mean']:.3f}. The asymptomatic deviation reaches {coefficients['cp_4']['mean']:.3f}, indicating substantial adjustment beyond the historical probability predictor.",
            f"TPE reduces the best cross-validation log loss from {search['random_best_cv_log_loss']:.4f} to {search['bo_best_cv_log_loss']:.4f}, approximately {search_gain:.1%} relative improvement. This supports a modest optimization advantage within this seeded run.",
        ],
        4: [
            f"Corrected M2 achieves {corrected_m2['accuracy']:.1%} accuracy: {fn} false negatives and {fp} false positives. Its {pair['auc_diff']:.4f} AUC advantage over M0 is uncertain; the paired interval includes zero.",
            f"Pooled hospital accuracy falls to {pooled['accuracy']:.1%}, compared with {m2['accuracy']:.1%} internally. VA specificity reaches {va_spec:.1%}, implying frequent false positives despite an overall pooled AUC of {pooled['roc_auc']:.4f}.",
        ],
        5: [
            f"Sequential updating reduces mean coefficient SD from {seq[0]['mean_posterior_sd']:.4f} to {seq[-1]['mean_posterior_sd']:.4f}. However, final log loss rises from {seq[-2]['test_log_loss']:.4f} to {seq[-1]['test_log_loss']:.4f}, separating parameter precision from predictive improvement.",
            f"The companion model reaches AUC {companion['roc_auc']:.4f} versus M2's {corrected_m2['roc_auc']:.4f}, with a difference interval crossing zero. Because predictor encoding also changes, individual prior contributions remain unresolved.",
        ],
        6: [
            f"Original M2 has lower Brier loss: {m2['brier']:.4f} versus M0's {m0['brier']:.4f}, but higher binned calibration error, {ece_m2:.4f} versus {ece_m0:.4f}. Lower loss does not guarantee better calibration.",
            f"The hospital analysis yields {pool_correct} correct classifications among {pool_n} records, below internal accuracy. Reporting {pooled['accuracy']:.1%} alongside {m2['accuracy']:.1%} demonstrates the practical performance gap associated with transfer.",
        ],
    }
    for step, items in observations.items():
        assert sum(len(item.split()) for item in items) == 50, (step, items)

    insights = f"""- Original M2 and the default forest both classify {original_correct} of {test_n} records correctly ({m2['accuracy']:.1%}), but M2 has higher AUC ({m2['roc_auc']:.4f} versus {rf_default['roc_auc']:.4f}) and lower log loss ({m2['log_loss']:.4f} versus {rf_default['log_loss']:.4f}). Equal accuracy therefore conceals differences in ranking and probability estimation. In the corrected fits, M2 minus M0 AUC is {pair['auc_diff']:.4f} with a 95% interval of {pair['auc_diff_95ci'][0]:.4f} to {pair['auc_diff_95ci'][1]:.4f}; superiority over M0 is not established.
- TPE improves the best CV log loss by {search_gain:.1%} relative to random search in the recorded {search['n_trials']}-trial comparison. Forest test AUC increases from {rf_default['roc_auc']:.4f} to {rf_tuned['roc_auc']:.4f}, while correct classifications decrease from {forest_default_correct} to {forest_tuned_correct}. The optimization improves its probability-based objective without improving every threshold-based outcome.
- The asymptomatic training subgroup has observed prevalence {observed_asym:.1%} and a historical estimate of {prior_asym:.1%}. Original M1 log loss is {m1['log_loss']:.4f}, compared with {m0['log_loss']:.4f} for M0. Together, these outputs indicate poor transportability of the unadjusted historical probabilities in this cohort; they do not identify the cause. The corrected companion comparison gives an AUC difference of {companion_pair['auc_diff']:.4f} (95% interval {companion_pair['auc_diff_95ci'][0]:.4f} to {companion_pair['auc_diff_95ci'][1]:.4f}) and also changes predictor encoding and shrinkage, so the individual contribution of priors remains unresolved.
- M2 accuracy decreases from {m2['accuracy']:.1%} on the internal split to {pooled['accuracy']:.1%} over hospital holdouts, a descriptive difference of {100 * (m2['accuracy'] - pooled['accuracy']):.1f} percentage points. At VA Long Beach, specificity {va_spec:.1%} corresponds to {va_fp} false positives among {va_negative_n} negative records. The pooled AUC of {pooled['roc_auc']:.4f} therefore does not imply adequate specificity at every site, and the difference between validation settings does not isolate a causal hospital effect.
- M2 has lower original Brier loss than M0 ({m2['brier']:.4f} versus {m0['brier']:.4f}) but higher binned calibration error ({ece_m2:.4f} versus {ece_m0:.4f}). The model ranking depends on the probability metric; these outputs do not support a claim that M2 is uniformly best calibrated.
- Sequential updating reduces mean coefficient SD by {sd_reduction:.1%}, from {seq[0]['mean_posterior_sd']:.4f} to {seq[-1]['mean_posterior_sd']:.4f}. Nevertheless, final AUC decreases from {seq[-2]['test_auc']:.4f} to {seq[-1]['test_auc']:.4f}, while log loss increases by {final_loss_increase:.4f}. Reduced parameter uncertainty is therefore not evidence of improved prediction after every update."""

    recommendations = f"""- Compare M0 and M2 through repeated validation before selecting a final model. The corrected AUC difference of {pair['auc_diff']:.4f} has an interval spanning {pair['auc_diff_95ci'][0]:.4f} to {pair['auc_diff_95ci'][1]:.4f}; the current sample cannot establish which has better discrimination.
- Evaluate prior strengths within training folds while holding predictor encoding, hospital structure, and shrinkage constant. The {asym_gap:.1f} percentage-point asymptomatic prevalence gap and M1 log loss of {m1['log_loss']:.4f} motivate reassessment, while the companion model's changed representation prevents attribution of its similar AUC to prior removal alone.
- Assess hospital-level calibration and threshold trade-offs using separate validation data. VA Long Beach produces {va_fp} false positives among {va_negative_n} negative cases at threshold 0.5, and Switzerland has only {swiss_negative_n} negative records. Additional negative cases and uncertainty intervals are required before selecting site-specific decision thresholds.
- Evaluate calibration curves, intercepts, slopes, and validation-based recalibration alongside proper probability scores. M2's lower Brier loss ({m2['brier']:.4f}) and higher binned error ({ece_m2:.4f}) relative to M0 ({m0['brier']:.4f} and {ece_m0:.4f}) show why a single probability metric is insufficient for model selection.
- Monitor predictive performance after each labelled-data update. Final sequential log loss increases from {seq[-2]['test_log_loss']:.4f} to {seq[-1]['test_log_loss']:.4f} despite a {sd_reduction:.1%} overall reduction in mean coefficient SD. Update acceptance criteria should therefore include prospective predictive scores rather than parameter precision alone.
- Prioritize independent hospital evaluation over additional internal tuning. M2's {original_correct}/{test_n} correct internal predictions become {pool_correct}/{pool_n} in the hospital evaluation, and source prevalence ranges from {sites['Hungary']['disease_rate']:.1%} to {sites['Switzerland']['disease_rate']:.1%}. These outputs justify evaluation across different prevalence and missingness patterns before considering clinical use."""

    trace = {
        "source_files": ["reports/metrics.json", "reports/review_verification.json", "reports/corrected/corrected_results.json"],
        "derivations": {
            "hospital_prevalence_gap_percentage_points": prevalence_gap,
            "test_always_positive_accuracy": positive_n / test_n,
            "asymptomatic_gap_percentage_points_from_displayed_training_outputs": asym_gap,
            "tpe_relative_cv_log_loss_improvement": search_gain,
            "internal_correct_classifications": original_correct,
            "corrected_hospital_correct_classifications": pool_correct,
            "va_negative_records": va_negative_n,
            "va_false_positives": va_fp,
            "swiss_negative_records": swiss_negative_n,
            "sequential_mean_coefficient_sd_relative_reduction": sd_reduction,
            "final_sequential_log_loss_increase": final_loss_increase,
        },
        "interpretation_limits": [
            "Original and corrected fits are identified separately.",
            "Intervals containing zero do not establish superiority or equivalence.",
            "Companion comparisons change predictor encoding and shrinkage as well as priors.",
            "Recommendations propose further analyses; they are not presented as completed experiments.",
        ],
    }
    return observations, insights, recommendations, trace
