import math

from interp_lab import stats


def test_t_interval_is_wider_than_normal_for_small_n():
    values = [0.2, 0.4]
    ci = stats.mean_confidence_interval(values)
    assert ci["method"] == "student_t"
    assert ci["n"] == 2
    # Sample std (n-1) of [0.2, 0.4] is 0.141..., std error 0.1. A normal z=1.96
    # interval would be +-0.196; the t_1 critical value (12.706) must be far wider.
    half_width = (ci["high"] - ci["low"]) / 2.0
    assert half_width > 1.0
    assert ci["low"] < ci["mean"] < ci["high"]


def test_t_interval_single_observation_reports_no_interval():
    ci = stats.mean_confidence_interval([0.5])
    assert ci["n"] == 1
    assert ci["method"] == "insufficient_n"
    assert ci["low"] is None and ci["high"] is None
    assert ci["mean"] == 0.5


def test_t_interval_empty_is_none():
    assert stats.mean_confidence_interval([]) is None


def test_t_critical_matches_table_and_converges_to_z():
    assert stats.t_critical(1) == 12.706
    assert stats.t_critical(2) == 4.303
    # For large df the t critical value collapses toward the normal z (~1.96).
    assert abs(stats.t_critical(100000) - 1.959964) < 1e-2


def test_bootstrap_interval_is_reproducible_by_seed():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    a = stats.bootstrap_mean_interval(values, seed=7, n_resamples=500)
    b = stats.bootstrap_mean_interval(values, seed=7, n_resamples=500)
    c = stats.bootstrap_mean_interval(values, seed=8, n_resamples=500)
    assert a == b
    assert a["method"] == "bootstrap_percentile"
    assert a["low"] < a["high"]
    # A different seed should (almost surely) move the percentile bounds.
    assert (a["low"], a["high"]) != (c["low"], c["high"])


def test_permutation_test_flags_strong_correlation():
    x = [float(i) for i in range(12)]
    y = [2.0 * i + 1.0 for i in range(12)]  # perfectly correlated
    result = stats.permutation_test(x, y, n_resamples=500, seed=0)
    assert result["method"] == "permutation_pearson"
    assert result["statistic"] > 0.99
    assert result["p_value"] is not None and result["p_value"] < 0.05


def test_permutation_test_insufficient_n():
    result = stats.permutation_test([1.0, 2.0], [1.0, 2.0])
    assert result["p_value"] is None
    assert result["note"] == "insufficient_n"
