"""Small, dependency-free statistics helpers for honest effect reporting.

Every estimate this toolkit grades comes from a finite, usually *small* sample of
prompts or intervention runs. The functions here make the uncertainty in those
estimates explicit instead of pretending a handful of points pins a mean exactly:

- :func:`mean_confidence_interval` uses a Student-t critical value (not a fixed
  ``z = 1.96``) and the sample standard deviation, so a CI from 3 points is wider
  than one from 300, and a single point reports *no* interval rather than a
  zero-width one.
- :func:`bootstrap_mean_interval` gives a seeded, reproducible percentile
  bootstrap for when the normal approximation is doubtful.
- :func:`permutation_test` provides a null distribution for an
  association/correlation statistic, so "this feature tracks the criterion" can be
  reported with a p-value instead of an uncorroborated correlation.

Pure stdlib (``math``, ``random``, ``statistics``); safe to import anywhere.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

# Two-sided Student-t critical values at 95% confidence, indexed by degrees of
# freedom. Small df is exactly where the toolkit operates (a few prompts), and it
# is exactly where t diverges most from the normal z, so we keep an exact table
# for df 1..30 and fall back to a Cornish-Fisher expansion beyond that.
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

_Z_95 = 1.959964


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation of the standard-normal quantile."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Two-sided Student-t critical value for ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if abs(confidence - 0.95) < 1e-9 and df in _T_TABLE_95:
        return _T_TABLE_95[df]
    z = _Z_95 if abs(confidence - 0.95) < 1e-9 else _inverse_normal_cdf(1 - (1 - confidence) / 2)
    # Cornish-Fisher expansion of the t-quantile in terms of the normal quantile.
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def mean_confidence_interval(values: Sequence[float], *, confidence: float = 0.95) -> dict | None:
    """Student-t confidence interval for the mean of ``values``.

    Returns a dict with ``low``/``high``/``mean``/``n``/``method``. With a single
    observation the interval is reported as ``None`` bounds (method
    ``insufficient_n``) rather than a misleading zero-width interval; with no
    observations the function returns ``None``.
    """
    n = len(values)
    if n == 0:
        return None
    m = mean(values)
    if n == 1:
        return {
            "mean": round(m, 6),
            "low": None,
            "high": None,
            "n": 1,
            "confidence": confidence,
            "method": "insufficient_n",
        }
    variance = sum((value - m) ** 2 for value in values) / (n - 1)
    std_error = math.sqrt(variance) / math.sqrt(n)
    half_width = t_critical(n - 1, confidence) * std_error
    return {
        "mean": round(m, 6),
        "low": round(m - half_width, 6),
        "high": round(m + half_width, 6),
        "n": n,
        "std_error": round(std_error, 6),
        "confidence": confidence,
        "method": "student_t",
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict | None:
    """Seeded percentile-bootstrap interval for the mean (reproducible by ``seed``)."""
    n = len(values)
    if n == 0:
        return None
    m = mean(values)
    if n == 1:
        return {
            "mean": round(m, 6),
            "low": None,
            "high": None,
            "n": 1,
            "confidence": confidence,
            "method": "insufficient_n",
        }
    rng = random.Random(seed)
    pool = list(values)
    resampled_means = []
    for _ in range(n_resamples):
        sample = [pool[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(sample) / n)
    resampled_means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(math.floor(alpha * n_resamples)))
    high_index = min(n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    return {
        "mean": round(m, 6),
        "low": round(resampled_means[low_index], 6),
        "high": round(resampled_means[high_index], 6),
        "n": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence": confidence,
        "method": "bootstrap_percentile",
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n < 2 or n != len(right):
        return 0.0
    lm = mean(left)
    rm = mean(right)
    cl = [value - lm for value in left]
    cr = [value - rm for value in right]
    denom = math.sqrt(sum(value * value for value in cl)) * math.sqrt(sum(value * value for value in cr))
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(cl, cr)) / denom


def permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Two-sided permutation test for the Pearson correlation between ``x`` and ``y``.

    Shuffles ``y`` against ``x`` ``n_resamples`` times and counts how often the
    permuted |correlation| matches or beats the observed one. Uses the (count+1)/
    (resamples+1) estimator so the p-value is never exactly zero. Reproducible by
    ``seed``; returns ``p_value = None`` when there are too few points to test.
    """
    n = len(x)
    observed = _pearson(x, y)
    if n < 3 or n != len(y):
        return {
            "statistic": round(observed, 6),
            "p_value": None,
            "n": n,
            "n_resamples": 0,
            "seed": seed,
            "method": "permutation_pearson",
            "note": "insufficient_n",
        }
    rng = random.Random(seed)
    permuted = list(y)
    threshold = abs(observed) - 1e-12
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        rng.shuffle(permuted)
        if abs(_pearson(x, permuted)) >= threshold:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_resamples + 1)
    return {
        "statistic": round(observed, 6),
        "p_value": round(p_value, 6),
        "n": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "permutation_pearson",
    }
