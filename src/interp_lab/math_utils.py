from __future__ import annotations

import math
from collections.abc import Sequence


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    # Vectors of different lengths are not comparable: silently trimming to the
    # shared prefix would cosine over an arbitrary, misaligned subspace (e.g. two
    # models with different hidden sizes). Treat that as "no evidence" instead.
    if len(left) != len(right):
        return 0.0
    denominator = norm(left) * norm(right)
    if denominator == 0:
        return 0.0
    return dot(left, right) / denominator


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    # Paired statistic: mismatched lengths mean the pairs are misaligned, so the
    # correlation is undefined rather than something to compute on a prefix.
    if len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = norm(centered_left) * norm(centered_right)
    if denominator == 0:
        return 0.0
    return dot(centered_left, centered_right) / denominator
