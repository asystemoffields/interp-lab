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
    size = min(len(left), len(right))
    left_trimmed = left[:size]
    right_trimmed = right[:size]
    denominator = norm(left_trimmed) * norm(right_trimmed)
    if denominator == 0:
        return 0.0
    return dot(left_trimmed, right_trimmed) / denominator


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    size = min(len(left), len(right))
    left_trimmed = left[:size]
    right_trimmed = right[:size]
    left_mean = mean(left_trimmed)
    right_mean = mean(right_trimmed)
    centered_left = [value - left_mean for value in left_trimmed]
    centered_right = [value - right_mean for value in right_trimmed]
    denominator = norm(centered_left) * norm(centered_right)
    if denominator == 0:
        return 0.0
    return dot(centered_left, centered_right) / denominator
