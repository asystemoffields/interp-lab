from __future__ import annotations

from oracle_sae.math_utils import clamp, cosine, mean
from oracle_sae.schema import Criterion, FeatureEvidence
from oracle_sae.text_vectors import hash_text_vector


def score_feature(evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
    criterion_vector = hash_text_vector(criterion.text)
    label_vector = hash_text_vector(" ".join([evidence.label, *evidence.examples]))
    association = (cosine(criterion_vector, label_vector) + 1.0) / 2.0
    causal_effect = clamp(float(evidence.causal_effects.get("criterion", 0.0)))
    specificity = clamp(float(evidence.causal_effects.get("specificity", association)))
    side_effect = clamp(float(evidence.causal_effects.get("side_effect", 0.0)))
    stability = _stability(evidence)
    importance = clamp(
        0.30 * association
        + 0.35 * causal_effect
        + 0.20 * specificity
        + 0.15 * stability
        - 0.10 * side_effect
    )
    return {
        "importance": round(importance, 6),
        "association": round(association, 6),
        "specificity": round(specificity, 6),
        "causal_effect": round(causal_effect, 6),
        "stability": round(stability, 6),
    }


def _stability(evidence: FeatureEvidence) -> float:
    values = [abs(value) for value in evidence.activation_signature[:16]]
    if not values:
        return 0.25
    average = mean(values)
    return clamp(0.35 + average)
