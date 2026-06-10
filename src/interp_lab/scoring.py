from __future__ import annotations

from interp_lab.matching import has_intervention_provenance
from interp_lab.math_utils import clamp, cosine, mean
from interp_lab.schema import Criterion, FeatureEvidence
from interp_lab.text_embedding import embed_text


def score_feature(evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
    association = _association(evidence, criterion)
    # The "criterion" key only counts as causal evidence when intervention provenance
    # is present (signed_causal_effect / intervention records). The records backend
    # publishes its correlational score under the same key
    # ({"criterion": abs(r), "signed_association": r}); without interventions that
    # Pearson r is already counted on the association axis, so crediting it here too
    # would double-count it AND surface a correlation as FeatureCard.causal_effect.
    if has_intervention_provenance(evidence.causal_effects, evidence.metadata):
        causal_effect = clamp(float(evidence.causal_effects.get("criterion", 0.0)))
    else:
        causal_effect = 0.0
    # When the side-effect/control-adjusted causal score is absent, it must default
    # to 0.0 -- NOT to causal_effect -- or the same causal number would be counted
    # twice (0.30 strong + 0.20 causal = 0.50). Likewise an absent specificity must
    # not borrow association, which would double-count association (0.25 + 0.15).
    strong_causal_score = clamp(float(evidence.causal_effects.get("strong_causal_score", 0.0)))
    specificity = clamp(float(evidence.causal_effects.get("specificity", 0.0)))
    side_effect = clamp(float(evidence.causal_effects.get("side_effect", 0.0)))
    stability = _stability(evidence)
    importance = clamp(
        0.25 * association
        + 0.30 * strong_causal_score
        + 0.20 * causal_effect
        + 0.15 * specificity
        + 0.10 * stability
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


def _association(evidence: FeatureEvidence, criterion: Criterion) -> float:
    signed = evidence.causal_effects.get("signed_association")
    if signed is not None:
        return clamp(abs(float(signed)))
    criterion_vector = embed_text(criterion.text)
    label_vector = embed_text(" ".join([evidence.label, *evidence.examples]))
    # No measured association: fall back to text similarity, but do NOT unit-map the
    # cosine -- (cosine+1)/2 hands unrelated hash-text vectors a free ~0.5 baseline
    # that outranks a measured weak association (0.10). Clamp instead so unrelated
    # text scores ~0 (mirrors the fingerprint-similarity gating in matching.py).
    return clamp(cosine(criterion_vector, label_vector))
