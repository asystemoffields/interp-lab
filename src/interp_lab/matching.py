from __future__ import annotations

import heapq

from interp_lab.math_utils import clamp, cosine
from interp_lab.schema import CandidateMatch, FeatureCard, FeatureFingerprint


DEFAULT_WEIGHTS = {
    "text": 0.35,
    "activation": 0.25,
    "decoder": 0.20,
    "causal": 0.20,
}

# Magnitude floor for treating two signed effects as a genuine opposite-direction
# conflict. Kept equal to match_validation.DEFAULT_MIN_ABS_SIGNED_EFFECT so the
# ranking layer and the validation layer agree on what counts as a real signed
# effect (otherwise a pair can score high here yet be graded "contradicted").
SIGNED_EFFECT_DIRECTION_MIN = 0.02


def fingerprint_similarity(
    left: FeatureFingerprint,
    right: FeatureFingerprint,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    weights = weights or DEFAULT_WEIGHTS
    components: dict[str, float] = {}
    active: dict[str, float] = {}
    # Activation and decoder signatures are only comparable when BOTH sides carry a
    # vector of the SAME length. An absent or differently-sized signature (two models
    # with different hidden sizes, or a feature dump with no decoder) would otherwise
    # make cosine() return 0.0, which _to_unit maps to a free 0.5 "half match" credited
    # at full weight. Exclude the component and renormalize the rest instead -- exactly
    # like the text and causal gates below -- so an unmeasured axis adds no evidence.
    if _vectors_comparable(left.activation_signature, right.activation_signature):
        components["activation"] = _to_unit(cosine(left.activation_signature, right.activation_signature))
        active["activation"] = weights.get("activation", 0.0)
    else:
        components["activation_absent"] = 1.0
    if _vectors_comparable(left.decoder_signature, right.decoder_signature):
        components["decoder"] = _to_unit(cosine(left.decoder_signature, right.decoder_signature))
        active["decoder"] = weights.get("decoder", 0.0)
    else:
        components["decoder_absent"] = 1.0
    # Only compare text vectors produced by the same embedder; mixing a lexical-hash
    # fingerprint with a semantic one would silently cosine over truncated, unrelated axes.
    if left.text_embedder == right.text_embedder and _vectors_comparable(left.text_vector, right.text_vector):
        components["text"] = _to_unit(cosine(left.text_vector, right.text_vector))
        active["text"] = weights.get("text", 0.0)
    else:
        components["text_embedder_mismatch"] = 1.0
    # Only credit causal similarity when BOTH sides carry a comparable causal vector of
    # the SAME, real provenance. An absent or provenance-"none" causal vector must not
    # earn a free contribution (it is excluded and the remaining weights are
    # renormalized), and a measured causal effect must never be cosine-compared against
    # a correlational proxy.
    if (
        _vectors_comparable(left.causal_vector, right.causal_vector)
        and left.causal_provenance == right.causal_provenance
        and left.causal_provenance != "none"
    ):
        components["causal"] = _causal_similarity(left.causal_vector, right.causal_vector)
        active["causal"] = weights.get("causal", 0.0)
    else:
        components["causal_absent"] = 1.0
    total = sum(active.values()) or 1.0
    score = sum(components[name] * active[name] for name in active) / total
    return round(clamp(score), 6), {key: round(value, 6) for key, value in components.items()}


def match_feature_cards(
    left_cards: list[FeatureCard],
    right_cards: list[FeatureCard],
    *,
    top_k: int = 10,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
) -> list[CandidateMatch]:
    if top_k <= 0:
        return []
    heap: list[tuple[float, int, CandidateMatch]] = []
    counter = 0
    for left in left_cards:
        for right in right_cards:
            score, components = fingerprint_similarity(left.fingerprint, right.fingerprint, weights=weights)
            left_signed = _signed_effect(left)
            right_signed = _signed_effect(right)
            if left_signed is not None and right_signed is not None:
                signed_component = _signed_effect_similarity(left_signed, right_signed)
                components["signed_effect"] = round(signed_component, 6)
                score = _score_with_signed_effect(score, signed_component, left_signed, right_signed)
            if score >= min_score:
                match = CandidateMatch(
                    left_feature_id=left.feature_id,
                    right_feature_id=right.feature_id,
                    left_model=left.model,
                    right_model=right.model,
                    score=score,
                    components=components,
                    left_label=left.label,
                    right_label=right.label,
                    left_signed_effect=left_signed,
                    right_signed_effect=right_signed,
                )
                item = (match.score, counter, match)
                counter += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif match.score > heap[0][0]:
                    heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _vectors_comparable(left: list[float], right: list[float]) -> bool:
    """Both vectors present and the same length, so a cosine is meaningful."""
    return bool(left) and bool(right) and len(left) == len(right)


def _to_unit(value: float) -> float:
    return clamp((value + 1.0) / 2.0)


def _signed_effect(card: FeatureCard) -> float | None:
    for key in ("signed_causal_effect", "signed_association"):
        if key in card.causal_effects:
            return float(card.causal_effects[key])
    return None


def _signed_effect_similarity(left: float, right: float) -> float:
    return clamp(1.0 - abs(left - right) / 2.0)


def _score_with_signed_effect(score: float, signed_component: float, left: float, right: float) -> float:
    adjusted = clamp(0.85 * score + 0.15 * signed_component)
    if left * right < 0 and min(abs(left), abs(right)) >= SIGNED_EFFECT_DIRECTION_MIN:
        adjusted = min(adjusted, 0.49)
    return round(adjusted, 6)


def _causal_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.5  # defensive: callers exclude absent causal vectors before this
    direction = _to_unit(cosine(left, right))
    strength = min(1.0, _vector_norm(left)) * min(1.0, _vector_norm(right))
    return clamp(direction * strength)


def _vector_norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5
