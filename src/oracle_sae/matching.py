from __future__ import annotations

from oracle_sae.math_utils import clamp, cosine
from oracle_sae.schema import CandidateMatch, FeatureCard, FeatureFingerprint


DEFAULT_WEIGHTS = {
    "text": 0.35,
    "activation": 0.25,
    "decoder": 0.20,
    "causal": 0.20,
}


def fingerprint_similarity(
    left: FeatureFingerprint,
    right: FeatureFingerprint,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    weights = weights or DEFAULT_WEIGHTS
    components = {
        "text": _to_unit(cosine(left.text_vector, right.text_vector)),
        "activation": _to_unit(cosine(left.activation_signature, right.activation_signature)),
        "decoder": _to_unit(cosine(left.decoder_signature, right.decoder_signature)),
        "causal": _causal_similarity(left.causal_vector, right.causal_vector),
    }
    score = sum(components[name] * weights.get(name, 0.0) for name in components)
    return round(clamp(score), 6), {key: round(value, 6) for key, value in components.items()}


def match_feature_cards(
    left_cards: list[FeatureCard],
    right_cards: list[FeatureCard],
    *,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[CandidateMatch]:
    matches: list[CandidateMatch] = []
    for left in left_cards:
        for right in right_cards:
            score, components = fingerprint_similarity(left.fingerprint, right.fingerprint)
            left_signed = _signed_effect(left)
            right_signed = _signed_effect(right)
            if left_signed is not None and right_signed is not None:
                signed_component = _signed_effect_similarity(left_signed, right_signed)
                components["signed_effect"] = round(signed_component, 6)
                score = _score_with_signed_effect(score, signed_component, left_signed, right_signed)
            if score >= min_score:
                matches.append(
                    CandidateMatch(
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
                )
    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:top_k]


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
    if left * right < 0 and min(abs(left), abs(right)) >= 0.05:
        adjusted = min(adjusted, 0.49)
    return round(adjusted, 6)


def _causal_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.5
    direction = _to_unit(cosine(left, right))
    strength = min(1.0, _vector_norm(left)) * min(1.0, _vector_norm(right))
    return clamp(direction * strength)


def _vector_norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5
