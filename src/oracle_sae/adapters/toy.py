from __future__ import annotations

import hashlib
import math
import random

from oracle_sae.schema import Criterion, FeatureEvidence
from oracle_sae.text_vectors import content_tokens


class ToyFeatureProvider:
    """Deterministic feature evidence for demos and tests."""

    def __init__(self, *, feature_count: int = 16, dimensions: int = 32):
        self.feature_count = feature_count
        self.dimensions = dimensions

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        tokens = content_tokens(criterion.text) or ["criterion"]
        features: list[FeatureEvidence] = []
        for index in range(self.feature_count):
            token = tokens[index % len(tokens)]
            label = self._label_for(index, token, criterion)
            seed = _seed(model, criterion.text, str(index))
            rng = random.Random(seed)
            activation = _unit_vector([rng.uniform(-1, 1) for _ in range(self.dimensions)])
            decoder = _unit_vector([rng.uniform(-1, 1) for _ in range(self.dimensions)])
            alignment = max(0.0, 1.0 - (index / max(1, self.feature_count - 1)))
            causal = {
                "criterion": round(0.16 + 0.74 * alignment + rng.uniform(-0.03, 0.03), 4),
                "specificity": round(0.22 + 0.60 * alignment + rng.uniform(-0.04, 0.04), 4),
                "side_effect": round(0.08 + 0.20 * (1.0 - alignment), 4),
            }
            features.append(
                FeatureEvidence(
                    feature_id=f"L{8 + index % 18}:F{abs(seed) % 100000}",
                    model=model,
                    layer=8 + index % 18,
                    label=label,
                    examples=[
                        f"Example with {token}: {criterion.positive_examples[0]}",
                        f"Near miss for {token}: {criterion.negative_examples[0]}",
                    ],
                    activation_signature=activation,
                    decoder_signature=decoder,
                    causal_effects=causal,
                    source="toy",
                    metadata={"rank_hint": alignment},
                )
            )
        return features

    def _label_for(self, index: int, token: str, criterion: Criterion) -> str:
        templates = [
            "{token} criterion detector",
            "latent state for {token}",
            "counterfactual sensitivity to {token}",
            "context pattern around {token}",
        ]
        template = templates[index % len(templates)]
        return template.format(token=token)


class ToyVerbalizer:
    def explain(self, evidence: FeatureEvidence, criterion: Criterion) -> str:
        direction = _direction_phrase(
            evidence.causal_effects.get(
                "signed_causal_effect",
                evidence.metadata.get("signed_association"),
            )
        )
        return (
            f"This feature appears to represent '{evidence.label}' in contexts related to "
            f"'{criterion.text}'{direction}. Treat this as a hypothesis, then check intervention results."
        )


class ToyInterventionRunner:
    def estimate(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
        return dict(evidence.causal_effects)


def _seed(*parts: str) -> int:
    joined = "\x1f".join(parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _unit_vector(values: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in values))
    if length == 0:
        return values
    return [round(value / length, 6) for value in values]


def _direction_phrase(raw_value: object) -> str:
    if raw_value is None:
        return ""
    value = float(raw_value)
    if value > 0.05:
        return " and is positively associated with the criterion"
    if value < -0.05:
        return " and is negatively associated with the criterion"
    return " and has a weak signed association with the criterion"
