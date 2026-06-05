from __future__ import annotations

import hashlib
import math
import random

from interp_lab.schema import Criterion, FeatureEvidence
from interp_lab.text_vectors import content_tokens

# Simulated number of prompts behind each toy "measured" intervention summary.
_MEASURED_RUN_COUNT = 5


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
        examples = _example_summary(evidence.examples)
        if examples:
            return f"Activation summary: {evidence.label}. Representative high-activation contexts include {examples}."
        return f"Activation summary: {evidence.label}."


class ToyInterventionRunner:
    """Deterministic causal estimates for demos and tests.

    By default this is purely correlational: ``estimate`` echoes the provider's
    causal fields unchanged. Pass ``measured=True`` to simulate having run real
    interventions -- it synthesizes a specificity-adjusted ``strong_causal_score``, a
    signed directed effect, and per-feature intervention metadata (n, controls, a
    confidence interval). The demo uses this so the toy tour actually *demonstrates*
    the correlational-vs-causal distinction the toolkit exists to make, instead of
    reporting "causal claims are untested".
    """

    def __init__(self, *, measured: bool = False):
        self.measured = measured

    def estimate(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
        effects = dict(evidence.causal_effects)
        if not self.measured:
            return effects
        directed = float(effects.get("criterion", 0.0))
        specificity = float(effects.get("specificity", 0.0))
        # Specificity-adjusted causal signal: a strong, specific feature scores high; a
        # feature that also moves unrelated behavior is discounted toward zero.
        effects["signed_causal_effect"] = round(directed, 4)
        effects["strong_causal_score"] = round(max(0.0, min(1.0, directed * specificity)), 4)
        effects["intervention_record_count"] = float(_MEASURED_RUN_COUNT)
        return effects

    def metadata_for(self, evidence: FeatureEvidence, criterion: Criterion) -> dict:
        if not self.measured:
            return {}
        effects = evidence.causal_effects
        directed = float(effects.get("criterion", 0.0))
        side = float(effects.get("side_effect", 0.0))
        # A small deterministic half-width so a believable 95% CI accompanies the mean.
        half_width = round(0.03 + 0.04 * (1.0 - directed), 4)
        # Keep the worked example internally consistent: baseline - suppressed == directed.
        baseline = round(min(1.0, directed + 0.05), 4)
        suppressed = round(max(0.0, baseline - directed), 4)
        return {
            "interventions": {
                "count": _MEASURED_RUN_COUNT,
                "mean_directed_effect": round(directed, 4),
                "mean_side_effect": round(side, 4),
                "criterion_ci_low": round(max(0.0, directed - half_width), 4),
                "criterion_ci_high": round(min(1.0, directed + half_width), 4),
                "controls": {
                    "count": 2,
                    "mean_abs_directed_effect": round(0.5 * side, 4),
                },
                "examples": [
                    f"suppress {evidence.feature_id}: criterion {baseline:.3f} -> {suppressed:.3f} "
                    f"(directed +{directed:.3f})",
                    f"random-feature control: criterion {baseline:.3f} -> {baseline:.3f} (directed +0.000)",
                ],
            }
        }


def _seed(*parts: str) -> int:
    joined = "\x1f".join(parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _unit_vector(values: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in values))
    if length == 0:
        return values
    return [round(value / length, 6) for value in values]


def _example_summary(examples: list[str]) -> str:
    cleaned = []
    for example in examples[:2]:
        text = " ".join(str(example).split())
        if len(text) > 120:
            text = text[:117].rstrip() + "..."
        cleaned.append(text)
    return "; ".join(cleaned)
