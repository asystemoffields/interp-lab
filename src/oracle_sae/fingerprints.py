from __future__ import annotations

from oracle_sae.schema import Criterion, FeatureEvidence, FeatureFingerprint
from oracle_sae.text_vectors import hash_text_vector


INTERVENTION_CAUSAL_KEYS = ("criterion", "signed_causal_effect", "specificity", "side_effect")
ASSOCIATION_CAUSAL_KEYS = ("criterion", "signed_association", "specificity", "side_effect")


def build_fingerprint(
    evidence: FeatureEvidence,
    criterion: Criterion,
    explanation: str,
) -> FeatureFingerprint:
    text = " ".join(
        item
        for item in [
            evidence.label,
            explanation,
            " ".join(evidence.examples[:3]),
            criterion.text,
        ]
        if item
    )
    return FeatureFingerprint(
        feature_id=evidence.feature_id,
        model=evidence.model,
        layer=evidence.layer,
        text=text,
        text_vector=hash_text_vector(text),
        activation_signature=evidence.activation_signature,
        decoder_signature=evidence.decoder_signature,
        causal_vector=_causal_vector(evidence.causal_effects),
        neighbor_labels=[],
    )


def _causal_vector(causal_effects: dict[str, float]) -> list[float]:
    keys = INTERVENTION_CAUSAL_KEYS if "signed_causal_effect" in causal_effects else ASSOCIATION_CAUSAL_KEYS
    return [float(causal_effects.get(key, 0.0)) for key in keys]
