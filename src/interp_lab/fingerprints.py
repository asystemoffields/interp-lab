from __future__ import annotations

from interp_lab.schema import Criterion, FeatureEvidence, FeatureFingerprint
from interp_lab.text_embedding import active_embedder_id, embed_text


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
        text_vector=embed_text(text),
        activation_signature=evidence.activation_signature,
        decoder_signature=evidence.decoder_signature,
        causal_vector=_causal_vector(evidence.causal_effects),
        neighbor_labels=[],
        text_embedder=active_embedder_id(),
        causal_provenance=_causal_provenance(evidence.causal_effects),
    )


def _causal_vector(causal_effects: dict[str, float]) -> list[float]:
    keys = INTERVENTION_CAUSAL_KEYS if "signed_causal_effect" in causal_effects else ASSOCIATION_CAUSAL_KEYS
    return [float(causal_effects.get(key, 0.0)) for key in keys]


def _causal_provenance(causal_effects: dict[str, float]) -> str:
    """Label the causal_vector's origin so matching never compares a measured
    causal effect against a correlational proxy on the same axis (bug-9).

    ``INTERVENTION_CAUSAL_KEYS`` and ``ASSOCIATION_CAUSAL_KEYS`` differ at index 1
    (signed_causal_effect vs signed_association), so two vectors built from
    different key sets are not on a common axis and must not be cosine-compared.
    """
    if "signed_causal_effect" in causal_effects:
        return "intervention"
    if "signed_association" in causal_effects or "criterion" in causal_effects:
        return "association"
    return "none"
