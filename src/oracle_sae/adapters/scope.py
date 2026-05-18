from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from oracle_sae.adapters.saelens import SAELensFeatureProvider
from oracle_sae.schema import Criterion, FeatureEvidence

SCOPE_SOURCES = {
    "gemma-scope": {
        "label": "Gemma Scope",
        "homepage": "https://deepmind.google/models/gemma/gemma-scope/",
        "citation": "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
    },
    "qwen-scope": {
        "label": "Qwen-Scope",
        "homepage": "https://qwen.ai/blog?id=qwen-scope",
        "citation": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models",
    },
}


class ScopeFeatureProvider:
    """Convenience loader for named SAE suites such as Gemma Scope and Qwen-Scope."""

    def __init__(
        self,
        *,
        source: str,
        release: str,
        sae_id: str,
        feature_indices: list[int] | None = None,
        max_features: int = 32,
        device: str = "cpu",
        force_download: bool = False,
        feature_metadata: dict[int, dict[str, Any]] | None = None,
        sae_loader: Callable[..., tuple[Any, dict[str, Any], Any]] | None = None,
    ):
        if source not in SCOPE_SOURCES:
            known = ", ".join(sorted(SCOPE_SOURCES))
            raise ValueError(f"Unknown scope source {source!r}. Known sources: {known}")
        self.source = source
        self.inner = SAELensFeatureProvider(
            release=release,
            sae_id=sae_id,
            feature_indices=feature_indices,
            max_features=max_features,
            device=device,
            force_download=force_download,
            feature_metadata=feature_metadata,
            sae_loader=sae_loader,
        )

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        source_metadata = SCOPE_SOURCES[self.source]
        evidence_items = []
        for evidence in self.inner.features_for(model, criterion):
            metadata = dict(evidence.metadata)
            metadata.update(
                {
                    "scope_source": self.source,
                    "scope_label": source_metadata["label"],
                    "scope_homepage": source_metadata["homepage"],
                    "scope_citation": source_metadata["citation"],
                }
            )
            evidence_items.append(replace(evidence, source=self.source, metadata=metadata))
        return evidence_items
