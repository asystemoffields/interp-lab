from __future__ import annotations

from typing import Protocol

from oracle_sae.schema import Criterion, FeatureEvidence


class FeatureProvider(Protocol):
    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        ...


class Verbalizer(Protocol):
    def explain(self, evidence: FeatureEvidence, criterion: Criterion) -> str:
        ...


class InterventionRunner(Protocol):
    def estimate(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
        ...
