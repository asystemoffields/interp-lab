from __future__ import annotations

import json
from pathlib import Path

from interp_lab.schema import Criterion, FeatureEvidence


class JsonlFeatureProvider:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        features: list[FeatureEvidence] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    message = f"{self.path}:{line_number}: invalid JSON: {exc.msg}"
                    raise ValueError(message) from exc
                evidence = FeatureEvidence.from_dict(data)
                if evidence.model == model:
                    features.append(evidence)
        return features
