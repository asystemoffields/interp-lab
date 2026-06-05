from __future__ import annotations

import json
import sys
from pathlib import Path

from interp_lab.schema import Criterion, FeatureEvidence


class JsonlFeatureProvider:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        features: list[FeatureEvidence] = []
        models_seen: set[str] = set()
        # utf-8-sig tolerates a UTF-8 BOM (common in files exported from Windows
        # tools like Notepad/PowerShell/Excel) instead of crashing on the leading
        # byte-order mark.
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    message = f"{self.path}:{line_number}: invalid JSON: {exc.msg}"
                    raise ValueError(message) from exc
                if not isinstance(data, dict):
                    raise ValueError(
                        f"{self.path}:{line_number}: each feature row must be a JSON object, "
                        f"got {type(data).__name__}"
                    )
                try:
                    evidence = FeatureEvidence.from_dict(data)
                except (KeyError, TypeError, ValueError) as exc:
                    field = str(exc).strip("'\"") if isinstance(exc, KeyError) else exc
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid feature record "
                        f"(missing or invalid field: {field}). Each row needs at least "
                        f'"feature_id" and "model".'
                    ) from exc
                models_seen.add(evidence.model)
                if evidence.model == model:
                    features.append(evidence)
        if not features and models_seen and model not in models_seen:
            # Almost always a typo in --model: the file has rows, but none for this
            # model, so the report would be silently empty. Nudge instead.
            available = ", ".join(sorted(models_seen))
            print(
                f"interp-lab: warning: {self.path} has no rows with model={model!r}; "
                f"found model(s): {available}. The report will be empty.",
                file=sys.stderr,
            )
        return features
