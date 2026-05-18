from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import clamp, mean
from oracle_sae.schema import Criterion, FeatureEvidence

PROMOTING_WHEN_SCORE_DROPS = {
    "ablate",
    "clamp_down",
    "knockout",
    "remove",
    "suppress",
    "zero",
}
PROMOTING_WHEN_SCORE_RISES = {
    "amplify",
    "clamp",
    "clamp_up",
    "patch",
    "patch_in",
    "steer",
}


@dataclass(frozen=True)
class InterventionRecord:
    model: str
    feature_id: str
    intervention: str
    baseline_score: float
    intervention_score: float
    criterion: str | None = None
    prompt_id: str = ""
    side_effect_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, line_label: str) -> "InterventionRecord":
        missing = [
            key
            for key in ["model", "feature_id", "intervention", "baseline_score", "intervention_score"]
            if key not in data
        ]
        if missing:
            raise ValueError(f"{line_label}: missing required fields: {', '.join(missing)}")
        side_effect_score = data.get("side_effect_score")
        return cls(
            model=str(data["model"]),
            feature_id=str(data["feature_id"]),
            intervention=str(data["intervention"]).lower(),
            baseline_score=float(data["baseline_score"]),
            intervention_score=float(data["intervention_score"]),
            criterion=str(data["criterion"]) if data.get("criterion") is not None else None,
            prompt_id=str(data.get("prompt_id", "")),
            side_effect_score=float(side_effect_score) if side_effect_score is not None else None,
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def raw_delta(self) -> float:
        return self.intervention_score - self.baseline_score

    @property
    def directed_effect(self) -> float:
        if self.intervention in PROMOTING_WHEN_SCORE_DROPS:
            return self.baseline_score - self.intervention_score
        if self.intervention in PROMOTING_WHEN_SCORE_RISES:
            return self.intervention_score - self.baseline_score
        return self.intervention_score - self.baseline_score


class InterventionRecordRunner:
    """Aggregates external causal intervention results for feature ranking."""

    def __init__(
        self,
        path: str | Path,
        *,
        fallback_runner: Any | None = None,
        require_criterion_match: bool = True,
        require_records: bool = False,
    ):
        self.path = Path(path)
        self.fallback_runner = fallback_runner
        self.require_criterion_match = require_criterion_match
        self.require_records = require_records
        self._records: list[InterventionRecord] | None = None

    def estimate(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, float]:
        effects = self._matching_records(evidence, criterion)
        fallback = (
            dict(self.fallback_runner.estimate(evidence, criterion))
            if self.fallback_runner is not None
            else {}
        )
        if not effects:
            if self.require_records:
                return {
                    "criterion": 0.0,
                    "specificity": 0.0,
                    "side_effect": float(fallback.get("side_effect", 0.0)),
                    "signed_causal_effect": 0.0,
                    "raw_intervention_delta": 0.0,
                    "intervention_record_count": 0.0,
                }
            return fallback

        directed = [record.directed_effect for record in effects]
        raw_deltas = [record.raw_delta for record in effects]
        side_effects = [
            abs(record.side_effect_score)
            for record in effects
            if record.side_effect_score is not None
        ]
        causal_strength = abs(mean(directed))
        side_effect = mean(side_effects) if side_effects else float(fallback.get("side_effect", 0.0))
        specificity = max(0.0, causal_strength - side_effect)

        merged = dict(fallback)
        merged.update(
            {
                "criterion": round(clamp(causal_strength), 6),
                "specificity": round(clamp(specificity), 6),
                "side_effect": round(clamp(side_effect), 6),
                "signed_causal_effect": round(mean(directed), 6),
                "raw_intervention_delta": round(mean(raw_deltas), 6),
                "intervention_record_count": float(len(effects)),
            }
        )
        return merged

    def metadata_for(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, Any]:
        effects = self._matching_records(evidence, criterion)
        if not effects:
            return {}
        directed = [record.directed_effect for record in effects]
        side_effects = [
            abs(record.side_effect_score)
            for record in effects
            if record.side_effect_score is not None
        ]
        summary = {
            "count": len(effects),
            "mean_directed_effect": round(mean(directed), 6),
            "mean_abs_directed_effect": round(mean([abs(value) for value in directed]), 6),
            "stdev_directed_effect": round(statistics.pstdev(directed), 6)
            if len(directed) > 1
            else 0.0,
            "mean_side_effect": round(mean(side_effects), 6) if side_effects else None,
            "examples": [_render_record(record) for record in effects[:5]],
        }
        return {"interventions": summary}

    def should_keep(self, evidence: FeatureEvidence, criterion: Criterion) -> bool:
        if not self.require_records:
            return True
        return bool(self._matching_records(evidence, criterion))

    def _matching_records(
        self,
        evidence: FeatureEvidence,
        criterion: Criterion,
    ) -> list[InterventionRecord]:
        criterion_key = _criterion_key(criterion.text)
        matches = []
        for record in self._load_records():
            if record.model != evidence.model or record.feature_id != evidence.feature_id:
                continue
            if self.require_criterion_match and record.criterion is not None:
                if _criterion_key(record.criterion) != criterion_key:
                    continue
            matches.append(record)
        return matches

    def _load_records(self) -> list[InterventionRecord]:
        if self._records is not None:
            return self._records
        records: list[InterventionRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                line_label = f"{self.path}:{line_number}"
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_label}: invalid JSON: {exc.msg}") from exc
                records.append(InterventionRecord.from_dict(data, line_label=line_label))
        self._records = records
        return records


def summarize_intervention_file(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    runner = InterventionRecordRunner(path, require_criterion_match=False)
    for record in runner._load_records():
        counts[record.intervention] += 1
    return dict(counts)


def _criterion_key(value: str) -> str:
    return " ".join(value.lower().split())


def _render_record(record: InterventionRecord) -> str:
    prompt = f"{record.prompt_id}: " if record.prompt_id else ""
    return (
        f"{prompt}{record.intervention} baseline={record.baseline_score:.3f}, "
        f"intervention={record.intervention_score:.3f}, "
        f"directed_effect={record.directed_effect:.3f}"
    )
