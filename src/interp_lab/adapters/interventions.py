from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interp_lab.math_utils import clamp, mean
from interp_lab.schema import Criterion, FeatureEvidence

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
CONTROL_TYPES = {
    "control",
    "matched_frequency",
    "negative_control",
    "placebo",
    "random",
    "random_feature",
}
SATURATED_BASELINE_MEAN = 0.85
SATURATED_BASELINE_MIN = 0.75
FLOOR_BASELINE_MEAN = 0.02
FLOOR_BASELINE_MAX = 0.05


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

    @property
    def control_type(self) -> str | None:
        value = self.metadata.get("control_type")
        if value is None:
            return None
        return str(value).lower()

    @property
    def is_control(self) -> bool:
        return self.control_type in CONTROL_TYPES


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
        matching = self._matching_records(evidence, criterion)
        effects = [record for record in matching if not record.is_control]
        controls = [record for record in matching if record.is_control]
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
        control_summary = _summarize_control_records(controls)
        causal_strength = abs(mean(directed))
        side_effect = mean(side_effects) if side_effects else float(fallback.get("side_effect", 0.0))
        specificity = max(0.0, causal_strength - side_effect)
        strong_causal_score = max(
            0.0,
            specificity - control_summary.get("mean_abs_directed_effect", 0.0),
        )
        ci_low, ci_high = _mean_confidence_interval(directed)

        merged = dict(fallback)
        merged.update(
            {
                "criterion": round(clamp(causal_strength), 6),
                "specificity": round(clamp(specificity), 6),
                "side_effect": round(clamp(side_effect), 6),
                "strong_causal_score": round(clamp(strong_causal_score), 6),
                "signed_causal_effect": round(mean(directed), 6),
                "raw_intervention_delta": round(mean(raw_deltas), 6),
                "intervention_record_count": float(len(effects)),
                "control_record_count": float(len(controls)),
                "control_mean_abs_effect": round(
                    clamp(control_summary.get("mean_abs_directed_effect", 0.0)),
                    6,
                ),
                "criterion_ci_low": round(ci_low, 6),
                "criterion_ci_high": round(ci_high, 6),
            }
        )
        return merged

    def metadata_for(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, Any]:
        matching = self._matching_records(evidence, criterion)
        effects = [record for record in matching if not record.is_control]
        controls = [record for record in matching if record.is_control]
        if not effects:
            return {}
        directed = [record.directed_effect for record in effects]
        side_effects = [
            abs(record.side_effect_score)
            for record in effects
            if record.side_effect_score is not None
        ]
        ci_low, ci_high = _mean_confidence_interval(directed)
        control_summary = _summarize_control_records(controls)
        summary = {
            "count": len(effects),
            "mean_directed_effect": round(mean(directed), 6),
            "mean_abs_directed_effect": round(mean([abs(value) for value in directed]), 6),
            "stdev_directed_effect": round(statistics.pstdev(directed), 6)
            if len(directed) > 1
            else 0.0,
            "criterion_ci_low": round(ci_low, 6),
            "criterion_ci_high": round(ci_high, 6),
            "mean_side_effect": round(mean(side_effects), 6) if side_effects else None,
            "controls": control_summary,
            "examples": [_render_record(record) for record in effects[:5]],
        }
        behavior_score = _summarize_behavior_scores(effects)
        if behavior_score:
            summary["behavior_score"] = behavior_score
        return {"interventions": summary}

    def report_metadata(self) -> dict[str, Any]:
        records = self._load_records()
        effect_records = [record for record in records if not record.is_control]
        control_records = [record for record in records if record.is_control]
        criteria = sorted({record.criterion for record in records if record.criterion})
        return {
            "interventions": {
                "path": str(self.path),
                "record_count": len(records),
                "effect_record_count": len(effect_records),
                "control_record_count": len(control_records),
                "feature_count": len({record.feature_id for record in records}),
                "criteria": criteria[:8],
            }
        }

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


def _summarize_behavior_scores(records: list[InterventionRecord]) -> dict[str, Any]:
    scored = [record for record in records if record.metadata.get("behavior_score")]
    if not scored:
        return {}
    baselines = [record.baseline_score for record in scored]
    interventions = [record.intervention_score for record in scored]
    deltas = [record.intervention_score - record.baseline_score for record in scored]
    baseline_mean = mean(baselines)
    baseline_min = min(baselines)
    baseline_max = max(baselines)
    summary: dict[str, Any] = {
        "name": str(_first_metadata(scored, "behavior_score") or "score"),
        "count": len(scored),
        "baseline_mean": round(baseline_mean, 6),
        "baseline_min": round(baseline_min, 6),
        "baseline_max": round(baseline_max, 6),
        "intervention_mean": round(mean(interventions), 6),
        "score_delta_mean": round(mean(deltas), 6),
    }
    strategy = _first_metadata(scored, "target_token_strategy")
    if strategy is not None:
        summary["target_token_strategy"] = str(strategy)
    target_tokens = _first_metadata(scored, "target_tokens")
    if isinstance(target_tokens, list):
        summary["target_token_count"] = len(target_tokens)
        summary["target_token_sample"] = [str(token) for token in target_tokens[:8]]
    if baseline_mean >= SATURATED_BASELINE_MEAN or baseline_min >= SATURATED_BASELINE_MIN:
        summary["diagnostic"] = "saturated_baseline"
        summary["advisory"] = (
            "Baseline score is already high; use a narrower target-token set, "
            "harder positive prompts, or a more specific behavior scorer."
        )
    elif baseline_mean <= FLOOR_BASELINE_MEAN and baseline_max <= FLOOR_BASELINE_MAX:
        summary["diagnostic"] = "near_zero_baseline"
        if str(strategy).lower() == "auto":
            summary["advisory"] = (
                "Baseline score is near zero even with auto-derived targets; inspect the target-token sample, "
                "pass explicit raw:/space: target tokens, or use positive prompts whose expected completions contain the behavior."
            )
        else:
            summary["advisory"] = (
                "Baseline score is near zero; use auto targets, raw tokenizer forms, "
                "or target tokens that appear in positive completions."
            )
    return summary


def _first_metadata(records: list[InterventionRecord], key: str) -> Any:
    for record in records:
        value = record.metadata.get(key)
        if value is not None:
            return value
    return None


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    average = mean(values)
    if len(values) == 1:
        return average, average
    stdev = statistics.pstdev(values)
    half_width = 1.96 * stdev / (len(values) ** 0.5)
    return average - half_width, average + half_width


def _summarize_control_records(records: list[InterventionRecord]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "mean_directed_effect": 0.0,
            "mean_abs_directed_effect": 0.0,
            "by_type": {},
        }
    directed = [record.directed_effect for record in records]
    by_type: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_type[record.control_type or "control"].append(record.directed_effect)
    return {
        "count": len(records),
        "mean_directed_effect": round(mean(directed), 6),
        "mean_abs_directed_effect": round(mean([abs(value) for value in directed]), 6),
        "by_type": {
            control_type: {
                "count": len(values),
                "mean_directed_effect": round(mean(values), 6),
                "mean_abs_directed_effect": round(mean([abs(value) for value in values]), 6),
            }
            for control_type, values in sorted(by_type.items())
        },
    }
