from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import clamp, mean
from oracle_sae.schema import Criterion, FeatureEvidence

LAYER_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])L(\d+)(?:[^A-Za-z0-9]|$)")


@dataclass(frozen=True)
class ActivationRecord:
    model: str
    prompt_id: str
    text: str
    criterion_score: float
    features: dict[str, float]
    feature_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, line_label: str) -> "ActivationRecord":
        features, inline_metadata = _parse_features(data.get("features"), line_label=line_label)
        explicit_metadata = data.get("feature_metadata", {})
        if explicit_metadata is None:
            explicit_metadata = {}
        if not isinstance(explicit_metadata, dict):
            raise ValueError(f"{line_label}: feature_metadata must be an object when provided")
        merged_metadata: dict[str, dict[str, Any]] = {}
        for feature_id in set(inline_metadata) | set(explicit_metadata):
            merged = dict(inline_metadata.get(feature_id, {}))
            explicit = explicit_metadata.get(feature_id, {})
            if explicit is not None:
                if not isinstance(explicit, dict):
                    raise ValueError(f"{line_label}: feature_metadata[{feature_id!r}] must be an object")
                merged.update(explicit)
            merged_metadata[feature_id] = merged

        if "criterion_score" not in data:
            raise ValueError(f"{line_label}: missing criterion_score")

        return cls(
            model=str(data["model"]),
            prompt_id=str(data.get("prompt_id", data.get("id", ""))),
            text=str(data.get("text", "")),
            criterion_score=float(data["criterion_score"]),
            features=features,
            feature_metadata=merged_metadata,
            metadata=dict(data.get("metadata", {})),
        )


class ActivationRecordFeatureProvider:
    """Aggregates per-prompt feature activations into criterion-ranked evidence.

    This backend is intentionally model-agnostic. It accepts records exported from
    SAEs, crosscoders, NLA probes, Neuronpedia scripts, or custom activation hooks.
    """

    def __init__(self, path: str | Path, *, signature_size: int = 128, examples_per_feature: int = 3):
        self.path = Path(path)
        self.signature_size = signature_size
        self.examples_per_feature = examples_per_feature
        self._last_summary: dict[str, Any] = {}

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        stats_by_feature: dict[str, _FeatureStats] = {}
        metadata_by_feature: dict[str, dict[str, Any]] = defaultdict(dict)
        summary = _RecordSummary()

        for record in self._iter_records():
            if record.model != model:
                continue
            summary.add(record.criterion_score)
            for feature_id, activation in record.features.items():
                stats = stats_by_feature.setdefault(
                    feature_id,
                    _FeatureStats(signature_size=self.signature_size, examples_per_feature=self.examples_per_feature),
                )
                stats.add(activation, record.criterion_score, record.prompt_id, record.text)
                metadata_by_feature[feature_id].update(record.feature_metadata.get(feature_id, {}))
        if not stats_by_feature:
            self._last_summary = summary.to_dict(feature_count=0)
            return []
        self._last_summary = summary.to_dict(feature_count=len(stats_by_feature))
        for record in self._iter_records():
            if record.model != model:
                continue
            for feature_id, activation in record.features.items():
                stats = stats_by_feature.get(feature_id)
                if stats is not None:
                    stats.add_separation(activation, record.criterion_score)

        evidence_items: list[FeatureEvidence] = []
        for feature_id, stats in stats_by_feature.items():
            signed_association = stats.pearson()
            signed_separation = stats.mean_separation()
            metadata = dict(metadata_by_feature.get(feature_id, {}))
            metadata.update(
                {
                    "record_count": stats.count,
                    "signed_association": round(signed_association, 6),
                    "signed_separation": round(signed_separation, 6),
                    "criterion_score_mean": round(stats.mean_score(), 6),
                }
            )

            label = str(metadata.get("label", feature_id))
            layer = _coerce_layer(metadata.get("layer"), feature_id)
            decoder_signature = _number_list(metadata.get("decoder_signature", []))

            evidence_items.append(
                FeatureEvidence(
                    feature_id=feature_id,
                    model=model,
                    layer=layer,
                    label=label,
                    examples=stats.top_examples(),
                    activation_signature=stats.activation_signature(),
                    decoder_signature=decoder_signature,
                    causal_effects={
                        "criterion": round(abs(signed_association), 6),
                        "signed_association": round(signed_association, 6),
                        "specificity": round(abs(signed_separation), 6),
                        "side_effect": float(metadata.get("side_effect", 0.0)),
                    },
                    source=str(metadata.get("source", "activation-records")),
                    metadata=metadata,
                )
            )
        evidence_items.sort(key=lambda item: item.causal_effects.get("criterion", 0.0), reverse=True)
        return evidence_items

    def _load_records(self) -> list[ActivationRecord]:
        return list(self._iter_records())

    def report_metadata(self) -> dict[str, Any]:
        return {"evidence": dict(self._last_summary)} if self._last_summary else {}

    def _iter_records(self):
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
                yield ActivationRecord.from_dict(data, line_label=line_label)


def _parse_features(raw_features: Any, *, line_label: str) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if isinstance(raw_features, dict):
        return {str(key): float(value) for key, value in raw_features.items()}, {}
    if isinstance(raw_features, list):
        features: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(raw_features):
            if not isinstance(item, dict):
                raise ValueError(f"{line_label}: features[{index}] must be an object")
            if "feature_id" not in item:
                raise ValueError(f"{line_label}: features[{index}] is missing feature_id")
            feature_id = str(item["feature_id"])
            activation = item.get("activation", item.get("value"))
            if activation is None:
                raise ValueError(f"{line_label}: features[{index}] is missing activation")
            features[feature_id] = float(activation)
            metadata[feature_id] = {
                key: value
                for key, value in item.items()
                if key not in {"feature_id", "activation", "value"}
            }
        return features, metadata
    raise ValueError(f"{line_label}: features must be an object or list")


def _mean_separation(activations: list[float], scores: list[float]) -> float:
    if not activations or not scores:
        return 0.0
    threshold = mean(scores)
    positive = [activation for activation, score in zip(activations, scores) if score >= threshold]
    negative = [activation for activation, score in zip(activations, scores) if score < threshold]
    if not positive or not negative:
        return 0.0
    spread = max(activations) - min(activations)
    if spread == 0:
        return 0.0
    return clamp((mean(positive) - mean(negative)) / spread, -1.0, 1.0)


def _signature(values: list[float], signature_size: int) -> list[float]:
    return [round(value, 6) for value in values[:signature_size]]


def _top_examples(examples: list[tuple[float, float, str, str]], limit: int) -> list[str]:
    sorted_examples = sorted(examples, key=lambda item: abs(item[0]), reverse=True)
    rendered: list[str] = []
    for activation, score, prompt_id, text in sorted_examples[:limit]:
        prefix = f"activation={activation:.3f}, criterion_score={score:.3f}"
        if prompt_id:
            prefix = f"{prompt_id}: {prefix}"
        rendered.append(f"{prefix} | {text}")
    return rendered


@dataclass
class _FeatureStats:
    signature_size: int
    examples_per_feature: int
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0
    min_x: float | None = None
    max_x: float | None = None
    positive_sum: float = 0.0
    positive_count: int = 0
    negative_sum: float = 0.0
    negative_count: int = 0
    signature_values: list[float] = field(default_factory=list)
    examples: list[tuple[float, float, str, str]] = field(default_factory=list)

    def add(self, activation: float, score: float, prompt_id: str, text: str) -> None:
        self.count += 1
        self.sum_x += activation
        self.sum_y += score
        self.sum_xx += activation * activation
        self.sum_yy += score * score
        self.sum_xy += activation * score
        self.min_x = activation if self.min_x is None else min(self.min_x, activation)
        self.max_x = activation if self.max_x is None else max(self.max_x, activation)
        if len(self.signature_values) < self.signature_size:
            self.signature_values.append(activation)
        self._add_example(activation, score, prompt_id, text)

    def pearson(self) -> float:
        if self.count < 2:
            return 0.0
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        x_denominator = self.count * self.sum_xx - self.sum_x * self.sum_x
        y_denominator = self.count * self.sum_yy - self.sum_y * self.sum_y
        if x_denominator <= 0.0 or y_denominator <= 0.0:
            return 0.0
        return numerator / math.sqrt(x_denominator * y_denominator)

    def mean_score(self) -> float:
        return self.sum_y / self.count if self.count else 0.0

    def mean_separation(self) -> float:
        if not self.count or self.min_x is None or self.max_x is None:
            return 0.0
        if not self.positive_count or not self.negative_count:
            return 0.0
        spread = self.max_x - self.min_x
        if spread == 0.0:
            return 0.0
        positive_mean = self.positive_sum / self.positive_count
        negative_mean = self.negative_sum / self.negative_count
        return clamp((positive_mean - negative_mean) / spread, -1.0, 1.0)

    def add_separation(self, activation: float, score: float) -> None:
        if score >= self.mean_score():
            self.positive_sum += activation
            self.positive_count += 1
        else:
            self.negative_sum += activation
            self.negative_count += 1

    def activation_signature(self) -> list[float]:
        return _signature(self.signature_values, self.signature_size)

    def top_examples(self) -> list[str]:
        return _top_examples(self.examples, self.examples_per_feature)

    def _add_example(self, activation: float, score: float, prompt_id: str, text: str) -> None:
        if not text:
            return
        self.examples.append((activation, score, prompt_id, text))
        self.examples.sort(key=lambda item: abs(item[0]), reverse=True)
        del self.examples[self.examples_per_feature :]


@dataclass
class _RecordSummary:
    count: int = 0
    score_sum: float = 0.0
    min_score: float | None = None
    max_score: float | None = None
    positive_count: int = 0
    negative_count: int = 0

    def add(self, score: float) -> None:
        self.count += 1
        self.score_sum += score
        self.min_score = score if self.min_score is None else min(self.min_score, score)
        self.max_score = score if self.max_score is None else max(self.max_score, score)
        if score > 0:
            self.positive_count += 1
        else:
            self.negative_count += 1

    def to_dict(self, *, feature_count: int) -> dict[str, Any]:
        return {
            "source": "activation-records",
            "record_count": self.count,
            "feature_count": feature_count,
            "criterion_score_mean": round(self.score_sum / self.count, 6) if self.count else 0.0,
            "criterion_score_min": round(float(self.min_score), 6) if self.min_score is not None else 0.0,
            "criterion_score_max": round(float(self.max_score), 6) if self.max_score is not None else 0.0,
            "positive_record_count": self.positive_count,
            "negative_record_count": self.negative_count,
        }


def _coerce_layer(raw_layer: Any, feature_id: str) -> int | None:
    if raw_layer is not None:
        return int(raw_layer)
    match = LAYER_PATTERN.search(feature_id)
    if match:
        return int(match.group(1))
    return None


def _number_list(raw_values: Any) -> list[float]:
    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError("decoder_signature must be a list when provided")
    return [float(value) for value in raw_values]
