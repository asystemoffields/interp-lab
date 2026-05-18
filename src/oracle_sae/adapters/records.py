from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import clamp, mean, pearson
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

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        records = [record for record in self._load_records() if record.model == model]
        if not records:
            return []

        activations: dict[str, list[float]] = defaultdict(list)
        scores: dict[str, list[float]] = defaultdict(list)
        examples: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
        metadata_by_feature: dict[str, dict[str, Any]] = defaultdict(dict)

        for record in records:
            for feature_id, activation in record.features.items():
                activations[feature_id].append(activation)
                scores[feature_id].append(record.criterion_score)
                if record.text:
                    examples[feature_id].append(
                        (activation, record.criterion_score, record.prompt_id, record.text)
                    )
                metadata_by_feature[feature_id].update(record.feature_metadata.get(feature_id, {}))

        evidence_items: list[FeatureEvidence] = []
        for feature_id, feature_activations in activations.items():
            feature_scores = scores[feature_id]
            signed_association = pearson(feature_activations, feature_scores)
            signed_separation = _mean_separation(feature_activations, feature_scores)
            metadata = dict(metadata_by_feature.get(feature_id, {}))
            metadata.update(
                {
                    "record_count": len(feature_activations),
                    "signed_association": round(signed_association, 6),
                    "signed_separation": round(signed_separation, 6),
                    "criterion_score_mean": round(mean(feature_scores), 6),
                }
            )

            label = str(metadata.get("label", feature_id))
            layer = _coerce_layer(metadata.get("layer"), feature_id)
            decoder_signature = _number_list(metadata.get("decoder_signature", []))
            top_examples = _top_examples(examples.get(feature_id, []), self.examples_per_feature)
            activation_signature = _signature(feature_activations, self.signature_size)

            evidence_items.append(
                FeatureEvidence(
                    feature_id=feature_id,
                    model=model,
                    layer=layer,
                    label=label,
                    examples=top_examples,
                    activation_signature=activation_signature,
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
        records: list[ActivationRecord] = []
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
                records.append(ActivationRecord.from_dict(data, line_label=line_label))
        return records


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
