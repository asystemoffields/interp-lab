from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


NumberVector = list[float]
INSPECTION_REPORT_SCHEMA = "interp-lab.inspection_report.v1"
MATCH_REPORT_SCHEMA = "interp-lab.match_report.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Criterion:
    text: str
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    scoring_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
            "scoring_notes": self.scoring_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Criterion":
        return cls(
            text=str(data["text"]),
            positive_examples=[str(item) for item in data.get("positive_examples", [])],
            negative_examples=[str(item) for item in data.get("negative_examples", [])],
            scoring_notes=[str(item) for item in data.get("scoring_notes", [])],
        )


@dataclass(frozen=True)
class FeatureEvidence:
    feature_id: str
    model: str
    layer: int | None
    label: str
    examples: list[str] = field(default_factory=list)
    activation_signature: NumberVector = field(default_factory=list)
    decoder_signature: NumberVector = field(default_factory=list)
    causal_effects: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "model": self.model,
            "layer": self.layer,
            "label": self.label,
            "examples": self.examples,
            "activation_signature": self.activation_signature,
            "decoder_signature": self.decoder_signature,
            "causal_effects": self.causal_effects,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEvidence":
        layer = data.get("layer")
        return cls(
            feature_id=str(data["feature_id"]),
            model=str(data["model"]),
            layer=int(layer) if layer is not None else None,
            label=str(data.get("label", "")),
            examples=[str(item) for item in data.get("examples", [])],
            activation_signature=[float(item) for item in data.get("activation_signature", [])],
            decoder_signature=[float(item) for item in data.get("decoder_signature", [])],
            causal_effects={str(k): float(v) for k, v in data.get("causal_effects", {}).items()},
            source=str(data.get("source", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class FeatureFingerprint:
    feature_id: str
    model: str
    layer: int | None
    text: str
    text_vector: NumberVector
    activation_signature: NumberVector
    decoder_signature: NumberVector
    causal_vector: NumberVector
    neighbor_labels: list[str] = field(default_factory=list)
    text_embedder: str = "hash-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "model": self.model,
            "layer": self.layer,
            "text": self.text,
            "text_vector": self.text_vector,
            "text_embedder": self.text_embedder,
            "activation_signature": self.activation_signature,
            "decoder_signature": self.decoder_signature,
            "causal_vector": self.causal_vector,
            "neighbor_labels": self.neighbor_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureFingerprint":
        return cls(
            feature_id=str(data["feature_id"]),
            model=str(data["model"]),
            layer=int(data["layer"]) if data.get("layer") is not None else None,
            text=str(data.get("text", "")),
            text_vector=[float(item) for item in data.get("text_vector", [])],
            activation_signature=[float(item) for item in data.get("activation_signature", [])],
            decoder_signature=[float(item) for item in data.get("decoder_signature", [])],
            causal_vector=[float(item) for item in data.get("causal_vector", [])],
            neighbor_labels=[str(item) for item in data.get("neighbor_labels", [])],
            text_embedder=str(data.get("text_embedder", "hash-v1")),
        )


@dataclass(frozen=True)
class FeatureCard:
    feature_id: str
    model: str
    layer: int | None
    label: str
    explanation: str
    importance: float
    association: float
    specificity: float
    causal_effect: float
    stability: float
    examples: list[str]
    source: str
    fingerprint: FeatureFingerprint
    metadata: dict[str, Any] = field(default_factory=dict)
    causal_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "model": self.model,
            "layer": self.layer,
            "label": self.label,
            "explanation": self.explanation,
            "importance": self.importance,
            "association": self.association,
            "specificity": self.specificity,
            "causal_effect": self.causal_effect,
            "stability": self.stability,
            "examples": self.examples,
            "source": self.source,
            "fingerprint": self.fingerprint.to_dict(),
            "metadata": self.metadata,
            "causal_effects": self.causal_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureCard":
        return cls(
            feature_id=str(data["feature_id"]),
            model=str(data["model"]),
            layer=int(data["layer"]) if data.get("layer") is not None else None,
            label=str(data.get("label", "")),
            explanation=str(data.get("explanation", "")),
            importance=float(data.get("importance", 0.0)),
            association=float(data.get("association", 0.0)),
            specificity=float(data.get("specificity", 0.0)),
            causal_effect=float(data.get("causal_effect", 0.0)),
            stability=float(data.get("stability", 0.0)),
            examples=[str(item) for item in data.get("examples", [])],
            source=str(data.get("source", "unknown")),
            fingerprint=FeatureFingerprint.from_dict(data["fingerprint"]),
            metadata=dict(data.get("metadata", {})),
            causal_effects={str(k): float(v) for k, v in data.get("causal_effects", {}).items()},
        )


@dataclass(frozen=True)
class CandidateMatch:
    left_feature_id: str
    right_feature_id: str
    left_model: str
    right_model: str
    score: float
    components: dict[str, float]
    left_label: str = ""
    right_label: str = ""
    left_signed_effect: float | None = None
    right_signed_effect: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_feature_id": self.left_feature_id,
            "right_feature_id": self.right_feature_id,
            "left_model": self.left_model,
            "right_model": self.right_model,
            "score": self.score,
            "components": self.components,
            "left_label": self.left_label,
            "right_label": self.right_label,
            "left_signed_effect": self.left_signed_effect,
            "right_signed_effect": self.right_signed_effect,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateMatch":
        return cls(
            left_feature_id=str(data["left_feature_id"]),
            right_feature_id=str(data["right_feature_id"]),
            left_model=str(data["left_model"]),
            right_model=str(data["right_model"]),
            score=float(data["score"]),
            components={str(k): float(v) for k, v in data.get("components", {}).items()},
            left_label=str(data.get("left_label", "")),
            right_label=str(data.get("right_label", "")),
            left_signed_effect=_optional_float(data.get("left_signed_effect")),
            right_signed_effect=_optional_float(data.get("right_signed_effect")),
        )


@dataclass(frozen=True)
class InspectionReport:
    model: str
    criterion: Criterion
    cards: list[FeatureCard]
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSPECTION_REPORT_SCHEMA,
            "model": self.model,
            "criterion": self.criterion.to_dict(),
            "cards": [card.to_dict() for card in self.cards],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InspectionReport":
        return cls(
            model=str(data["model"]),
            criterion=Criterion.from_dict(data["criterion"]),
            cards=[FeatureCard.from_dict(item) for item in data.get("cards", [])],
            created_at=str(data.get("created_at", utc_now_iso())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MatchReport:
    left_model: str
    right_model: str
    matches: list[CandidateMatch]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCH_REPORT_SCHEMA,
            "left_model": self.left_model,
            "right_model": self.right_model,
            "matches": [match.to_dict() for match in self.matches],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchReport":
        return cls(
            left_model=str(data["left_model"]),
            right_model=str(data["right_model"]),
            matches=[CandidateMatch.from_dict(item) for item in data.get("matches", [])],
            created_at=str(data.get("created_at", utc_now_iso())),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
