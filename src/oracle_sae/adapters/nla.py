from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle_sae.adapters.toy import ToyVerbalizer
from oracle_sae.schema import Criterion, FeatureEvidence


@dataclass(frozen=True)
class NlaExplanationRecord:
    feature_id: str
    explanation: str
    label: str = ""
    confidence: float | None = None
    source: str = "nla"
    paraphrases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NlaVerbalizer:
    """Use Natural Language Autoencoder or autointerp records as feature explanations."""

    def __init__(
        self,
        explanations: str | Path | dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        min_confidence: float | None = None,
        fallback: ToyVerbalizer | None = None,
    ):
        self.records = _load_records(explanations) if explanations is not None else {}
        self.min_confidence = min_confidence
        self.fallback = fallback

    def explain(self, evidence: FeatureEvidence, criterion: Criterion) -> str:
        record = self._record_for(evidence)
        if record is not None and _passes_confidence(record, self.min_confidence):
            return record.explanation
        metadata_explanation = _metadata_explanation(evidence.metadata)
        if metadata_explanation:
            return metadata_explanation
        if self.fallback is not None:
            return self.fallback.explain(evidence, criterion)
        return evidence.label

    def metadata_for(self, evidence: FeatureEvidence, criterion: Criterion) -> dict[str, Any]:
        record = self._record_for(evidence)
        metadata_explanation = _metadata_explanation(evidence.metadata)
        fallback_available = self.fallback is not None
        if record is None:
            return {
                "verbalizer": {
                    "type": "nla",
                    "source": _fallback_source(metadata_explanation, fallback_available),
                    "used_record": False,
                    "used_fallback": not metadata_explanation and fallback_available,
                }
            }
        used_record = _passes_confidence(record, self.min_confidence)
        source = "record" if used_record else _fallback_source(metadata_explanation, fallback_available)
        metadata: dict[str, Any] = {
            "type": "nla",
            "source": source,
            "record_source": record.source,
            "confidence": record.confidence,
            "used_record": used_record,
            "used_fallback": not used_record and not metadata_explanation and fallback_available,
            "paraphrases": record.paraphrases,
            "metadata": record.metadata,
        }
        if not used_record:
            metadata["rejected_record"] = {
                "source": record.source,
                "confidence": record.confidence,
                "reason": "below_min_confidence",
            }
        return {
            "verbalizer": metadata
        }

    def report_metadata(self) -> dict[str, Any]:
        return {
            "verbalizer": {
                "type": "nla",
                "record_count": len({record.feature_id for record in self.records.values()}),
                "min_confidence": self.min_confidence,
                "fallback": self.fallback is not None,
            }
        }

    def _record_for(self, evidence: FeatureEvidence) -> NlaExplanationRecord | None:
        return self.records.get(evidence.feature_id) or self.records.get(_normalize_feature_id(evidence.feature_id))


def load_nla_explanation_records(path: str | Path) -> dict[str, NlaExplanationRecord]:
    return _load_records(path)


def _load_records(source: str | Path | dict[str, Any] | list[dict[str, Any]]) -> dict[str, NlaExplanationRecord]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"NLA explanation file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return _records_from_rows(rows)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _load_records(payload)
    if isinstance(source, list):
        return _records_from_rows(source)
    if "features" in source and isinstance(source["features"], list):
        return _records_from_rows(source["features"])
    if "explanations" in source and isinstance(source["explanations"], list):
        return _records_from_rows(source["explanations"])
    rows: list[dict[str, Any]] = []
    for feature_id, value in source.items():
        if isinstance(value, str):
            rows.append({"feature_id": feature_id, "explanation": value})
        elif isinstance(value, dict):
            row = dict(value)
            row.setdefault("feature_id", feature_id)
            rows.append(row)
    return _records_from_rows(rows)


def _records_from_rows(rows: list[dict[str, Any]]) -> dict[str, NlaExplanationRecord]:
    records: dict[str, NlaExplanationRecord] = {}
    for row in rows:
        feature_id = str(row.get("feature_id") or row.get("id") or "").strip()
        explanation = str(row.get("explanation") or row.get("description") or row.get("text") or "").strip()
        if not feature_id or not explanation:
            continue
        record = NlaExplanationRecord(
            feature_id=feature_id,
            explanation=explanation,
            label=str(row.get("label", "")).strip(),
            confidence=_optional_float(row.get("confidence", row.get("score"))),
            source=str(row.get("source", "nla")),
            paraphrases=[str(item) for item in row.get("paraphrases", [])],
            metadata=dict(row.get("metadata", {})),
        )
        records[record.feature_id] = record
        records[_normalize_feature_id(record.feature_id)] = record
    return records


def _metadata_explanation(metadata: dict[str, Any]) -> str:
    for key in ("nla_explanation", "nla_description", "autointerp_explanation", "explanation"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _passes_confidence(record: NlaExplanationRecord, min_confidence: float | None) -> bool:
    if min_confidence is None or record.confidence is None:
        return True
    return record.confidence >= min_confidence


def _fallback_source(metadata_explanation: str, fallback_available: bool) -> str:
    if metadata_explanation:
        return "metadata"
    if fallback_available:
        return "fallback"
    return "label"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _normalize_feature_id(value: str) -> str:
    return value.strip().lower()
