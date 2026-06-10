from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from interp_lab.math_utils import clamp
from interp_lab.schema import Criterion, FeatureEvidence

DEFAULT_BASE_URL = "https://www.neuronpedia.org"
FEATURE_ID_PATTERN = re.compile(
    r"^(?P<model>[^@/\s]+)@(?P<source>[^:\s]+):(?P<index>[0-9]+)$"
)
FEATURE_URL_PATTERN = re.compile(
    r"/(?:(?:api/feature)/)?(?P<model>[^/\s]+)/(?P<source>[^/\s]+)/(?P<index>[0-9]+)(?:[/?#].*)?$"
)


@dataclass(frozen=True)
class NeuronpediaFeatureRef:
    model: str
    source: str
    index: int

    @property
    def feature_id(self) -> str:
        return f"{self.model}@{self.source}:{self.index}"

    @classmethod
    def parse(cls, value: str) -> "NeuronpediaFeatureRef":
        stripped = value.strip()
        match = FEATURE_ID_PATTERN.match(stripped)
        if not match:
            match = FEATURE_URL_PATTERN.search(stripped)
        if not match:
            raise ValueError(
                "Neuronpedia feature refs must look like "
                "'model@source:index' or a Neuronpedia feature URL"
            )
        return cls(
            model=match.group("model"),
            source=match.group("source"),
            index=int(match.group("index")),
        )


class NeuronpediaClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 20.0,
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fetch_json = fetch_json

    def feature_url(self, ref: NeuronpediaFeatureRef) -> str:
        parts = [
            self.base_url,
            "api",
            "feature",
            urllib.parse.quote(ref.model, safe=""),
            urllib.parse.quote(ref.source, safe=""),
            str(ref.index),
        ]
        return "/".join(parts)

    def fetch_feature(self, ref: NeuronpediaFeatureRef) -> dict[str, Any]:
        url = self.feature_url(ref)
        if self._fetch_json is not None:
            return self._fetch_json(url)
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Neuronpedia returned HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Neuronpedia at {url}: {exc.reason}") from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Neuronpedia returned non-JSON content for {url}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "interp-lab/0.1"}
        api_key = os.environ.get("NEURONPEDIA_API_KEY")
        if api_key:
            headers["X-Api-Key"] = api_key
        return headers


class NeuronpediaFeatureProvider:
    def __init__(
        self,
        feature_refs: list[str | NeuronpediaFeatureRef],
        *,
        client: NeuronpediaClient | None = None,
        examples_per_feature: int = 3,
    ):
        self.refs = [
            ref if isinstance(ref, NeuronpediaFeatureRef) else NeuronpediaFeatureRef.parse(ref)
            for ref in feature_refs
        ]
        self.client = client or NeuronpediaClient()
        self.examples_per_feature = examples_per_feature

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        evidence_items: list[FeatureEvidence] = []
        for ref in self.refs:
            payload = self.client.fetch_feature(ref)
            evidence = neuronpedia_payload_to_evidence(
                payload,
                examples_per_feature=self.examples_per_feature,
            )
            if evidence.model == model:
                evidence_items.append(evidence)
        return evidence_items


def load_neuronpedia_feature_refs(path: str | Path) -> list[str]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if file_path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict) and isinstance(data.get("features"), list):
            return [str(item) for item in data["features"]]
        raise ValueError("Neuronpedia JSON feature files must be a list or {'features': [...]}")
    refs: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            refs.append(stripped)
    return refs


def neuronpedia_payload_to_evidence(
    payload: dict[str, Any],
    *,
    examples_per_feature: int = 3,
) -> FeatureEvidence:
    ref = NeuronpediaFeatureRef(
        model=str(payload["modelId"]),
        source=str(payload["layer"]),
        index=int(payload["index"]),
    )
    explanations = _explanations(payload)
    label = _best_label(payload, explanations)
    examples = _activation_examples(payload.get("activations", []), examples_per_feature)
    autointerp_score = _best_explanation_score(explanations)
    activation_signature = _activation_signature(payload)
    decoder_signature = _number_list(payload.get("vector", [])) or _number_list(
        payload.get("decoder_weights_dist", [])
    )
    url = f"{DEFAULT_BASE_URL}/{ref.model}/{ref.source}/{ref.index}"
    metadata = {
        "url": url,
        "source_set_name": payload.get("sourceSetName"),
        "max_act_approx": payload.get("maxActApprox"),
        "frac_nonzero": payload.get("frac_nonzero"),
        "pos_tokens": payload.get("pos_str", []),
        "pos_values": payload.get("pos_values", []),
        "neg_tokens": payload.get("neg_str", []),
        "neg_values": payload.get("neg_values", []),
        "explanations": explanations,
        # Autointerp explanation-fidelity score. This measures how well the text
        # explanation predicts activations -- it is not a causal measurement, so
        # it must not go into causal_effects.
        "autointerp_score": autointerp_score,
    }
    return FeatureEvidence(
        feature_id=ref.feature_id,
        model=ref.model,
        layer=_parse_layer(ref.source),
        label=label,
        examples=examples,
        activation_signature=activation_signature,
        decoder_signature=decoder_signature,
        causal_effects={},
        source="neuronpedia",
        metadata=metadata,
    )


def _best_label(payload: dict[str, Any], explanations: list[dict[str, Any]]) -> str:
    if explanations:
        return str(explanations[0]["description"]).strip()
    vector_label = payload.get("vectorLabel")
    if vector_label:
        return str(vector_label)
    pos_tokens = [str(token).strip() for token in payload.get("pos_str", [])[:5]]
    if pos_tokens:
        return " / ".join(token for token in pos_tokens if token)
    return f"Neuronpedia feature {payload.get('index')}"


def _explanations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explanations = []
    for item in payload.get("explanations", []) or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        explanations.append(
            {
                "description": description,
                "model": item.get("explanationModelName"),
                "type": item.get("typeName"),
                "score": _best_score(item.get("scores", [])),
            }
        )
    explanations.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    return explanations


def _best_score(scores: Any) -> float | None:
    if not isinstance(scores, list) or not scores:
        return None
    values = []
    for item in scores:
        if isinstance(item, dict) and item.get("value") is not None:
            values.append(float(item["value"]))
    return max(values) if values else None


def _best_explanation_score(explanations: list[dict[str, Any]]) -> float:
    scores = [float(item["score"]) for item in explanations if item.get("score") is not None]
    if not scores:
        return 0.0
    return clamp(max(scores))


def _activation_examples(raw_activations: Any, limit: int) -> list[str]:
    if not isinstance(raw_activations, list):
        return []
    rendered: list[str] = []
    for item in raw_activations[:limit]:
        if not isinstance(item, dict):
            continue
        tokens = [str(token) for token in item.get("tokens", [])]
        values = _number_list(item.get("values", []))
        if not tokens:
            continue
        max_value = max(values) if values else _coerce_float(item.get("maxValue"))
        text = "".join(tokens).replace("\n", "\\n")
        rendered.append(f"max_activation={max_value:.3f} | {text[:500]}")
    return rendered


def _activation_signature(payload: dict[str, Any]) -> list[float]:
    # _coerce_float: a non-numeric API payload value (string, dict, list) must not
    # raise and abort the whole features_for loop.
    values = [
        _coerce_float(payload.get("maxActApprox")),
        _coerce_float(payload.get("frac_nonzero")),
    ]
    values.extend(_number_list(payload.get("pos_values", []))[:20])
    values.extend(_number_list(payload.get("neg_values", []))[:20])
    values.extend(_number_list(payload.get("topkCosSimValues", []))[:20])
    values.extend(_number_list(payload.get("neuron_alignment_values", []))[:20])
    return [round(value, 6) for value in values]


def _parse_layer(source: str) -> int | None:
    match = re.match(r"^(\d+)", source)
    if match:
        return int(match.group(1))
    return None


def _number_list(raw_values: Any) -> list[float]:
    if not isinstance(raw_values, list):
        return []
    values = []
    for value in raw_values:
        # Skip nulls, strings, and nested objects from the API instead of
        # raising and aborting the whole features_for loop.
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
