from __future__ import annotations

import importlib
import os
from typing import Any

from interp_lab.schema import Criterion, FeatureEvidence


class GoodfireFeatureProvider:
    """Imports semantically searched features from the Goodfire SDK.

    The SDK stays optional. Users can install `interp-lab[goodfire]` and provide
    a Goodfire API key through GOODFIRE_API_KEY or the configured environment
    variable.
    """

    def __init__(
        self,
        *,
        top_k: int = 32,
        api_key_env: str = "GOODFIRE_API_KEY",
        client: Any | None = None,
        variant_factory: Any | None = None,
    ):
        self.top_k = top_k
        self.api_key_env = api_key_env
        self.client = client
        self.variant_factory = variant_factory

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        client, variant_factory = self._client_and_variant_factory()
        model_ref = variant_factory(model) if variant_factory is not None else model
        features = client.features.search(criterion.text, model=model_ref, top_k=self.top_k)
        return [
            goodfire_feature_to_evidence(feature, model=model, rank=rank)
            for rank, feature in enumerate(features, start=1)
        ]

    def _client_and_variant_factory(self) -> tuple[Any, Any | None]:
        if self.client is not None:
            return self.client, self.variant_factory
        try:
            goodfire = importlib.import_module("goodfire")
        except ImportError as exc:
            raise RuntimeError(
                "Goodfire SDK is not installed. Install it with `python -m pip install interp-lab[goodfire]`."
            ) from exc
        Client = getattr(goodfire, "Client", None)
        if Client is None:
            raise RuntimeError("The installed goodfire package does not expose Client")
        api_key = os.environ.get(self.api_key_env)
        client = Client(api_key=api_key) if api_key else Client()
        variant_factory = self.variant_factory
        if variant_factory is None:
            variant_factory = getattr(goodfire, "Variant", None)
        return client, variant_factory


def goodfire_feature_to_evidence(feature: Any, *, model: str, rank: int = 1) -> FeatureEvidence:
    feature_id = _feature_id(feature)
    label = str(_get(feature, "label", _get(feature, "description", feature_id)))
    metadata = _metadata(feature)
    metadata.update(
        {
            "goodfire_rank": rank,
            "goodfire_index": _get(feature, "index", None),
            "goodfire_uuid": _get(feature, "uuid", _get(feature, "id", None)),
        }
    )
    examples = [str(item) for item in _list_value(_get(feature, "examples", []))]
    activation_signature = _number_list(
        _get(feature, "vector", _get(feature, "activations", _get(feature, "activation", [])))
    )
    decoder_signature = _number_list(_get(feature, "decoder", _get(feature, "decoder_signature", [])))
    layer = _optional_int(_get(feature, "layer", _get(feature, "layer_index", None)))
    return FeatureEvidence(
        feature_id=f"goodfire:{feature_id}",
        model=model,
        layer=layer,
        label=label,
        examples=examples,
        activation_signature=activation_signature,
        decoder_signature=decoder_signature,
        causal_effects={"specificity": 0.5},
        source="goodfire",
        metadata=metadata,
    )


def _feature_id(feature: Any) -> str:
    for key in ("uuid", "id", "index"):
        value = _get(feature, key, None)
        if value is not None:
            return str(value)
    return str(_get(feature, "label", "feature")).strip().replace(" ", "-")


def _metadata(feature: Any) -> dict[str, Any]:
    if isinstance(feature, dict):
        return {
            str(key): value
            for key, value in feature.items()
            if key not in {"vector", "activations", "decoder", "decoder_signature"}
        }
    data = {}
    for key in ("label", "description", "index", "uuid", "id", "layer"):
        value = _get(feature, key, None)
        if value is not None:
            data[key] = value
    return data


def _get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, (int, float)):
                output.append(float(item))
        return output
    return []


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
