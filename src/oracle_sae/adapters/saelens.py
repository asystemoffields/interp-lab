from __future__ import annotations

import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from oracle_sae.schema import Criterion, FeatureEvidence

LAYER_PATTERN = re.compile(r"(?:blocks\.|layers\.|^)(\d+)")


class SAELensFeatureProvider:
    """Loads selected features from a pretrained SAE Lens SAE.

    This adapter intentionally avoids importing `sae_lens` at module import time.
    Projects that want it can install `oracle-sae[saelens]`.
    """

    def __init__(
        self,
        *,
        release: str,
        sae_id: str,
        feature_indices: list[int] | None = None,
        max_features: int = 32,
        device: str = "cpu",
        force_download: bool = False,
        feature_metadata: dict[int, dict[str, Any]] | None = None,
        sae_loader: Callable[..., tuple[Any, dict[str, Any], Any]] | None = None,
        signature_size: int = 128,
    ):
        self.release = release
        self.sae_id = sae_id
        self.feature_indices = feature_indices
        self.max_features = max_features
        self.device = device
        self.force_download = force_download
        self.feature_metadata = feature_metadata or {}
        self.sae_loader = sae_loader
        self.signature_size = signature_size

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        sae, cfg, sparsity = self._load_sae()
        decoder = _require_decoder_matrix(sae)
        feature_count = _matrix_length(decoder)
        indices = self._selected_indices(feature_count, sparsity)
        hook_name = str(cfg.get("hook_name") or cfg.get("hook_point") or "")
        layer = _coerce_layer(cfg.get("hook_layer"), hook_name, self.sae_id)
        evidence_items: list[FeatureEvidence] = []
        for index in indices:
            metadata = dict(self.feature_metadata.get(index, {}))
            decoder_row = _row(decoder, index, self.signature_size)
            sparsity_value = _sparsity_value(sparsity, index)
            metadata.update(
                {
                    "release": self.release,
                    "sae_id": self.sae_id,
                    "feature_index": index,
                    "hook_name": hook_name,
                    "sparsity": sparsity_value,
                    "decoder_norm": round(_norm(decoder_row), 6),
                }
            )
            label = str(metadata.get("label", f"SAELens feature {index}"))
            evidence_items.append(
                FeatureEvidence(
                    feature_id=f"{self.release}@{self.sae_id}:{index}",
                    model=model,
                    layer=layer,
                    label=label,
                    examples=[str(item) for item in metadata.get("examples", [])],
                    activation_signature=[
                        value
                        for value in [
                            float(sparsity_value or 0.0),
                            _norm(decoder_row),
                        ]
                    ],
                    decoder_signature=decoder_row,
                    causal_effects={},
                    source="saelens",
                    metadata=metadata,
                )
            )
        return evidence_items

    def _load_sae(self) -> tuple[Any, dict[str, Any], Any]:
        if self.sae_loader is not None:
            return self.sae_loader(
                self.release,
                self.sae_id,
                device=self.device,
                force_download=self.force_download,
            )
        try:
            sae_lens = importlib.import_module("sae_lens")
        except ImportError as exc:
            raise RuntimeError(
                "SAELens is not installed. Install it with `python -m pip install oracle-sae[saelens]` "
                "or provide activation records exported from SAELens."
            ) from exc
        SAE = getattr(sae_lens, "SAE", None)
        if SAE is None:
            raise RuntimeError("The installed sae_lens package does not expose SAE")
        if hasattr(SAE, "from_pretrained_with_cfg_and_sparsity"):
            return SAE.from_pretrained_with_cfg_and_sparsity(
                self.release,
                self.sae_id,
                device=self.device,
                force_download=self.force_download,
            )
        sae = SAE.from_pretrained(
            self.release,
            self.sae_id,
            device=self.device,
            force_download=self.force_download,
        )
        cfg = getattr(sae, "cfg", {})
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        elif not isinstance(cfg, dict):
            cfg = vars(cfg) if cfg is not None else {}
        return sae, dict(cfg), None

    def _selected_indices(self, feature_count: int, sparsity: Any) -> list[int]:
        if self.feature_indices is not None:
            for index in self.feature_indices:
                if index < 0 or index >= feature_count:
                    raise ValueError(f"Feature index {index} is outside 0..{feature_count - 1}")
            return self.feature_indices
        sparsity_values = _tensor_to_list(sparsity)
        if sparsity_values and len(sparsity_values) >= feature_count:
            ranked = sorted(range(feature_count), key=lambda index: sparsity_values[index], reverse=True)
            return ranked[: self.max_features]
        return list(range(min(self.max_features, feature_count)))


def load_saelens_feature_metadata(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("SAELens feature metadata must be a JSON object keyed by feature index")
    return {int(key): dict(value) for key, value in data.items()}


def parse_feature_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indices: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending feature range: {chunk}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(chunk))
    return indices


def _require_decoder_matrix(sae: Any) -> Any:
    for name in ("W_dec", "decoder", "decoder_weight"):
        if hasattr(sae, name):
            matrix = getattr(sae, name)
            if matrix is not None:
                return matrix
    state_dict = sae.state_dict() if hasattr(sae, "state_dict") else {}
    for key in ("W_dec", "decoder.weight", "decoder_weight"):
        if key in state_dict:
            return state_dict[key]
    raise RuntimeError("Could not find a decoder matrix on the SAELens SAE")


def _matrix_length(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(matrix)


def _row(matrix: Any, index: int, limit: int) -> list[float]:
    row = matrix[index]
    return [round(float(value), 6) for value in _tensor_to_list(row)[:limit]]


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _sparsity_value(sparsity: Any, index: int) -> float | None:
    values = _tensor_to_list(sparsity)
    if not values or index >= len(values):
        return None
    return float(values[index])


def _coerce_layer(raw_layer: Any, *fallbacks: str) -> int | None:
    if raw_layer is not None:
        return int(raw_layer)
    for value in fallbacks:
        match = LAYER_PATTERN.search(value)
        if match:
            return int(match.group(1))
    return None


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
