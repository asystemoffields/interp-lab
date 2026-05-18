from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import pearson


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    criterion_score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, line_label: str) -> "PromptRecord":
        if "text" not in data:
            raise ValueError(f"{line_label}: missing text")
        if "criterion_score" not in data:
            raise ValueError(f"{line_label}: missing criterion_score")
        return cls(
            prompt_id=str(data.get("prompt_id", data.get("id", ""))),
            text=str(data["text"]),
            criterion_score=float(data["criterion_score"]),
        )


def export_hf_activation_records(
    *,
    model_name: str,
    dataset_path: str | Path,
    out_path: str | Path,
    layers: list[int] | None = None,
    features_per_layer: int = 16,
    pool: str = "last",
    device: str = "cpu",
    max_length: int = 128,
) -> Path:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to export Hugging Face activations.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to export Hugging Face activations.",
    )
    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    layer_vectors: dict[int, list[list[float]]] = {}
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            selected_layers = _resolve_layers(layers, len(hidden_states))
            for layer in selected_layers:
                vector = _pool_hidden_state(
                    hidden_states[layer],
                    encoded.get("attention_mask"),
                    pool=pool,
                )
                layer_vectors.setdefault(layer, []).append(vector)

    scores = [prompt.criterion_score for prompt in prompts]
    selected_features = _select_features(layer_vectors, scores, features_per_layer)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt in enumerate(prompts):
            features = []
            feature_metadata = {}
            for layer, dimensions in selected_features.items():
                vector = layer_vectors[layer][prompt_index]
                for dimension, association in dimensions:
                    feature_id = f"L{layer}:D{dimension}"
                    activation = vector[dimension]
                    features.append(
                        {
                            "feature_id": feature_id,
                            "activation": activation,
                            "label": f"hidden dimension {dimension} at hidden-state layer {layer}",
                            "layer": layer,
                        }
                    )
                    feature_metadata[feature_id] = {
                        "label": f"hidden dimension {dimension} at hidden-state layer {layer}",
                        "layer": layer,
                        "source": "hf-hidden-state",
                        "signed_selection_correlation": round(association, 6),
                    }
            row = {
                "model": model_name,
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "criterion_score": prompt.criterion_score,
                "features": features,
                "feature_metadata": feature_metadata,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_label = f"{file_path}:{line_number}"
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_label}: invalid JSON: {exc.msg}") from exc
            records.append(PromptRecord.from_dict(data, line_label=line_label))
    return records


def split_prompt_record_indexes(prompts: list[PromptRecord]) -> tuple[set[int], set[int]]:
    if not prompts:
        return set(), set()
    scores = [prompt.criterion_score for prompt in prompts]
    if min(scores) == max(scores):
        return set(range(len(prompts))), set()
    threshold = sum(scores) / len(scores)
    positive_indexes = {
        index
        for index, prompt in enumerate(prompts)
        if prompt.criterion_score >= threshold
    }
    negative_indexes = set(range(len(prompts))) - positive_indexes
    return positive_indexes, negative_indexes


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    layers: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {chunk}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(chunk))
    return layers


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Hugging Face hidden states as activation records.")
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument("--layers", help="Hidden-state layers, e.g. 0,4,8 or 1-3. Defaults to final layer.")
    parser.add_argument("--features-per-layer", type=int, default=16)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    return parser


def run_export_from_args(args: argparse.Namespace) -> Path:
    try:
        layers = parse_layers(args.layers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return export_hf_activation_records(
        model_name=args.model,
        dataset_path=args.dataset,
        out_path=args.out,
        layers=layers,
        features_per_layer=args.features_per_layer,
        pool=args.pool,
        device=args.device,
        max_length=args.max_length,
    )


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _resolve_layers(layers: list[int] | None, hidden_state_count: int) -> list[int]:
    if layers is None:
        return [hidden_state_count - 1]
    for layer in layers:
        if layer < 0 or layer >= hidden_state_count:
            raise ValueError(f"Layer {layer} is outside 0..{hidden_state_count - 1}")
    return layers


def _pool_hidden_state(hidden_state, attention_mask, *, pool: str) -> list[float]:
    if pool == "mean":
        if attention_mask is None:
            pooled = hidden_state[0].mean(dim=0)
        else:
            mask = attention_mask[0].to(hidden_state.dtype)
            pooled = (hidden_state[0] * mask[:, None]).sum(dim=0) / mask.sum().clamp(min=1)
    else:
        if attention_mask is None:
            token_index = hidden_state.shape[1] - 1
        else:
            token_index = int(attention_mask[0].sum().item()) - 1
        pooled = hidden_state[0, token_index]
    return [float(value) for value in pooled.detach().cpu().tolist()]


def _select_features(
    layer_vectors: dict[int, list[list[float]]],
    scores: list[float],
    features_per_layer: int,
) -> dict[int, list[tuple[int, float]]]:
    selected: dict[int, list[tuple[int, float]]] = {}
    for layer, vectors in layer_vectors.items():
        if not vectors:
            selected[layer] = []
            continue
        dimensions = len(vectors[0])
        associations = []
        for dimension in range(dimensions):
            values = [vector[dimension] for vector in vectors]
            association = pearson(values, scores)
            associations.append((dimension, association))
        associations.sort(key=lambda item: abs(item[1]), reverse=True)
        selected[layer] = associations[:features_per_layer]
    return selected
