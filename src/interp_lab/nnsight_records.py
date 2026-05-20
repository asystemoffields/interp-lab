from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

from interp_lab.hf_records import load_prompt_records
from interp_lab.math_utils import pearson

PATH_PART_PATTERN = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)?(?P<indexes>(?:\[[0-9]+\])*)$")
INDEX_PATTERN = re.compile(r"\[([0-9]+)\]")


def export_nnsight_activation_records(
    *,
    model_name: str,
    dataset_path: str | Path,
    out_path: str | Path,
    activation_paths: list[str],
    features_per_path: int = 16,
    pool: str = "last",
    device_map: str | None = None,
    remote: bool = False,
    model_factory: Any | None = None,
) -> Path:
    torch = _optional_import("torch", "Install `interp-lab[nnsight]` to export NNsight activations.")
    nnsight = _optional_import("nnsight", "Install `interp-lab[nnsight]` to export NNsight activations.")
    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")
    if not activation_paths:
        raise ValueError("At least one --activation-path is required")

    LanguageModel = model_factory or getattr(nnsight, "LanguageModel", None)
    if LanguageModel is None:
        raise RuntimeError("The installed nnsight package does not expose LanguageModel")
    model = _build_language_model(LanguageModel, model_name, device_map=device_map)

    path_vectors: dict[str, list[list[float]]] = {path: [] for path in activation_paths}
    for prompt in prompts:
        saved_by_path = {}
        trace_kwargs = {"remote": True} if remote else {}
        with model.trace(prompt.text, **trace_kwargs):
            for activation_path in activation_paths:
                saved_by_path[activation_path] = _resolve_activation_path(model, activation_path).save()
        for activation_path, saved in saved_by_path.items():
            value = getattr(saved, "value", saved)
            path_vectors[activation_path].append(_pool_activation(value, torch=torch, pool=pool))

    scores = [prompt.criterion_score for prompt in prompts]
    selected = _select_dimensions(path_vectors, scores, features_per_path)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt in enumerate(prompts):
            features = []
            feature_metadata = {}
            for activation_path, dimensions in selected.items():
                vector = path_vectors[activation_path][prompt_index]
                layer = _layer_for_path(activation_path)
                for dimension, association in dimensions:
                    feature_id = f"{activation_path}:D{dimension}"
                    features.append(
                        {
                            "feature_id": feature_id,
                            "activation": vector[dimension],
                            "label": f"{activation_path} dimension {dimension}",
                            "layer": layer,
                        }
                    )
                    feature_metadata[feature_id] = {
                        "label": f"{activation_path} dimension {dimension}",
                        "layer": layer,
                        "source": "nnsight",
                        "activation_path": activation_path,
                        "dimension": dimension,
                        "signed_selection_correlation": round(association, 6),
                    }
            handle.write(
                json.dumps(
                    {
                        "model": model_name,
                        "prompt_id": prompt.prompt_id,
                        "text": prompt.text,
                        "criterion_score": prompt.criterion_score,
                        "features": features,
                        "feature_metadata": feature_metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def parse_activation_paths(values: list[str] | None) -> list[str]:
    if not values:
        return []
    paths: list[str] = []
    for value in values:
        for chunk in value.split(","):
            path = chunk.strip()
            if path:
                paths.append(path)
    return paths


def build_nnsight_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export NNsight trace activations as activation records.")
    parser.add_argument("--model", required=True, help="Model name passed to nnsight.LanguageModel.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument(
        "--activation-path",
        action="append",
        required=True,
        help="NNsight path such as transformer.h[6].output[0]. Repeat or comma-separate.",
    )
    parser.add_argument("--features-per-path", type=int, default=16)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device-map", help="Optional device_map passed to nnsight.LanguageModel.")
    parser.add_argument("--remote", action="store_true", help="Run the trace with remote=True.")
    return parser


def run_nnsight_export_from_args(args: argparse.Namespace) -> Path:
    return export_nnsight_activation_records(
        model_name=args.model,
        dataset_path=args.dataset,
        out_path=args.out,
        activation_paths=parse_activation_paths(args.activation_path),
        features_per_path=args.features_per_path,
        pool=args.pool,
        device_map=args.device_map,
        remote=args.remote,
    )


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _build_language_model(LanguageModel: Any, model_name: str, *, device_map: str | None):
    if device_map:
        try:
            return LanguageModel(model_name, device_map=device_map)
        except TypeError:
            return LanguageModel(model_name, device=device_map)
    return LanguageModel(model_name)


def _resolve_activation_path(model: Any, path: str) -> Any:
    current = model
    for raw_part in path.split("."):
        if not raw_part:
            continue
        match = PATH_PART_PATTERN.fullmatch(raw_part)
        if not match:
            raise ValueError(f"Invalid NNsight activation path part {raw_part!r}")
        name = match.group("name")
        if name:
            current = getattr(current, name)
        for raw_index in INDEX_PATTERN.findall(match.group("indexes")):
            current = current[int(raw_index)]
    return current


def _pool_activation(value: Any, *, torch: Any, pool: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if not hasattr(value, "reshape"):
        value = torch.as_tensor(value)
    shape = getattr(value, "shape", ())
    if len(shape) < 2:
        values = value.reshape(-1)
    else:
        if len(shape) >= 3:
            value = value[0]
        value = value.reshape(value.shape[0], -1)
        values = value.mean(dim=0) if pool == "mean" else value[-1]
    return [float(item) for item in values.tolist()]


def _select_dimensions(
    path_vectors: dict[str, list[list[float]]],
    scores: list[float],
    features_per_path: int,
) -> dict[str, list[tuple[int, float]]]:
    selected: dict[str, list[tuple[int, float]]] = {}
    for activation_path, vectors in path_vectors.items():
        if not vectors:
            selected[activation_path] = []
            continue
        dimensions = len(vectors[0])
        associations = []
        for dimension in range(dimensions):
            values = [vector[dimension] for vector in vectors]
            associations.append((dimension, pearson(values, scores)))
        associations.sort(key=lambda item: abs(item[1]), reverse=True)
        selected[activation_path] = associations[:features_per_path]
    return selected


def _layer_for_path(path: str) -> int | None:
    for pattern in (r"\.h\[(\d+)\]", r"\.layers\[(\d+)\]", r"\.blocks\[(\d+)\]"):
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None
