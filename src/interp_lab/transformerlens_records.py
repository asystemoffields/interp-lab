from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from interp_lab.hf_records import LayerSelection, PromptRecord, load_prompt_records, parse_layers
from interp_lab.math_utils import pearson

DEFAULT_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_post"


def export_transformerlens_activation_records(
    *,
    model_name: str,
    dataset_path: str | Path,
    out_path: str | Path,
    hook_names: list[str] | None = None,
    layers: LayerSelection = None,
    hook_template: str = DEFAULT_HOOK_TEMPLATE,
    features_per_hook: int = 16,
    pool: str = "last",
    device: str = "cpu",
    prepend_bos: bool = True,
) -> Path:
    torch = _optional_import(
        "torch",
        "Install `interp-lab[transformerlens]` to export TransformerLens activations.",
    )
    transformer_lens = _optional_import(
        "transformer_lens",
        "Install `interp-lab[transformerlens]` to export TransformerLens activations.",
    )
    HookedTransformer = getattr(transformer_lens, "HookedTransformer", None)
    if HookedTransformer is None:
        raise RuntimeError("The installed transformer_lens package does not expose HookedTransformer")

    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")

    model = HookedTransformer.from_pretrained(model_name, device=device)
    if hasattr(model, "eval"):
        model.eval()
    resolved_hooks = _resolve_hook_names(model, hook_names, layers, hook_template)
    hook_vectors: dict[str, list[list[float]]] = {hook_name: [] for hook_name in resolved_hooks}

    with torch.no_grad():
        for prompt in prompts:
            tokens = model.to_tokens(prompt.text, prepend_bos=prepend_bos)
            _logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in resolved_hooks,
                remove_batch_dim=False,
            )
            for hook_name in resolved_hooks:
                if hook_name not in cache:
                    raise RuntimeError(f"TransformerLens cache did not include hook {hook_name!r}")
                hook_vectors[hook_name].append(_pool_activation(cache[hook_name], pool=pool))

    scores = [prompt.criterion_score for prompt in prompts]
    selected = _select_dimensions(hook_vectors, scores, features_per_hook)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt in enumerate(prompts):
            features = []
            feature_metadata = {}
            for hook_name, dimensions in selected.items():
                vector = hook_vectors[hook_name][prompt_index]
                layer = _layer_for_hook(hook_name)
                for dimension, association in dimensions:
                    feature_id = f"{hook_name}:D{dimension}"
                    features.append(
                        {
                            "feature_id": feature_id,
                            "activation": vector[dimension],
                            "label": f"{hook_name} dimension {dimension}",
                            "layer": layer,
                        }
                    )
                    feature_metadata[feature_id] = {
                        "label": f"{hook_name} dimension {dimension}",
                        "layer": layer,
                        "source": "transformerlens",
                        "hook_name": hook_name,
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


def parse_hook_names(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    hooks: list[str] = []
    for value in values:
        for chunk in value.split(","):
            hook_name = chunk.strip()
            if hook_name:
                hooks.append(hook_name)
    return hooks


def build_transformerlens_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export TransformerLens cached activations as activation records."
    )
    parser.add_argument("--model", required=True, help="TransformerLens model name.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument(
        "--hook-name",
        action="append",
        help="Hook name or comma-separated hook names. Overrides --layers.",
    )
    parser.add_argument("--layers", help="Layers for --hook-template, e.g. all, 0,4,8, or 1-3.")
    parser.add_argument(
        "--hook-template",
        default=DEFAULT_HOOK_TEMPLATE,
        help="Template used with --layers. It must include {layer}.",
    )
    parser.add_argument("--features-per-hook", type=int, default=16)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-prepend-bos",
        action="store_true",
        help="Pass prepend_bos=False to TransformerLens tokenization.",
    )
    return parser


def run_transformerlens_export_from_args(args: argparse.Namespace) -> Path:
    try:
        layers = parse_layers(args.layers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return export_transformerlens_activation_records(
        model_name=args.model,
        dataset_path=args.dataset,
        out_path=args.out,
        hook_names=parse_hook_names(args.hook_name),
        layers=layers,
        hook_template=args.hook_template,
        features_per_hook=args.features_per_hook,
        pool=args.pool,
        device=args.device,
        prepend_bos=not args.no_prepend_bos,
    )


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _resolve_hook_names(
    model: Any,
    hook_names: list[str] | None,
    layers: LayerSelection,
    hook_template: str,
) -> list[str]:
    if hook_names:
        return hook_names
    if "{layer}" not in hook_template:
        raise ValueError("--hook-template must include {layer}")
    cfg = getattr(model, "cfg", None)
    n_layers = getattr(cfg, "n_layers", None)
    if layers is None:
        if n_layers is None:
            raise ValueError("--layers is required when the TransformerLens model has no cfg.n_layers")
        layers = [int(n_layers) - 1]
    elif layers == "all":
        if n_layers is None:
            raise ValueError("--layers all requires the TransformerLens model to expose cfg.n_layers")
        layers = list(range(int(n_layers)))
    elif isinstance(layers, str):
        raise ValueError("--layers must be 'all' or layer indexes")
    return [hook_template.format(layer=layer) for layer in layers]


def _pool_activation(activation: Any, *, pool: str) -> list[float]:
    if hasattr(activation, "detach"):
        activation = activation.detach()
    if hasattr(activation, "cpu"):
        activation = activation.cpu()
    shape = getattr(activation, "shape", ())
    if len(shape) < 2:
        values = activation.reshape(-1)
    else:
        if len(shape) >= 3:
            activation = activation[0]
        activation = activation.reshape(activation.shape[0], -1)
        values = activation.mean(dim=0) if pool == "mean" else activation[-1]
    return [float(value) for value in values.tolist()]


def _select_dimensions(
    hook_vectors: dict[str, list[list[float]]],
    scores: list[float],
    features_per_hook: int,
) -> dict[str, list[tuple[int, float]]]:
    selected: dict[str, list[tuple[int, float]]] = {}
    for hook_name, vectors in hook_vectors.items():
        if not vectors:
            selected[hook_name] = []
            continue
        dimensions = len(vectors[0])
        associations = []
        for dimension in range(dimensions):
            values = [vector[dimension] for vector in vectors]
            associations.append((dimension, pearson(values, scores)))
        associations.sort(key=lambda item: abs(item[1]), reverse=True)
        selected[hook_name] = associations[:features_per_hook]
    return selected


def _layer_for_hook(hook_name: str) -> int | None:
    chunks = hook_name.split(".")
    for index, chunk in enumerate(chunks[:-1]):
        if chunk == "blocks" and chunks[index + 1].isdigit():
            return int(chunks[index + 1])
    return None
