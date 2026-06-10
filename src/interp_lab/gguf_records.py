"""Bridge GGUF / llama.cpp runtimes into the activation-records pipeline.

interp-lab's record export normally requires torch + transformers, but the
activation-records JSONL format is model-agnostic: any runtime that can dump
hidden states can feed `inspect --backend records`. This module provides the
two entry points for CPU GGUF labs:

Path A — ``export_gguf_records``: drive llama-cpp-python directly (optional
    extra ``gguf``). llama.cpp's embedding API, with pooling disabled, returns
    one vector per token from the FINAL hidden layer (after the final norm).
    That yields genuine last-layer activation records — and nothing else.
    llama.cpp does not expose intermediate layers through this API, so every
    record is stamped ``"layers_available": "final_only"``.

Path B — ``convert_hidden_state_dump``: convert a simple documented JSONL dump
    (one ``{prompt_index, layer, tokens, hidden}`` object per line) that ANY
    runtime can produce — a patched llama.cpp, a custom GGML harness, or ten
    lines of Python. This is the full-fidelity route: every layer the runtime
    dumps becomes a record layer.

Both paths emit the existing activation-records format (no new schema) and
re-validate their output through the real records loader
(`interp_lab.adapters.records.ActivationRecord`) before returning.

See ``docs/GGUF_BRIDGE.md`` for the dump format spec and worked examples.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interp_lab.adapters.records import ActivationRecord
from interp_lab.hf_records import PromptRecord, load_prompt_records

GGUF_INSTALL_MESSAGE = "Install `interp-lab[gguf]` to export GGUF activation records with llama-cpp-python."

HIDDEN_STATE_DUMP_SCHEMA = "interp-lab.hidden_state_dump.v1"

# Documented duck-typed interface for the object returned by `_load_llama` /
# `llama_factory`. Tests (and alternative runtimes) only need:
#   embed(text) -> list[list[float]]  per-token final-layer vectors
#                  (pooling disabled), or list[float] for a single
#                  already-pooled vector.
# and ONE way to discover the transformer block count:
#   n_layer() -> int                                 (method), or
#   metadata -> {"general.architecture": "llama", "llama.block_count": "30", ...}
# Callers can bypass discovery entirely with `n_layers=`.


@dataclass(frozen=True)
class GgufExportSummary:
    """What an export/convert wrote, for human and --json CLI output."""

    path: Path
    record_count: int
    feature_count: int
    layers: list[int]
    layers_available: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "out": str(self.path),
            "record_count": self.record_count,
            "feature_count": self.feature_count,
            "layers": self.layers,
            "layers_available": self.layers_available,
            "source": self.source,
        }


def export_gguf_records(
    *,
    model_path: str | Path,
    dataset_path: str | Path,
    out_path: str | Path,
    n_ctx: int = 2048,
    n_threads: int | None = None,
    top_features: int = 64,
    pool: str = "last",
    n_layers: int | None = None,
    model_name: str | None = None,
    llama_factory: Any | None = None,
) -> Path:
    """Export FINAL-layer activation records from a GGUF model via llama.cpp.

    Honest limitation: llama.cpp's embedding API only exposes the final
    hidden state (post final norm). Feature ids are ``L{n_layers}:D{dim}``,
    matching the Hugging Face hidden-state convention where
    ``hidden_states[n_layers]`` is the final-norm output that
    ``export-hf-records`` labels when no ``--layers`` is given. Records are
    stamped ``"source": "llama_cpp_embeddings"`` and
    ``"layers_available": "final_only"`` so downstream reports cannot mistake
    this for multi-layer evidence. For intermediate layers, use
    ``convert_hidden_state_dump`` (Path B).

    Dimensions are ranked by activation variance across prompts and the top
    ``top_features`` are kept (variance is well defined even when every prompt
    shares one criterion score, unlike the correlation ranking the HF exporter
    uses; the records backend recomputes criterion associations on load).

    ``llama_factory`` is the test/integration seam: any callable returning an
    object with the duck-typed interface documented at module top. When it is
    None, llama-cpp-python is imported (optional extra ``gguf``).
    """
    _validate_pool(pool)
    if top_features < 1:
        raise ValueError("top_features must be at least 1")
    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")

    factory = llama_factory or _load_llama
    llama = factory(str(model_path), n_ctx=n_ctx, n_threads=n_threads)
    layer_count = _resolve_n_layers(llama, n_layers)
    final_layer = layer_count  # hidden_states index of the final-norm output.

    resolved_model_name = model_name or Path(model_path).name
    vectors = [_pool_token_vectors(_embed_prompt(llama, prompt.text), pool=pool) for prompt in prompts]
    _require_uniform_width(vectors, label=str(model_path))

    selected = _select_dimensions_by_variance(vectors, top_features)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt, vector in zip(prompts, vectors):
            features = []
            feature_metadata = {}
            for dimension, variance in selected:
                feature_id = f"L{final_layer}:D{dimension}"
                features.append(
                    {
                        "feature_id": feature_id,
                        "activation": vector[dimension],
                        "label": f"hidden dimension {dimension} at hidden-state layer {final_layer}",
                        "layer": final_layer,
                    }
                )
                feature_metadata[feature_id] = {
                    "label": f"hidden dimension {dimension} at hidden-state layer {final_layer}",
                    "layer": final_layer,
                    "layer_convention": "hidden_state_index",
                    "source": "llama_cpp_embeddings",
                    "layers_available": "final_only",
                    "selection_variance": round(variance, 6),
                }
            row = {
                "model": resolved_model_name,
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "criterion_score": prompt.criterion_score,
                "features": features,
                "feature_metadata": feature_metadata,
                "metadata": {
                    "source": "llama_cpp_embeddings",
                    "layers_available": "final_only",
                    "layer_convention": "hidden_state_index",
                    "model_path": str(model_path),
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _validate_with_records_loader(path)
    return path


def summarize_records(path: str | Path) -> GgufExportSummary:
    """Summarize a written records file (used by the CLI, incl. --json)."""
    record_count = 0
    feature_ids: set[str] = set()
    layers: set[int] = set()
    sources: set[str] = set()
    layers_available: set[str] = set()
    for record in _load_validated_records(Path(path)):
        record_count += 1
        feature_ids.update(record.features)
        for metadata in record.feature_metadata.values():
            if metadata.get("layer") is not None:
                layers.add(int(metadata["layer"]))
            if metadata.get("source"):
                sources.add(str(metadata["source"]))
        if record.metadata.get("layers_available"):
            layers_available.add(str(record.metadata["layers_available"]))
    return GgufExportSummary(
        path=Path(path),
        record_count=record_count,
        feature_count=len(feature_ids),
        layers=sorted(layers),
        layers_available=",".join(sorted(layers_available)) or "unknown",
        source=",".join(sorted(sources)) or "unknown",
    )


def convert_hidden_state_dump(
    *,
    dump_path: str | Path,
    out_path: str | Path,
    prompts_path: str | Path | None = None,
    layer_convention: str = "hidden_state_index",
    pool: str = "last",
    features_per_layer: int | None = 64,
    model_name: str | None = None,
) -> Path:
    """Convert a hidden-state dump JSONL into activation records (all layers).

    Dump format (``interp-lab.hidden_state_dump.v1``), one JSON object per line:

        {"prompt_index": 0, "layer": 5,
         "tokens": ["The", " sky"],                # optional
         "hidden": [[0.1, ...], [0.2, ...]],        # one row per token,
                                                    # or a single flat vector
         "prompt_id": "p1", "text": "The sky",     # optional
         "criterion_score": 1.0, "model": "m"}      # optional

    Criterion scores and text come from ``prompts_path`` (the scored-prompts
    JSONL used everywhere else: ``{"prompt_id", "text", "criterion_score"}``,
    matched by ``prompt_index`` order) or inline on the dump lines; inline
    values win. Every prompt must cover every layer that appears in the dump.

    This is the full-fidelity GGUF path: any layer the producing runtime dumps
    becomes a record layer, stamped with ``layer_convention`` so cross-tool
    layer comparisons stay honest. The output is re-read through the real
    records loader before this function returns.
    """
    _validate_pool(pool)
    if features_per_layer is not None and features_per_layer < 1:
        raise ValueError("features_per_layer must be at least 1 or None for all dimensions")
    prompts = load_prompt_records(prompts_path) if prompts_path is not None else None
    entries = _load_dump_entries(Path(dump_path), prompts=prompts, pool=pool)
    if not entries:
        raise ValueError(f"{dump_path}: no dump lines found")

    prompt_indexes = sorted({entry.prompt_index for entry in entries})
    layers = sorted({entry.layer for entry in entries})
    by_key = {(entry.prompt_index, entry.layer): entry for entry in entries}
    for prompt_index in prompt_indexes:
        for layer in layers:
            if (prompt_index, layer) not in by_key:
                raise ValueError(
                    f"{dump_path}: prompt_index {prompt_index} has no line for layer {layer}; "
                    "every prompt must cover every dumped layer"
                )

    layer_vectors = {
        layer: [by_key[(prompt_index, layer)].vector for prompt_index in prompt_indexes]
        for layer in layers
    }
    for layer, vectors in layer_vectors.items():
        _require_uniform_width(vectors, label=f"{dump_path} layer {layer}")
    selected = {
        layer: _select_dimensions_by_variance(
            vectors, features_per_layer if features_per_layer is not None else len(vectors[0])
        )
        for layer, vectors in layer_vectors.items()
    }

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for position, prompt_index in enumerate(prompt_indexes):
            first_entry = by_key[(prompt_index, layers[0])]
            resolved = _resolve_prompt_fields(
                first_entry,
                prompts=prompts,
                model_name=model_name,
                dump_path=Path(dump_path),
            )
            features = []
            feature_metadata = {}
            for layer in layers:
                vector = layer_vectors[layer][position]
                for dimension, variance in selected[layer]:
                    feature_id = f"L{layer}:D{dimension}"
                    features.append(
                        {
                            "feature_id": feature_id,
                            "activation": vector[dimension],
                            "label": f"hidden dimension {dimension} at layer {layer}",
                            "layer": layer,
                        }
                    )
                    feature_metadata[feature_id] = {
                        "label": f"hidden dimension {dimension} at layer {layer}",
                        "layer": layer,
                        "layer_convention": layer_convention,
                        "source": "hidden_state_dump",
                        "selection_variance": round(variance, 6),
                    }
            row = {
                "model": resolved.model,
                "prompt_id": resolved.prompt_id,
                "text": resolved.text,
                "criterion_score": resolved.criterion_score,
                "features": features,
                "feature_metadata": feature_metadata,
                "metadata": {
                    "source": "hidden_state_dump",
                    "dump_format": HIDDEN_STATE_DUMP_SCHEMA,
                    "layer_convention": layer_convention,
                    "layers_available": "as_dumped",
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _validate_with_records_loader(path)
    return path


def build_gguf_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export final-layer activation records from a GGUF model via llama-cpp-python. "
            "llama.cpp only exposes the final hidden state; use convert-hidden-dump for all layers."
        )
    )
    parser.add_argument("--model", required=True, help="Path to the GGUF model file.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument("--top-features", type=int, default=64, help="Dimensions kept, ranked by variance.")
    parser.add_argument("--n-ctx", type=int, default=2048, help="llama.cpp context size.")
    parser.add_argument("--threads", type=int, help="llama.cpp thread count (default: library default).")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument(
        "--n-layers",
        type=int,
        help="Transformer block count override when GGUF metadata lacks block_count.",
    )
    parser.add_argument("--model-name", help="Model name stamped on records (default: GGUF file name).")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary instead of text.")
    return parser


def build_hidden_dump_convert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a hidden-state dump JSONL from any runtime into activation records."
    )
    parser.add_argument("--dump", required=True, help="Hidden-state dump JSONL (see docs/GGUF_BRIDGE.md).")
    parser.add_argument("--out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument(
        "--dataset",
        help="Optional scored prompt JSONL supplying text and criterion_score by prompt_index order.",
    )
    parser.add_argument(
        "--layer-convention",
        default="hidden_state_index",
        help="Convention stamped on every feature (default: hidden_state_index).",
    )
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument(
        "--features-per-layer",
        type=int,
        default=64,
        help="Dimensions kept per layer, ranked by variance. Use 0 for all dimensions.",
    )
    parser.add_argument("--model-name", help="Model name stamped on records when the dump has none.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary instead of text.")
    return parser


def run_gguf_export_from_args(args: argparse.Namespace) -> Path:
    return export_gguf_records(
        model_path=args.model,
        dataset_path=args.dataset,
        out_path=args.out,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        top_features=args.top_features,
        pool=args.pool,
        n_layers=args.n_layers,
        model_name=args.model_name,
    )


def run_hidden_dump_convert_from_args(args: argparse.Namespace) -> Path:
    features_per_layer = args.features_per_layer if args.features_per_layer > 0 else None
    return convert_hidden_state_dump(
        dump_path=args.dump,
        out_path=args.out,
        prompts_path=args.dataset,
        layer_convention=args.layer_convention,
        pool=args.pool,
        features_per_layer=features_per_layer,
        model_name=args.model_name,
    )


def _load_llama(model_path: str, *, n_ctx: int, n_threads: int | None) -> Any:
    """Default Path A seam: construct a llama_cpp.Llama for per-token embeddings.

    Uses ``embedding=True`` plus ``pooling_type=LLAMA_POOLING_TYPE_NONE`` so
    ``embed(text)`` returns one final-layer vector per token. ``pooling_type``
    is supported by llama-cpp-python >= 0.2.57; the ``gguf`` extra pins >= 0.3.
    """
    llama_cpp = _optional_import("llama_cpp", GGUF_INSTALL_MESSAGE)
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "embedding": True,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    pooling_none = getattr(llama_cpp, "LLAMA_POOLING_TYPE_NONE", None)
    if pooling_none is not None:
        kwargs["pooling_type"] = pooling_none
    if n_threads is not None:
        kwargs["n_threads"] = n_threads
    return llama_cpp.Llama(**kwargs)


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _embed_prompt(llama: Any, text: str) -> Any:
    embed = getattr(llama, "embed", None)
    if embed is None:
        raise RuntimeError(
            "The loaded llama object does not provide embed(); "
            "llama-cpp-python >= 0.3 with embedding=True is required"
        )
    return embed(text)


def _resolve_n_layers(llama: Any, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("n_layers must be at least 1")
        return override
    n_layer = getattr(llama, "n_layer", None)
    if callable(n_layer):
        return int(n_layer())
    metadata = getattr(llama, "metadata", None)
    if isinstance(metadata, dict):
        architecture = metadata.get("general.architecture")
        if architecture and f"{architecture}.block_count" in metadata:
            return int(metadata[f"{architecture}.block_count"])
        for key, value in metadata.items():
            if key.endswith(".block_count"):
                return int(value)
    raise ValueError(
        "Could not determine the transformer block count from the GGUF model "
        "(no n_layer() and no *.block_count metadata); pass n_layers=/--n-layers explicitly "
        "so feature ids use the correct final-layer number"
    )


def _validate_pool(pool: str) -> None:
    if pool not in {"last", "mean"}:
        raise ValueError("pool must be 'last' or 'mean'")


def _pool_token_vectors(raw: Any, *, pool: str) -> list[float]:
    """Pool per-token vectors (or accept a single already-pooled vector)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("embeddings must be a non-empty list of numbers or token vectors")
    if not isinstance(raw[0], list):
        return [float(value) for value in raw]
    token_vectors = [[float(value) for value in token] for token in raw]
    if pool == "mean":
        width = len(token_vectors[0])
        return [
            sum(vector[dimension] for vector in token_vectors) / len(token_vectors)
            for dimension in range(width)
        ]
    return token_vectors[-1]


def _require_uniform_width(vectors: list[list[float]], *, label: str) -> None:
    widths = {len(vector) for vector in vectors}
    if len(widths) > 1:
        raise ValueError(f"{label}: hidden vectors have inconsistent widths {sorted(widths)}")
    if widths == {0}:
        raise ValueError(f"{label}: hidden vectors are empty")


def _select_dimensions_by_variance(vectors: list[list[float]], limit: int) -> list[tuple[int, float]]:
    """Rank dimensions by activation variance across prompts; keep the top ``limit``.

    Deterministic tie-break: lower dimension index first. Returned in
    ascending dimension order so feature ids are stable across runs.
    """
    if not vectors:
        return []
    width = len(vectors[0])
    count = len(vectors)
    variances: list[tuple[int, float]] = []
    for dimension in range(width):
        values = [vector[dimension] for vector in vectors]
        mean_value = sum(values) / count
        variance = sum((value - mean_value) ** 2 for value in values) / count
        variances.append((dimension, variance))
    ranked = sorted(variances, key=lambda item: (-item[1], item[0]))[: max(limit, 0)]
    return sorted(ranked, key=lambda item: item[0])


@dataclass(frozen=True)
class _DumpEntry:
    line_label: str
    prompt_index: int
    layer: int
    vector: list[float]
    prompt_id: str | None
    text: str | None
    criterion_score: float | None
    model: str | None


@dataclass(frozen=True)
class _ResolvedPrompt:
    prompt_id: str
    text: str
    criterion_score: float
    model: str


def _load_dump_entries(dump_path: Path, *, prompts: list[PromptRecord] | None, pool: str) -> list[_DumpEntry]:
    entries: list[_DumpEntry] = []
    seen: set[tuple[int, int]] = set()
    with dump_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_label = f"{dump_path}:{line_number}"
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_label}: invalid JSON: {exc.msg}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{line_label}: each dump line must be a JSON object")
            entry = _parse_dump_entry(data, line_label=line_label, prompts=prompts, pool=pool)
            key = (entry.prompt_index, entry.layer)
            if key in seen:
                raise ValueError(
                    f"{line_label}: duplicate dump line for prompt_index {entry.prompt_index} "
                    f"layer {entry.layer}"
                )
            seen.add(key)
            entries.append(entry)
    return entries


def _parse_dump_entry(
    data: dict[str, Any],
    *,
    line_label: str,
    prompts: list[PromptRecord] | None,
    pool: str,
) -> _DumpEntry:
    for field_name in ("prompt_index", "layer", "hidden"):
        if field_name not in data:
            raise ValueError(f"{line_label}: missing {field_name}")
    try:
        prompt_index = int(data["prompt_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{line_label}: prompt_index must be an integer, got {data['prompt_index']!r}") from exc
    if prompt_index < 0:
        raise ValueError(f"{line_label}: prompt_index must be non-negative, got {prompt_index}")
    if prompts is not None and prompt_index >= len(prompts):
        raise ValueError(
            f"{line_label}: prompt_index {prompt_index} is outside the {len(prompts)}-prompt dataset"
        )
    try:
        layer = int(data["layer"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{line_label}: layer must be an integer, got {data['layer']!r}") from exc
    if layer < 0:
        raise ValueError(f"{line_label}: layer must be non-negative, got {layer}")

    hidden = data["hidden"]
    if not isinstance(hidden, list) or not hidden:
        raise ValueError(f"{line_label}: hidden must be a non-empty list")
    if isinstance(hidden[0], list):
        widths = {len(token) for token in hidden if isinstance(token, list)}
        if any(not isinstance(token, list) for token in hidden):
            raise ValueError(f"{line_label}: hidden mixes token vectors and scalars")
        if len(widths) > 1:
            raise ValueError(f"{line_label}: hidden token vectors have inconsistent widths {sorted(widths)}")
        tokens = data.get("tokens")
        if tokens is not None:
            if not isinstance(tokens, list):
                raise ValueError(f"{line_label}: tokens must be a list of strings")
            if len(tokens) != len(hidden):
                raise ValueError(
                    f"{line_label}: tokens has {len(tokens)} entries but hidden has {len(hidden)} rows"
                )
    try:
        vector = _pool_token_vectors(hidden, pool=pool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{line_label}: hidden must contain finite numbers: {exc}") from exc
    for dimension, value in enumerate(vector):
        if not math.isfinite(value):
            raise ValueError(f"{line_label}: hidden[{dimension}] is not a finite number")

    criterion_score: float | None = None
    if "criterion_score" in data:
        try:
            criterion_score = float(data["criterion_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{line_label}: criterion_score must be a number, got {data['criterion_score']!r}"
            ) from exc
        if not math.isfinite(criterion_score):
            raise ValueError(f"{line_label}: criterion_score must be finite")
    if criterion_score is None and prompts is None:
        raise ValueError(
            f"{line_label}: missing criterion_score; add it to the dump line or pass a scored "
            "prompt dataset (--dataset) so records carry criterion scores"
        )
    return _DumpEntry(
        line_label=line_label,
        prompt_index=prompt_index,
        layer=layer,
        vector=vector,
        prompt_id=str(data["prompt_id"]) if "prompt_id" in data else None,
        text=str(data["text"]) if "text" in data else None,
        criterion_score=criterion_score,
        model=str(data["model"]) if "model" in data else None,
    )


def _resolve_prompt_fields(
    entry: _DumpEntry,
    *,
    prompts: list[PromptRecord] | None,
    model_name: str | None,
    dump_path: Path,
) -> _ResolvedPrompt:
    prompt = prompts[entry.prompt_index] if prompts is not None else None
    criterion_score = entry.criterion_score
    if criterion_score is None and prompt is not None:
        criterion_score = prompt.criterion_score
    if criterion_score is None:
        raise ValueError(
            f"{entry.line_label}: no criterion_score on the dump line or in the prompt dataset"
        )
    text = entry.text or (prompt.text if prompt is not None else "")
    prompt_id = entry.prompt_id or (prompt.prompt_id if prompt is not None else "")
    if not prompt_id:
        prompt_id = f"{dump_path.stem}-{entry.prompt_index:03d}"
    model = model_name or entry.model or "hidden-state-dump"
    return _ResolvedPrompt(prompt_id=prompt_id, text=text, criterion_score=criterion_score, model=model)


def _validate_with_records_loader(path: Path) -> None:
    """Re-read the written file through the real records loader as a final check."""
    for _ in _load_validated_records(path):
        pass


def _load_validated_records(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_label = f"{path}:{line_number}"
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_label}: invalid JSON: {exc.msg}") from exc
            yield ActivationRecord.from_dict(data, line_label=line_label)
