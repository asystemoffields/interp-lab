from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interp_lab.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from interp_lab.math_utils import pearson

LayerSelection = list[int] | str | None


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


@dataclass(frozen=True)
class PromptDatasetSummary:
    path: Path
    record_count: int
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class PromptDatasetSplitSummary:
    out_dir: Path
    train_path: Path
    causal_path: Path
    validation_path: Path
    manifest_path: Path
    counts: dict[str, Any]
    advisories: list[str]


@dataclass(frozen=True)
class _PromptGroup:
    key: str
    records: list[PromptRecord]
    score: float


def export_hf_activation_records(
    *,
    model_name: str,
    dataset_path: str | Path,
    out_path: str | Path,
    layers: LayerSelection = None,
    features_per_layer: int = 16,
    pool: str = "last",
    device: str = "cpu",
    max_length: int = 128,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to export Hugging Face activations.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to export Hugging Face activations.",
    )
    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")

    tokenizer, model, runtime_device = load_hf_text_model(
        transformers=transformers,
        torch=torch,
        model_name=model_name,
        device=device,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    layer_vectors: dict[int, list[list[float]]] = {}
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(runtime_device) for key, value in encoded.items()}
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
    with file_path.open("r", encoding="utf-8-sig") as handle:
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


def build_prompt_dataset(
    *,
    out_path: str | Path,
    positive_paths: list[str | Path] | None = None,
    negative_paths: list[str | Path] | None = None,
    positive_prompts: list[str] | None = None,
    negative_prompts: list[str] | None = None,
    split: str = "paragraphs",
    delimiter: str | None = None,
    positive_score: float = 1.0,
    negative_score: float = 0.0,
    id_prefix: str = "prompt",
) -> PromptDatasetSummary:
    records = build_prompt_records(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        split=split,
        delimiter=delimiter,
        positive_score=positive_score,
        negative_score=negative_score,
        id_prefix=id_prefix,
    )
    if not records:
        raise ValueError("Add at least one positive or negative prompt")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": record.prompt_id,
                        "text": record.text,
                        "criterion_score": record.criterion_score,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    positive_count = sum(
        1 for record in records if record.prompt_id.startswith(f"{id_prefix}-positive-")
    )
    negative_count = sum(
        1 for record in records if record.prompt_id.startswith(f"{id_prefix}-negative-")
    )
    return PromptDatasetSummary(
        path=path,
        record_count=len(records),
        positive_count=positive_count,
        negative_count=negative_count,
    )


def prepare_sae_prompt_datasets(
    *,
    dataset_path: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.7,
    causal_ratio: float = 0.15,
    validation_ratio: float = 0.15,
    seed: str = "0",
    latent_dim: int | None = None,
    max_length: int | None = None,
    min_rows_per_latent: float = 4.0,
) -> PromptDatasetSplitSummary:
    records = load_prompt_records(dataset_path)
    if not records:
        raise ValueError(f"{dataset_path}: no prompt records found")
    _validate_split_ratios(train_ratio, causal_ratio, validation_ratio)

    groups, duplicate_count, conflicting_duplicates = _prompt_groups_by_text(records)
    labels = [_label_for_score(group.score, groups) for group in groups]
    positive_groups = [group for group, label in zip(groups, labels) if label == "positive"]
    negative_groups = [group for group, label in zip(groups, labels) if label == "negative"]
    advisories: list[str] = []
    if duplicate_count:
        advisories.append("Duplicate prompt text was kept in one split to avoid train/eval leakage.")
    if conflicting_duplicates:
        advisories.append("Some duplicate prompt text has conflicting scores; review the source dataset.")
    if not positive_groups or not negative_groups:
        advisories.append("Prompt pack has only one score side; causal and held-out validation will be weaker.")

    split_groups: dict[str, list[_PromptGroup]] = {"train": [], "causal": [], "validation": []}
    for label, label_groups in (("positive", positive_groups), ("negative", negative_groups)):
        ordered = _stable_prompt_group_order(label_groups, seed=f"{seed}:{label}")
        allocation = _allocate_split_counts(
            len(ordered),
            train_ratio=train_ratio,
            causal_ratio=causal_ratio,
            validation_ratio=validation_ratio,
        )
        start = 0
        for split_name in ("train", "causal", "validation"):
            count = allocation[split_name]
            split_groups[split_name].extend(ordered[start : start + count])
            start += count

    split_records = {
        split_name: _flatten_groups(_stable_prompt_group_order(groups_for_split, seed=f"{seed}:{split_name}"))
        for split_name, groups_for_split in split_groups.items()
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    causal_path = out / "causal.jsonl"
    validation_path = out / "validation.jsonl"
    manifest_path = out / "manifest.json"
    _write_prompt_records_jsonl(train_path, split_records["train"])
    _write_prompt_records_jsonl(causal_path, split_records["causal"])
    _write_prompt_records_jsonl(validation_path, split_records["validation"])

    counts = _prompt_split_counts(
        records_by_split=split_records,
        all_groups=groups,
        latent_dim=latent_dim,
        max_length=max_length,
    )
    for split_name in ("causal", "validation"):
        split_counts = counts["splits"][split_name]
        if split_counts["record_count"] and (
            split_counts["positive_count"] == 0 or split_counts["negative_count"] == 0
        ):
            advisories.append(f"{split_name} split lacks both positive and negative prompts.")
    estimated_rows_per_latent = counts["splits"]["train"].get("estimated_rows_per_latent")
    if (
        estimated_rows_per_latent is not None
        and estimated_rows_per_latent < min_rows_per_latent
    ):
        advisories.append(
            "Estimated training rows per latent are low; add prompts, increase max_length, or reduce latent_dim."
        )
    if counts["total"]["record_count"] < 30:
        advisories.append("Prompt pack is small; treat SAE labels as early hypotheses until held-out validation passes.")

    manifest = {
        "format": "interp-lab.sae_prompt_pack.v1",
        "source_dataset": str(dataset_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "causal": causal_ratio,
            "validation": validation_ratio,
        },
        "outputs": {
            "train": str(train_path),
            "causal": str(causal_path),
            "validation": str(validation_path),
        },
        "counts": counts,
        "advisories": advisories,
        "agent_next_actions": [
            "Use train.jsonl as train-sae --dataset.",
            "Use causal.jsonl as train-sae --causal-dataset when writing --causal-out.",
            "Use validation.jsonl for held-out path validation or repeated causal checks.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PromptDatasetSplitSummary(
        out_dir=out,
        train_path=train_path,
        causal_path=causal_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        counts=counts,
        advisories=advisories,
    )


def build_prompt_records(
    *,
    positive_paths: list[str | Path] | None = None,
    negative_paths: list[str | Path] | None = None,
    positive_prompts: list[str] | None = None,
    negative_prompts: list[str] | None = None,
    split: str = "paragraphs",
    delimiter: str | None = None,
    positive_score: float = 1.0,
    negative_score: float = 0.0,
    id_prefix: str = "prompt",
) -> list[PromptRecord]:
    if split not in {"lines", "paragraphs"}:
        raise ValueError("--split must be 'lines' or 'paragraphs'")
    records: list[PromptRecord] = []
    next_index = 1
    next_index = _extend_prompt_records(
        records,
        texts=list(positive_prompts or []),
        score=positive_score,
        label="positive",
        id_prefix=id_prefix,
        start_index=next_index,
    )
    for path in positive_paths or []:
        next_index = _extend_prompt_records(
            records,
            texts=_read_prompt_texts(path, split=split, delimiter=delimiter),
            score=positive_score,
            label="positive",
            id_prefix=id_prefix,
            start_index=next_index,
        )
    next_index = _extend_prompt_records(
        records,
        texts=list(negative_prompts or []),
        score=negative_score,
        label="negative",
        id_prefix=id_prefix,
        start_index=next_index,
    )
    for path in negative_paths or []:
        next_index = _extend_prompt_records(
            records,
            texts=_read_prompt_texts(path, split=split, delimiter=delimiter),
            score=negative_score,
            label="negative",
            id_prefix=id_prefix,
            start_index=next_index,
        )
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


def parse_layers(value: str | None) -> LayerSelection:
    if value is None or not value.strip():
        return None
    if value.strip().lower() in {"all", "*"}:
        return "all"
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
    parser.add_argument(
        "--layers",
        help="Hidden-state layers, e.g. all, 0,4,8, or 1-3. Defaults to final layer.",
    )
    parser.add_argument("--features-per-layer", type=int, default=16)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    add_hf_loading_args(parser)
    return parser


def build_prompt_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a prompt JSONL from user-written prompts.")
    parser.add_argument(
        "--positive",
        action="append",
        default=[],
        help="Text file of positive prompts. Repeatable.",
    )
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        help="Text file of negative prompts. Repeatable.",
    )
    parser.add_argument(
        "--positive-prompt",
        action="append",
        default=[],
        help="Inline positive prompt. Repeatable.",
    )
    parser.add_argument(
        "--negative-prompt",
        action="append",
        default=[],
        help="Inline negative prompt. Repeatable.",
    )
    parser.add_argument("--out", required=True, help="Output prompt JSONL path.")
    parser.add_argument(
        "--split",
        choices=["lines", "paragraphs"],
        default="paragraphs",
        help="How to split prompt files when --delimiter is omitted.",
    )
    parser.add_argument("--delimiter", help="Literal delimiter between prompts in prompt files.")
    parser.add_argument("--positive-score", type=float, default=1.0)
    parser.add_argument("--negative-score", type=float, default=0.0)
    parser.add_argument("--id-prefix", default="prompt")
    return parser


def build_prepare_sae_prompt_datasets_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split scored prompts into train, causal, and held-out SAE datasets."
    )
    parser.add_argument("--dataset", required=True, help="Scored prompt JSONL with text and criterion_score.")
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        required=True,
        help="Output directory for train/causal/validation JSONL (--out is an accepted alias).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--causal-ratio", type=float, default=0.15)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", default="0", help="Deterministic split seed.")
    parser.add_argument("--latent-dim", type=int, help="Optional SAE latent count for data-volume advisories.")
    parser.add_argument("--max-length", type=int, help="Optional max prompt length for token-row estimates.")
    parser.add_argument("--min-rows-per-latent", type=float, default=4.0)
    return parser


def run_export_from_args(args: argparse.Namespace) -> Path:
    try:
        layers = parse_layers(args.layers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        loading_options = hf_loading_options_from_args(args)
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
        **loading_options,
    )


def run_build_prompt_dataset_from_args(args: argparse.Namespace) -> PromptDatasetSummary:
    try:
        return build_prompt_dataset(
            out_path=args.out,
            positive_paths=args.positive,
            negative_paths=args.negative,
            positive_prompts=args.positive_prompt,
            negative_prompts=args.negative_prompt,
            split=args.split,
            delimiter=args.delimiter,
            positive_score=args.positive_score,
            negative_score=args.negative_score,
            id_prefix=args.id_prefix,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def run_prepare_sae_prompt_datasets_from_args(args: argparse.Namespace) -> PromptDatasetSplitSummary:
    try:
        return prepare_sae_prompt_datasets(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            causal_ratio=args.causal_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            latent_dim=args.latent_dim,
            max_length=args.max_length,
            min_rows_per_latent=args.min_rows_per_latent,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _resolve_layers(layers: LayerSelection, hidden_state_count: int) -> list[int]:
    if layers is None:
        return [hidden_state_count - 1]
    if layers == "all":
        return list(range(hidden_state_count))
    if isinstance(layers, str):
        raise ValueError("layers must be 'all' or a list of layer indexes")
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


def _read_prompt_texts(path: str | Path, *, split: str, delimiter: str | None) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    if delimiter:
        chunks = raw.split(delimiter)
    elif split == "lines":
        chunks = raw.splitlines()
    else:
        chunks = re.split(r"\n[ \t]*\n+", raw.replace("\r\n", "\n").replace("\r", "\n"))
    return [_clean_prompt_text(chunk) for chunk in chunks if _clean_prompt_text(chunk)]


def _extend_prompt_records(
    records: list[PromptRecord],
    *,
    texts: list[str],
    score: float,
    label: str,
    id_prefix: str,
    start_index: int,
) -> int:
    next_index = start_index
    for text in texts:
        cleaned = _clean_prompt_text(text)
        if not cleaned:
            continue
        records.append(
            PromptRecord(
                prompt_id=f"{id_prefix}-{label}-{next_index:03d}",
                text=cleaned,
                criterion_score=score,
            )
        )
        next_index += 1
    return next_index


def _clean_prompt_text(value: str) -> str:
    return value.strip("\ufeff \t\r\n")


def _validate_split_ratios(train_ratio: float, causal_ratio: float, validation_ratio: float) -> None:
    ratios = [train_ratio, causal_ratio, validation_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    if sum(ratios) <= 0:
        raise ValueError("At least one split ratio must be positive")


def _prompt_groups_by_text(records: list[PromptRecord]) -> tuple[list[_PromptGroup], int, bool]:
    grouped: dict[str, list[PromptRecord]] = {}
    for record in records:
        key = _canonical_prompt_text(record.text)
        grouped.setdefault(key, []).append(record)
    groups: list[_PromptGroup] = []
    duplicate_count = 0
    conflicting_duplicates = False
    for key, group_records in grouped.items():
        if len(group_records) > 1:
            duplicate_count += len(group_records) - 1
            if len({record.criterion_score for record in group_records}) > 1:
                conflicting_duplicates = True
        score = sum(record.criterion_score for record in group_records) / len(group_records)
        groups.append(_PromptGroup(key=key, records=group_records, score=score))
    return groups, duplicate_count, conflicting_duplicates


def _canonical_prompt_text(text: str) -> str:
    return " ".join(text.lower().split())


def _label_for_score(score: float, groups: list[_PromptGroup]) -> str:
    scores = [group.score for group in groups]
    if min(scores) == max(scores):
        return "positive" if score > 0 else "negative"
    threshold = sum(scores) / len(scores)
    return "positive" if score >= threshold else "negative"


def _stable_prompt_group_order(groups: list[_PromptGroup], *, seed: str) -> list[_PromptGroup]:
    return sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group.key}".encode("utf-8")).hexdigest(),
    )


def _allocate_split_counts(
    count: int,
    *,
    train_ratio: float,
    causal_ratio: float,
    validation_ratio: float,
) -> dict[str, int]:
    ratios = {
        "train": train_ratio,
        "causal": causal_ratio,
        "validation": validation_ratio,
    }
    total_ratio = sum(ratios.values())
    if count <= 0:
        return {name: 0 for name in ratios}
    raw = {name: count * ratio / total_ratio for name, ratio in ratios.items()}
    allocated = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = count - sum(allocated.values())
    for name, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0] == "train"),
        reverse=True,
    ):
        if remaining <= 0:
            break
        allocated[name] += 1
        remaining -= 1
    positive_splits = [name for name, ratio in ratios.items() if ratio > 0]
    if count >= len(positive_splits):
        for name in positive_splits:
            if allocated[name] == 0:
                donor = max(positive_splits, key=lambda split_name: allocated[split_name])
                if allocated[donor] > 1:
                    allocated[donor] -= 1
                    allocated[name] = 1
    return allocated


def _flatten_groups(groups: list[_PromptGroup]) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for group in groups:
        records.extend(group.records)
    return records


def _write_prompt_records_jsonl(path: Path, records: list[PromptRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            prompt_id = record.prompt_id or f"{path.stem}-{index:03d}"
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "text": record.text,
                        "criterion_score": record.criterion_score,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _prompt_split_counts(
    *,
    records_by_split: dict[str, list[PromptRecord]],
    all_groups: list[_PromptGroup],
    latent_dim: int | None,
    max_length: int | None,
) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "total": _count_prompt_records(_flatten_groups(all_groups), all_groups=all_groups),
        "splits": {},
    }
    for split_name, records in records_by_split.items():
        split_counts = _count_prompt_records(records, all_groups=all_groups)
        estimated_token_rows = sum(_estimated_token_rows(record.text, max_length=max_length) for record in records)
        split_counts["estimated_token_rows"] = estimated_token_rows
        if latent_dim:
            split_counts["estimated_rows_per_latent"] = round(estimated_token_rows / latent_dim, 6)
        counts["splits"][split_name] = split_counts
    return counts


def _count_prompt_records(records: list[PromptRecord], *, all_groups: list[_PromptGroup]) -> dict[str, int]:
    positive_count = 0
    negative_count = 0
    for record in records:
        label = _label_for_score(record.criterion_score, all_groups)
        if label == "positive":
            positive_count += 1
        else:
            negative_count += 1
    return {
        "record_count": len(records),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _estimated_token_rows(text: str, *, max_length: int | None) -> int:
    estimate = max(1, len(re.findall(r"\S+", text)))
    if max_length is not None:
        return min(estimate, max_length)
    return estimate
