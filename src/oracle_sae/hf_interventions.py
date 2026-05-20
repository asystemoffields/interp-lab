from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

from oracle_sae.hf_hooks import register_hidden_ablations
from oracle_sae.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from oracle_sae.hf_records import PromptRecord, load_prompt_records, split_prompt_record_indexes
from oracle_sae.reporting import load_inspection_report

FEATURE_PATTERN = re.compile(r"^L(?P<layer>\d+):D(?P<dimension>\d+)$")
AUTO_TARGET_TOKEN = "auto"
DEFAULT_TARGET_TOKENS = [
    " meters",
    " meter",
    " feet",
    " foot",
    " miles",
    " mile",
    " kilograms",
    " kilogram",
    " grams",
    " gram",
    " centimeters",
    " centimeter",
    " millimeters",
    " millimeter",
    " inches",
    " inch",
    " liters",
    " liter",
    " milliliters",
    " milliliter",
]


def export_hf_intervention_records(
    *,
    model_name: str,
    report_path: str | Path,
    dataset_path: str | Path,
    out_path: str | Path,
    criterion: str,
    top_k: int = 8,
    target_tokens: list[str] | None = None,
    device: str = "cpu",
    max_length: int = 128,
    ablate_value: float = 0.0,
    group_top_k: int | None = None,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to export Hugging Face interventions.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to export Hugging Face interventions.",
    )
    report = load_inspection_report(report_path)
    features = [_parse_hidden_dimension(card.feature_id) for card in report.cards[:top_k]]
    group_feature = _group_feature(features[:group_top_k], group_top_k) if group_top_k else None
    prompts = load_prompt_records(dataset_path)
    if not features:
        raise ValueError(f"{report_path}: no hidden-dimension features found")
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")
    positive_indexes, negative_indexes = split_prompt_record_indexes(prompts)
    if not positive_indexes:
        raise ValueError("HF interventions need at least one positive-scored prompt")
    requested_target_tokens = target_tokens
    score_target_tokens = requested_target_tokens or DEFAULT_TARGET_TOKENS
    token_strategy = target_token_strategy(requested_target_tokens)

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
    target_ids, resolved_target_tokens = resolve_target_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_tokens=score_target_tokens,
        device=runtime_device,
        max_length=max_length,
    )
    if not target_ids:
        raise ValueError("No target token ids resolved for intervention scoring")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for layer, dimension, feature_id in features:
                rows, side_effect = _score_intervention_rows(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    target_ids=target_ids,
                    criterion=criterion,
                    feature_id=feature_id,
                    intervention="ablate",
                    metadata={
                        "behavior_score": "target_token_probability_mass",
                        "negative_prompt_count": len(negative_indexes),
                        "positive_prompt_count": len(positive_indexes),
                        "target_token_strategy": token_strategy,
                        "target_tokens": resolved_target_tokens,
                    },
                    positive_indexes=positive_indexes,
                    negative_indexes=negative_indexes,
                    device=runtime_device,
                    max_length=max_length,
                    ablation=(layer, dimension, ablate_value),
                )
                for row in rows:
                    row["side_effect_score"] = round(side_effect, 8)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            if group_feature is not None:
                feature_id, group_ablations = group_feature
                rows, side_effect = _score_intervention_rows(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    target_ids=target_ids,
                    criterion=criterion,
                    feature_id=feature_id,
                    intervention="ablate",
                    metadata={
                        "behavior_score": "target_token_probability_mass",
                        "group_members": [item[2] for item in group_ablations],
                        "negative_prompt_count": len(negative_indexes),
                        "positive_prompt_count": len(positive_indexes),
                        "target_token_strategy": token_strategy,
                        "target_tokens": resolved_target_tokens,
                    },
                    positive_indexes=positive_indexes,
                    negative_indexes=negative_indexes,
                    device=runtime_device,
                    max_length=max_length,
                    ablations=[
                        (layer, dimension, ablate_value)
                        for layer, dimension, _ in group_ablations
                    ],
                )
                for row in rows:
                    row["side_effect_score"] = round(side_effect, 8)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _score_intervention_rows(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    target_ids: list[int],
    criterion: str,
    feature_id: str,
    intervention: str,
    metadata: dict[str, Any],
    positive_indexes: set[int],
    negative_indexes: set[int],
    device: str,
    max_length: int,
    ablation: tuple[int, int, float] | None = None,
    ablations: list[tuple[int, int, float]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    side_effects = []
    for prompt_index, prompt in enumerate(prompts):
        baseline_score = _score_prompt(
            model,
            tokenizer,
            prompt,
            target_ids,
            device=device,
            max_length=max_length,
        )
        intervention_score = _score_prompt(
            model,
            tokenizer,
            prompt,
            target_ids,
            device=device,
            max_length=max_length,
            ablation=ablation,
            ablations=ablations,
        )
        if prompt_index in positive_indexes:
            rows.append(
                {
                    "schema_version": "interp-lab.intervention_record.v1",
                    "model": model_name,
                    "feature_id": feature_id,
                    "criterion": criterion,
                    "intervention": intervention,
                    "prompt_id": prompt.prompt_id,
                    "baseline_score": baseline_score,
                    "intervention_score": intervention_score,
                    "metadata": dict(metadata),
                }
            )
        elif prompt_index in negative_indexes:
            side_effects.append(abs(intervention_score - baseline_score))
    return rows, _mean(side_effects)


def append_hf_group_activation_record(
    *,
    records_path: str | Path,
    report_path: str | Path,
    out_path: str | Path,
    group_top_k: int,
) -> str:
    report = load_inspection_report(report_path)
    members = [card for card in report.cards[:group_top_k]]
    if not members:
        raise ValueError(f"{report_path}: no report cards found")
    group_id = _group_id([card.feature_id for card in members])
    member_ids = [card.feature_id for card in members]
    member_signs = {
        card.feature_id: _effect_sign(
            card.causal_effects.get(
                "signed_causal_effect",
                card.causal_effects.get("signed_association", 0.0),
            )
        )
        for card in members
    }
    input_path = Path(records_path)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            feature_values = _feature_value_map(row.get("features", []))
            values = [
                member_signs[feature_id] * feature_values[feature_id]
                for feature_id in member_ids
                if feature_id in feature_values
            ]
            activation = sum(values) / len(values) if values else 0.0
            features = list(row.get("features", []))
            features.append(
                {
                    "feature_id": group_id,
                    "activation": activation,
                    "label": f"grouped top-{group_top_k} hidden dimensions",
                    "layer": members[0].layer,
                }
            )
            metadata = dict(row.get("feature_metadata", {}))
            metadata[group_id] = {
                "label": f"grouped top-{group_top_k} hidden dimensions",
                "layer": members[0].layer,
                "source": "hf-hidden-state-group",
                "group_members": member_ids,
            }
            row["features"] = features
            row["feature_metadata"] = metadata
            target.write(json.dumps(row, sort_keys=True) + "\n")
    return group_id


def parse_target_tokens(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    tokens: list[str] = []
    for value in values:
        for chunk in value.split(","):
            token = chunk.strip()
            if token:
                if token.lower() == AUTO_TARGET_TOKEN:
                    tokens.append(AUTO_TARGET_TOKEN)
                    continue
                raw = False
                if token.startswith("raw:"):
                    raw = True
                    token = token[len("raw:") :]
                elif token.startswith("space:"):
                    token = token[len("space:") :]
                if not raw and not token.startswith(" "):
                    token = " " + token
                if token:
                    tokens.append(token)
    return tokens


def target_tokens_are_auto(target_tokens: list[str]) -> bool:
    return len(target_tokens) == 1 and target_tokens[0].lower() == AUTO_TARGET_TOKEN


def target_token_strategy(target_tokens: list[str] | None) -> str:
    if target_tokens is None:
        return "default"
    if target_tokens_are_auto(target_tokens):
        return "auto"
    return "explicit"


def resolve_target_token_ids(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    target_tokens: list[str],
    device: str,
    max_length: int,
    auto_top_k: int = 16,
) -> tuple[list[int], list[str]]:
    if target_tokens_are_auto(target_tokens):
        token_ids = _auto_target_token_ids(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_length=max_length,
            top_k=auto_top_k,
        )
        labels = [_decode_token(tokenizer, token_id) for token_id in token_ids]
        return token_ids, labels
    return _target_token_ids(tokenizer, target_tokens), target_tokens


def build_intervention_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export HF hidden-dimension ablation results as intervention records."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--report", required=True, help="Oracle inspection report JSON.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL used for behavior scoring.")
    parser.add_argument("--out", required=True, help="Output intervention-record JSONL path.")
    parser.add_argument("--criterion", required=True, help="Criterion text stored in intervention rows.")
    parser.add_argument("--top-k", type=int, default=8, help="Top report features to ablate.")
    parser.add_argument(
        "--target-token",
        action="append",
        help="Target token or comma-separated tokens. Defaults to physical unit tokens.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--ablate-value", type=float, default=0.0)
    parser.add_argument(
        "--group-top-k",
        type=int,
        help="Also ablate the top-k report features together as one grouped feature.",
    )
    parser.add_argument(
        "--append-group-records",
        help="Optional activation-record output path with a grouped top-k feature appended.",
    )
    parser.add_argument(
        "--records",
        help="Activation-record JSONL to copy when --append-group-records is set.",
    )
    add_hf_loading_args(parser)
    return parser


def run_interventions_from_args(args: argparse.Namespace) -> Path:
    try:
        loading_options = hf_loading_options_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = export_hf_intervention_records(
        model_name=args.model,
        report_path=args.report,
        dataset_path=args.dataset,
        out_path=args.out,
        criterion=args.criterion,
        top_k=args.top_k,
        target_tokens=parse_target_tokens(args.target_token),
        device=args.device,
        max_length=args.max_length,
        ablate_value=args.ablate_value,
        group_top_k=args.group_top_k,
        **loading_options,
    )
    if args.append_group_records:
        if not args.group_top_k:
            raise SystemExit("--append-group-records requires --group-top-k")
        if not args.records:
            raise SystemExit("--append-group-records requires --records")
        append_hf_group_activation_record(
            records_path=args.records,
            report_path=args.report,
            out_path=args.append_group_records,
            group_top_k=args.group_top_k,
        )
    return path


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _parse_hidden_dimension(feature_id: str) -> tuple[int, int, str]:
    match = FEATURE_PATTERN.match(feature_id)
    if not match:
        raise ValueError(f"{feature_id!r} is not an HF hidden-dimension feature id like L6:D512")
    return int(match.group("layer")), int(match.group("dimension")), feature_id


def _target_token_ids(tokenizer: Any, target_tokens: list[str]) -> list[int]:
    token_ids: set[int] = set()
    for token in target_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[-1]))
    return sorted(token_ids)


def _auto_target_token_ids(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    device: str,
    max_length: int,
    top_k: int,
) -> list[int]:
    positive_indexes, negative_indexes = split_prompt_record_indexes(prompts)
    totals: dict[int, list[float]] = {}
    try:
        torch = importlib.import_module("torch")
        no_grad = torch.no_grad()
    except ImportError:
        no_grad = contextlib.nullcontext()
    with no_grad:
        for index in sorted(positive_indexes | negative_indexes):
            prompt = prompts[index]
            encoded = tokenizer(
                prompt.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, use_cache=False)
            probabilities = outputs.logits[0, -1].softmax(dim=-1)
            count = min(top_k, int(probabilities.shape[-1]))
            values, indices = probabilities.topk(count)
            side = 0 if index in positive_indexes else 2
            for value, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                stats = totals.setdefault(int(token_id), [0.0, 0.0, 0.0, 0.0])
                stats[side] += float(value)
                stats[side + 1] += 1.0
    ranked = []
    for token_id, (positive_sum, positive_count, negative_sum, negative_count) in totals.items():
        positive_mean = positive_sum / positive_count if positive_count else 0.0
        negative_mean = negative_sum / negative_count if negative_count else 0.0
        ranked.append((positive_mean - negative_mean, positive_mean, token_id))
    ranked.sort(reverse=True)
    return [token_id for score, positive_mean, token_id in ranked[:top_k] if score > 0 or positive_mean > 0]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        text = ""
    return text if text else f"<token:{token_id}>"


def _score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    target_ids: list[int],
    *,
    device: str,
    max_length: int,
    ablation: tuple[int, int, float] | None = None,
    ablations: list[tuple[int, int, float]] | None = None,
) -> float:
    encoded = tokenizer(
        prompt.text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hook_handle = None
    if ablation is not None:
        layer, dimension, value = ablation
        ablations = [(layer, dimension, value)]
    if ablations:
        hook_handle = register_hidden_ablations(model, ablations)
    try:
        outputs = model(**encoded, use_cache=False)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    logits = outputs.logits[0, -1]
    probabilities = logits.softmax(dim=-1)
    return round(float(probabilities[target_ids].sum().detach().cpu().item()), 8)


def _register_gpt2_hidden_ablations(model: Any, ablations: list[tuple[int, int, float]]):
    return register_hidden_ablations(model, ablations)


def _group_feature(
    features: list[tuple[int, int, str]],
    group_top_k: int | None,
) -> tuple[str, list[tuple[int, int, str]]] | None:
    if not group_top_k or not features:
        return None
    selected = features[:group_top_k]
    return _group_id([feature_id for _, _, feature_id in selected]), selected


def _group_id(feature_ids: list[str]) -> str:
    return "group:top" + str(len(feature_ids)) + ":" + "+".join(feature_ids)


def _feature_value_map(raw_features: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(raw_features, dict):
        return {str(key): float(value) for key, value in raw_features.items()}
    if isinstance(raw_features, list):
        for item in raw_features:
            if isinstance(item, dict) and "feature_id" in item:
                activation = item.get("activation", item.get("value", 0.0))
                values[str(item["feature_id"])] = float(activation)
    return values


def _effect_sign(value: float) -> float:
    return 1.0 if value >= 0 else -1.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
