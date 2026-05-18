from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

from oracle_sae.hf_interventions import DEFAULT_TARGET_TOKENS, parse_target_tokens
from oracle_sae.hf_records import PromptRecord, load_prompt_records, split_prompt_record_indexes


def export_hf_contrast_feature(
    *,
    model_name: str,
    dataset_path: str | Path,
    records_out: str | Path,
    interventions_out: str | Path | None,
    criterion: str,
    layer: int | None = None,
    pool: str = "last",
    device: str = "cpu",
    max_length: int = 128,
    target_tokens: list[str] | None = None,
    steer_strength: float = 3.0,
    strength_sweep: list[float] | None = None,
) -> tuple[Path, Path | None, str]:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to export Hugging Face contrast features.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to export Hugging Face contrast features.",
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

    vectors: list[list[float]] = []
    resolved_layer: int | None = None
    with torch.no_grad():
        for prompt in prompts:
            encoded = _encode(tokenizer, prompt.text, device=device, max_length=max_length)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            resolved_layer = _resolve_layer(layer, len(hidden_states))
            vector = _pool_hidden_state(
                hidden_states[resolved_layer],
                encoded.get("attention_mask"),
                pool=pool,
            )
            vectors.append(vector)
    assert resolved_layer is not None

    direction = _contrast_direction(vectors, [prompt.criterion_score for prompt in prompts])
    feature_id = f"direction:contrast:L{resolved_layer}:{_slug(criterion)}"
    records_path = _write_contrast_records(
        model_name=model_name,
        prompts=prompts,
        vectors=vectors,
        direction=direction,
        feature_id=feature_id,
        layer=resolved_layer,
        out_path=records_out,
    )
    intervention_path = None
    if interventions_out is not None:
        target_ids = _target_token_ids(tokenizer, target_tokens or DEFAULT_TARGET_TOKENS)
        if not target_ids:
            raise ValueError("No target token ids resolved for intervention scoring")
        intervention_path = _write_steering_interventions(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            direction=direction,
            feature_id=feature_id,
            layer=resolved_layer,
            criterion=criterion,
            target_ids=target_ids,
            target_tokens=target_tokens or DEFAULT_TARGET_TOKENS,
            steer_strength=steer_strength,
            strength_sweep=strength_sweep,
            device=device,
            max_length=max_length,
            out_path=interventions_out,
        )
    return records_path, intervention_path, feature_id


def build_contrast_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a contrast-direction feature and optional steering interventions."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--records-out", required=True, help="Output activation-record JSONL path.")
    parser.add_argument("--interventions-out", help="Optional output intervention-record JSONL path.")
    parser.add_argument("--criterion", required=True, help="Criterion text for labels and interventions.")
    parser.add_argument("--layer", type=int, help="Hidden-state layer. Defaults to final hidden state.")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-token", action="append")
    parser.add_argument("--steer-strength", type=float, default=3.0)
    parser.add_argument(
        "--strength-sweep",
        help="Comma-separated signed steering strengths to evaluate; writes the most specific setting.",
    )
    return parser


def run_contrast_from_args(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    return export_hf_contrast_feature(
        model_name=args.model,
        dataset_path=args.dataset,
        records_out=args.records_out,
        interventions_out=args.interventions_out,
        criterion=args.criterion,
        layer=args.layer,
        pool=args.pool,
        device=args.device,
        max_length=args.max_length,
        target_tokens=parse_target_tokens(args.target_token),
        steer_strength=args.steer_strength,
        strength_sweep=parse_strength_sweep(args.strength_sweep),
    )


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def parse_strength_sweep(value: str | None) -> list[float] | None:
    if not value:
        return None
    strengths: list[float] = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        strengths.append(float(stripped))
    if not strengths:
        raise ValueError("--strength-sweep did not contain any numeric strengths")
    return strengths


def _encode(tokenizer: Any, text: str, *, device: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _resolve_layer(layer: int | None, hidden_state_count: int) -> int:
    resolved = hidden_state_count - 1 if layer is None else layer
    if resolved < 0 or resolved >= hidden_state_count:
        raise ValueError(f"Layer {resolved} is outside 0..{hidden_state_count - 1}")
    return resolved


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


def _contrast_direction(vectors: list[list[float]], scores: list[float]) -> list[float]:
    threshold = sum(scores) / len(scores)
    positives = [vector for vector, score in zip(vectors, scores) if score >= threshold]
    negatives = [vector for vector, score in zip(vectors, scores) if score < threshold]
    if not positives or not negatives:
        raise ValueError("Contrast feature needs both positive and negative scored prompts")
    direction = [
        _mean([vector[index] for vector in positives]) - _mean([vector[index] for vector in negatives])
        for index in range(len(vectors[0]))
    ]
    norm = sum(value * value for value in direction) ** 0.5
    if norm == 0:
        raise ValueError("Contrast direction has zero norm")
    return [value / norm for value in direction]


def _write_contrast_records(
    *,
    model_name: str,
    prompts: list[PromptRecord],
    vectors: list[list[float]],
    direction: list[float],
    feature_id: str,
    layer: int,
    out_path: str | Path,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt, vector in zip(prompts, vectors):
            activation = _dot(vector, direction)
            row = {
                "model": model_name,
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "criterion_score": prompt.criterion_score,
                "features": [
                    {
                        "feature_id": feature_id,
                        "activation": activation,
                        "label": "contrast direction for criterion",
                        "layer": layer,
                    }
                ],
                "feature_metadata": {
                    feature_id: {
                        "label": "contrast direction for criterion",
                        "layer": layer,
                        "source": "hf-contrast-direction",
                        "direction_norm": 1.0,
                    }
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _write_steering_interventions(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    direction: list[float],
    feature_id: str,
    layer: int,
    criterion: str,
    target_ids: list[int],
    target_tokens: list[str],
    steer_strength: float,
    strength_sweep: list[float] | None,
    device: str,
    max_length: int,
    out_path: str | Path,
) -> Path:
    torch = importlib.import_module("torch")
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
    strengths = strength_sweep or [steer_strength]
    rows_by_strength: dict[float, list[dict[str, Any]]] = {strength: [] for strength in strengths}
    side_effects_by_strength: dict[float, list[float]] = {strength: [] for strength in strengths}
    positive_indexes, negative_indexes = split_prompt_record_indexes(prompts)
    if not positive_indexes:
        raise ValueError("Contrast interventions need at least one positive-scored prompt")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for prompt_index, prompt in enumerate(prompts):
            baseline_score = _score_prompt(
                model,
                tokenizer,
                prompt,
                target_ids,
                direction=None,
                layer=layer,
                strength=0.0,
                device=device,
                max_length=max_length,
            )
            for strength in strengths:
                intervention_score = _score_prompt(
                    model,
                    tokenizer,
                    prompt,
                    target_ids,
                    direction=direction_tensor,
                    layer=layer,
                    strength=strength,
                    device=device,
                    max_length=max_length,
                )
                if prompt_index in positive_indexes:
                    rows_by_strength[strength].append(
                        {
                            "model": model_name,
                            "feature_id": feature_id,
                            "criterion": criterion,
                            "intervention": "steer",
                            "prompt_id": prompt.prompt_id,
                            "baseline_score": baseline_score,
                            "intervention_score": intervention_score,
                            "metadata": {
                                "behavior_score": "target_token_probability_mass",
                                "negative_prompt_count": len(negative_indexes),
                                "positive_prompt_count": len(positive_indexes),
                                "steer_strength": strength,
                                "target_tokens": target_tokens,
                            },
                        }
                    )
                else:
                    side_effects_by_strength[strength].append(abs(intervention_score - baseline_score))
    selected_strength, sweep_summary = _select_best_strength(rows_by_strength, side_effects_by_strength)
    selected_side_effect = _mean(side_effects_by_strength.get(selected_strength, []))
    for row in rows_by_strength[selected_strength]:
        row["side_effect_score"] = round(selected_side_effect, 8)
        if strength_sweep:
            row["metadata"]["selected_strength"] = selected_strength
            row["metadata"]["strength_sweep"] = sweep_summary
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_by_strength[selected_strength]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _select_best_strength(
    rows_by_strength: dict[float, list[dict[str, Any]]],
    side_effects_by_strength: dict[float, list[float]] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    if not rows_by_strength:
        raise ValueError("No steering strengths were evaluated")
    summary = []
    side_effects_by_strength = side_effects_by_strength or {}
    for strength, rows in rows_by_strength.items():
        effects = [row["intervention_score"] - row["baseline_score"] for row in rows]
        side_effect = _mean(side_effects_by_strength.get(strength, []))
        mean_effect = _mean(effects)
        summary.append(
            {
                "steer_strength": strength,
                "mean_directed_effect": round(mean_effect, 8),
                "mean_side_effect": round(side_effect, 8),
                "specificity": round(mean_effect - side_effect, 8),
            }
        )
    selected = max(
        summary,
        key=lambda item: (item["specificity"], item["mean_directed_effect"]),
    )["steer_strength"]
    return float(selected), summary


def _score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    target_ids: list[int],
    *,
    direction: Any | None,
    layer: int,
    strength: float,
    device: str,
    max_length: int,
) -> float:
    encoded = _encode(tokenizer, prompt.text, device=device, max_length=max_length)
    hook_handle = None
    if direction is not None:
        hook_handle = _register_gpt2_steering(model, layer, direction, strength)
    try:
        outputs = model(**encoded, use_cache=False)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    logits = outputs.logits[0, -1]
    probabilities = logits.softmax(dim=-1)
    return round(float(probabilities[target_ids].sum().detach().cpu().item()), 8)


def _register_gpt2_steering(model: Any, layer: int, direction: Any, strength: float):
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    final_layer_norm = getattr(transformer, "ln_f", None)
    if blocks is None or final_layer_norm is None:
        raise RuntimeError("HF contrast steering currently supports GPT-2-style models with transformer.h and ln_f")
    if layer == len(blocks):
        def final_hook(_module, _inputs, output):
            hidden = output.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + strength * direction.to(hidden.dtype)
            return hidden

        return final_layer_norm.register_forward_hook(final_hook)
    block_index = layer - 1
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(f"Layer {layer} cannot be steered through transformer.h")

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + strength * direction.to(hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + strength * direction.to(hidden.dtype)
        return hidden

    return blocks[block_index].register_forward_hook(hook)


def _target_token_ids(tokenizer: Any, target_tokens: list[str]) -> list[int]:
    token_ids: set[int] = set()
    for token in target_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[-1]))
    return sorted(token_ids)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "criterion"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
