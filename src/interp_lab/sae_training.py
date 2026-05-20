from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interp_lab.adapters.records import ActivationRecord
from interp_lab.hf_contrast import _register_gpt2_steering, _select_best_strength, parse_strength_sweep
from interp_lab.hf_interventions import (
    DEFAULT_TARGET_TOKENS,
    parse_target_tokens,
    resolve_target_token_ids,
    target_token_strategy,
)
from interp_lab.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from interp_lab.hf_records import PromptRecord, _pool_hidden_state, load_prompt_records
from interp_lab.math_utils import mean, norm, pearson

TRAINING_PRESETS = {
    "minimal": {
        "expansion_factor": 1.0,
        "method": "auto",
        "sparsity": "relu-l1",
        "top_k": None,
        "jump_threshold": 0.0,
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "l1": 1e-3,
        "validation_fraction": 0.1,
        "dead_latent_threshold": 0.0,
        "token_mode": None,
        "max_length": 128,
        "top_k_features": 0,
        "decoder_signature_size": 128,
        "causal_top_k": 4,
    },
    "production": {
        "expansion_factor": 4.0,
        "method": "auto",
        "sparsity": "topk",
        "top_k": 32,
        "jump_threshold": 0.0,
        "epochs": 500,
        "batch_size": 256,
        "lr": 1e-3,
        "l1": 1e-3,
        "validation_fraction": 0.1,
        "dead_latent_threshold": 1e-6,
        "token_mode": "all",
        "max_length": 256,
        "top_k_features": 0,
        "decoder_signature_size": 256,
        "causal_top_k": 16,
    },
    "custom": {
        "expansion_factor": 2.0,
        "method": "auto",
        "sparsity": "relu-l1",
        "top_k": None,
        "jump_threshold": 0.0,
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-3,
        "l1": 1e-3,
        "validation_fraction": 0.1,
        "dead_latent_threshold": 0.0,
        "token_mode": None,
        "max_length": 128,
        "top_k_features": 0,
        "decoder_signature_size": 128,
        "causal_top_k": 8,
    },
}


@dataclass(frozen=True)
class MatrixRow:
    model: str
    prompt_id: str
    text: str
    criterion_score: float
    token_index: int | None = None
    token_text: str = ""


@dataclass(frozen=True)
class ActivationMatrix:
    model: str
    rows: list[MatrixRow]
    source_feature_ids: list[str]
    values: list[list[float]]
    layer: int | None = None
    source: dict[str, Any] | None = None


def train_sae_from_records(
    *,
    records_path: str | Path,
    out_path: str | Path,
    records_out: str | Path | None = None,
    model_name: str | None = None,
    latent_dim: int | None = None,
    expansion_factor: float = 2.0,
    method: str = "auto",
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
    sparsity: str = "relu-l1",
    top_k: int | None = None,
    jump_threshold: float = 0.0,
    validation_fraction: float = 0.1,
    dead_latent_threshold: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    max_records: int | None = None,
    top_k_features: int | None = None,
    decoder_signature_size: int = 128,
) -> tuple[Path, Path | None]:
    matrix = load_activation_matrix_from_records(
        records_path,
        model_name=model_name,
        max_records=max_records,
        seed=seed,
    )
    artifact = train_sae(
        matrix,
        latent_dim=_resolve_latent_dim(latent_dim, expansion_factor, len(matrix.source_feature_ids)),
        method=method,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        sparsity=sparsity,
        top_k=top_k,
        jump_threshold=jump_threshold,
        validation_fraction=validation_fraction,
        dead_latent_threshold=dead_latent_threshold,
        seed=seed,
        device=device,
    )
    artifact_path = write_sae_artifact(artifact, out_path)
    activation_records_path = None
    if records_out is not None:
        activation_records_path = write_sae_activation_records(
            matrix,
            artifact,
            records_out,
            top_k_features=top_k_features,
            decoder_signature_size=decoder_signature_size,
        )
    return artifact_path, activation_records_path


def train_sae_from_hf(
    *,
    model_name: str,
    dataset_path: str | Path,
    causal_dataset_path: str | Path | None = None,
    out_path: str | Path,
    records_out: str | Path | None = None,
    layer: int | None = None,
    pool: str = "last",
    token_mode: str | None = None,
    latent_dim: int | None = None,
    expansion_factor: float = 2.0,
    method: str = "auto",
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
    sparsity: str = "relu-l1",
    top_k: int | None = None,
    jump_threshold: float = 0.0,
    validation_fraction: float = 0.1,
    dead_latent_threshold: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    max_length: int = 128,
    max_records: int | None = None,
    top_k_features: int | None = None,
    decoder_signature_size: int = 128,
    causal_out: str | Path | None = None,
    criterion: str | None = None,
    causal_top_k: int = 8,
    causal_strength_sweep: list[float] | None = None,
    target_tokens: list[str] | None = None,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    matrix = load_activation_matrix_from_hf(
        model_name=model_name,
        dataset_path=dataset_path,
        layer=layer,
        pool=pool,
        token_mode=token_mode,
        device=device,
        max_length=max_length,
        max_records=max_records,
        seed=seed,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    artifact = train_sae(
        matrix,
        latent_dim=_resolve_latent_dim(latent_dim, expansion_factor, len(matrix.source_feature_ids)),
        method=method,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        sparsity=sparsity,
        top_k=top_k,
        jump_threshold=jump_threshold,
        validation_fraction=validation_fraction,
        dead_latent_threshold=dead_latent_threshold,
        seed=seed,
        device=device,
    )
    artifact_path = write_sae_artifact(artifact, out_path)
    activation_records_path = None
    causal_latent_indexes = (
        {index for index, _score in _rank_latents_by_association(matrix, artifact)[:causal_top_k]}
        if causal_out is not None
        else set()
    )
    if records_out is not None:
        activation_records_path = write_sae_activation_records(
            matrix,
            artifact,
            records_out,
            top_k_features=top_k_features,
            decoder_signature_size=decoder_signature_size,
            force_latent_indexes=causal_latent_indexes,
        )
    if causal_out is not None:
        if criterion is None:
            raise ValueError("criterion is required when causal_out is set")
        export_hf_sae_interventions(
            model_name=model_name,
            dataset_path=causal_dataset_path or dataset_path,
            artifact=artifact,
            matrix=matrix,
            criterion=criterion,
            out_path=causal_out,
            top_k=causal_top_k,
            strength_sweep=causal_strength_sweep or [3.0, 10.0, 30.0],
            target_tokens=target_tokens,
            device=device,
            max_length=max_length,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    return artifact_path, activation_records_path


def load_activation_matrix_from_records(
    path: str | Path,
    *,
    model_name: str | None = None,
    max_records: int | None = None,
    seed: int = 0,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> ActivationMatrix:
    records: list[ActivationRecord] = []
    file_path = Path(path)
    seen = 0
    rng = random.Random(seed)
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
            record = ActivationRecord.from_dict(data, line_label=line_label)
            if model_name is None or record.model == model_name:
                seen += 1
                if max_records is None or len(records) < max_records:
                    records.append(record)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < max_records:
                        records[replacement] = record
    if not records:
        raise ValueError(f"{file_path}: no activation records matched")

    source_feature_ids = _ordered_feature_ids(records)
    values = [
        [record.features.get(feature_id, 0.0) for feature_id in source_feature_ids]
        for record in records
    ]
    rows = [
        MatrixRow(
            model=record.model,
            prompt_id=record.prompt_id,
            text=record.text,
            criterion_score=record.criterion_score,
        )
        for record in records
    ]
    layer = _common_layer(records, source_feature_ids)
    return ActivationMatrix(
        model=model_name or records[0].model,
        rows=rows,
        source_feature_ids=source_feature_ids,
        values=values,
        layer=layer,
        source={"kind": "activation-records", "path": str(file_path)},
    )


def load_activation_matrix_from_hf(
    *,
    model_name: str,
    dataset_path: str | Path,
    layer: int | None = None,
    pool: str = "last",
    token_mode: str | None = None,
    device: str = "cpu",
    max_length: int = 128,
    max_records: int | None = None,
    seed: int = 0,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> ActivationMatrix:
    torch = _optional_import("torch", "Install `interp-lab[hf]` or `interp-lab[train]` to train from HF activations.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to train directly from Hugging Face models.",
    )
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

    values: list[list[float]] = []
    rows: list[MatrixRow] = []
    resolved_layer: int | None = None
    rng = random.Random(seed)
    seen_rows = 0
    with torch.no_grad():
        for prompt in _iter_prompt_records(dataset_path):
            encoded = tokenizer(
                prompt.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(runtime_device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            resolved_layer = _resolve_layer(layer, len(hidden_states))
            prompt_vectors = _vectors_from_hidden_state(
                hidden_states[resolved_layer],
                encoded,
                tokenizer,
                prompt,
                token_mode=token_mode or pool,
            )
            for row, vector in prompt_vectors:
                seen_rows += 1
                if max_records is None or len(values) < max_records:
                    rows.append(row)
                    values.append(vector)
                else:
                    replacement = rng.randrange(seen_rows)
                    if replacement < max_records:
                        rows[replacement] = row
                        values[replacement] = vector
    if not values:
        raise ValueError(f"{dataset_path}: no prompt activation rows found")
    if resolved_layer is None:
        raise ValueError(f"{dataset_path}: no prompt records found")
    source_feature_ids = [f"H{resolved_layer}:D{index}" for index in range(len(values[0]))]
    return ActivationMatrix(
        model=model_name,
        rows=rows,
        source_feature_ids=source_feature_ids,
        values=values,
        layer=resolved_layer,
        source={
            "kind": "hf-hidden-state",
            "model": model_name,
            "dataset": str(dataset_path),
            "token_mode": token_mode or pool,
        },
    )


def train_sae(
    matrix: ActivationMatrix,
    *,
    latent_dim: int,
    method: str = "auto",
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
    sparsity: str = "relu-l1",
    top_k: int | None = None,
    jump_threshold: float = 0.0,
    validation_fraction: float = 0.1,
    dead_latent_threshold: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if not matrix.values or not matrix.source_feature_ids:
        raise ValueError("SAE training needs a non-empty activation matrix")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    method_key = method.lower()
    if method_key not in {"auto", "torch", "fallback"}:
        raise ValueError("method must be one of: auto, torch, fallback")
    if method_key in {"auto", "torch"}:
        try:
            return _train_torch_sae(
                matrix,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l1_coefficient=l1_coefficient,
                sparsity=sparsity,
                top_k=top_k,
                jump_threshold=jump_threshold,
                validation_fraction=validation_fraction,
                dead_latent_threshold=dead_latent_threshold,
                seed=seed,
                device=device,
            )
        except ImportError:
            if method_key == "torch":
                raise RuntimeError("PyTorch is required for --method torch")
    return _train_fallback_dictionary(
        matrix,
        latent_dim=latent_dim,
        seed=seed,
        l1_coefficient=l1_coefficient,
        sparsity=sparsity,
        top_k=top_k,
        jump_threshold=jump_threshold,
        validation_fraction=validation_fraction,
        dead_latent_threshold=dead_latent_threshold,
    )


def write_sae_artifact(artifact: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_sae_activation_records(
    matrix: ActivationMatrix,
    artifact: dict[str, Any],
    out_path: str | Path,
    *,
    top_k_features: int | None = None,
    decoder_signature_size: int = 128,
    force_latent_indexes: set[int] | None = None,
) -> Path:
    activations = encode_with_artifact(matrix.values, artifact)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = artifact.get("layer")
    source_feature_ids = list(artifact["source_feature_ids"])
    decoder_rows = list(artifact["decoder_weight"])
    training_summary = _sae_training_summary(artifact, sample_count=len(matrix.rows))
    with path.open("w", encoding="utf-8") as handle:
        for row, latent_values in zip(matrix.rows, activations):
            selected = _select_latents(
                latent_values,
                top_k_features,
                force_latent_indexes=force_latent_indexes,
            )
            features = []
            metadata = {}
            for latent_index, activation in selected:
                feature_id = _latent_feature_id(layer, latent_index)
                decoder = decoder_rows[latent_index]
                features.append(
                    {
                        "feature_id": feature_id,
                        "activation": round(activation, 8),
                        "label": f"trained SAE latent {latent_index}",
                        "layer": layer,
                        "decoder_signature": _signature(decoder, decoder_signature_size),
                    }
                )
                metadata[feature_id] = {
                    "label": f"trained SAE latent {latent_index}",
                    "layer": layer,
                    "source": "trained-sae",
                    "decoder_top_sources": _top_decoder_sources(decoder, source_feature_ids, limit=8),
                    "sae_training": training_summary,
                    "training_method": artifact["method"],
                }
            handle.write(
                json.dumps(
                    {
                        "model": matrix.model,
                        "prompt_id": _row_prompt_id(row),
                        "text": _row_text(row),
                        "criterion_score": row.criterion_score,
                        "features": features,
                        "feature_metadata": metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _sae_training_summary(artifact: dict[str, Any], *, sample_count: int) -> dict[str, Any]:
    metrics = dict(artifact.get("metrics", {}))
    config = dict(artifact.get("config", {}))
    latent_dim = int(artifact.get("latent_dim", 0) or 0)
    dead_count = int(metrics.get("dead_latent_count", 0) or 0)
    active_fraction = float(metrics.get("active_latent_fraction", 0.0) or 0.0)
    rows_per_latent = sample_count / latent_dim if latent_dim else 0.0
    summary: dict[str, Any] = {
        "method": str(artifact.get("method", "unknown")),
        "sample_count": int(sample_count),
        "latent_dim": latent_dim,
        "rows_per_latent": round(rows_per_latent, 6),
        "active_latent_fraction": round(active_fraction, 6),
        "dead_latent_count": dead_count,
        "average_l0": _round_optional(metrics.get("average_l0")),
        "train_reconstruction_mse": _round_optional(metrics.get("train_reconstruction_mse")),
        "validation_reconstruction_mse": _round_optional(metrics.get("validation_reconstruction_mse")),
        "sparsity": str(config.get("sparsity", "")),
        "top_k": config.get("top_k"),
    }
    advisories = []
    dead_fraction = dead_count / latent_dim if latent_dim else 0.0
    if sample_count < latent_dim:
        advisories.append("Training rows are fewer than latents; collect more activations or reduce latent_dim.")
    elif latent_dim and rows_per_latent < 4:
        advisories.append(
            "Training rows are sparse for this latent count; use a broader prompt corpus before relying on latent labels."
        )
    if dead_fraction >= 0.5:
        advisories.append("Dead-latent fraction is high; increase data, lower latent_dim, or retune sparsity.")
    train_mse = _optional_float(metrics.get("train_reconstruction_mse"))
    validation_mse = _optional_float(metrics.get("validation_reconstruction_mse"))
    if train_mse is not None and validation_mse is not None and train_mse > 0:
        summary["validation_train_mse_ratio"] = round(validation_mse / train_mse, 6)
        if validation_mse > train_mse * 1.5:
            advisories.append(
                "Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set."
            )
    if advisories:
        summary["advisories"] = advisories
    return summary


def _round_optional(value: Any) -> float | None:
    parsed = _optional_float(value)
    return round(parsed, 6) if parsed is not None else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_hf_sae_interventions(
    *,
    model_name: str,
    dataset_path: str | Path,
    artifact: dict[str, Any],
    matrix: ActivationMatrix,
    criterion: str,
    out_path: str | Path,
    top_k: int,
    strength_sweep: list[float],
    target_tokens: list[str] | None,
    device: str,
    max_length: int,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to validate SAE latents causally.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to validate SAE latents causally.",
    )
    layer = artifact.get("layer")
    if layer is None:
        raise ValueError("HF SAE causal validation requires an artifact with a layer")
    if artifact.get("source", {}).get("kind") != "hf-hidden-state":
        raise ValueError("HF SAE causal validation requires an SAE trained from HF hidden states")

    prompts = load_prompt_records(dataset_path)
    positive_prompt_indexes, negative_prompt_indexes = _split_prompt_indexes(prompts)
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
        raise ValueError("No target token ids resolved for SAE causal validation")

    ranked_latents = _rank_latents_by_association(matrix, artifact)
    decoder_rows = list(artifact["decoder_weight"])
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for latent_index, signed_association in ranked_latents[:top_k]:
                sign = 1.0 if signed_association >= 0 else -1.0
                base_direction = [sign * value for value in decoder_rows[latent_index]]
                direction_tensor = torch.tensor(base_direction, dtype=torch.float32, device=runtime_device)
                rows_by_strength: dict[float, list[dict[str, Any]]] = {
                    strength: [] for strength in strength_sweep
                }
                side_effects_by_strength: dict[float, list[float]] = {
                    strength: [] for strength in strength_sweep
                }
                for prompt_index, prompt in enumerate(prompts):
                    baseline_score = _score_prompt(
                        model,
                        tokenizer,
                        prompt.text,
                        target_ids,
                        direction=None,
                        layer=int(layer),
                        strength=0.0,
                        device=runtime_device,
                        max_length=max_length,
                    )
                    for strength in strength_sweep:
                        intervention_score = _score_prompt(
                            model,
                            tokenizer,
                            prompt.text,
                            target_ids,
                            direction=direction_tensor,
                            layer=int(layer),
                            strength=strength,
                            device=runtime_device,
                            max_length=max_length,
                        )
                        if prompt_index in positive_prompt_indexes:
                            rows_by_strength[strength].append(
                                {
                                    "schema_version": "interp-lab.intervention_record.v1",
                                    "model": model_name,
                                    "feature_id": _latent_feature_id(layer, latent_index),
                                    "criterion": criterion,
                                    "intervention": "steer",
                                    "prompt_id": prompt.prompt_id,
                                    "baseline_score": baseline_score,
                                    "intervention_score": intervention_score,
                                    "metadata": {
                                        "behavior_score": "target_token_probability_mass",
                                        "signed_association": round(signed_association, 8),
                                        "steer_sign": sign,
                                        "steer_strength": strength,
                                        "target_token_strategy": token_strategy,
                                        "target_tokens": resolved_target_tokens,
                                    },
                                }
                            )
                        elif prompt_index in negative_prompt_indexes:
                            side_effects_by_strength[strength].append(abs(intervention_score - baseline_score))
                selected_strength, sweep_summary = _select_best_strength(
                    rows_by_strength,
                    side_effects_by_strength,
                )
                selected_side_effect = mean(side_effects_by_strength.get(selected_strength, []))
                for row in rows_by_strength[selected_strength]:
                    row["side_effect_score"] = round(selected_side_effect, 8)
                    row["metadata"]["selected_strength"] = selected_strength
                    row["metadata"]["strength_sweep"] = sweep_summary
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output_path


def encode_with_artifact(values: list[list[float]], artifact: dict[str, Any]) -> list[list[float]]:
    mean_vector = list(artifact["mean"])
    encoder_weight = list(artifact["encoder_weight"])
    encoder_bias = list(artifact["encoder_bias"])
    validate_sae_artifact_shapes(artifact, values=values)
    config = dict(artifact.get("config", {}))
    sparsity = str(config.get("sparsity", "relu-l1"))
    top_k = config.get("top_k")
    jump_threshold = float(config.get("jump_threshold", 0.0))
    encoded = []
    for row in values:
        centered = [value - mean_value for value, mean_value in zip(row, mean_vector)]
        latent = []
        for weights, bias in zip(encoder_weight, encoder_bias):
            latent.append(sum(value * weight for value, weight in zip(centered, weights)) + bias)
        encoded.append(_apply_sparsity_list(latent, sparsity=sparsity, top_k=top_k, jump_threshold=jump_threshold))
    return encoded


def validate_sae_artifact_shapes(artifact: dict[str, Any], *, values: list[list[float]] | None = None) -> None:
    mean_vector = list(artifact.get("mean", []))
    encoder_weight = [list(row) for row in artifact.get("encoder_weight", [])]
    encoder_bias = list(artifact.get("encoder_bias", []))
    decoder_weight = [list(row) for row in artifact.get("decoder_weight", [])]
    input_dim = int(artifact.get("input_dim", len(mean_vector)) or 0)
    latent_dim = int(artifact.get("latent_dim", len(encoder_weight)) or 0)
    if input_dim <= 0:
        raise ValueError("SAE artifact input_dim must be positive")
    if latent_dim <= 0:
        raise ValueError("SAE artifact latent_dim must be positive")
    if len(mean_vector) != input_dim:
        raise ValueError(f"SAE artifact mean length {len(mean_vector)} does not match input_dim={input_dim}")
    if len(encoder_weight) != latent_dim:
        raise ValueError(
            f"SAE artifact encoder_weight row count {len(encoder_weight)} does not match latent_dim={latent_dim}"
        )
    if len(encoder_bias) != latent_dim:
        raise ValueError(f"SAE artifact encoder_bias length {len(encoder_bias)} does not match latent_dim={latent_dim}")
    for index, row in enumerate(encoder_weight):
        if len(row) != input_dim:
            raise ValueError(
                f"SAE artifact encoder_weight[{index}] length {len(row)} does not match input_dim={input_dim}"
            )
    if decoder_weight:
        if len(decoder_weight) != latent_dim:
            raise ValueError(
                f"SAE artifact decoder_weight row count {len(decoder_weight)} does not match latent_dim={latent_dim}"
            )
        for index, row in enumerate(decoder_weight):
            if len(row) != input_dim:
                raise ValueError(
                    f"SAE artifact decoder_weight[{index}] length {len(row)} does not match input_dim={input_dim}"
                )
    source_feature_ids = artifact.get("source_feature_ids")
    if source_feature_ids is not None and len(list(source_feature_ids)) != input_dim:
        raise ValueError("SAE artifact source_feature_ids length does not match input_dim")
    for index, row in enumerate(values or []):
        if len(row) != input_dim:
            raise ValueError(f"activation row {index} length {len(row)} does not match SAE input_dim={input_dim}")


def build_train_sae_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an on-demand SAE from activation records or HF activations.")
    parser.add_argument(
        "--preset",
        choices=["minimal", "production", "custom"],
        default="minimal",
        help="Training profile. Minimal is quick; production uses token rows, top-k sparsity, and broader eval defaults.",
    )
    parser.add_argument("--records", help="Activation-record JSONL to train from.")
    parser.add_argument("--model", help="Model id to filter/use with --records.")
    parser.add_argument("--hf-model", help="Hugging Face model name to collect activations from directly.")
    parser.add_argument("--dataset", help="Prompt JSONL with text and criterion_score for --hf-model.")
    parser.add_argument(
        "--causal-dataset",
        help="Optional criterion-scored prompt JSONL for --causal-out. Defaults to --dataset.",
    )
    parser.add_argument("--layer", type=int, help="HF hidden-state layer. Defaults to final hidden state.")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument(
        "--token-mode",
        choices=["last", "mean", "all"],
        help="HF activation rows to train on. Use all for token-level SAE training.",
    )
    parser.add_argument("--out", required=True, help="Output SAE artifact JSON path.")
    parser.add_argument("--records-out", help="Optional SAE activation-record JSONL output.")
    parser.add_argument("--latent-dim", type=int, help="Number of SAE latents to train.")
    parser.add_argument("--expansion-factor", type=float)
    parser.add_argument("--method", choices=["auto", "torch", "fallback"])
    parser.add_argument("--sparsity", choices=["relu-l1", "topk", "jumprelu"])
    parser.add_argument("--top-k", type=int, help="Active latents per token for --sparsity topk.")
    parser.add_argument("--jump-threshold", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l1", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--dead-latent-threshold", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--top-k-features",
        type=int,
        help="Latents to write per prompt; 0 writes every latent.",
    )
    parser.add_argument("--decoder-signature-size", type=int)
    parser.add_argument("--causal-out", help="Optional intervention-record JSONL for SAE latent steering.")
    parser.add_argument("--criterion", help="Criterion text for --causal-out.")
    parser.add_argument("--causal-top-k", type=int)
    parser.add_argument(
        "--causal-strength-sweep",
        help="Comma-separated steering strengths for SAE latent causal validation.",
    )
    parser.add_argument("--target-token", action="append")
    add_hf_loading_args(parser)
    return parser


def run_train_sae_from_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.records and args.hf_model:
        raise SystemExit("Use either --records or --hf-model, not both")
    settings = _training_settings(args)
    if args.records:
        if args.causal_out:
            raise SystemExit("--causal-out currently requires --hf-model")
        return train_sae_from_records(
            records_path=args.records,
            out_path=args.out,
            records_out=args.records_out,
            model_name=args.model,
            latent_dim=args.latent_dim,
            expansion_factor=settings["expansion_factor"],
            method=settings["method"],
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["lr"],
            l1_coefficient=settings["l1"],
            sparsity=settings["sparsity"],
            top_k=settings["top_k"],
            jump_threshold=settings["jump_threshold"],
            validation_fraction=settings["validation_fraction"],
            dead_latent_threshold=settings["dead_latent_threshold"],
            seed=args.seed,
            device=args.device,
            max_records=args.max_records,
            top_k_features=_feature_limit(settings["top_k_features"]),
            decoder_signature_size=settings["decoder_signature_size"],
        )
    if args.hf_model:
        if not args.dataset:
            raise SystemExit("--dataset is required with --hf-model")
        try:
            loading_options = hf_loading_options_from_args(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return train_sae_from_hf(
            model_name=args.hf_model,
            dataset_path=args.dataset,
            causal_dataset_path=args.causal_dataset,
            out_path=args.out,
            records_out=args.records_out,
            layer=args.layer,
            pool=args.pool,
            token_mode=settings["token_mode"],
            latent_dim=args.latent_dim,
            expansion_factor=settings["expansion_factor"],
            method=settings["method"],
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["lr"],
            l1_coefficient=settings["l1"],
            sparsity=settings["sparsity"],
            top_k=settings["top_k"],
            jump_threshold=settings["jump_threshold"],
            validation_fraction=settings["validation_fraction"],
            dead_latent_threshold=settings["dead_latent_threshold"],
            seed=args.seed,
            device=args.device,
            max_length=settings["max_length"],
            max_records=args.max_records,
            top_k_features=_feature_limit(settings["top_k_features"]),
            decoder_signature_size=settings["decoder_signature_size"],
            causal_out=args.causal_out,
            criterion=args.criterion,
            causal_top_k=settings["causal_top_k"],
            causal_strength_sweep=parse_strength_sweep(args.causal_strength_sweep),
            target_tokens=parse_target_tokens(args.target_token),
            **loading_options,
        )
    raise SystemExit("Either --records or --hf-model is required")


def _training_settings(args: argparse.Namespace) -> dict[str, Any]:
    preset = dict(TRAINING_PRESETS[str(args.preset)])
    for key in preset:
        value = getattr(args, key, None)
        if value is not None:
            preset[key] = value
    return preset


def _feature_limit(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _train_torch_sae(
    matrix: ActivationMatrix,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_coefficient: float,
    sparsity: str,
    top_k: int | None,
    jump_threshold: float,
    validation_fraction: float,
    dead_latent_threshold: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    x = torch.tensor(matrix.values, dtype=torch.float32, device=device)
    train_indexes, validation_indexes = _split_indexes(x.shape[0], validation_fraction, seed)
    train_index_tensor = torch.tensor(train_indexes, dtype=torch.long, device=device)
    mean_vector = x[train_index_tensor].mean(dim=0)
    centered = x - mean_vector
    train_centered = centered[train_index_tensor]
    validation_centered = (
        centered[torch.tensor(validation_indexes, dtype=torch.long, device=device)]
        if validation_indexes
        else None
    )
    input_dim = centered.shape[1]
    encoder = torch.nn.Linear(input_dim, latent_dim, bias=True, device=device)
    decoder = torch.nn.Linear(latent_dim, input_dim, bias=False, device=device)
    torch.nn.init.normal_(encoder.weight, mean=0.0, std=1.0 / math.sqrt(max(1, input_dim)))
    torch.nn.init.zeros_(encoder.bias)
    torch.nn.init.normal_(decoder.weight, mean=0.0, std=1.0 / math.sqrt(max(1, latent_dim)))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    row_count = train_centered.shape[0]
    batch_size = max(1, min(batch_size, row_count))
    for _epoch in range(max(1, epochs)):
        permutation = torch.randperm(row_count, device=device)
        for start in range(0, row_count, batch_size):
            batch = train_centered[permutation[start : start + batch_size]]
            hidden = _apply_sparsity_torch(
                encoder(batch),
                sparsity=sparsity,
                top_k=top_k,
                jump_threshold=jump_threshold,
            )
            reconstruction = decoder(hidden)
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            sparsity_loss = hidden.abs().mean()
            loss = reconstruction_loss + l1_coefficient * sparsity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norms = decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-6)
                decoder.weight.data = decoder.weight.data / norms
    with torch.no_grad():
        hidden = _apply_sparsity_torch(
            encoder(centered),
            sparsity=sparsity,
            top_k=top_k,
            jump_threshold=jump_threshold,
        )
        reconstruction = decoder(hidden)
        train_hidden = _apply_sparsity_torch(
            encoder(train_centered),
            sparsity=sparsity,
            top_k=top_k,
            jump_threshold=jump_threshold,
        )
        train_reconstruction = decoder(train_hidden)
        validation_mse = None
        if validation_centered is not None:
            validation_hidden = _apply_sparsity_torch(
                encoder(validation_centered),
                sparsity=sparsity,
                top_k=top_k,
                jump_threshold=jump_threshold,
            )
            validation_reconstruction = decoder(validation_hidden)
            validation_mse = float(
                torch.mean((validation_reconstruction - validation_centered) ** 2).detach().cpu().item()
            )
        reconstruction_mse = float(torch.mean((reconstruction - centered) ** 2).detach().cpu().item())
        train_mse = float(torch.mean((train_reconstruction - train_centered) ** 2).detach().cpu().item())
        average_l0 = float((hidden > 1e-6).float().sum(dim=1).mean().detach().cpu().item())
        firing_rates = (hidden > 1e-6).float().mean(dim=0).detach().cpu().tolist()
        dead_latents = [
            index
            for index, firing_rate in enumerate(firing_rates)
            if firing_rate <= dead_latent_threshold
        ]
        encoder_weight = _round_matrix(encoder.weight.detach().cpu().tolist())
        encoder_bias = _round_list(encoder.bias.detach().cpu().tolist())
        decoder_weight = _round_matrix(decoder.weight.detach().cpu().transpose(0, 1).tolist())
        latent_means = _round_list(hidden.mean(dim=0).detach().cpu().tolist())
    return _artifact(
        matrix,
        method="torch",
        latent_dim=latent_dim,
        mean_vector=_round_list(mean_vector.detach().cpu().tolist()),
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        metrics={
            "reconstruction_mse": round(reconstruction_mse, 8),
            "train_reconstruction_mse": round(train_mse, 8),
            "validation_reconstruction_mse": round(validation_mse, 8) if validation_mse is not None else None,
            "average_l0": round(average_l0, 6),
            "latent_activation_mean": latent_means,
            "latent_firing_rate": _round_list(firing_rates),
            "dead_latent_count": len(dead_latents),
            "dead_latent_indices": dead_latents,
            "active_latent_fraction": round(1.0 - len(dead_latents) / max(1, latent_dim), 6),
        },
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l1_coefficient": l1_coefficient,
            "sparsity": sparsity,
            "top_k": top_k,
            "jump_threshold": jump_threshold,
            "validation_fraction": validation_fraction,
            "dead_latent_threshold": dead_latent_threshold,
            "seed": seed,
        },
    )


def _train_fallback_dictionary(
    matrix: ActivationMatrix,
    *,
    latent_dim: int,
    seed: int,
    l1_coefficient: float,
    sparsity: str,
    top_k: int | None,
    jump_threshold: float,
    validation_fraction: float,
    dead_latent_threshold: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    train_indexes, validation_indexes = _split_indexes(len(matrix.values), validation_fraction, seed)
    mean_vector = _column_means([matrix.values[index] for index in train_indexes])
    centered = [[value - mean_value for value, mean_value in zip(row, mean_vector)] for row in matrix.values]
    input_dim = len(matrix.source_feature_ids)
    variances = _column_variances(centered)
    ranked_dimensions = sorted(range(input_dim), key=lambda index: variances[index], reverse=True)
    directions = []
    for latent_index in range(latent_dim):
        if latent_index < len(ranked_dimensions):
            direction = [0.0] * input_dim
            direction[ranked_dimensions[latent_index]] = 1.0
        else:
            direction = [rng.gauss(0.0, 1.0) for _ in range(input_dim)]
            direction = _normalize(direction)
        directions.append(direction)
    activations = []
    for row in centered:
        raw_hidden = [
            sum(value * weight for value, weight in zip(row, direction)) - l1_coefficient
            for direction in directions
        ]
        activations.append(
            _apply_sparsity_list(
                raw_hidden,
                sparsity=sparsity,
                top_k=top_k,
                jump_threshold=jump_threshold,
            )
        )
    reconstruction = [
        [
            sum(hidden_value * direction[dimension] for hidden_value, direction in zip(hidden, directions))
            for dimension in range(input_dim)
        ]
        for hidden in activations
    ]
    reconstruction_mse = _matrix_mse(centered, reconstruction)
    train_mse = _matrix_mse(
        [centered[index] for index in train_indexes],
        [reconstruction[index] for index in train_indexes],
    )
    validation_mse = (
        _matrix_mse(
            [centered[index] for index in validation_indexes],
            [reconstruction[index] for index in validation_indexes],
        )
        if validation_indexes
        else None
    )
    average_l0 = mean([sum(1 for value in hidden if value > 1e-6) for hidden in activations])
    firing_rates = [
        mean([1.0 if hidden[index] > 1e-6 else 0.0 for hidden in activations])
        for index in range(latent_dim)
    ]
    dead_latents = [
        index
        for index, firing_rate in enumerate(firing_rates)
        if firing_rate <= dead_latent_threshold
    ]
    return _artifact(
        matrix,
        method="fallback-dictionary",
        latent_dim=latent_dim,
        mean_vector=_round_list(mean_vector),
        encoder_weight=_round_matrix(directions),
        encoder_bias=[0.0] * latent_dim,
        decoder_weight=_round_matrix(directions),
        metrics={
            "reconstruction_mse": round(reconstruction_mse, 8),
            "train_reconstruction_mse": round(train_mse, 8),
            "validation_reconstruction_mse": round(validation_mse, 8) if validation_mse is not None else None,
            "average_l0": round(average_l0, 6),
            "latent_activation_mean": _round_list([mean(column) for column in zip(*activations)]),
            "latent_firing_rate": _round_list(firing_rates),
            "dead_latent_count": len(dead_latents),
            "dead_latent_indices": dead_latents,
            "active_latent_fraction": round(1.0 - len(dead_latents) / max(1, latent_dim), 6),
        },
        config={
            "seed": seed,
            "l1_coefficient": l1_coefficient,
            "sparsity": sparsity,
            "top_k": top_k,
            "jump_threshold": jump_threshold,
            "validation_fraction": validation_fraction,
            "dead_latent_threshold": dead_latent_threshold,
        },
    )


def _artifact(
    matrix: ActivationMatrix,
    *,
    method: str,
    latent_dim: int,
    mean_vector: list[float],
    encoder_weight: list[list[float]],
    encoder_bias: list[float],
    decoder_weight: list[list[float]],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "interp-lab.sae.v1",
        "model": matrix.model,
        "source": matrix.source or {},
        "layer": matrix.layer,
        "input_dim": len(matrix.source_feature_ids),
        "latent_dim": latent_dim,
        "source_feature_ids": matrix.source_feature_ids,
        "mean": mean_vector,
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_weight": decoder_weight,
        "method": method,
        "metrics": metrics,
        "config": config,
    }


def _vectors_from_hidden_state(
    hidden_state: Any,
    encoded: dict[str, Any],
    tokenizer: Any,
    prompt: Any,
    *,
    token_mode: str,
) -> list[tuple[MatrixRow, list[float]]]:
    mode = token_mode.lower()
    if mode in {"last", "mean"}:
        return [
            (
                MatrixRow(
                    model="",
                    prompt_id=prompt.prompt_id,
                    text=prompt.text,
                    criterion_score=prompt.criterion_score,
                ),
                _pool_hidden_state(hidden_state, encoded.get("attention_mask"), pool=mode),
            )
        ]
    if mode != "all":
        raise ValueError("token_mode must be one of: last, mean, all")
    attention_mask = encoded.get("attention_mask")
    token_count = hidden_state.shape[1]
    if attention_mask is not None:
        token_count = int(attention_mask[0].sum().item())
    input_ids = encoded.get("input_ids")
    rows = []
    for token_index in range(token_count):
        token_text = ""
        if input_ids is not None:
            token_text = tokenizer.decode([int(input_ids[0, token_index].item())])
        rows.append(
            (
                MatrixRow(
                    model="",
                    prompt_id=prompt.prompt_id,
                    text=prompt.text,
                    criterion_score=prompt.criterion_score,
                    token_index=token_index,
                    token_text=token_text,
                ),
                [float(value) for value in hidden_state[0, token_index].detach().cpu().tolist()],
            )
        )
    return rows


def _iter_prompt_records(path: str | Path):
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
            yield PromptRecord.from_dict(data, line_label=line_label)


def _load_prompt_records_sampled(path: str | Path, *, max_records: int | None, seed: int) -> list[PromptRecord]:
    if max_records is None:
        return load_prompt_records(path)
    prompts: list[PromptRecord] = []
    seen = 0
    rng = random.Random(seed)
    for prompt in _iter_prompt_records(path):
        seen += 1
        if len(prompts) < max_records:
            prompts.append(prompt)
        else:
            replacement = rng.randrange(seen)
            if replacement < max_records:
                prompts[replacement] = prompt
    return prompts


def _apply_sparsity_torch(raw_hidden: Any, *, sparsity: str, top_k: int | None, jump_threshold: float):
    torch = importlib.import_module("torch")
    key = sparsity.lower()
    if key == "relu-l1":
        return torch.relu(raw_hidden)
    if key == "jumprelu":
        return torch.where(raw_hidden > jump_threshold, raw_hidden, torch.zeros_like(raw_hidden))
    if key == "topk":
        hidden = torch.relu(raw_hidden)
        if top_k is None or top_k <= 0 or top_k >= hidden.shape[-1]:
            return hidden
        values, indexes = torch.topk(hidden, k=top_k, dim=-1)
        sparse = torch.zeros_like(hidden)
        return sparse.scatter(dim=-1, index=indexes, src=values)
    raise ValueError("sparsity must be one of: relu-l1, topk, jumprelu")


def _apply_sparsity_list(
    raw_hidden: list[float],
    *,
    sparsity: str,
    top_k: Any,
    jump_threshold: float,
) -> list[float]:
    key = sparsity.lower()
    if key == "relu-l1" or key == "fallback-relu":
        return [max(0.0, value) for value in raw_hidden]
    if key == "jumprelu":
        return [value if value > jump_threshold else 0.0 for value in raw_hidden]
    if key == "topk":
        relu = [max(0.0, value) for value in raw_hidden]
        k = int(top_k) if top_k is not None else len(relu)
        if k <= 0 or k >= len(relu):
            return relu
        keep = {index for index, _ in sorted(enumerate(relu), key=lambda item: item[1], reverse=True)[:k]}
        return [value if index in keep else 0.0 for index, value in enumerate(relu)]
    return [max(0.0, value) for value in raw_hidden]


def _split_indexes(row_count: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indexes = list(range(row_count))
    if row_count <= 1 or validation_fraction <= 0:
        return indexes, []
    rng = random.Random(seed)
    rng.shuffle(indexes)
    validation_count = int(round(row_count * validation_fraction))
    validation_count = max(1, min(validation_count, row_count - 1))
    validation = sorted(indexes[:validation_count])
    train = sorted(indexes[validation_count:])
    return train, validation


def _rank_latents_by_association(matrix: ActivationMatrix, artifact: dict[str, Any]) -> list[tuple[int, float]]:
    encoded = encode_with_artifact(matrix.values, artifact)
    scores = [row.criterion_score for row in matrix.rows]
    ranked = []
    for latent_index in range(int(artifact["latent_dim"])):
        values = [row[latent_index] for row in encoded]
        ranked.append((latent_index, pearson(values, scores)))
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return ranked


def _split_prompt_indexes(prompts: list[Any]) -> tuple[set[int], set[int]]:
    if not prompts:
        return set(), set()
    scores = [prompt.criterion_score for prompt in prompts]
    if min(scores) == max(scores):
        return set(range(len(prompts))), set()
    threshold = mean(scores)
    positive = {
        index
        for index, prompt in enumerate(prompts)
        if prompt.criterion_score >= threshold
    }
    negative = set(range(len(prompts))) - positive
    return positive, negative


def _split_prompt_ids(prompts: list[Any]) -> tuple[set[str], set[str]]:
    positive_indexes, negative_indexes = _split_prompt_indexes(prompts)
    positive = {prompts[index].prompt_id for index in positive_indexes}
    negative = {prompts[index].prompt_id for index in negative_indexes}
    return positive, negative


def _score_prompt(
    model: Any,
    tokenizer: Any,
    text: str,
    target_ids: list[int],
    *,
    direction: Any | None,
    layer: int,
    strength: float,
    device: str,
    max_length: int,
) -> float:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    handle = None
    if direction is not None:
        handle = _register_gpt2_steering(model, layer, direction, strength)
    try:
        outputs = model(**encoded, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    probabilities = outputs.logits[0, -1].softmax(dim=-1)
    return round(float(probabilities[target_ids].sum().detach().cpu().item()), 8)


def _target_token_ids(tokenizer: Any, target_tokens: list[str]) -> list[int]:
    token_ids: set[int] = set()
    for token in target_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[-1]))
    return sorted(token_ids)


def _row_prompt_id(row: MatrixRow) -> str:
    if row.token_index is None:
        return row.prompt_id
    return f"{row.prompt_id}:tok{row.token_index}"


def _row_text(row: MatrixRow) -> str:
    if row.token_index is None:
        return row.text
    token = row.token_text.replace("\n", "\\n")
    return f"{row.text} | token[{row.token_index}]={token!r}"


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(message) from exc


def _resolve_layer(layer: int | None, hidden_state_count: int) -> int:
    resolved = hidden_state_count - 1 if layer is None else layer
    if resolved < 0 or resolved >= hidden_state_count:
        raise ValueError(f"Layer {resolved} is outside 0..{hidden_state_count - 1}")
    return resolved


def _resolve_latent_dim(latent_dim: int | None, expansion_factor: float, input_dim: int) -> int:
    if latent_dim is not None:
        return latent_dim
    if expansion_factor <= 0:
        raise ValueError("expansion_factor must be positive")
    return max(1, int(round(input_dim * expansion_factor)))


def _ordered_feature_ids(records: list[ActivationRecord]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        for feature_id in record.features:
            if feature_id not in seen:
                seen.add(feature_id)
                ordered.append(feature_id)
    return ordered


def _common_layer(records: list[ActivationRecord], feature_ids: list[str]) -> int | None:
    layers = set()
    for record in records:
        for feature_id in feature_ids:
            raw_layer = record.feature_metadata.get(feature_id, {}).get("layer")
            if raw_layer is not None:
                layers.add(int(raw_layer))
    return next(iter(layers)) if len(layers) == 1 else None


def _select_latents(
    latent_values: list[float],
    top_k_features: int | None,
    *,
    force_latent_indexes: set[int] | None = None,
) -> list[tuple[int, float]]:
    indexed = [(index, value) for index, value in enumerate(latent_values)]
    indexed.sort(key=lambda item: item[1], reverse=True)
    if top_k_features is None or top_k_features <= 0:
        return indexed
    selected = indexed[:top_k_features]
    selected_indexes = {index for index, _value in selected}
    for index in sorted(force_latent_indexes or set()):
        if 0 <= index < len(latent_values) and index not in selected_indexes:
            selected.append((index, latent_values[index]))
    selected.sort(key=lambda item: item[1], reverse=True)
    return selected


def _latent_feature_id(layer: Any, latent_index: int) -> str:
    if layer is None:
        return f"SAE:F{latent_index}"
    return f"SAE:L{layer}:F{latent_index}"


def _signature(values: list[float], size: int) -> list[float]:
    return _round_list(values[: max(0, size)])


def _top_decoder_sources(decoder: list[float], source_feature_ids: list[str], *, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        zip(source_feature_ids, decoder),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {"source_feature_id": feature_id, "weight": round(weight, 6)}
        for feature_id, weight in ranked[:limit]
    ]


def _column_means(values: list[list[float]]) -> list[float]:
    return [mean(column) for column in zip(*values)]


def _column_variances(centered: list[list[float]]) -> list[float]:
    return [mean([value * value for value in column]) for column in zip(*centered)]


def _normalize(values: list[float]) -> list[float]:
    length = norm(values)
    if length == 0:
        return values
    return [value / length for value in values]


def _matrix_mse(left: list[list[float]], right: list[list[float]]) -> float:
    errors = []
    for left_row, right_row in zip(left, right):
        errors.extend((a - b) ** 2 for a, b in zip(left_row, right_row))
    return mean(errors)


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 8) for value in values]


def _round_matrix(values: list[list[float]]) -> list[list[float]]:
    return [_round_list(row) for row in values]
