from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int8": 1,
}
COUNT_SUFFIXES = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}
BYTE_SUFFIXES = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
    "pb": 1024**5,
}
PROFILE_PRESETS = {
    "local-cpu": {
        "target_shard_size_bytes": 512 * 1024**2,
        "artifact_format": "jsonl-gzip",
        "max_dense_bytes": 100 * 1024**3,
    },
    "single-gpu": {
        "target_shard_size_bytes": 4 * 1024**3,
        "artifact_format": "safetensors",
        "max_dense_bytes": 2 * 1024**4,
    },
    "cluster": {
        "target_shard_size_bytes": 32 * 1024**3,
        "artifact_format": "safetensors",
        "max_dense_bytes": 100 * 1024**4,
    },
    "remote-api": {
        "target_shard_size_bytes": 1 * 1024**3,
        "artifact_format": "jsonl-gzip",
        "max_dense_bytes": 10 * 1024**4,
    },
    "frontier-lab": {
        "target_shard_size_bytes": 64 * 1024**3,
        "artifact_format": "zarr-or-safetensors",
        "max_dense_bytes": 10 * 1024**5,
    },
}
FORMAT_EVENT_BYTES = {
    "jsonl": 160,
    "jsonl-gzip": 64,
    "parquet": 32,
    "safetensors": 16,
    "zarr": 16,
    "zarr-or-safetensors": 16,
    "auto": 32,
}


@dataclass(frozen=True)
class ScalePlan:
    model_params: float
    tokens: int
    d_model: int
    selected_layers: int
    latent_dim: int
    dtype: str
    shards: int | None = None
    profile: str = "auto"
    artifact_format: str = "auto"
    target_shard_size_bytes: int | None = None
    top_k_active: int = 64
    causal_features: int = 128
    causal_prompts: int = 256
    interventions_per_feature: int = 2
    train_batch_size: int = 4096

    def to_dict(self) -> dict[str, Any]:
        dtype_bytes = DTYPE_BYTES[self.dtype]
        activation_bytes = self.tokens * self.d_model * self.selected_layers * dtype_bytes
        profile, profile_reason = _resolve_profile(self.profile, self.model_params, activation_bytes)
        preset = PROFILE_PRESETS[profile]
        artifact_format = (
            str(preset["artifact_format"]) if self.artifact_format == "auto" else self.artifact_format
        )
        target_shard_size_bytes = self.target_shard_size_bytes or int(preset["target_shard_size_bytes"])
        recommended_shards = max(1, _ceil_div(activation_bytes, target_shard_size_bytes))
        shards = self.shards or recommended_shards
        sae_parameter_bytes = self.d_model * self.latent_dim * 2 * dtype_bytes
        sparse_event_bytes = FORMAT_EVENT_BYTES.get(artifact_format, FORMAT_EVENT_BYTES["auto"])
        sparse_record_bytes = (
            self.tokens
            * self.selected_layers
            * self.top_k_active
            * sparse_event_bytes
        )
        batch_activation_bytes = self.train_batch_size * self.d_model * dtype_bytes
        training_memory_floor_bytes = sae_parameter_bytes * 4 + batch_activation_bytes * 2
        causal_forward_passes = self.causal_prompts * (
            1 + self.causal_features * self.interventions_per_feature
        )
        assumptions = _assumptions(
            dtype_bytes=dtype_bytes,
            sparse_event_bytes=sparse_event_bytes,
            artifact_format=artifact_format,
        )
        risks = risk_flags_for_plan(
            model_params=self.model_params,
            activation_bytes=activation_bytes,
            sparse_record_bytes=sparse_record_bytes,
            profile=profile,
            shards=shards,
            target_shard_size_bytes=target_shard_size_bytes,
            max_dense_bytes=int(preset["max_dense_bytes"]),
        )
        recommendations = recommendations_for_plan(
            model_params=self.model_params,
            activation_bytes=activation_bytes,
            sparse_record_bytes=sparse_record_bytes,
            shards=shards,
            profile=profile,
            artifact_format=artifact_format,
            risks=risks,
        )
        next_actions = agent_next_actions_for_plan(
            profile=profile,
            artifact_format=artifact_format,
            activation_bytes=activation_bytes,
            sparse_record_bytes=sparse_record_bytes,
        )
        return {
            "schema_version": "interp-lab.scale_plan.v2",
            "inputs": {
                "model_params": self.model_params,
                "tokens": self.tokens,
                "d_model": self.d_model,
                "selected_layers": self.selected_layers,
                "latent_dim": self.latent_dim,
                "dtype": self.dtype,
                "profile": self.profile,
                "artifact_format": self.artifact_format,
                "target_shard_size_bytes": self.target_shard_size_bytes,
                "top_k_active": self.top_k_active,
                "causal_features": self.causal_features,
                "causal_prompts": self.causal_prompts,
                "interventions_per_feature": self.interventions_per_feature,
                "train_batch_size": self.train_batch_size,
            },
            "profile": profile,
            "profile_reason": profile_reason,
            "artifact_format": artifact_format,
            "assumptions": assumptions,
            "estimates": {
                "dense_activation_storage": _byte_estimate(activation_bytes),
                "sparse_feature_record_storage": _byte_estimate(sparse_record_bytes),
                "sae_parameter_storage": _byte_estimate(sae_parameter_bytes),
                "sae_training_memory_floor": _byte_estimate(training_memory_floor_bytes),
                "causal_validation": {
                    "features": self.causal_features,
                    "prompts": self.causal_prompts,
                    "interventions_per_feature": self.interventions_per_feature,
                    "estimated_forward_passes": causal_forward_passes,
                },
            },
            "shard_plan": {
                "shards": shards,
                "recommended_shards_for_target": recommended_shards,
                "target_shard_size_bytes": target_shard_size_bytes,
                "target_shard_size_human": _format_bytes(target_shard_size_bytes),
                "dense_bytes_per_shard": activation_bytes // max(shards, 1),
                "dense_human_per_shard": _format_bytes(activation_bytes // max(shards, 1)),
                "sparse_bytes_per_shard": sparse_record_bytes // max(shards, 1),
                "sparse_human_per_shard": _format_bytes(sparse_record_bytes // max(shards, 1)),
            },
            "risk_flags": risks,
            "recommendations": recommendations,
            "agent_next_actions": next_actions,
            # Backward-compatible aliases for simple callers.
            "model_params": self.model_params,
            "tokens": self.tokens,
            "d_model": self.d_model,
            "selected_layers": self.selected_layers,
            "latent_dim": self.latent_dim,
            "dtype": self.dtype,
            "dtype_bytes": dtype_bytes,
            "shards": shards,
            "activation_storage_bytes": activation_bytes,
            "activation_storage_human": _format_bytes(activation_bytes),
            "activation_storage_per_shard_bytes": activation_bytes // max(shards, 1),
            "activation_storage_per_shard_human": _format_bytes(activation_bytes // max(shards, 1)),
            "sae_parameter_bytes": sae_parameter_bytes,
            "sae_parameter_human": _format_bytes(sae_parameter_bytes),
        }


def build_scale_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate storage and execution shape for large interp-lab jobs.")
    parser.add_argument("--model-params", type=parse_count_float, required=True, help="Model parameter count, e.g. 1T, 70B, or 1e12.")
    parser.add_argument("--tokens", type=parse_count_int, required=True, help="Activation tokens to process, e.g. 1B.")
    parser.add_argument("--d-model", type=parse_count_int, required=True, help="Hidden width of the activation stream.")
    parser.add_argument("--selected-layers", type=int, default=1, help="Number of layers or hook points captured.")
    parser.add_argument("--latent-dim", type=parse_count_int, default=131072, help="SAE latent width.")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument(
        "--profile",
        choices=["auto", *sorted(PROFILE_PRESETS)],
        default="auto",
        help="Execution profile used for defaults and recommendations.",
    )
    parser.add_argument(
        "--artifact-format",
        choices=sorted(FORMAT_EVENT_BYTES),
        default="auto",
        help="Planned activation or feature-record artifact format.",
    )
    parser.add_argument(
        "--target-shard-size",
        type=parse_bytes,
        help="Target dense shard size, e.g. 4GB or 64GB. Used when --shards is omitted.",
    )
    parser.add_argument("--shards", type=parse_shards, default="auto", help="Number of activation shards or 'auto'.")
    parser.add_argument("--top-k-active", type=int, default=64, help="Estimated active features per token after sparsification.")
    parser.add_argument("--causal-features", type=int, default=128, help="Features to causally validate.")
    parser.add_argument("--causal-prompts", type=int, default=256, help="Prompts per causal validation batch.")
    parser.add_argument("--interventions-per-feature", type=int, default=2, help="Interventions per feature.")
    parser.add_argument("--train-batch-size", type=parse_count_int, default=4096, help="SAE train batch size for memory floor estimate.")
    parser.add_argument("--out", help="Optional JSON path to write the plan for agents or workflows.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def run_scale_plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.shards is not None and args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    for name in [
        "selected_layers",
        "top_k_active",
        "causal_features",
        "causal_prompts",
        "interventions_per_feature",
        "train_batch_size",
    ]:
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    plan = ScalePlan(
        model_params=args.model_params,
        tokens=args.tokens,
        d_model=args.d_model,
        selected_layers=args.selected_layers,
        latent_dim=args.latent_dim,
        dtype=args.dtype,
        shards=args.shards,
        profile=args.profile,
        artifact_format=args.artifact_format,
        target_shard_size_bytes=args.target_shard_size,
        top_k_active=args.top_k_active,
        causal_features=args.causal_features,
        causal_prompts=args.causal_prompts,
        interventions_per_feature=args.interventions_per_feature,
        train_batch_size=args.train_batch_size,
    ).to_dict()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(render_scale_plan(plan))
    return plan


def render_scale_plan(plan: dict[str, Any]) -> str:
    estimates = plan["estimates"]
    shard_plan = plan["shard_plan"]
    lines = [
        "interp-lab scale plan",
        "",
        f"Model parameters: {plan['model_params']:.3g}",
        f"Profile: {plan['profile']} ({plan['profile_reason']})",
        f"Artifact format: {plan['artifact_format']}",
        f"Activation tokens: {plan['tokens']:,}",
        f"Captured hook points: {plan['selected_layers']}",
        f"Dense activation storage: {estimates['dense_activation_storage']['human']}",
        f"Sparse feature-record storage: {estimates['sparse_feature_record_storage']['human']}",
        f"Per dense shard: {shard_plan['dense_human_per_shard']} across {shard_plan['shards']} shard(s)",
        f"SAE parameter storage: {estimates['sae_parameter_storage']['human']}",
        f"SAE training memory floor: {estimates['sae_training_memory_floor']['human']}",
        (
            "Causal validation: "
            f"{estimates['causal_validation']['estimated_forward_passes']:,} estimated forward passes"
        ),
    ]
    if plan["risk_flags"]:
        lines.extend(["", "Risk flags:"])
        for item in plan["risk_flags"]:
            lines.append(f"- [{item['level']}] {item['message']} Mitigation: {item['mitigation']}")
    lines.extend(["", "Recommendations:"])
    for item in plan["recommendations"]:
        lines.append(f"- {item}")
    lines.extend(["", "Agent next actions:"])
    for item in plan["agent_next_actions"]:
        command = f" Command: {item['command']}" if item.get("command") else ""
        lines.append(f"- {item['id']}: {item['title']}.{command}")
    return "\n".join(lines)


def recommendations_for_plan(
    *,
    model_params: float,
    activation_bytes: int,
    sparse_record_bytes: int,
    shards: int,
    profile: str,
    artifact_format: str,
    risks: list[dict[str, str]],
) -> list[str]:
    recommendations = [
        f"Use the {profile} profile assumptions for this estimate.",
        f"Write dense activations as {artifact_format} shards, then emit compact activation records after feature extraction.",
        "Run causal validation as resumable batches and merge intervention records after each batch.",
    ]
    if model_params >= 1e12:
        recommendations.append("Use remote inference or colocated activation harvesting for 1T+ models.")
    if activation_bytes > 10 * 1024**4:
        recommendations.append("Store activations in shards and stream them into SAE training or feature ranking.")
    if sparse_record_bytes < activation_bytes:
        recommendations.append("Prefer sparse feature records for report generation and cross-model matching.")
    if shards < 128 and activation_bytes > 1024**4:
        recommendations.append("Use more shards so retries and uploads stay granular.")
    if any(item["level"] == "high" for item in risks):
        recommendations.append("Create a small pilot shard before launching the full harvest.")
    return recommendations


def risk_flags_for_plan(
    *,
    model_params: float,
    activation_bytes: int,
    sparse_record_bytes: int,
    profile: str,
    shards: int,
    target_shard_size_bytes: int,
    max_dense_bytes: int,
) -> list[dict[str, str]]:
    risks = []
    dense_per_shard = activation_bytes // max(shards, 1)
    if activation_bytes > max_dense_bytes:
        risks.append(
            {
                "level": "high",
                "message": f"Dense activations exceed the comfortable range for profile {profile}.",
                "mitigation": "Capture fewer hook points per run or move harvesting to a larger profile.",
            }
        )
    if dense_per_shard > target_shard_size_bytes * 2:
        risks.append(
            {
                "level": "medium",
                "message": "Dense shards are much larger than the target shard size.",
                "mitigation": "Use auto shards or increase --shards.",
            }
        )
    if model_params >= 1e12 and profile in {"local-cpu", "single-gpu"}:
        risks.append(
            {
                "level": "high",
                "message": "The selected profile is too small for direct 1T+ model execution.",
                "mitigation": "Use remote-api, cluster, or frontier-lab profile for model execution.",
            }
        )
    if sparse_record_bytes > 10 * 1024**4:
        risks.append(
            {
                "level": "medium",
                "message": "Sparse feature records are still large.",
                "mitigation": "Lower --top-k-active, reduce captured hook points, or use columnar storage.",
            }
        )
    return risks


def agent_next_actions_for_plan(
    *,
    profile: str,
    artifact_format: str,
    activation_bytes: int,
    sparse_record_bytes: int,
) -> list[dict[str, str]]:
    actions = [
        {
            "id": "choose_adapter",
            "title": "Choose the model execution adapter",
            "why": "Model execution should happen where the model can run efficiently.",
        },
        {
            "id": "pilot_shard",
            "title": "Run one pilot shard and inspect record quality",
            "command": "interp-lab inspect --backend records --records <pilot-records.jsonl> --model <model> --criterion <criterion>",
        },
        {
            "id": "causal_batch",
            "title": "Run a small causal validation batch before scaling out",
        },
    ]
    if activation_bytes > sparse_record_bytes:
        actions.insert(
            1,
            {
                "id": "sparsify_records",
                "title": "Convert dense activations into sparse feature records",
                "why": f"The planned {artifact_format} evidence layer is smaller after feature extraction.",
            },
        )
    if profile in {"remote-api", "frontier-lab", "cluster"}:
        actions.append(
            {
                "id": "publish_manifest",
                "title": "Publish or archive manifests next to artifacts",
                "command": "interp-lab publish-hf-artifact --repo-id <repo> --path <artifact-dir>",
            }
        )
    return actions


def parse_count_int(value: str | int) -> int:
    return int(parse_count_float(value))


def parse_count_float(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().replace("_", "").lower()
    if not text:
        raise argparse.ArgumentTypeError("value is required")
    suffix = text[-1]
    if suffix in COUNT_SUFFIXES:
        return float(text[:-1]) * COUNT_SUFFIXES[suffix]
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse count {value!r}") from exc


def parse_bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().replace("_", "").lower()
    if not text:
        raise argparse.ArgumentTypeError("value is required")
    for suffix in sorted(BYTE_SUFFIXES, key=len, reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * BYTE_SUFFIXES[suffix])
    try:
        return int(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse byte size {value!r}") from exc


def parse_shards(value: str | int) -> int | None:
    if isinstance(value, int):
        return value
    if value.strip().lower() == "auto":
        return None
    return parse_count_int(value)


def _resolve_profile(profile: str, model_params: float, activation_bytes: int) -> tuple[str, str]:
    if profile != "auto":
        return profile, "selected explicitly"
    if model_params >= 1e12 or activation_bytes >= 100 * 1024**4:
        return "frontier-lab", "auto-selected for frontier-scale model or activation volume"
    if activation_bytes >= 2 * 1024**4:
        return "cluster", "auto-selected for multi-terabyte activation volume"
    if activation_bytes >= 100 * 1024**3:
        return "single-gpu", "auto-selected for medium activation volume"
    return "local-cpu", "auto-selected for small activation volume"


def _assumptions(*, dtype_bytes: int, sparse_event_bytes: int, artifact_format: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "dense_activation_storage",
            "formula": "tokens * d_model * selected_layers * dtype_bytes",
            "dtype_bytes": dtype_bytes,
        },
        {
            "id": "sparse_feature_record_storage",
            "formula": "tokens * selected_layers * top_k_active * sparse_event_bytes",
            "artifact_format": artifact_format,
            "sparse_event_bytes": sparse_event_bytes,
        },
        {
            "id": "sae_parameter_storage",
            "formula": "d_model * latent_dim * 2 * dtype_bytes",
            "note": "The factor of 2 covers encoder and decoder weights.",
        },
        {
            "id": "causal_forward_passes",
            "formula": "causal_prompts * (1 + causal_features * interventions_per_feature)",
            "note": "The 1 covers a shared baseline pass per prompt.",
        },
    ]


def _byte_estimate(value: int) -> dict[str, Any]:
    return {"bytes": value, "human": _format_bytes(value)}


def _ceil_div(left: int, right: int) -> int:
    return -(-left // max(right, 1))


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} EB"
