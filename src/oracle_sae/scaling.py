from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScalePlan:
    model_params: float
    tokens: int
    d_model: int
    selected_layers: int
    latent_dim: int
    dtype: str
    shards: int

    def to_dict(self) -> dict[str, Any]:
        dtype_bytes = DTYPE_BYTES[self.dtype]
        activation_bytes = self.tokens * self.d_model * self.selected_layers * dtype_bytes
        sae_parameter_bytes = self.d_model * self.latent_dim * 2 * dtype_bytes
        return {
            "schema_version": "interp-lab.scale_plan.v1",
            "model_params": self.model_params,
            "tokens": self.tokens,
            "d_model": self.d_model,
            "selected_layers": self.selected_layers,
            "latent_dim": self.latent_dim,
            "dtype": self.dtype,
            "dtype_bytes": dtype_bytes,
            "shards": self.shards,
            "activation_storage_bytes": activation_bytes,
            "activation_storage_human": _format_bytes(activation_bytes),
            "activation_storage_per_shard_bytes": activation_bytes // max(self.shards, 1),
            "activation_storage_per_shard_human": _format_bytes(activation_bytes // max(self.shards, 1)),
            "sae_parameter_bytes": sae_parameter_bytes,
            "sae_parameter_human": _format_bytes(sae_parameter_bytes),
            "recommendations": recommendations_for_plan(
                model_params=self.model_params,
                activation_bytes=activation_bytes,
                shards=self.shards,
            ),
        }


def build_scale_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate storage and execution shape for large interp-lab jobs.")
    parser.add_argument("--model-params", type=float, required=True, help="Model parameter count, e.g. 1e12.")
    parser.add_argument("--tokens", type=int, required=True, help="Activation tokens to process.")
    parser.add_argument("--d-model", type=int, required=True, help="Hidden width of the activation stream.")
    parser.add_argument("--selected-layers", type=int, default=1, help="Number of layers or hook points captured.")
    parser.add_argument("--latent-dim", type=int, default=131072, help="SAE latent width.")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument("--shards", type=int, default=256, help="Number of activation shards.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def run_scale_plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    plan = ScalePlan(
        model_params=args.model_params,
        tokens=args.tokens,
        d_model=args.d_model,
        selected_layers=args.selected_layers,
        latent_dim=args.latent_dim,
        dtype=args.dtype,
        shards=args.shards,
    ).to_dict()
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(render_scale_plan(plan))
    return plan


def render_scale_plan(plan: dict[str, Any]) -> str:
    lines = [
        "interp-lab scale plan",
        "",
        f"Model parameters: {plan['model_params']:.3g}",
        f"Activation tokens: {plan['tokens']:,}",
        f"Captured hook points: {plan['selected_layers']}",
        f"Activation storage: {plan['activation_storage_human']}",
        f"Per shard: {plan['activation_storage_per_shard_human']} across {plan['shards']} shard(s)",
        f"SAE parameter storage: {plan['sae_parameter_human']}",
        "",
        "Recommendations:",
    ]
    for item in plan["recommendations"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def recommendations_for_plan(*, model_params: float, activation_bytes: int, shards: int) -> list[str]:
    recommendations = [
        "Use activation-record JSONL or sharded tensor manifests as the boundary between model execution and interp-lab ranking.",
        "Keep model execution behind HF, TransformerLens, NNsight, Goodfire, or a custom remote adapter.",
        "Run causal validation as resumable batches and merge intervention records after each batch.",
    ]
    if model_params >= 1e12:
        recommendations.append("Use remote inference or colocated activation harvesting for 1T+ models.")
    if activation_bytes > 10 * 1024**4:
        recommendations.append("Store activations in shards and stream them into SAE training or feature ranking.")
    if shards < 128 and activation_bytes > 1024**4:
        recommendations.append("Increase shard count so retries and uploads stay granular.")
    return recommendations


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
