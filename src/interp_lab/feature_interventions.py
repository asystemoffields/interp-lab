from __future__ import annotations

import argparse
import importlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interp_lab.hf_hooks import register_hidden_ablations, register_hidden_steering
from interp_lab.hf_interventions import (
    DEFAULT_TARGET_TOKENS,
    FEATURE_PATTERN,
    parse_target_tokens,
    resolve_target_token_ids,
    target_token_strategy,
)
from interp_lab.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from interp_lab.hf_records import PromptRecord, load_prompt_records, split_prompt_record_indexes
from interp_lab.hf_sae_paths import SAE_FEATURE_PATTERN, parse_sae_feature_ref
from interp_lab.reporting import load_inspection_report

DEFAULT_STEERING_STRENGTHS = [1.0, 3.0, 10.0]
INTERVENTION_SCHEMA = "interp-lab.intervention_result.v1"
PLAN_SCHEMA = "interp-lab.intervention_plan.v1"


@dataclass(frozen=True)
class FeatureInterventionResult:
    records_path: Path | None
    plan_path: Path | None
    plan: dict[str, Any]
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTION_SCHEMA,
            "dry_run": self.dry_run,
            "records_path": str(self.records_path) if self.records_path is not None else None,
            "plan_path": str(self.plan_path) if self.plan_path is not None else None,
            "plan": self.plan,
        }


@dataclass(frozen=True)
class _FeatureSpec:
    feature_id: str
    kind: str
    layer: int
    dimension: int | None = None
    latent_index: int | None = None
    label: str = ""


def build_intervene_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Amplify, suppress, or ablate selected features and write intervention records."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--criterion", required=True, help="Criterion text stored in intervention rows.")
    parser.add_argument("--out", required=True, help="Output intervention-record JSONL path.")
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Feature id to intervene on. Repeat for multiple features, e.g. L6:D512 or SAE:L6:F30.",
    )
    parser.add_argument(
        "--report",
        help="Optional inspection report JSON. Top report features are used when --feature is omitted.",
    )
    parser.add_argument(
        "--records",
        help="Optional activation-record JSONL used to generate a directly runnable inspect next-action command.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top report features to use when --report is set.")
    parser.add_argument(
        "--sae",
        help="interp-lab SAE artifact JSON required for SAE:* latent interventions.",
    )
    parser.add_argument(
        "--mode",
        choices=["amplify", "suppress", "ablate"],
        default="suppress",
        help="Feature edit to test. Amplify/suppress steer residual directions; ablate sets hidden dimensions.",
    )
    parser.add_argument(
        "--strength-sweep",
        help="Comma-separated steering strengths. Sign is inferred from --mode.",
    )
    parser.add_argument("--ablate-value", type=float, default=0.0)
    parser.add_argument(
        "--target-token",
        action="append",
        help="Target token, comma-separated tokens, or auto. Defaults to the shared interp-lab behavior token set.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--plan-out",
        help="Optional JSON path for a machine-readable intervention plan/manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/print the plan without loading the model or running interventions.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    add_hf_loading_args(parser)
    return parser


def run_intervene_from_args(args: argparse.Namespace) -> FeatureInterventionResult:
    try:
        loading_options = hf_loading_options_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return intervene_on_features(
        model_name=args.model,
        dataset_path=args.dataset,
        criterion=args.criterion,
        out_path=args.out,
        features=args.feature,
        report_path=args.report,
        activation_records_path=args.records,
        top_k=args.top_k,
        sae_path=args.sae,
        mode=args.mode,
        strength_sweep=parse_strength_sweep(args.strength_sweep),
        ablate_value=args.ablate_value,
        target_tokens=parse_target_tokens(args.target_token),
        device=args.device,
        max_length=args.max_length,
        plan_out=args.plan_out,
        dry_run=args.dry_run,
        **loading_options,
    )


def intervene_on_features(
    *,
    model_name: str,
    dataset_path: str | Path,
    criterion: str,
    out_path: str | Path,
    features: list[str] | None = None,
    report_path: str | Path | None = None,
    activation_records_path: str | Path | None = None,
    top_k: int = 8,
    sae_path: str | Path | None = None,
    mode: str = "suppress",
    strength_sweep: list[float] | None = None,
    ablate_value: float = 0.0,
    target_tokens: list[str] | None = None,
    device: str = "cpu",
    max_length: int = 128,
    plan_out: str | Path | None = None,
    dry_run: bool = False,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> FeatureInterventionResult:
    prompts = load_prompt_records(dataset_path)
    if not prompts:
        raise ValueError(f"{dataset_path}: no prompt records found")
    positive_indexes, negative_indexes = split_prompt_record_indexes(prompts)
    if not positive_indexes:
        raise ValueError("interventions need at least one positive-scored prompt")
    specs = resolve_feature_specs(
        features=features,
        report_path=report_path,
        top_k=top_k,
        sae_path=sae_path,
    )
    if not specs:
        raise ValueError("No features selected. Pass --feature or --report.")
    if mode == "ablate" and any(spec.kind == "sae" for spec in specs):
        raise ValueError("SAE latents cannot be directly ablated yet; use --mode suppress or --mode amplify")
    strengths = resolve_strengths(mode=mode, strength_sweep=strength_sweep)
    plan = build_intervention_plan(
        model_name=model_name,
        dataset_path=dataset_path,
        criterion=criterion,
        out_path=out_path,
        specs=specs,
        mode=mode,
        strengths=strengths,
        ablate_value=ablate_value,
        target_tokens=target_tokens,
        report_path=report_path,
        activation_records_path=activation_records_path,
        sae_path=sae_path,
        prompt_count=len(prompts),
        positive_count=len(positive_indexes),
        negative_count=len(negative_indexes),
        device=device,
        max_length=max_length,
        dry_run=dry_run,
    )
    plan_path = write_plan(plan, plan_out)
    if dry_run:
        return FeatureInterventionResult(records_path=None, plan_path=plan_path, plan=plan, dry_run=True)

    records_path = export_feature_intervention_records(
        model_name=model_name,
        prompts=prompts,
        positive_indexes=positive_indexes,
        negative_indexes=negative_indexes,
        criterion=criterion,
        out_path=out_path,
        specs=specs,
        mode=mode,
        strengths=strengths,
        ablate_value=ablate_value,
        target_tokens=target_tokens,
        sae_path=sae_path,
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
    return FeatureInterventionResult(records_path=records_path, plan_path=plan_path, plan=plan, dry_run=False)


def resolve_feature_specs(
    *,
    features: list[str] | None,
    report_path: str | Path | None,
    top_k: int,
    sae_path: str | Path | None,
) -> list[_FeatureSpec]:
    selected: dict[str, str] = {}
    labels: dict[str, str] = {}
    explicit = [feature for feature in features or [] if feature]
    for feature_id in explicit:
        selected[feature_id] = feature_id
    if report_path is not None and not explicit:
        report = load_inspection_report(report_path)
        for card in report.cards[: max(0, top_k)]:
            selected[card.feature_id] = card.feature_id
            labels[card.feature_id] = card.label
    artifact = _load_sae_artifact(sae_path) if sae_path is not None else None
    specs = []
    for feature_id in selected:
        specs.append(_parse_feature_spec(feature_id, label=labels.get(feature_id, ""), artifact=artifact))
    return specs


def build_intervention_plan(
    *,
    model_name: str,
    dataset_path: str | Path,
    criterion: str,
    out_path: str | Path,
    specs: list[_FeatureSpec],
    mode: str,
    strengths: list[float],
    ablate_value: float,
    target_tokens: list[str] | None,
    report_path: str | Path | None,
    activation_records_path: str | Path | None,
    sae_path: str | Path | None,
    prompt_count: int,
    positive_count: int,
    negative_count: int,
    device: str,
    max_length: int,
    dry_run: bool,
) -> dict[str, Any]:
    target_strategy = target_token_strategy(target_tokens)
    display_tokens = target_tokens if target_tokens is not None else DEFAULT_TARGET_TOKENS
    plan = {
        "schema_version": PLAN_SCHEMA,
        "model": model_name,
        "criterion": criterion,
        "dataset": str(dataset_path),
        "report": str(report_path) if report_path is not None else None,
        "activation_records": str(activation_records_path) if activation_records_path is not None else None,
        "sae": str(sae_path) if sae_path is not None else None,
        "out": str(out_path),
        "mode": mode,
        "strengths": strengths,
        "ablate_value": ablate_value,
        "target_token_strategy": target_strategy,
        "target_tokens": display_tokens[:16],
        "device": device,
        "max_length": max_length,
        "dry_run": dry_run,
        "features": [_feature_plan_entry(spec) for spec in specs],
        "prompt_counts": {
            "total": prompt_count,
            "positive": positive_count,
            "negative": negative_count,
        },
        "estimated_forward_passes": _estimated_forward_passes(
            feature_count=len(specs),
            prompt_count=prompt_count,
            mode=mode,
            strength_count=len(strengths),
        ),
        "agent_actions": _agent_actions(model_name, criterion, out_path, report_path, activation_records_path),
        "advisories": _plan_advisories(specs, mode, negative_count, target_tokens),
    }
    return plan


def write_plan(plan: dict[str, Any], plan_out: str | Path | None) -> Path | None:
    if plan_out is None:
        return None
    path = Path(plan_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def export_feature_intervention_records(
    *,
    model_name: str,
    prompts: list[PromptRecord],
    positive_indexes: set[int],
    negative_indexes: set[int],
    criterion: str,
    out_path: str | Path,
    specs: list[_FeatureSpec],
    mode: str,
    strengths: list[float],
    ablate_value: float,
    target_tokens: list[str] | None,
    sae_path: str | Path | None,
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
    torch = _optional_import("torch", "Install `interp-lab[hf]` to run feature interventions.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to run feature interventions.",
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
    requested_target_tokens = target_tokens
    score_target_tokens = requested_target_tokens or DEFAULT_TARGET_TOKENS
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
    artifact = _load_sae_artifact(sae_path) if sae_path is not None else None
    hidden_size = _hidden_size(model)
    baselines = _baseline_scores(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_ids=target_ids,
        device=runtime_device,
        max_length=max_length,
        torch=torch,
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for spec in specs:
                rows = _evaluate_feature(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    baseline_scores=baselines,
                    target_ids=target_ids,
                    criterion=criterion,
                    spec=spec,
                    mode=mode,
                    strengths=strengths,
                    ablate_value=ablate_value,
                    positive_indexes=positive_indexes,
                    negative_indexes=negative_indexes,
                    device=runtime_device,
                    max_length=max_length,
                    torch=torch,
                    artifact=artifact,
                    sae_path=sae_path,
                    hidden_size=hidden_size,
                    requested_target_tokens=requested_target_tokens,
                    resolved_target_tokens=resolved_target_tokens,
                )
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output_path


def _evaluate_feature(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    baseline_scores: list[float],
    target_ids: list[int],
    criterion: str,
    spec: _FeatureSpec,
    mode: str,
    strengths: list[float],
    ablate_value: float,
    positive_indexes: set[int],
    negative_indexes: set[int],
    device: str,
    max_length: int,
    torch: Any,
    artifact: dict[str, Any] | None,
    sae_path: str | Path | None,
    hidden_size: int,
    requested_target_tokens: list[str] | None,
    resolved_target_tokens: list[str],
) -> list[dict[str, Any]]:
    if mode == "ablate":
        rows_by_strength, side_effects_by_strength = _evaluate_ablation(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            baseline_scores=baseline_scores,
            target_ids=target_ids,
            criterion=criterion,
            spec=spec,
            ablate_value=ablate_value,
            positive_indexes=positive_indexes,
            negative_indexes=negative_indexes,
            device=device,
            max_length=max_length,
            requested_target_tokens=requested_target_tokens,
            resolved_target_tokens=resolved_target_tokens,
        )
    else:
        direction = _direction_for_spec(spec, artifact=artifact, torch=torch, device=device, hidden_size=hidden_size)
        rows_by_strength, side_effects_by_strength = _evaluate_steering(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            baseline_scores=baseline_scores,
            target_ids=target_ids,
            criterion=criterion,
            spec=spec,
            mode=mode,
            strengths=strengths,
            direction=direction,
            positive_indexes=positive_indexes,
            negative_indexes=negative_indexes,
            device=device,
            max_length=max_length,
            sae_path=sae_path,
            requested_target_tokens=requested_target_tokens,
            resolved_target_tokens=resolved_target_tokens,
        )
    selected_strength, summary = _select_intervention_strength(
        rows_by_strength,
        side_effects_by_strength,
        mode=mode,
    )
    selected_side_effect = _mean(side_effects_by_strength.get(selected_strength, []))
    selected_rows = rows_by_strength[selected_strength]
    for row in selected_rows:
        row["side_effect_score"] = round(selected_side_effect, 8)
        row["metadata"]["selected_strength"] = selected_strength
        row["metadata"]["strength_sweep"] = summary
    return selected_rows


def _evaluate_ablation(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    baseline_scores: list[float],
    target_ids: list[int],
    criterion: str,
    spec: _FeatureSpec,
    ablate_value: float,
    positive_indexes: set[int],
    negative_indexes: set[int],
    device: str,
    max_length: int,
    requested_target_tokens: list[str] | None,
    resolved_target_tokens: list[str],
) -> tuple[dict[float, list[dict[str, Any]]], dict[float, list[float]]]:
    if spec.kind != "hidden":
        raise ValueError("ablation currently supports hidden-dimension features only")
    key = 0.0
    rows_by_strength: dict[float, list[dict[str, Any]]] = {key: []}
    side_effects_by_strength: dict[float, list[float]] = {key: []}
    for prompt_index, prompt in enumerate(prompts):
        score = _score_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_ids=target_ids,
            device=device,
            max_length=max_length,
            hook_factory=lambda: register_hidden_ablations(
                model,
                [(spec.layer, int(spec.dimension), ablate_value)],
            ),
        )
        if prompt_index in positive_indexes:
            rows_by_strength[key].append(
                _intervention_row(
                    model_name=model_name,
                    criterion=criterion,
                    prompt=prompt,
                    spec=spec,
                    intervention="ablate",
                    baseline_score=baseline_scores[prompt_index],
                    intervention_score=score,
                    metadata={
                        "ablate_value": ablate_value,
                        "target_token_strategy": target_token_strategy(requested_target_tokens),
                        "target_tokens": resolved_target_tokens,
                    },
                )
            )
        elif prompt_index in negative_indexes:
            side_effects_by_strength[key].append(abs(score - baseline_scores[prompt_index]))
    return rows_by_strength, side_effects_by_strength


def _evaluate_steering(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    baseline_scores: list[float],
    target_ids: list[int],
    criterion: str,
    spec: _FeatureSpec,
    mode: str,
    strengths: list[float],
    direction: Any,
    positive_indexes: set[int],
    negative_indexes: set[int],
    device: str,
    max_length: int,
    sae_path: str | Path | None,
    requested_target_tokens: list[str] | None,
    resolved_target_tokens: list[str],
) -> tuple[dict[float, list[dict[str, Any]]], dict[float, list[float]]]:
    signed_strengths = [_signed_strength(mode, strength) for strength in strengths]
    rows_by_strength: dict[float, list[dict[str, Any]]] = {strength: [] for strength in signed_strengths}
    side_effects_by_strength: dict[float, list[float]] = {strength: [] for strength in signed_strengths}
    for prompt_index, prompt in enumerate(prompts):
        for signed_strength in signed_strengths:
            score = _score_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_ids=target_ids,
                device=device,
                max_length=max_length,
                hook_factory=lambda signed_strength=signed_strength: register_hidden_steering(
                    model,
                    spec.layer,
                    direction,
                    signed_strength,
                ),
            )
            if prompt_index in positive_indexes:
                rows_by_strength[signed_strength].append(
                    _intervention_row(
                        model_name=model_name,
                        criterion=criterion,
                        prompt=prompt,
                        spec=spec,
                        intervention=mode,
                        baseline_score=baseline_scores[prompt_index],
                        intervention_score=score,
                        metadata={
                            "requested_strength": abs(float(signed_strength)),
                            "signed_strength": signed_strength,
                            "sae": str(sae_path) if spec.kind == "sae" and sae_path is not None else None,
                            "target_token_strategy": target_token_strategy(requested_target_tokens),
                            "target_tokens": resolved_target_tokens,
                        },
                    )
                )
            elif prompt_index in negative_indexes:
                side_effects_by_strength[signed_strength].append(abs(score - baseline_scores[prompt_index]))
    return rows_by_strength, side_effects_by_strength


def _baseline_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    target_ids: list[int],
    device: str,
    max_length: int,
    torch: Any,
) -> list[float]:
    with torch.no_grad():
        return [
            _score_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_ids=target_ids,
                device=device,
                max_length=max_length,
            )
            for prompt in prompts
        ]


def _score_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    target_ids: list[int],
    device: str,
    max_length: int,
    hook_factory: Any | None = None,
) -> float:
    encoded = tokenizer(prompt.text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hook = hook_factory() if hook_factory is not None else None
    try:
        outputs = model(**encoded, use_cache=False)
    finally:
        if hook is not None:
            hook.remove()
    probabilities = outputs.logits[0, -1].softmax(dim=-1)
    return round(float(probabilities[target_ids].sum().detach().cpu().item()), 8)


def _intervention_row(
    *,
    model_name: str,
    criterion: str,
    prompt: PromptRecord,
    spec: _FeatureSpec,
    intervention: str,
    baseline_score: float,
    intervention_score: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row_metadata = {
        "behavior_score": "target_token_probability_mass",
        "feature_type": spec.kind,
        "layer": spec.layer,
        "label": spec.label,
    }
    if spec.dimension is not None:
        row_metadata["dimension"] = spec.dimension
    if spec.latent_index is not None:
        row_metadata["latent_index"] = spec.latent_index
    row_metadata.update({key: value for key, value in metadata.items() if value is not None})
    return {
        "schema_version": "interp-lab.intervention_record.v1",
        "model": model_name,
        "feature_id": spec.feature_id,
        "criterion": criterion,
        "intervention": intervention,
        "prompt_id": prompt.prompt_id,
        "baseline_score": baseline_score,
        "intervention_score": intervention_score,
        "metadata": row_metadata,
    }


def _select_intervention_strength(
    rows_by_strength: dict[float, list[dict[str, Any]]],
    side_effects_by_strength: dict[float, list[float]],
    *,
    mode: str,
) -> tuple[float, list[dict[str, float]]]:
    summary = []
    best_key: tuple[float, float] | None = None
    best_strength: float | None = None
    for strength, rows in rows_by_strength.items():
        directed = [_directed_delta(row, mode) for row in rows]
        side_effect = _mean(side_effects_by_strength.get(strength, []))
        mean_directed_effect = _mean(directed)
        specificity = mean_directed_effect - side_effect
        summary.append(
            {
                "strength": round(float(strength), 8),
                "mean_directed_effect": round(mean_directed_effect, 8),
                "mean_side_effect": round(side_effect, 8),
                "specificity": round(specificity, 8),
            }
        )
        # Select on the *raw* dict key, not the 8dp-rounded display value: the
        # rows are keyed by the unrounded float, so returning the rounded value
        # would KeyError when the two differ (e.g. a high-precision sweep value).
        key = (specificity, mean_directed_effect)
        if best_key is None or key > best_key:
            best_key = key
            best_strength = float(strength)
    if best_strength is None:
        raise ValueError("No intervention strengths were evaluated")
    return best_strength, summary


def _directed_delta(row: dict[str, Any], mode: str) -> float:
    baseline = float(row["baseline_score"])
    intervention = float(row["intervention_score"])
    if mode in {"suppress", "ablate"}:
        return baseline - intervention
    return intervention - baseline


def _direction_for_spec(
    spec: _FeatureSpec,
    *,
    artifact: dict[str, Any] | None,
    torch: Any,
    device: str,
    hidden_size: int,
) -> Any:
    if spec.kind == "hidden":
        dimension = int(spec.dimension)
        if dimension < 0 or dimension >= hidden_size:
            raise ValueError(f"{spec.feature_id}: dimension {dimension} is outside hidden_size={hidden_size}")
        direction = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        direction[dimension] = 1.0
        return direction
    if artifact is None:
        raise ValueError(f"{spec.feature_id}: --sae is required for SAE latent interventions")
    decoder_rows = artifact.get("decoder_weight", [])
    latent_index = int(spec.latent_index)
    if latent_index < 0 or latent_index >= len(decoder_rows):
        raise ValueError(f"{spec.feature_id}: latent index is outside decoder_weight")
    return torch.tensor(decoder_rows[latent_index], dtype=torch.float32, device=device)


def _parse_feature_spec(
    feature_id: str,
    *,
    label: str,
    artifact: dict[str, Any] | None,
) -> _FeatureSpec:
    hidden = FEATURE_PATTERN.match(feature_id)
    if hidden:
        return _FeatureSpec(
            feature_id=feature_id,
            kind="hidden",
            layer=int(hidden.group("layer")),
            dimension=int(hidden.group("dimension")),
            label=label,
        )
    if SAE_FEATURE_PATTERN.match(feature_id):
        if artifact is None:
            match = SAE_FEATURE_PATTERN.match(feature_id)
            layer_text = match.group("layer") if match is not None else None
            if layer_text is None:
                raise ValueError(f"{feature_id!r} needs --sae so interp-lab can infer the SAE layer")
            return _FeatureSpec(
                feature_id=feature_id,
                kind="sae",
                layer=int(layer_text),
                latent_index=int(match.group("latent")),
                label=label,
            )
        ref = parse_sae_feature_ref(feature_id, artifact=artifact, role="intervention", label=label)
        if ref.layer is None:
            raise ValueError(f"{feature_id!r} needs an SAE artifact with a hidden-state layer")
        return _FeatureSpec(
            feature_id=ref.feature_id,
            kind="sae",
            layer=int(ref.layer),
            latent_index=ref.latent_index,
            label=label or ref.label,
        )
    raise ValueError(f"{feature_id!r} is not a supported feature id. Use L<layer>:D<dim> or SAE:L<layer>:F<latent>.")


def _feature_plan_entry(spec: _FeatureSpec) -> dict[str, Any]:
    entry = {
        "feature_id": spec.feature_id,
        "type": spec.kind,
        "layer": spec.layer,
        "label": spec.label,
    }
    if spec.dimension is not None:
        entry["dimension"] = spec.dimension
    if spec.latent_index is not None:
        entry["latent_index"] = spec.latent_index
    return entry


def _load_sae_artifact(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("format") != "interp-lab.sae.v1":
        raise ValueError(f"{artifact_path}: expected an interp-lab SAE artifact")
    for key in ["layer", "latent_dim", "decoder_weight"]:
        if key not in artifact:
            raise ValueError(f"{artifact_path}: missing {key!r}")
    return artifact


def parse_strength_sweep(value: str | None) -> list[float] | None:
    if value is None:
        return None
    strengths = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if stripped:
            strengths.append(float(stripped))
    if not strengths:
        raise ValueError("--strength-sweep did not contain any numeric strengths")
    return strengths


def resolve_strengths(*, mode: str, strength_sweep: list[float] | None) -> list[float]:
    if mode == "ablate":
        return [0.0]
    strengths = strength_sweep or DEFAULT_STEERING_STRENGTHS
    resolved = [abs(float(strength)) for strength in strengths if float(strength) != 0.0]
    if not resolved:
        raise ValueError("steering interventions need at least one nonzero strength")
    return resolved


def _signed_strength(mode: str, strength: float) -> float:
    value = abs(float(strength))
    return -value if mode == "suppress" else value


def _hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    candidates = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidates.append(text_config)
    for candidate in candidates:
        for name in ["hidden_size", "n_embd", "d_model"]:
            value = getattr(candidate, name, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer model hidden size for hidden-dimension steering")


def _estimated_forward_passes(*, feature_count: int, prompt_count: int, mode: str, strength_count: int) -> int:
    if mode == "ablate":
        return prompt_count * (1 + feature_count)
    return prompt_count * (1 + feature_count * strength_count)


def _agent_actions(
    model_name: str,
    criterion: str,
    out_path: str | Path,
    report_path: str | Path | None,
    activation_records_path: str | Path | None,
) -> list[dict[str, Any]]:
    records_arg = str(activation_records_path) if activation_records_path is not None else "<activation-records.jsonl>"
    actions = [
        {
            "id": "inspect_with_interventions",
            "title": "Re-rank the feature report with causal evidence",
            "argv": [
                "interp-lab",
                "inspect",
                "--model",
                model_name,
                "--criterion",
                criterion,
                "--backend",
                "records",
                "--records",
                records_arg,
                "--interventions",
                str(out_path),
                "--out",
                str(Path(out_path).with_suffix("")),
            ],
        },
        {
            "id": "export_graph",
            "title": "Build an attribution graph after re-inspection",
            "argv": [
                "interp-lab",
                "export-attribution-graph",
                "--report",
                "<report.json>",
                "--out",
                str(Path(out_path).with_suffix(".graph.json")),
                "--html-out",
                str(Path(out_path).with_suffix(".graph.html")),
            ],
        },
    ]
    if report_path is not None:
        actions[0]["inputs"] = {"source_report": str(report_path)}
    if activation_records_path is not None:
        actions[0].setdefault("inputs", {})["activation_records"] = str(activation_records_path)
    else:
        actions[0]["requires"] = ["activation records JSONL from the original inspection run"]
    for action in actions:
        action["command"] = _format_command(action["argv"])
    return actions


def _plan_advisories(
    specs: list[_FeatureSpec],
    mode: str,
    negative_count: int,
    target_tokens: list[str] | None,
) -> list[str]:
    advisories = []
    if negative_count == 0:
        advisories.append("Add negative/control prompts to estimate side effects.")
    if target_tokens is None:
        advisories.append("Default target tokens are generic; use --target-token auto or explicit behavior tokens when possible.")
    if mode == "suppress" and any(spec.kind == "sae" for spec in specs):
        advisories.append("SAE suppression is decoder-direction steering, not an exact latent clamp.")
    if any(spec.kind == "hidden" and spec.layer == 0 for spec in specs):
        advisories.append("Layer 0 hidden-dimension edits may not map to a decoder block hook on all architectures.")
    return advisories


def _format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item).replace("\\", "/")) for item in argv)


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
