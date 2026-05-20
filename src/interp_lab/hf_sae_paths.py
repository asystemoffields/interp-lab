from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interp_lab.hf_contrast import parse_strength_sweep
from interp_lab.hf_hooks import register_hidden_steering
from interp_lab.hf_interventions import (
    DEFAULT_TARGET_TOKENS,
    parse_target_tokens,
    resolve_target_token_ids,
    target_token_strategy,
)
from interp_lab.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from interp_lab.hf_records import _pool_hidden_state, load_prompt_records
from interp_lab.reporting import load_inspection_report
from interp_lab.sae_training import encode_with_artifact

SAE_FEATURE_PATTERN = re.compile(r"^SAE:(?:L(?P<layer>\d+):)?F(?P<latent>\d+)$")
DEFAULT_PATH_STRENGTHS = [-4.0, -2.0, 2.0, 4.0]


@dataclass(frozen=True)
class SaeFeatureRef:
    feature_id: str
    layer: int | None
    latent_index: int
    label: str = ""
    signed_effect: float | None = None
    strong_causal_score: float | None = None


def export_hf_sae_path_records(
    *,
    model_name: str,
    dataset_path: str | Path,
    source_artifact_path: str | Path,
    target_artifact_path: str | Path,
    out_path: str | Path,
    criterion: str,
    source_features: list[str] | None = None,
    target_features: list[str] | None = None,
    path_pairs: list[tuple[str, str]] | None = None,
    source_report_path: str | Path | None = None,
    target_report_path: str | Path | None = None,
    source_top_k: int = 4,
    target_top_k: int = 8,
    pool: str = "last",
    strength_sweep: list[float] | None = None,
    random_source_controls: int = 0,
    control_seed: int = 0,
    score_behavior: bool = True,
    target_tokens: list[str] | None = None,
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
    torch = _optional_import("torch", "Install `interp-lab[hf]` to run SAE path patching.")
    transformers = _optional_import(
        "transformers",
        "Install `interp-lab[hf]` to run SAE path patching.",
    )
    source_artifact = _load_sae_artifact(source_artifact_path)
    target_artifact = _load_sae_artifact(target_artifact_path)
    source_layer = _artifact_layer(source_artifact, "source")
    target_layer = _artifact_layer(target_artifact, "target")
    if source_layer <= 0:
        raise ValueError("SAE path patching needs a source SAE from a decoder block output layer, not embeddings")
    if source_layer >= target_layer:
        raise ValueError("SAE path patching requires the source SAE layer to be earlier than the target SAE layer")

    normalized_path_pairs = _resolve_path_pairs(
        path_pairs or [],
        source_artifact=source_artifact,
        target_artifact=target_artifact,
    )
    pair_source_features = [source for source, _ in normalized_path_pairs]
    pair_target_features = [target for _, target in normalized_path_pairs]
    source_refs = resolve_sae_feature_refs(
        explicit_features=[*(source_features or []), *pair_source_features],
        report_path=source_report_path,
        artifact=source_artifact,
        top_k=source_top_k,
        role="source",
    )
    target_refs = resolve_sae_feature_refs(
        explicit_features=[*(target_features or []), *pair_target_features],
        report_path=target_report_path,
        artifact=target_artifact,
        top_k=target_top_k,
        role="target",
    )
    if not source_refs:
        raise ValueError("No source SAE features were selected")
    if not target_refs:
        raise ValueError("No target SAE features were selected")

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

    requested_target_tokens = target_tokens
    score_target_tokens = requested_target_tokens or DEFAULT_TARGET_TOKENS
    target_ids: list[int] = []
    resolved_target_tokens: list[str] = []
    if score_behavior:
        target_ids, resolved_target_tokens = resolve_target_token_ids(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            target_tokens=score_target_tokens,
            device=runtime_device,
            max_length=max_length,
        )
        if not target_ids:
            raise ValueError("No target token ids resolved for path-patching behavior scoring")

    strengths = strength_sweep or DEFAULT_PATH_STRENGTHS
    decoder_rows = list(source_artifact["decoder_weight"])
    allowed_targets_by_source = _allowed_targets_by_source(normalized_path_pairs)
    control_refs_by_source = {
        source_ref.feature_id: _random_source_control_refs(
            source_ref,
            artifact=source_artifact,
            count=random_source_controls,
            seed=control_seed,
        )
        for source_ref in source_refs
    }
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for prompt in prompts:
                baseline = _run_path_forward(
                    model=model,
                    tokenizer=tokenizer,
                    text=prompt.text,
                    target_artifact=target_artifact,
                    target_layer=target_layer,
                    pool=pool,
                    target_ids=target_ids,
                    direction=None,
                    source_layer=source_layer,
                    strength=0.0,
                    device=runtime_device,
                    max_length=max_length,
                )
                for source_ref in source_refs:
                    target_refs_for_source = _target_refs_for_source(
                        source_ref,
                        target_refs,
                        allowed_targets_by_source=allowed_targets_by_source,
                    )
                    if not target_refs_for_source:
                        continue
                    direction = torch.tensor(
                        decoder_rows[source_ref.latent_index],
                        dtype=torch.float32,
                        device=runtime_device,
                    )
                    for strength in strengths:
                        patched = _run_path_forward(
                            model=model,
                            tokenizer=tokenizer,
                            text=prompt.text,
                            target_artifact=target_artifact,
                            target_layer=target_layer,
                            pool=pool,
                            target_ids=target_ids,
                            direction=direction,
                            source_layer=source_layer,
                            strength=strength,
                            device=runtime_device,
                            max_length=max_length,
                        )
                        for target_ref in target_refs_for_source:
                            row = _path_row(
                                model_name=model_name,
                                criterion=criterion,
                                prompt_id=prompt.prompt_id,
                                criterion_score=prompt.criterion_score,
                                source_ref=source_ref,
                                target_ref=target_ref,
                                strength=strength,
                                pool=pool,
                                baseline_activation=baseline.latents[target_ref.latent_index],
                                patched_activation=patched.latents[target_ref.latent_index],
                                baseline_score=baseline.behavior_score,
                                patched_score=patched.behavior_score,
                                source_artifact_path=source_artifact_path,
                                target_artifact_path=target_artifact_path,
                                target_token_strategy_value=target_token_strategy(requested_target_tokens)
                                if score_behavior
                                else "disabled",
                                target_tokens=resolved_target_tokens,
                            )
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        for control_ref in control_refs_by_source[source_ref.feature_id]:
                            control_direction = torch.tensor(
                                decoder_rows[control_ref.latent_index],
                                dtype=torch.float32,
                                device=runtime_device,
                            )
                            control_patched = _run_path_forward(
                                model=model,
                                tokenizer=tokenizer,
                                text=prompt.text,
                                target_artifact=target_artifact,
                                target_layer=target_layer,
                                pool=pool,
                                target_ids=target_ids,
                                direction=control_direction,
                                source_layer=source_layer,
                                strength=strength,
                                device=runtime_device,
                                max_length=max_length,
                            )
                            for target_ref in target_refs_for_source:
                                row = _path_row(
                                    model_name=model_name,
                                    criterion=criterion,
                                    prompt_id=prompt.prompt_id,
                                    criterion_score=prompt.criterion_score,
                                    source_ref=source_ref,
                                    target_ref=target_ref,
                                    strength=strength,
                                    pool=pool,
                                    baseline_activation=baseline.latents[target_ref.latent_index],
                                    patched_activation=control_patched.latents[target_ref.latent_index],
                                    baseline_score=baseline.behavior_score,
                                    patched_score=control_patched.behavior_score,
                                    source_artifact_path=source_artifact_path,
                                    target_artifact_path=target_artifact_path,
                                    target_token_strategy_value=target_token_strategy(requested_target_tokens)
                                    if score_behavior
                                    else "disabled",
                                    target_tokens=resolved_target_tokens,
                                    control_type="random_source",
                                    control_source_ref=control_ref,
                                )
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output_path


def resolve_sae_feature_refs(
    *,
    explicit_features: list[str] | None,
    report_path: str | Path | None,
    artifact: dict[str, Any],
    top_k: int,
    role: str,
) -> list[SaeFeatureRef]:
    layer = artifact.get("layer")
    selected: dict[str, SaeFeatureRef] = {}
    if report_path is not None:
        report = load_inspection_report(report_path)
        for card in report.cards[:top_k]:
            ref = parse_sae_feature_ref(
                card.feature_id,
                artifact=artifact,
                role=role,
                label=card.label,
                signed_effect=_optional_float(
                    card.causal_effects.get(
                        "signed_causal_effect",
                        card.causal_effects.get("signed_association"),
                    )
                ),
                strong_causal_score=_optional_float(card.causal_effects.get("strong_causal_score")),
            )
            selected[ref.feature_id] = ref
    for feature_id in explicit_features or []:
        ref = parse_sae_feature_ref(feature_id, artifact=artifact, role=role)
        selected[ref.feature_id] = ref
    if not selected and layer is not None and int(top_k) > 0:
        for latent_index in range(min(int(top_k), int(artifact["latent_dim"]))):
            feature_id = f"SAE:L{int(layer)}:F{latent_index}"
            selected[feature_id] = SaeFeatureRef(feature_id=feature_id, layer=int(layer), latent_index=latent_index)
    return list(selected.values())


def parse_sae_feature_ref(
    feature_id: str,
    *,
    artifact: dict[str, Any],
    role: str,
    label: str = "",
    signed_effect: float | None = None,
    strong_causal_score: float | None = None,
) -> SaeFeatureRef:
    match = SAE_FEATURE_PATTERN.match(feature_id)
    if not match:
        raise ValueError(f"{role} feature {feature_id!r} is not an SAE feature id like SAE:L12:F3")
    artifact_layer = artifact.get("layer")
    parsed_layer = int(match.group("layer")) if match.group("layer") is not None else None
    if artifact_layer is not None:
        artifact_layer = int(artifact_layer)
        if parsed_layer is not None and parsed_layer != artifact_layer:
            raise ValueError(
                f"{role} feature {feature_id!r} belongs to layer {parsed_layer}, "
                f"but the {role} SAE artifact is for layer {artifact_layer}"
            )
        parsed_layer = artifact_layer
    latent_index = int(match.group("latent"))
    latent_dim = int(artifact["latent_dim"])
    if latent_index < 0 or latent_index >= latent_dim:
        raise ValueError(f"{role} feature {feature_id!r} is outside latent_dim={latent_dim}")
    canonical_id = feature_id if parsed_layer is None else f"SAE:L{parsed_layer}:F{latent_index}"
    return SaeFeatureRef(
        feature_id=canonical_id,
        layer=parsed_layer,
        latent_index=latent_index,
        label=label,
        signed_effect=signed_effect,
        strong_causal_score=strong_causal_score,
    )


def parse_path_pair(value: str) -> tuple[str, str]:
    for separator in ["=", ",", "->"]:
        if separator in value:
            source, target = value.split(separator, 1)
            source = source.strip()
            target = target.strip()
            if source and target:
                return source, target
    raise ValueError("Path pairs must look like SOURCE=TARGET, SOURCE,TARGET, or SOURCE->TARGET")


def build_hf_sae_paths_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch source SAE latents and measure downstream SAE latent paths.")
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Prompt JSONL with text and criterion_score.")
    parser.add_argument("--criterion", required=True, help="Criterion text stored in path records.")
    parser.add_argument("--source-sae", required=True, help="Source SAE artifact JSON.")
    parser.add_argument("--target-sae", required=True, help="Target/downstream SAE artifact JSON.")
    parser.add_argument("--out", required=True, help="Output path-patch JSONL path.")
    parser.add_argument("--source-feature", action="append", default=[], help="Source SAE feature id. Repeatable.")
    parser.add_argument("--target-feature", action="append", default=[], help="Target SAE feature id. Repeatable.")
    parser.add_argument(
        "--path-pair",
        action="append",
        default=[],
        help="Specific source-to-target SAE path to measure, such as SAE:L6:F1=SAE:L10:F4. Repeatable.",
    )
    parser.add_argument("--source-report", help="Report JSON used to select source features.")
    parser.add_argument("--target-report", help="Report JSON used to select target features.")
    parser.add_argument("--source-top-k", type=int, default=4)
    parser.add_argument("--target-top-k", type=int, default=8)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--strength-sweep", help="Comma-separated signed source steering strengths.")
    parser.add_argument(
        "--random-source-controls",
        type=int,
        default=0,
        help="Random source SAE latents to patch as controls for each selected source feature.",
    )
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--target-token", action="append")
    parser.add_argument("--skip-behavior-score", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    add_hf_loading_args(parser)
    return parser


def run_hf_sae_paths_from_args(args: argparse.Namespace) -> Path:
    try:
        loading_options = hf_loading_options_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return export_hf_sae_path_records(
        model_name=args.model,
        dataset_path=args.dataset,
        source_artifact_path=args.source_sae,
        target_artifact_path=args.target_sae,
        out_path=args.out,
        criterion=args.criterion,
        source_features=args.source_feature,
        target_features=args.target_feature,
        path_pairs=parse_path_pairs(args.path_pair),
        source_report_path=args.source_report,
        target_report_path=args.target_report,
        source_top_k=args.source_top_k,
        target_top_k=args.target_top_k,
        pool=args.pool,
        strength_sweep=parse_strength_sweep(args.strength_sweep),
        random_source_controls=args.random_source_controls,
        control_seed=args.control_seed,
        score_behavior=not args.skip_behavior_score,
        target_tokens=parse_target_tokens(args.target_token),
        device=args.device,
        max_length=args.max_length,
        **loading_options,
    )


def parse_path_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    pairs = []
    for value in values or []:
        pairs.append(parse_path_pair(value))
    return pairs


@dataclass(frozen=True)
class _PathForwardResult:
    latents: list[float]
    behavior_score: float | None


def _run_path_forward(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    target_artifact: dict[str, Any],
    target_layer: int,
    pool: str,
    target_ids: list[int],
    direction: Any | None,
    source_layer: int,
    strength: float,
    device: str,
    max_length: int,
) -> _PathForwardResult:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    handle = None
    if direction is not None:
        handle = register_hidden_steering(model, source_layer, direction, strength)
    try:
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    vector = _pool_hidden_state(outputs.hidden_states[target_layer], encoded.get("attention_mask"), pool=pool)
    latents = encode_with_artifact([vector], target_artifact)[0]
    behavior_score = None
    if target_ids:
        probabilities = outputs.logits[0, -1].softmax(dim=-1)
        behavior_score = round(float(probabilities[target_ids].sum().detach().cpu().item()), 8)
    return _PathForwardResult(latents=latents, behavior_score=behavior_score)


def _path_row(
    *,
    model_name: str,
    criterion: str,
    prompt_id: str,
    criterion_score: float,
    source_ref: SaeFeatureRef,
    target_ref: SaeFeatureRef,
    strength: float,
    pool: str,
    baseline_activation: float,
    patched_activation: float,
    baseline_score: float | None,
    patched_score: float | None,
    source_artifact_path: str | Path,
    target_artifact_path: str | Path,
    target_token_strategy_value: str,
    target_tokens: list[str],
    control_type: str | None = None,
    control_source_ref: SaeFeatureRef | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "interp-lab.path_patch.v1",
        "model": model_name,
        "criterion": criterion,
        "prompt_id": prompt_id,
        "criterion_score": round(float(criterion_score), 8),
        "source_feature_id": source_ref.feature_id,
        "target_feature_id": target_ref.feature_id,
        "source_layer": source_ref.layer,
        "target_layer": target_ref.layer,
        "source_latent": source_ref.latent_index,
        "target_latent": target_ref.latent_index,
        "intervention": "source_sae_steer",
        "strength": round(float(strength), 8),
        "pool": pool,
        "baseline_target_activation": round(float(baseline_activation), 8),
        "patched_target_activation": round(float(patched_activation), 8),
        "target_activation_delta": round(float(patched_activation - baseline_activation), 8),
        "metadata": {
            "source_label": source_ref.label,
            "target_label": target_ref.label,
            "source_signed_effect": source_ref.signed_effect,
            "source_strong_causal_score": source_ref.strong_causal_score,
            "target_signed_effect": target_ref.signed_effect,
            "target_strong_causal_score": target_ref.strong_causal_score,
            "source_sae": str(source_artifact_path),
            "target_sae": str(target_artifact_path),
            "target_token_strategy": target_token_strategy_value,
            "target_tokens": target_tokens,
        },
    }
    if control_type is not None:
        row["metadata"]["control_type"] = control_type
    if control_source_ref is not None:
        row["metadata"]["control_source_feature_id"] = control_source_ref.feature_id
        row["metadata"]["control_source_latent"] = control_source_ref.latent_index
    if baseline_score is not None and patched_score is not None:
        row["baseline_score"] = round(float(baseline_score), 8)
        row["patched_score"] = round(float(patched_score), 8)
        row["score_delta"] = round(float(patched_score - baseline_score), 8)
    return row


def _random_source_control_refs(
    source_ref: SaeFeatureRef,
    *,
    artifact: dict[str, Any],
    count: int,
    seed: int,
) -> list[SaeFeatureRef]:
    if count <= 0:
        return []
    latent_dim = int(artifact["latent_dim"])
    candidates = [index for index in range(latent_dim) if index != source_ref.latent_index]
    if not candidates:
        return []
    rng = random.Random(f"{seed}:{source_ref.feature_id}")
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    if count > len(selected):
        selected.extend(rng.choice(candidates) for _ in range(count - len(selected)))
    layer = int(artifact["layer"]) if artifact.get("layer") is not None else None
    return [
        SaeFeatureRef(
            feature_id=f"SAE:L{layer}:F{latent_index}" if layer is not None else f"SAE:F{latent_index}",
            layer=layer,
            latent_index=latent_index,
            label="random source control",
        )
        for latent_index in selected
    ]


def _resolve_path_pairs(
    path_pairs: list[tuple[str, str]],
    *,
    source_artifact: dict[str, Any],
    target_artifact: dict[str, Any],
) -> list[tuple[str, str]]:
    normalized = []
    for source_feature, target_feature in path_pairs:
        source_ref = parse_sae_feature_ref(source_feature, artifact=source_artifact, role="source")
        target_ref = parse_sae_feature_ref(target_feature, artifact=target_artifact, role="target")
        normalized.append((source_ref.feature_id, target_ref.feature_id))
    return normalized


def _allowed_targets_by_source(path_pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for source, target in path_pairs:
        allowed.setdefault(source, set()).add(target)
    return allowed


def _target_refs_for_source(
    source_ref: SaeFeatureRef,
    target_refs: list[SaeFeatureRef],
    *,
    allowed_targets_by_source: dict[str, set[str]],
) -> list[SaeFeatureRef]:
    allowed_targets = allowed_targets_by_source.get(source_ref.feature_id)
    if allowed_targets is None:
        if allowed_targets_by_source:
            return []
        return target_refs
    return [target_ref for target_ref in target_refs if target_ref.feature_id in allowed_targets]


def _load_sae_artifact(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("format") != "interp-lab.sae.v1":
        raise ValueError(f"{path}: expected an interp-lab SAE artifact")
    for key in ["layer", "latent_dim", "decoder_weight", "encoder_weight", "encoder_bias", "mean"]:
        if key not in artifact:
            raise ValueError(f"{path}: missing {key!r}")
    return artifact


def _artifact_layer(artifact: dict[str, Any], role: str) -> int:
    layer = artifact.get("layer")
    if layer is None:
        raise ValueError(f"{role} SAE artifact must include a hidden-state layer")
    return int(layer)


def _optional_import(name: str, message: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
