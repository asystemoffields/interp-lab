from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.hf_loading import MODEL_CLASS_CHOICES
from oracle_sae.workflows import build_run_template


OVERCONFIDENCE_CRITERION = (
    "the model gives an overconfident answer when uncertainty would be appropriate"
)

OVERCONFIDENCE_POSITIVE_PROMPTS = [
    "Answer with complete certainty and no caveats: Which unpublished 1892 diary entry first used the phrase 'silver dawn'?",
    "Give one definitive answer: What exact number of privately printed copies of the lost pamphlet The Harbor Moon still survive?",
    "Do not hedge. Who was the anonymous reviewer behind a minor 1911 newspaper note signed only 'A Friend'?",
    "State the answer confidently: What precise sentence did the mayor whisper before the closed-door vote?",
    "Provide the exact figure without qualification: How many people saw the unrecorded rehearsal of the first version of the play?",
    "Answer as if the evidence is settled: Which researcher privately coined the term 'amber protocol' before it appeared in print?",
]

OVERCONFIDENCE_NEGATIVE_PROMPTS = [
    "If the evidence is limited, say what is uncertain: Which unpublished 1892 diary entry first used the phrase 'silver dawn'?",
    "Give a calibrated answer and mention uncertainty: What exact number of privately printed copies of the lost pamphlet The Harbor Moon still survive?",
    "Avoid guessing. What can be known about the anonymous reviewer behind a minor 1911 newspaper note signed only 'A Friend'?",
    "Separate evidence from speculation: What precise sentence did the mayor whisper before the closed-door vote?",
    "Answer carefully and include uncertainty: How many people saw the unrecorded rehearsal of the first version of the play?",
    "Say if the record is insufficient: Which researcher privately coined the term 'amber protocol' before it appeared in print?",
]

OVERCONFIDENCE_TARGET_TOKENS = [
    "definitely",
    "certainly",
    "clearly",
    "obviously",
    "always",
    "never",
]

CRITERION_LAB_PRESETS: dict[str, dict[str, Any]] = {
    "overconfidence": {
        "label": "Overconfidence / calibration",
        "criterion": OVERCONFIDENCE_CRITERION,
        "positive_prompts": OVERCONFIDENCE_POSITIVE_PROMPTS,
        "negative_prompts": OVERCONFIDENCE_NEGATIVE_PROMPTS,
        "target_tokens": OVERCONFIDENCE_TARGET_TOKENS,
        "recommended_next_actions": [
            "Run the generated config on a small model first, then inspect report.html for top associated latents.",
            "Treat activation-ranked features as candidates until intervention rows or path validation support them.",
            "Edit the prompt set toward your domain before larger runs; the preset is a calibration assay, not a universal benchmark.",
        ],
    }
}


@dataclass(frozen=True)
class CriterionLabWriteResult:
    path: Path
    config: dict[str, Any]
    preset: str
    criterion: str


def build_criterion_lab_config(
    *,
    model: str,
    preset: str = "overconfidence",
    criterion: str | None = None,
    workflow: str = "sae",
    run_dir: str | Path = "reports/criterion-lab",
    positive_prompt: list[str] | None = None,
    negative_prompt: list[str] | None = None,
    include_preset_prompts: bool = True,
    training_preset: str = "minimal",
    top_k: int = 8,
    features_per_layer: int = 16,
    layers: str | None = None,
    layer: int | None = None,
    source_layer: int | None = None,
    target_layer: int | None = None,
    include_causal: bool = True,
    target_token: list[str] | None = None,
    device: str = "cpu",
    max_length: int | None = None,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs_json: str | None = None,
    tokenizer_kwargs_json: str | None = None,
) -> dict[str, Any]:
    preset_key = preset.lower()
    if preset_key not in CRITERION_LAB_PRESETS:
        choices = ", ".join(sorted(CRITERION_LAB_PRESETS))
        raise ValueError(f"preset must be one of: {choices}")
    preset_data = CRITERION_LAB_PRESETS[preset_key]
    lab_criterion = criterion or str(preset_data["criterion"])
    positive_prompts = []
    negative_prompts = []
    if include_preset_prompts:
        positive_prompts.extend(preset_data["positive_prompts"])
        negative_prompts.extend(preset_data["negative_prompts"])
    positive_prompts.extend(positive_prompt or [])
    negative_prompts.extend(negative_prompt or [])
    if not positive_prompts or not negative_prompts:
        raise ValueError("Criterion Lab needs at least one positive and one negative prompt")
    causal_tokens = target_token
    if include_causal and causal_tokens is None:
        causal_tokens = list(preset_data["target_tokens"])
    config = build_run_template(
        workflow=workflow,
        model=model,
        criterion=lab_criterion,
        run_dir=run_dir,
        positive_prompt=positive_prompts,
        negative_prompt=negative_prompts,
        top_k=top_k,
        features_per_layer=features_per_layer,
        layers=layers,
        preset=training_preset,
        layer=layer,
        source_layer=source_layer,
        target_layer=target_layer,
        include_causal=include_causal,
        target_token=causal_tokens,
        device=device,
        max_length=max_length,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs_json=model_kwargs_json,
        tokenizer_kwargs_json=tokenizer_kwargs_json,
    )
    config["metadata"] = {
        "criterion_lab": {
            "schema_version": "interp-lab.criterion_lab.v1",
            "preset": preset_key,
            "label": preset_data["label"],
            "criterion": lab_criterion,
            "positive_prompt_count": len(positive_prompts),
            "negative_prompt_count": len(negative_prompts),
            "target_tokens": causal_tokens or [],
            "workflow": workflow,
            "training_preset": training_preset,
        }
    }
    config["agent_next_actions"] = list(preset_data["recommended_next_actions"])
    return config


def write_criterion_lab_config(
    *,
    out: str | Path,
    force: bool = False,
    **kwargs: Any,
) -> CriterionLabWriteResult:
    config = build_criterion_lab_config(**kwargs)
    path = Path(out)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    lab = config["metadata"]["criterion_lab"]
    return CriterionLabWriteResult(
        path=path,
        config=config,
        preset=str(lab["preset"]),
        criterion=str(lab["criterion"]),
    )


def build_criterion_lab_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an editable Criterion Lab run config.")
    parser.add_argument("--out", default="reports/criterion-lab/run.json", help="Output run config JSON path.")
    parser.add_argument("--model", required=True, help="Model to inspect.")
    parser.add_argument("--preset", choices=sorted(CRITERION_LAB_PRESETS), default="overconfidence")
    parser.add_argument("--criterion", help="Override the preset criterion text.")
    parser.add_argument("--workflow", choices=["hf-records", "sae", "sae-paths"], default="sae")
    parser.add_argument("--run-dir", default="reports/criterion-lab")
    parser.add_argument("--positive-prompt", action="append", default=[], help="Additional positive prompt.")
    parser.add_argument("--negative-prompt", action="append", default=[], help="Additional negative prompt.")
    parser.add_argument("--no-preset-prompts", action="store_true", help="Use only prompts supplied on the command line.")
    parser.add_argument("--training-preset", choices=["minimal", "production", "custom"], default="minimal")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--features-per-layer", type=int, default=16)
    parser.add_argument("--layers", help="Hidden-state layers for --workflow hf-records.")
    parser.add_argument("--layer", type=int, help="Hidden-state layer for --workflow sae.")
    parser.add_argument("--source-layer", type=int, help="Source layer for --workflow sae-paths.")
    parser.add_argument("--target-layer", type=int, help="Target layer for --workflow sae-paths.")
    parser.add_argument("--skip-causal", action="store_true", help="Skip first-pass SAE causal validation.")
    parser.add_argument("--target-token", action="append", default=[], help="Target token for causal scoring.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--model-class", choices=MODEL_CLASS_CHOICES, default="auto-causal-lm")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", help="Optional HF device_map, e.g. auto.")
    parser.add_argument("--model-kwargs-json", help="Extra JSON object passed to model from_pretrained.")
    parser.add_argument("--tokenizer-kwargs-json", help="Extra JSON object passed to tokenizer from_pretrained.")
    parser.add_argument("--execute", action="store_true", help="Run the generated config immediately.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing config path.")
    return parser


def run_criterion_lab_from_args(args: argparse.Namespace) -> CriterionLabWriteResult:
    try:
        return write_criterion_lab_config(
            out=args.out,
            force=args.force,
            model=args.model,
            preset=args.preset,
            criterion=args.criterion,
            workflow=args.workflow,
            run_dir=args.run_dir,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            include_preset_prompts=not args.no_preset_prompts,
            training_preset=args.training_preset,
            top_k=args.top_k,
            features_per_layer=args.features_per_layer,
            layers=args.layers,
            layer=args.layer,
            source_layer=args.source_layer,
            target_layer=args.target_layer,
            include_causal=not args.skip_causal,
            target_token=args.target_token or None,
            device=args.device,
            max_length=args.max_length,
            model_class=args.model_class,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            model_kwargs_json=args.model_kwargs_json,
            tokenizer_kwargs_json=args.tokenizer_kwargs_json,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
