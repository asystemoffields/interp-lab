from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from oracle_sae.hf_loading import MODEL_CLASS_CHOICES
from oracle_sae.workflows import build_run_template

PRESET_SCHEMA_VERSION = "interp-lab.criterion_lab_preset.v1"
PRESET_PACKAGE = "oracle_sae.presets"


@dataclass(frozen=True)
class CriterionLabWriteResult:
    path: Path
    config: dict[str, Any]
    preset: str
    criterion: str


@dataclass(frozen=True)
class CriterionLabPresetInfo:
    name: str
    label: str
    criterion: str
    source: str


def build_criterion_lab_config(
    *,
    model: str,
    preset: str | None = None,
    preset_file: str | Path | None = None,
    preset_dirs: list[str | Path] | None = None,
    criterion: str | None = None,
    workflow: str | None = None,
    run_dir: str | Path = "reports/criterion-lab",
    positive_prompt: list[str] | None = None,
    negative_prompt: list[str] | None = None,
    include_preset_prompts: bool = True,
    use_preset_target_hints: bool = False,
    training_preset: str | None = None,
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
    if not model:
        raise ValueError("model is required")
    preset_data = load_criterion_lab_preset(
        preset=preset,
        preset_file=preset_file,
        preset_dirs=preset_dirs,
    )
    defaults = dict(preset_data.get("defaults", {})) if preset_data else {}
    lab_criterion = criterion or (str(preset_data["criterion"]) if preset_data else None)
    if not lab_criterion:
        raise ValueError("criterion is required when no preset supplies one")

    positive_prompts: list[str] = []
    negative_prompts: list[str] = []
    if include_preset_prompts and preset_data:
        positive_prompts.extend(preset_data.get("positive_prompts", []))
        negative_prompts.extend(preset_data.get("negative_prompts", []))
    positive_prompts.extend(positive_prompt or [])
    negative_prompts.extend(negative_prompt or [])
    if not positive_prompts or not negative_prompts:
        raise ValueError("Criterion Lab needs at least one positive and one negative prompt")

    resolved_workflow = workflow or str(defaults.get("workflow", "discovery"))
    template_workflow = "hf-records" if resolved_workflow == "discovery" else resolved_workflow
    resolved_training_preset = training_preset or str(defaults.get("training_preset", "minimal"))
    resolved_layers = layers
    if resolved_layers is None and template_workflow == "hf-records":
        resolved_layers = str(defaults.get("layers", "all"))
    resolved_layer = _default_int(layer, defaults.get("layer"))
    resolved_source_layer = _default_int(source_layer, defaults.get("source_layer"))
    resolved_target_layer = _default_int(target_layer, defaults.get("target_layer"))
    causal_tokens = target_token
    if include_causal and causal_tokens is None:
        preset_tokens = preset_data.get("target_token_hints", []) if preset_data else []
        if use_preset_target_hints and preset_tokens:
            causal_tokens = list(preset_tokens)
        elif template_workflow in {"sae", "sae-paths"}:
            causal_tokens = ["auto"]

    config = build_run_template(
        workflow=template_workflow,
        model=model,
        criterion=lab_criterion,
        run_dir=run_dir,
        positive_prompt=positive_prompts,
        negative_prompt=negative_prompts,
        top_k=top_k,
        features_per_layer=features_per_layer,
        layers=resolved_layers,
        preset=resolved_training_preset,
        layer=resolved_layer,
        source_layer=resolved_source_layer,
        target_layer=resolved_target_layer,
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

    preset_name = str(preset_data.get("name", "custom")) if preset_data else "custom"
    label = str(preset_data.get("label", preset_name)) if preset_data else "Custom criterion"
    source = str(preset_data.get("source", "")) if preset_data else ""
    target_hints = list(preset_data.get("target_token_hints", [])) if preset_data else []
    config["metadata"] = {
        "criterion_lab": {
            "schema_version": "interp-lab.criterion_lab.v1",
            "preset": preset_name,
            "preset_source": source,
            "label": label,
            "criterion": lab_criterion,
            "positive_prompt_count": len(positive_prompts),
            "negative_prompt_count": len(negative_prompts),
            "discovery_first": template_workflow == "hf-records",
            "template_workflow": template_workflow,
            "layers": resolved_layers,
            "target_token_hints": target_hints,
            "use_preset_target_hints": use_preset_target_hints,
            "target_tokens": causal_tokens or [],
            "workflow": resolved_workflow,
            "training_preset": resolved_training_preset,
        }
    }
    config["agent_next_actions"] = _recommended_next_actions(preset_data)
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


def load_criterion_lab_preset(
    *,
    preset: str | None = None,
    preset_file: str | Path | None = None,
    preset_dirs: list[str | Path] | None = None,
) -> dict[str, Any] | None:
    if preset and preset_file:
        raise ValueError("Use either --preset or --preset-file, not both")
    if preset_file is not None:
        return _load_preset_path(Path(preset_file))
    if not preset:
        return None
    preset_path = Path(preset)
    if preset_path.exists() or preset_path.suffix.lower() == ".json" or any(sep in preset for sep in ("/", "\\")):
        return _load_preset_path(preset_path)
    for path in _candidate_named_preset_paths(preset, preset_dirs=preset_dirs):
        if path.exists():
            return _load_preset_path(path)
    bundled = _load_bundled_preset(preset)
    if bundled is not None:
        return bundled
    names = ", ".join(info.name for info in available_criterion_lab_presets(preset_dirs=preset_dirs))
    suffix = f" Available presets: {names}" if names else ""
    raise ValueError(f"unknown Criterion Lab preset: {preset}.{suffix}")


def available_criterion_lab_presets(
    *,
    preset_dirs: list[str | Path] | None = None,
) -> list[CriterionLabPresetInfo]:
    presets: dict[str, CriterionLabPresetInfo] = {}
    for path in _iter_preset_dir_files(preset_dirs or []):
        try:
            data = _load_preset_path(path)
        except ValueError:
            continue
        presets[str(data["name"])] = CriterionLabPresetInfo(
            name=str(data["name"]),
            label=str(data.get("label", data["name"])),
            criterion=str(data["criterion"]),
            source=str(data["source"]),
        )
    for item in _iter_bundled_preset_files():
        try:
            data = _load_preset_resource(item)
        except ValueError:
            continue
        presets.setdefault(
            str(data["name"]),
            CriterionLabPresetInfo(
                name=str(data["name"]),
                label=str(data.get("label", data["name"])),
                criterion=str(data["criterion"]),
                source=str(data["source"]),
            ),
        )
    return sorted(presets.values(), key=lambda item: item.name)


def format_available_presets(*, preset_dirs: list[str | Path] | None = None) -> str:
    presets = available_criterion_lab_presets(preset_dirs=preset_dirs)
    if not presets:
        return "No Criterion Lab presets found."
    lines = ["Available Criterion Lab presets:"]
    for info in presets:
        lines.append(f"- {info.name}: {info.label} ({info.source})")
    return "\n".join(lines)


def build_criterion_lab_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an editable Criterion Lab run config.")
    parser.add_argument("--out", default="reports/criterion-lab/run.json", help="Output run config JSON path.")
    parser.add_argument("--model", help="Model to inspect.")
    parser.add_argument("--preset", help="Preset name or JSON path.")
    parser.add_argument("--preset-file", help="Explicit preset JSON path.")
    parser.add_argument("--preset-dir", action="append", default=[], help="Directory of preset JSON files. Repeatable.")
    parser.add_argument("--list-presets", action="store_true", help="List discoverable Criterion Lab presets.")
    parser.add_argument("--criterion", help="Criterion text for custom or preset-overridden runs.")
    parser.add_argument("--workflow", choices=["discovery", "hf-records", "sae", "sae-paths"])
    parser.add_argument("--run-dir", default="reports/criterion-lab")
    parser.add_argument("--positive-prompt", action="append", default=[], help="Additional positive prompt.")
    parser.add_argument("--negative-prompt", action="append", default=[], help="Additional negative prompt.")
    parser.add_argument("--no-preset-prompts", action="store_true", help="Use only prompts supplied on the command line.")
    parser.add_argument(
        "--use-preset-target-hints",
        action="store_true",
        help="Use target-token hints from a preset during SAE causal scoring.",
    )
    parser.add_argument("--training-preset", choices=["minimal", "production", "custom"])
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
    if not args.model:
        raise SystemExit("--model is required unless --list-presets is used")
    try:
        return write_criterion_lab_config(
            out=args.out,
            force=args.force,
            model=args.model,
            preset=args.preset,
            preset_file=args.preset_file,
            preset_dirs=args.preset_dir,
            criterion=args.criterion,
            workflow=args.workflow,
            run_dir=args.run_dir,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            include_preset_prompts=not args.no_preset_prompts,
            use_preset_target_hints=args.use_preset_target_hints,
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


def _load_preset_path(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"preset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    return _normalize_preset(raw, source=str(path), fallback_name=path.stem)


def _load_preset_resource(resource: Traversable) -> dict[str, Any]:
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resource}: invalid JSON: {exc.msg}") from exc
    return _normalize_preset(raw, source=f"bundled:{resource.name}", fallback_name=Path(resource.name).stem)


def _load_bundled_preset(name: str) -> dict[str, Any] | None:
    target = f"{name}.json"
    for item in _iter_bundled_preset_files():
        if item.name == target:
            return _load_preset_resource(item)
        try:
            data = _load_preset_resource(item)
        except ValueError:
            continue
        if data["name"] == name:
            return data
    return None


def _normalize_preset(raw: Any, *, source: str, fallback_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: preset must be a JSON object")
    name = str(raw.get("name", fallback_name)).strip()
    if not name:
        raise ValueError(f"{source}: preset needs a name")
    criterion = str(raw.get("criterion", "")).strip()
    if not criterion:
        raise ValueError(f"{source}: preset needs a criterion")
    positive = _string_list(raw.get("positive_prompts"), f"{source}: positive_prompts")
    negative = _string_list(raw.get("negative_prompts"), f"{source}: negative_prompts")
    if not positive or not negative:
        raise ValueError(f"{source}: preset needs positive_prompts and negative_prompts")
    target_tokens = _string_list(
        raw.get("target_token_hints", raw.get("target_tokens", [])),
        f"{source}: target_token_hints",
    )
    recommended = _string_list(
        raw.get("recommended_next_actions", []),
        f"{source}: recommended_next_actions",
    )
    defaults = raw.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{source}: defaults must be an object")
    return {
        "schema_version": str(raw.get("schema_version", PRESET_SCHEMA_VERSION)),
        "name": name,
        "label": str(raw.get("label", name)),
        "criterion": criterion,
        "positive_prompts": positive,
        "negative_prompts": negative,
        "target_token_hints": target_tokens,
        "recommended_next_actions": recommended,
        "defaults": dict(defaults),
        "source": source,
    }


def _recommended_next_actions(preset_data: dict[str, Any] | None) -> list[str]:
    if preset_data and preset_data.get("recommended_next_actions"):
        return list(preset_data["recommended_next_actions"])
    return [
        "Run the generated config on a small model first, then inspect report.html for top associated latents.",
        "Treat activation-ranked features as candidates until intervention rows or path validation support them.",
        "Edit the prompt set toward your domain before larger runs.",
    ]


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _default_int(value: int | None, default: Any) -> int | None:
    if value is not None:
        return value
    if default in (None, ""):
        return None
    return int(default)


def _candidate_named_preset_paths(
    name: str,
    *,
    preset_dirs: list[str | Path] | None,
) -> list[Path]:
    paths = []
    for directory in preset_dirs or []:
        paths.append(Path(directory) / f"{name}.json")
    return paths


def _iter_preset_dir_files(preset_dirs: list[str | Path]):
    for directory in preset_dirs:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            continue
        yield from sorted(path.glob("*.json"))


def _iter_bundled_preset_files() -> list[Traversable]:
    try:
        root = resources.files(PRESET_PACKAGE)
    except ModuleNotFoundError:
        return []
    return sorted(
        (item for item in root.iterdir() if item.name.endswith(".json")),
        key=lambda item: item.name,
    )
