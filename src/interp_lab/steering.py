"""Steering-vector artifacts: turn validated report features into reusable directions.

``export-steering`` resolves a report card's direction -- an SAE decoder row for
``SAE:L<n>:F<i>`` latents, or a one-hot hidden direction for ``L<n>:D<i>``
features -- into a portable ``interp-lab.steering_vector.v1`` JSON artifact.

Provenance gate (the same discipline as matching/scoring): by default a card is
only exportable when it carries intervention-measured evidence (intervention
provenance plus a ``signed_causal_effect``). A correlational card can still be
exported with ``--allow-unvalidated``, but the artifact is then stamped
``"provenance": "unvalidated"`` and carries an explicit warning, so a steered
demo can never silently present an untested direction as a measured one.

``apply-steering`` loads an artifact and generates baseline vs steered
continuations through the EXISTING :func:`interp_lab.hf_hooks.register_hidden_steering`
machinery (no reimplemented hooks). ``model_loader``/``generate_fn`` are seams
so tests can stub the heavy HF path exactly like ``tests/test_interventions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from interp_lab import __version__
from interp_lab.hf_hooks import register_hidden_steering
from interp_lab.hf_interventions import FEATURE_PATTERN
from interp_lab.hf_loading import add_hf_loading_args, hf_loading_options_from_args, load_hf_text_model
from interp_lab.hf_sae_paths import SAE_FEATURE_PATTERN, parse_sae_feature_ref
from interp_lab.matching import has_intervention_provenance, signed_effect_with_provenance
from interp_lab.reporting import load_inspection_report
from interp_lab.schema import FeatureCard, InspectionReport, utc_now_iso

STEERING_SCHEMA = "interp-lab.steering_vector.v1"
STEERING_GENERATION_SCHEMA = "interp-lab.steering_generation.v1"

_UNVALIDATED_WARNING = (
    "UNVALIDATED: no intervention-measured evidence backs this direction. "
    "Steered generations from this artifact are exploratory demos, not measured causal claims."
)


# --- export -------------------------------------------------------------------


def export_steering_vector(
    report: InspectionReport | dict | str | Path,
    feature_id: str,
    *,
    sae: dict | str | Path | None = None,
    out: str | Path,
    strength: float | None = None,
    allow_unvalidated: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Export one report feature as a reusable steering-vector artifact.

    ``report`` may be an ``InspectionReport``, a report dict, or a path to a
    ``report.json`` (path inputs also get a sha256 recorded). ``sae`` (artifact
    dict or path) is required for ``SAE:*`` latent features; the decoder row is
    the steering direction. ``strength`` overrides the derived
    ``recommended_strength``. ``now`` is injectable for tests.
    """
    report_obj, report_path = _resolve_report(report)
    card = _find_card(report_obj, feature_id)
    validated = has_intervention_provenance(card.causal_effects, card.metadata) and (
        "signed_causal_effect" in card.causal_effects
    )
    if not validated and not allow_unvalidated:
        raise ValueError(
            f"{feature_id}: this card carries no intervention-measured evidence "
            "(intervention provenance plus a signed_causal_effect are required), so every "
            "causal-looking number on it is correlational. Run `interp-lab intervene` on this "
            "feature first, or pass --allow-unvalidated to export anyway with the artifact "
            'stamped "provenance": "unvalidated".'
        )
    sae_artifact, sae_path = _resolve_sae(sae)
    direction, layer = _resolve_direction(feature_id, card=card, sae_artifact=sae_artifact)
    recommended, strength_source, strength_reason = _recommended_strength(card, explicit=strength)
    signed, signed_provenance = signed_effect_with_provenance(card.causal_effects, card.metadata)
    payload: dict[str, Any] = {
        "schema_version": STEERING_SCHEMA,
        "model": report_obj.model,
        "criterion": report_obj.criterion.text,
        "feature_id": feature_id,
        "label": card.label,
        "layer": layer,
        "direction": direction,
        "recommended_strength": recommended,
        "recommended_strength_source": strength_source,
        "recommended_strength_reason": strength_reason,
        "measured_signed_effect": None if signed is None else round(float(signed), 6),
        "signed_effect_provenance": signed_provenance,
        "provenance": "intervention" if validated else "unvalidated",
        "source": {
            "report_path": str(report_path) if report_path is not None else None,
            "report_sha256": _sha256(report_path) if report_path is not None else None,
            "sae_path": str(sae_path) if sae_path is not None else None,
            "sae_sha256": _sha256(sae_path) if sae_path is not None else None,
        },
        "created_at": _timestamp(now),
        "tool": {"name": "interp-lab", "version": __version__},
    }
    if not validated:
        # Stamped prominently (top-level, next to provenance) so any reader of the
        # JSON sees the caveat before the direction values.
        payload["unvalidated_warning"] = _UNVALIDATED_WARNING
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_steering_artifact(artifact: dict | str | Path) -> dict[str, Any]:
    """Load and minimally validate a steering-vector artifact (dict or path)."""
    if isinstance(artifact, (str, Path)):
        path = Path(artifact)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid steering artifact JSON: {exc.msg}") from exc
        label = str(path)
    else:
        data = artifact
        label = "steering artifact"
    if not isinstance(data, dict):
        raise ValueError(f"{label}: steering artifact must be a JSON object")
    schema = data.get("schema_version")
    if schema != STEERING_SCHEMA:
        raise ValueError(f"{label}: unsupported steering artifact schema_version {schema!r} (expected {STEERING_SCHEMA})")
    for key in ["model", "feature_id", "layer", "direction"]:
        if data.get(key) is None:
            raise ValueError(f"{label}: steering artifact is missing {key!r}")
    if not isinstance(data["direction"], dict):
        raise ValueError(f"{label}: steering artifact direction must be an object")
    return data


# --- apply --------------------------------------------------------------------


def apply_steering(
    artifact: dict | str | Path,
    *,
    prompts: list[str] | str | Path,
    out: str | Path,
    strength: float | None = None,
    max_new_tokens: int = 32,
    model_loader: Callable[[], tuple[Any, Any, str]] | None = None,
    generate_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Generate baseline vs steered continuations from a steering artifact.

    ``prompts`` is a list of strings or a path to a prompt file (JSONL rows with
    ``text``/``prompt`` fields, or plain text lines). ``strength`` falls back to
    the artifact's ``recommended_strength``; when both are absent this raises so
    an arbitrary strength is never invented silently.

    Seams: ``model_loader() -> (tokenizer, model, device)`` replaces the default
    HF loading path, and ``generate_fn(model=..., tokenizer=..., prompt=...,
    device=..., max_new_tokens=...) -> str`` replaces ``model.generate``. The
    steering hook itself always goes through ``register_hidden_steering``.
    """
    data = load_steering_artifact(artifact)
    prompt_texts = _resolve_prompts(prompts)
    strength_used = strength if strength is not None else data.get("recommended_strength")
    if strength_used is None:
        raise ValueError(
            f"{data['feature_id']}: the steering artifact carries no recommended_strength "
            f"({data.get('recommended_strength_reason') or 'none derived'}); pass --strength explicitly."
        )
    strength_used = float(strength_used)
    layer = int(data["layer"])
    if model_loader is None:
        tokenizer, model, device = _default_model_loader(str(data["model"]))()
    else:
        tokenizer, model, device = model_loader()
    generate = generate_fn if generate_fn is not None else _default_generate
    direction = _direction_for_model(data["direction"], model=model, feature_id=str(data["feature_id"]))
    provenance = str(data.get("provenance", "unknown"))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for prompt in prompt_texts:
            baseline_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            hook = register_hidden_steering(model, layer, direction, strength_used)
            try:
                steered_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                hook.remove()
            row = {
                "schema_version": STEERING_GENERATION_SCHEMA,
                "model": data["model"],
                "criterion": data.get("criterion"),
                "feature_id": data["feature_id"],
                "label": data.get("label", ""),
                "layer": layer,
                "prompt": prompt,
                "baseline_text": baseline_text,
                "steered_text": steered_text,
                "strength": strength_used,
                "provenance": provenance,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            row_count += 1
    return {
        "rows": row_count,
        "out": str(out_path),
        "strength_used": strength_used,
        "provenance": provenance,
        "feature_id": data["feature_id"],
        "model": data["model"],
    }


# --- CLI wiring (module-owned, evidence_planner/dossier house pattern) ---------


def build_export_steering_parser() -> argparse.ArgumentParser:
    # Default add_help=True so `export-steering --help` works standalone; the cli
    # subparser adopting this as a parent passes add_help=False.
    parser = argparse.ArgumentParser(
        description="Export a report feature as a reusable steering-vector artifact."
    )
    parser.add_argument("--report", required=True, help="Inspection report.json with the feature card.")
    parser.add_argument("--feature", required=True, help="Feature id to export, e.g. L6:D512 or SAE:L6:F30.")
    parser.add_argument("--sae", help="interp-lab SAE artifact JSON, required for SAE:* latent features.")
    parser.add_argument("--out", required=True, help="Output steering-vector JSON path.")
    parser.add_argument("--strength", type=float, help="Explicit recommended strength to stamp on the artifact.")
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help='Export a card without intervention-measured evidence (artifact is stamped provenance "unvalidated").',
    )
    parser.add_argument("--json", action="store_true", help="Print the artifact as JSON.")
    return parser


def run_export_steering_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return export_steering_vector(
        args.report,
        args.feature,
        sae=args.sae,
        out=args.out,
        strength=args.strength,
        allow_unvalidated=args.allow_unvalidated,
    )


def build_apply_steering_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate baseline vs steered continuations from a steering-vector artifact."
    )
    parser.add_argument("--artifact", required=True, help="Steering-vector artifact JSON from export-steering.")
    parser.add_argument(
        "--prompts",
        required=True,
        help="Prompt file: JSONL rows with text/prompt fields, or plain text lines.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL of baseline/steered continuations.")
    parser.add_argument(
        "--strength",
        type=float,
        help="Steering strength. Defaults to the artifact's recommended_strength.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="Print the run summary as JSON.")
    add_hf_loading_args(parser)
    return parser


def run_apply_steering_from_args(args: argparse.Namespace) -> dict[str, Any]:
    try:
        loading_options = hf_loading_options_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    data = load_steering_artifact(args.artifact)
    return apply_steering(
        data,
        prompts=args.prompts,
        out=args.out,
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        model_loader=_default_model_loader(str(data["model"]), device=args.device, **loading_options),
    )


# --- direction / strength resolution -------------------------------------------


def _resolve_report(report: InspectionReport | dict | str | Path) -> tuple[InspectionReport, Path | None]:
    if isinstance(report, InspectionReport):
        return report, None
    if isinstance(report, dict):
        return InspectionReport.from_dict(report), None
    report_path = Path(report)
    return load_inspection_report(report_path), report_path


def _find_card(report: InspectionReport, feature_id: str) -> FeatureCard:
    for card in report.cards:
        if card.feature_id == feature_id:
            return card
    available = ", ".join(card.feature_id for card in report.cards[:8]) or "none"
    raise ValueError(f"{feature_id!r} is not in this report. Available features: {available}")


def _resolve_sae(sae: dict | str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if sae is None:
        return None, None
    if isinstance(sae, dict):
        artifact, sae_path = sae, None
        label = "SAE artifact"
    else:
        sae_path = Path(sae)
        artifact = json.loads(sae_path.read_text(encoding="utf-8"))
        label = str(sae_path)
    if artifact.get("format") != "interp-lab.sae.v1":
        raise ValueError(f"{label}: expected an interp-lab SAE artifact")
    for key in ["layer", "latent_dim", "decoder_weight"]:
        if key not in artifact:
            raise ValueError(f"{label}: missing {key!r}")
    return artifact, sae_path


def _resolve_direction(
    feature_id: str,
    *,
    card: FeatureCard,
    sae_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    if SAE_FEATURE_PATTERN.match(feature_id):
        if sae_artifact is None:
            raise ValueError(
                f"{feature_id}: SAE latent directions come from the SAE decoder; pass the matching "
                "interp-lab SAE artifact (--sae)."
            )
        ref = parse_sae_feature_ref(feature_id, artifact=sae_artifact, role="steering", label=card.label)
        if ref.layer is None:
            raise ValueError(f"{feature_id}: the SAE artifact carries no hidden-state layer to steer at")
        values = [float(value) for value in sae_artifact["decoder_weight"][ref.latent_index]]
        return {"kind": "vector", "values": values, "dim": len(values)}, int(ref.layer)
    hidden = FEATURE_PATTERN.match(feature_id)
    if hidden:
        layer = int(hidden.group("layer"))
        if layer < 1:
            # Same constraint as validate_hookable_feature_layers: layer 0 is the
            # embedding output and has no decoder-block hook point.
            raise ValueError(
                f"{feature_id}: layer 0 features are embedding hidden states and cannot be steered "
                "through decoder hooks."
            )
        index = int(hidden.group("dimension"))
        # dim is resolved at apply time from the loaded model's hidden size; the
        # report alone cannot know it for a one-hot hidden direction.
        return {"kind": "hidden_dim", "index": index, "dim": None}, layer
    raise ValueError(
        f"{feature_id!r} is not a steerable feature id. Use L<layer>:D<dim> or SAE:L<layer>:F<latent>."
    )


def _recommended_strength(
    card: FeatureCard,
    *,
    explicit: float | None,
) -> tuple[float | None, str | None, str | None]:
    """(value, source, reason). Derive from what the card actually stores.

    Cards built by the current inspect pipeline retain an ``interventions``
    summary (count, mean/stdev directed effect, CI, controls) but the
    aggregation does NOT retain the row-level ``selected_strength`` /
    ``strength_sweep`` values from the intervene CLI. When a card (or a future
    aggregation) does carry them they are used; otherwise recommended_strength
    is honestly null with a reason instead of a guessed magnitude.
    """
    if explicit is not None:
        return float(explicit), "explicit", None
    interventions = card.metadata.get("interventions")
    if isinstance(interventions, dict):
        selected = interventions.get("selected_strength")
        if isinstance(selected, (int, float)) and not isinstance(selected, bool):
            return float(selected), "interventions.selected_strength", None
        sweep = interventions.get("strength_sweep")
        if isinstance(sweep, list):
            best: tuple[float, float] | None = None
            for entry in sweep:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("strength")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                specificity = float(entry.get("specificity", entry.get("mean_directed_effect", 0.0)) or 0.0)
                if best is None or specificity > best[0]:
                    best = (specificity, float(value))
            if best is not None:
                return best[1], "interventions.strength_sweep_best_specificity", None
    return (
        None,
        None,
        "the card's interventions summary retains no measured steering strength "
        "(no selected_strength or strength_sweep was stored); pass --strength when applying",
    )


def _direction_for_model(direction: dict[str, Any], *, model: Any, feature_id: str) -> Any:
    kind = direction.get("kind")
    if kind == "vector":
        values = [float(value) for value in direction.get("values", [])]
        if not values:
            raise ValueError(f"{feature_id}: steering artifact direction has no values")
    elif kind == "hidden_dim":
        index = int(direction["index"])
        hidden_size = _model_hidden_size(model)
        if index < 0 or index >= hidden_size:
            raise ValueError(f"{feature_id}: dimension {index} is outside hidden_size={hidden_size}")
        values = [0.0] * hidden_size
        values[index] = 1.0
    else:
        raise ValueError(f"{feature_id}: unsupported steering direction kind {kind!r}")
    return _maybe_tensor(values)


def _model_hidden_size(model: Any) -> int:
    # Same lookup as feature_interventions._hidden_size (kept private there).
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


def _maybe_tensor(values: list[float]) -> Any:
    """Tensorize when torch is available; otherwise return the plain list.

    The list path only exists for stubbed tests (no torch installed): a real
    forward pass needs torch anyway, and register_hidden_steering only touches
    the direction inside the forward hook.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return list(values)
    return torch.tensor(values, dtype=torch.float32)


def _default_model_loader(model_name: str, *, device: str = "cpu", **loading_options: Any):
    def load() -> tuple[Any, Any, str]:
        torch = _optional_import("torch", "Install `interp-lab[hf]` to apply steering vectors.")
        transformers = _optional_import("transformers", "Install `interp-lab[hf]` to apply steering vectors.")
        return load_hf_text_model(
            transformers=transformers,
            torch=torch,
            model_name=model_name,
            device=device,
            **loading_options,
        )

    return load


def _default_generate(*, model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    torch = _optional_import("torch", "Install `interp-lab[hf]` to apply steering vectors.")
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False}
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    with torch.no_grad():
        output = model.generate(**encoded, **kwargs)
    prompt_length = int(encoded["input_ids"].shape[1])
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)


def _resolve_prompts(prompts: list[str] | str | Path) -> list[str]:
    if isinstance(prompts, (str, Path)):
        path = Path(prompts)
        texts: list[str] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid prompt JSON: {exc.msg}") from exc
                text = data.get("text", data.get("prompt"))
                if text is None:
                    raise ValueError(f"{path}:{line_number}: prompt rows need a text or prompt field")
                texts.append(str(text))
            else:
                texts.append(stripped)
        resolved = texts
    else:
        resolved = [str(prompt) for prompt in prompts]
    if not resolved:
        raise ValueError("apply_steering needs at least one prompt")
    return resolved


def _timestamp(now: datetime | None) -> str:
    if now is None:
        return utc_now_iso()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat()


def _sha256(path: Path) -> str:
    # Same local helper runs.py, demo_sweep.py, and dossier.py keep privately.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc
