from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.adapters.interventions import InterventionRecordRunner
from oracle_sae.adapters.jsonl import JsonlFeatureProvider
from oracle_sae.adapters.neuronpedia import (
    NeuronpediaClient,
    NeuronpediaFeatureProvider,
    load_neuronpedia_feature_refs,
)
from oracle_sae.adapters.records import ActivationRecordFeatureProvider
from oracle_sae.adapters.saelens import (
    SAELensFeatureProvider,
    load_saelens_feature_metadata,
    parse_feature_indices,
)
from oracle_sae.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from oracle_sae.cli import main as _cli_main
from oracle_sae.doctor import collect_diagnostics
from oracle_sae.pipeline import inspect_model, match_reports
from oracle_sae.reporting import (
    load_inspection_report,
    write_inspection_report,
    write_match_markdown,
    write_match_report,
)
from oracle_sae.runs import RunOptions, run_config_file
from oracle_sae.sae_training import (
    TRAINING_PRESETS,
    train_sae_from_hf,
    train_sae_from_records,
)
from oracle_sae.schema import InspectionReport, MatchReport


@dataclass(frozen=True)
class WrittenInspection:
    report: InspectionReport
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class WrittenMatch:
    report: MatchReport
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class SaeTrainingResult:
    artifact_path: Path
    records_path: Path | None = None
    interventions_path: Path | None = None


def inspect(
    model: str,
    criterion: str,
    *,
    backend: str = "toy",
    features: str | Path | None = None,
    records: str | Path | None = None,
    interventions: str | Path | None = None,
    out: str | Path | None = None,
    top_k: int = 8,
    require_interventions: bool = False,
    allow_intervention_criterion_mismatch: bool = False,
    neuronpedia_feature: str | list[str] | None = None,
    neuronpedia_features: str | Path | None = None,
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    saelens_release: str | None = None,
    saelens_sae_id: str | None = None,
    saelens_feature_indexes: str | None = None,
    saelens_max_features: int = 32,
    saelens_device: str = "cpu",
    saelens_force_download: bool = False,
    saelens_feature_metadata: str | Path | None = None,
) -> InspectionReport | WrittenInspection:
    """Rank and explain features for a natural-language criterion.

    When `out` is supplied, the report is written as JSON and Markdown and a
    `WrittenInspection` is returned. Otherwise this returns the in-memory
    `InspectionReport`.
    """
    provider = _feature_provider(
        backend=backend,
        features=features,
        records=records,
        neuronpedia_feature=neuronpedia_feature,
        neuronpedia_features=neuronpedia_features,
        neuronpedia_base_url=neuronpedia_base_url,
        saelens_release=saelens_release,
        saelens_sae_id=saelens_sae_id,
        saelens_feature_indexes=saelens_feature_indexes,
        saelens_max_features=saelens_max_features,
        saelens_device=saelens_device,
        saelens_force_download=saelens_force_download,
        saelens_feature_metadata=saelens_feature_metadata,
    )
    intervention_runner = _intervention_runner(
        interventions=interventions,
        require_interventions=require_interventions,
        allow_intervention_criterion_mismatch=allow_intervention_criterion_mismatch,
    )
    report = inspect_model(
        model=model,
        criterion_text=criterion,
        feature_provider=provider,
        verbalizer=ToyVerbalizer(),
        intervention_runner=intervention_runner,
        top_k=top_k,
    )
    if out is None:
        return report
    json_path, markdown_path = write_inspection_report(report, out)
    return WrittenInspection(report=report, json_path=json_path, markdown_path=markdown_path)


def compare(
    left: InspectionReport | str | Path,
    right: InspectionReport | str | Path,
    *,
    top_k: int = 10,
    out: str | Path | None = None,
) -> MatchReport | WrittenMatch:
    """Match candidate equivalent features across two inspection reports."""
    left_report = _load_report(left)
    right_report = _load_report(right)
    report = match_reports(left_report, right_report, top_k=top_k)
    if out is None:
        return report
    json_path = write_match_report(report, out)
    markdown_path = write_match_markdown(report, _match_markdown_path(out))
    return WrittenMatch(report=report, json_path=json_path, markdown_path=markdown_path)


def train_sae(
    *,
    out: str | Path,
    records: str | Path | None = None,
    model: str | None = None,
    hf_model: str | None = None,
    dataset: str | Path | None = None,
    records_out: str | Path | None = None,
    preset: str = "minimal",
    layer: int | None = None,
    pool: str = "last",
    token_mode: str | None = None,
    latent_dim: int | None = None,
    expansion_factor: float | None = None,
    method: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    l1: float | None = None,
    sparsity: str | None = None,
    top_k: int | None = None,
    jump_threshold: float | None = None,
    validation_fraction: float | None = None,
    dead_latent_threshold: float | None = None,
    seed: int = 0,
    device: str = "cpu",
    max_length: int | None = None,
    max_records: int | None = None,
    top_k_features: int | None = None,
    decoder_signature_size: int | None = None,
    causal_out: str | Path | None = None,
    criterion: str | None = None,
    causal_top_k: int | None = None,
    causal_strength_sweep: list[float] | None = None,
    target_tokens: list[str] | None = None,
) -> SaeTrainingResult:
    """Train an SAE from activation records or Hugging Face hidden states."""
    settings = _training_settings(
        preset=preset,
        expansion_factor=expansion_factor,
        method=method,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        l1=l1,
        sparsity=sparsity,
        top_k=top_k,
        jump_threshold=jump_threshold,
        validation_fraction=validation_fraction,
        dead_latent_threshold=dead_latent_threshold,
        token_mode=token_mode,
        max_length=max_length,
        top_k_features=top_k_features,
        decoder_signature_size=decoder_signature_size,
        causal_top_k=causal_top_k,
    )
    if records and hf_model:
        raise ValueError("Use either records or hf_model, not both")
    if records:
        if causal_out is not None:
            raise ValueError("causal_out currently requires hf_model")
        artifact_path, activation_records_path = train_sae_from_records(
            records_path=records,
            out_path=out,
            records_out=records_out,
            model_name=model,
            latent_dim=latent_dim,
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
            seed=seed,
            device=device,
            max_records=max_records,
            top_k_features=_feature_limit(settings["top_k_features"]),
            decoder_signature_size=settings["decoder_signature_size"],
        )
        return SaeTrainingResult(artifact_path=artifact_path, records_path=activation_records_path)
    if hf_model:
        if dataset is None:
            raise ValueError("dataset is required with hf_model")
        artifact_path, activation_records_path = train_sae_from_hf(
            model_name=hf_model,
            dataset_path=dataset,
            out_path=out,
            records_out=records_out,
            layer=layer,
            pool=pool,
            token_mode=settings["token_mode"],
            latent_dim=latent_dim,
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
            seed=seed,
            device=device,
            max_length=settings["max_length"],
            max_records=max_records,
            top_k_features=_feature_limit(settings["top_k_features"]),
            decoder_signature_size=settings["decoder_signature_size"],
            causal_out=causal_out,
            criterion=criterion,
            causal_top_k=settings["causal_top_k"],
            causal_strength_sweep=causal_strength_sweep,
            target_tokens=target_tokens,
        )
        return SaeTrainingResult(
            artifact_path=artifact_path,
            records_path=activation_records_path,
            interventions_path=Path(causal_out) if causal_out is not None else None,
        )
    raise ValueError("Either records or hf_model is required")


def run(
    config: str | Path,
    *,
    dry_run: bool = False,
    variables: dict[str, str] | None = None,
) -> int:
    """Run a reproducible workflow config."""
    return run_config_file(
        RunOptions(config_path=Path(config), dry_run=dry_run, variables=variables),
        command_runner=_cli_main,
    )


def doctor() -> dict[str, Any]:
    """Return environment diagnostics for optional adapters and core runtime."""
    return collect_diagnostics()


def _feature_provider(
    *,
    backend: str,
    features: str | Path | None,
    records: str | Path | None,
    neuronpedia_feature: str | list[str] | None,
    neuronpedia_features: str | Path | None,
    neuronpedia_base_url: str,
    saelens_release: str | None,
    saelens_sae_id: str | None,
    saelens_feature_indexes: str | None,
    saelens_max_features: int,
    saelens_device: str,
    saelens_force_download: bool,
    saelens_feature_metadata: str | Path | None,
):
    if backend == "toy":
        return ToyFeatureProvider()
    if backend == "jsonl":
        if features is None:
            raise ValueError("features is required with backend='jsonl'")
        return JsonlFeatureProvider(features)
    if backend == "records":
        if records is None:
            raise ValueError("records is required with backend='records'")
        return ActivationRecordFeatureProvider(records)
    if backend == "neuronpedia":
        refs = _as_list(neuronpedia_feature)
        if neuronpedia_features is not None:
            refs.extend(load_neuronpedia_feature_refs(neuronpedia_features))
        if not refs:
            raise ValueError("neuronpedia_feature or neuronpedia_features is required with backend='neuronpedia'")
        return NeuronpediaFeatureProvider(
            refs,
            client=NeuronpediaClient(base_url=neuronpedia_base_url),
        )
    if backend == "saelens":
        if saelens_release is None:
            raise ValueError("saelens_release is required with backend='saelens'")
        if saelens_sae_id is None:
            raise ValueError("saelens_sae_id is required with backend='saelens'")
        return SAELensFeatureProvider(
            release=saelens_release,
            sae_id=saelens_sae_id,
            feature_indices=parse_feature_indices(saelens_feature_indexes),
            max_features=saelens_max_features,
            device=saelens_device,
            force_download=saelens_force_download,
            feature_metadata=load_saelens_feature_metadata(saelens_feature_metadata),
        )
    raise ValueError("backend must be one of: toy, jsonl, records, neuronpedia, saelens")


def _intervention_runner(
    *,
    interventions: str | Path | None,
    require_interventions: bool,
    allow_intervention_criterion_mismatch: bool,
):
    fallback = ToyInterventionRunner()
    if interventions is None:
        return fallback
    return InterventionRecordRunner(
        interventions,
        fallback_runner=fallback,
        require_criterion_match=not allow_intervention_criterion_mismatch,
        require_records=require_interventions,
    )


def _load_report(value: InspectionReport | str | Path) -> InspectionReport:
    if isinstance(value, InspectionReport):
        return value
    return load_inspection_report(value)


def _match_markdown_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    if path.suffix:
        return path.with_suffix(".md")
    return path / "matches.md"


def _training_settings(preset: str, **overrides: Any) -> dict[str, Any]:
    if preset not in TRAINING_PRESETS:
        raise ValueError("preset must be one of: minimal, production, custom")
    settings = dict(TRAINING_PRESETS[preset])
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _feature_limit(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
