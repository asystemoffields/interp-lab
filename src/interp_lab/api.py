from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.adapters.goodfire import GoodfireFeatureProvider
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
from oracle_sae.adapters.scope import ScopeFeatureProvider
from oracle_sae.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from oracle_sae.cli import main as _cli_main
from oracle_sae.doctor import collect_diagnostics
from oracle_sae.env_profile import collect_environment_profile, load_environment_profile
from oracle_sae.graphs import (
    build_attribution_graph,
    export_attribution_graph,
    export_attribution_graph_summary,
    load_graph_report,
    load_path_patch_records,
    write_attribution_graph_html,
    summarize_attribution_graph,
    write_attribution_graph_markdown,
)
from oracle_sae.graph_validation import (
    annotate_graph_with_validation,
    build_graph_validation_report,
    export_graph_validation_report,
    render_graph_validation_markdown,
)
from oracle_sae.hf_publish import PublishResult, publish_hf_artifact as _publish_hf_artifact
from oracle_sae.hf_records import PromptDatasetSummary, build_prompt_dataset
from oracle_sae.hf_sae_paths import export_hf_sae_path_records
from oracle_sae.hf_sae_validation import export_hf_sae_path_validation
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
from oracle_sae.scaling import ScalePlan
from oracle_sae.workflows import RunTemplateWriteResult, write_run_template


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


@dataclass(frozen=True)
class WrittenGraph:
    graph: dict[str, Any]
    json_path: Path | None = None
    markdown_path: Path | None = None
    html_path: Path | None = None


@dataclass(frozen=True)
class WrittenGraphSummary:
    summary: dict[str, Any]
    json_path: Path


@dataclass(frozen=True)
class WrittenGraphValidation:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    annotated_graph_path: Path | None = None
    annotated_graph_markdown_path: Path | None = None
    annotated_graph_html_path: Path | None = None


@dataclass(frozen=True)
class HfSaePathValidation:
    selected_path_pairs: list[tuple[str, str]]
    path_records_path: Path
    validation_json_path: Path
    validation_markdown_path: Path
    validation_report: dict[str, Any]
    annotated_graph_path: Path | None = None
    annotated_graph_markdown_path: Path | None = None
    annotated_graph_html_path: Path | None = None


@dataclass(frozen=True)
class PathPatchResult:
    path_records: Path


def build_prompts(
    *,
    out: str | Path,
    positive: str | Path | list[str | Path] | None = None,
    negative: str | Path | list[str | Path] | None = None,
    positive_prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    split: str = "paragraphs",
    delimiter: str | None = None,
    positive_score: float = 1.0,
    negative_score: float = 0.0,
    id_prefix: str = "prompt",
) -> PromptDatasetSummary:
    """Write a scored prompt JSONL dataset from user-authored prompts."""
    return build_prompt_dataset(
        out_path=out,
        positive_paths=_as_optional_list(positive),
        negative_paths=_as_optional_list(negative),
        positive_prompts=_as_optional_list(positive_prompt),
        negative_prompts=_as_optional_list(negative_prompt),
        split=split,
        delimiter=delimiter,
        positive_score=positive_score,
        negative_score=negative_score,
        id_prefix=id_prefix,
    )


def scaffold_run(
    *,
    out: str | Path,
    workflow: str,
    model: str,
    criterion: str,
    run_dir: str | Path = "reports/interp-run",
    records: str | Path | None = None,
    interventions: str | Path | None = None,
    dataset: str | Path | None = None,
    validation_dataset: str | Path | None = None,
    positive: str | Path | list[str | Path] | None = None,
    negative: str | Path | list[str | Path] | None = None,
    positive_prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    split: str = "paragraphs",
    delimiter: str | None = None,
    top_k: int = 8,
    features_per_layer: int = 16,
    layers: str | None = None,
    preset: str = "minimal",
    layer: int | None = None,
    source_layer: int | None = None,
    target_layer: int | None = None,
    path_top_k: int = 8,
    source_top_k: int = 4,
    target_top_k: int = 8,
    random_source_controls: int = 2,
    validate_paths: bool = False,
    pool: str = "last",
    device: str = "cpu",
    max_length: int | None = None,
    include_causal: bool = False,
    target_token: str | list[str] | None = None,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs_json: str | None = None,
    tokenizer_kwargs_json: str | None = None,
    force: bool = False,
) -> RunTemplateWriteResult:
    """Write an editable `interp-lab run` config for a common workflow."""
    return write_run_template(
        out=out,
        force=force,
        workflow=workflow,
        model=model,
        criterion=criterion,
        run_dir=run_dir,
        records=records,
        interventions=interventions,
        dataset=dataset,
        validation_dataset=validation_dataset,
        positive=_as_optional_list(positive),
        negative=_as_optional_list(negative),
        positive_prompt=_as_optional_list(positive_prompt),
        negative_prompt=_as_optional_list(negative_prompt),
        split=split,
        delimiter=delimiter,
        top_k=top_k,
        features_per_layer=features_per_layer,
        layers=layers,
        preset=preset,
        layer=layer,
        source_layer=source_layer,
        target_layer=target_layer,
        path_top_k=path_top_k,
        source_top_k=source_top_k,
        target_top_k=target_top_k,
        random_source_controls=random_source_controls,
        validate_paths=validate_paths,
        pool=pool,
        device=device,
        max_length=max_length,
        include_causal=include_causal,
        target_token=_as_optional_list(target_token),
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs_json=model_kwargs_json,
        tokenizer_kwargs_json=tokenizer_kwargs_json,
    )


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
    goodfire_top_k: int = 32,
    goodfire_api_key_env: str = "GOODFIRE_API_KEY",
    scope_source: str | None = None,
    scope_release: str | None = None,
    scope_sae_id: str | None = None,
    scope_feature_indexes: str | None = None,
    scope_max_features: int = 32,
    scope_device: str = "cpu",
    scope_force_download: bool = False,
    scope_feature_metadata: str | Path | None = None,
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
        goodfire_top_k=goodfire_top_k,
        goodfire_api_key_env=goodfire_api_key_env,
        scope_source=scope_source,
        scope_release=scope_release,
        scope_sae_id=scope_sae_id,
        scope_feature_indexes=scope_feature_indexes,
        scope_max_features=scope_max_features,
        scope_device=scope_device,
        scope_force_download=scope_force_download,
        scope_feature_metadata=scope_feature_metadata,
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
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
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
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        return SaeTrainingResult(
            artifact_path=artifact_path,
            records_path=activation_records_path,
            interventions_path=Path(causal_out) if causal_out is not None else None,
        )
    raise ValueError("Either records or hf_model is required")


def path_patch(
    *,
    model: str,
    dataset: str | Path,
    criterion: str,
    source_sae: str | Path,
    target_sae: str | Path,
    out: str | Path,
    source_features: list[str] | None = None,
    target_features: list[str] | None = None,
    source_report: str | Path | None = None,
    target_report: str | Path | None = None,
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
) -> PathPatchResult:
    """Patch source SAE latents and write measured downstream SAE path records."""
    path = export_hf_sae_path_records(
        model_name=model,
        dataset_path=dataset,
        source_artifact_path=source_sae,
        target_artifact_path=target_sae,
        out_path=out,
        criterion=criterion,
        source_features=source_features,
        target_features=target_features,
        source_report_path=source_report,
        target_report_path=target_report,
        source_top_k=source_top_k,
        target_top_k=target_top_k,
        pool=pool,
        strength_sweep=strength_sweep,
        random_source_controls=random_source_controls,
        control_seed=control_seed,
        score_behavior=score_behavior,
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
    return PathPatchResult(path_records=path)


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


def profile_environment(path: str | Path = ".") -> dict[str, Any]:
    """Return a sanitized compute, storage, and route profile for an environment."""
    return collect_environment_profile(path=path)


def attribution_graph(
    report: InspectionReport | str | Path | list[str | Path],
    *,
    out: str | Path | None = None,
    markdown_out: str | Path | None = None,
    html_out: str | Path | None = None,
    include_similarity_edges: bool = False,
    similarity_threshold: float = 0.9,
    include_coactivation_edges: bool = True,
    coactivation_threshold: float = 0.65,
    include_supernodes: bool = True,
    strong_causal_threshold: float = 0.05,
    path_records: str | Path | list[str | Path] | None = None,
) -> dict[str, Any] | WrittenGraph:
    """Build or write a candidate mechanism graph from one or more inspection reports."""
    loaded_path_records = load_path_patch_records(path_records) if path_records is not None else []
    if isinstance(report, InspectionReport):
        graph = build_attribution_graph(
            report,
            include_similarity_edges=include_similarity_edges,
            similarity_threshold=similarity_threshold,
            include_coactivation_edges=include_coactivation_edges,
            coactivation_threshold=coactivation_threshold,
            include_supernodes=include_supernodes,
            strong_causal_threshold=strong_causal_threshold,
            path_records=loaded_path_records,
        )
        if out is None:
            return graph
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
        markdown_path = write_attribution_graph_markdown(graph, markdown_out) if markdown_out is not None else None
        html_path = write_attribution_graph_html(graph, html_out) if html_out is not None else None
        return WrittenGraph(graph=graph, json_path=path, markdown_path=markdown_path, html_path=html_path)
    if out is None:
        loaded = load_graph_report(report)
        return build_attribution_graph(
            loaded,
            include_similarity_edges=include_similarity_edges,
            similarity_threshold=similarity_threshold,
            include_coactivation_edges=include_coactivation_edges,
            coactivation_threshold=coactivation_threshold,
            include_supernodes=include_supernodes,
            strong_causal_threshold=strong_causal_threshold,
            path_records=loaded_path_records,
        )
    path = export_attribution_graph(
        report_path=report,
        out_path=out,
        markdown_out_path=markdown_out,
        html_out_path=html_out,
        include_similarity_edges=include_similarity_edges,
        similarity_threshold=similarity_threshold,
        include_coactivation_edges=include_coactivation_edges,
        coactivation_threshold=coactivation_threshold,
        include_supernodes=include_supernodes,
        strong_causal_threshold=strong_causal_threshold,
        path_records_path=path_records,
    )
    graph = build_attribution_graph(
        load_graph_report(report),
        include_similarity_edges=include_similarity_edges,
        similarity_threshold=similarity_threshold,
        include_coactivation_edges=include_coactivation_edges,
        coactivation_threshold=coactivation_threshold,
        include_supernodes=include_supernodes,
        strong_causal_threshold=strong_causal_threshold,
        path_records=loaded_path_records,
    )
    markdown_path = Path(markdown_out) if markdown_out is not None else None
    html_path = Path(html_out) if html_out is not None else None
    return WrittenGraph(graph=graph, json_path=path, markdown_path=markdown_path, html_path=html_path)


def attribution_graph_summary(
    graph: dict[str, Any] | str | Path,
    *,
    out: str | Path | None = None,
) -> dict[str, Any] | WrittenGraphSummary:
    """Build or write a compact attribution graph summary for agents and scripts."""
    if isinstance(graph, dict):
        summary = summarize_attribution_graph(graph)
        if out is None:
            return summary
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return WrittenGraphSummary(summary=summary, json_path=path)
    if out is None:
        loaded = json.loads(Path(graph).read_text(encoding="utf-8"))
        return summarize_attribution_graph(loaded)
    path = export_attribution_graph_summary(graph_path=graph, out_path=out)
    summary = json.loads(path.read_text(encoding="utf-8"))
    return WrittenGraphSummary(summary=summary, json_path=path)


def validate_attribution_graph(
    graph: dict[str, Any] | str | Path,
    *,
    path_records: str | Path | list[str | Path],
    out: str | Path | None = None,
    markdown_out: str | Path | None = None,
    graph_out: str | Path | None = None,
    graph_markdown_out: str | Path | None = None,
    graph_html_out: str | Path | None = None,
    top_k: int = 8,
    min_effect: float = 0.05,
    min_specificity: float = 0.02,
    min_effect_control_ratio: float = 1.5,
    min_prompt_count: int = 3,
    min_sign_consistency: float = 0.75,
    require_controls: bool = True,
) -> dict[str, Any] | WrittenGraphValidation:
    """Validate measured attribution graph paths with path-patching records."""
    if isinstance(graph, dict):
        records = load_path_patch_records(path_records)
        report = build_graph_validation_report(
            graph,
            path_records=records,
            top_k=top_k,
            min_effect=min_effect,
            min_specificity=min_specificity,
            min_effect_control_ratio=min_effect_control_ratio,
            min_prompt_count=min_prompt_count,
            min_sign_consistency=min_sign_consistency,
            require_controls=require_controls,
        )
        if out is None:
            return report
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        markdown_path = Path(markdown_out) if markdown_out is not None else path.with_suffix(".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_graph_validation_markdown(report), encoding="utf-8")
        annotated_graph_path = None
        annotated_graph_markdown_path = None
        if graph_out is not None:
            annotated_graph = annotate_graph_with_validation(graph, report)
            annotated_graph_path = Path(graph_out)
            annotated_graph_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_graph_path.write_text(json.dumps(annotated_graph, indent=2, sort_keys=True), encoding="utf-8")
            annotated_graph_markdown_path = write_attribution_graph_markdown(
                annotated_graph,
                graph_markdown_out if graph_markdown_out is not None else annotated_graph_path.with_suffix(".md"),
            )
            annotated_graph_html_path = write_attribution_graph_html(
                annotated_graph,
                graph_html_out if graph_html_out is not None else annotated_graph_path.with_suffix(".html"),
            )
        elif graph_markdown_out is not None:
            annotated_graph = annotate_graph_with_validation(graph, report)
            annotated_graph_markdown_path = write_attribution_graph_markdown(annotated_graph, graph_markdown_out)
            annotated_graph_path = None
            annotated_graph_html_path = (
                write_attribution_graph_html(annotated_graph, graph_html_out) if graph_html_out is not None else None
            )
        elif graph_html_out is not None:
            annotated_graph = annotate_graph_with_validation(graph, report)
            annotated_graph_path = None
            annotated_graph_markdown_path = None
            annotated_graph_html_path = write_attribution_graph_html(annotated_graph, graph_html_out)
        else:
            annotated_graph_html_path = None
        return WrittenGraphValidation(
            report=report,
            json_path=path,
            markdown_path=markdown_path,
            annotated_graph_path=annotated_graph_path,
            annotated_graph_markdown_path=annotated_graph_markdown_path,
            annotated_graph_html_path=annotated_graph_html_path,
        )
    if out is None:
        loaded_graph = json.loads(Path(graph).read_text(encoding="utf-8"))
        records = load_path_patch_records(path_records)
        return build_graph_validation_report(
            loaded_graph,
            path_records=records,
            graph_path=str(graph),
            top_k=top_k,
            min_effect=min_effect,
            min_specificity=min_specificity,
            min_effect_control_ratio=min_effect_control_ratio,
            min_prompt_count=min_prompt_count,
            min_sign_consistency=min_sign_consistency,
            require_controls=require_controls,
        )
    result = export_graph_validation_report(
        graph_path=graph,
        path_records_path=path_records,
        out_path=out,
        markdown_out_path=markdown_out,
        graph_out_path=graph_out,
        graph_markdown_out_path=graph_markdown_out,
        graph_html_out_path=graph_html_out,
        top_k=top_k,
        min_effect=min_effect,
        min_specificity=min_specificity,
        min_effect_control_ratio=min_effect_control_ratio,
        min_prompt_count=min_prompt_count,
        min_sign_consistency=min_sign_consistency,
        require_controls=require_controls,
    )
    return WrittenGraphValidation(
        report=result.report,
        json_path=result.json_path,
        markdown_path=result.markdown_path,
        annotated_graph_path=result.annotated_graph_path,
        annotated_graph_markdown_path=result.annotated_graph_markdown_path,
        annotated_graph_html_path=result.annotated_graph_html_path,
    )


def validate_hf_sae_paths(
    *,
    graph: str | Path,
    model: str,
    dataset: str | Path,
    source_sae: str | Path,
    target_sae: str | Path,
    path_records_out: str | Path,
    out: str | Path,
    criterion: str | None = None,
    markdown_out: str | Path | None = None,
    graph_out: str | Path | None = None,
    graph_markdown_out: str | Path | None = None,
    graph_html_out: str | Path | None = None,
    source_report: str | Path | None = None,
    target_report: str | Path | None = None,
    top_k: int = 8,
    pool: str = "last",
    strength_sweep: list[float] | None = None,
    random_source_controls: int = 2,
    control_seed: int = 0,
    score_behavior: bool = True,
    target_tokens: list[str] | None = None,
    device: str = "cpu",
    max_length: int = 128,
    min_effect: float = 0.05,
    min_specificity: float = 0.02,
    min_effect_control_ratio: float = 1.5,
    min_prompt_count: int = 3,
    min_sign_consistency: float = 0.75,
    require_controls: bool = True,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> HfSaePathValidation:
    """Rerun graph candidate SAE paths on a prompt set and validate the results."""
    result = export_hf_sae_path_validation(
        graph_path=graph,
        model_name=model,
        dataset_path=dataset,
        source_artifact_path=source_sae,
        target_artifact_path=target_sae,
        path_records_out_path=path_records_out,
        validation_out_path=out,
        criterion=criterion,
        markdown_out_path=markdown_out,
        graph_out_path=graph_out,
        graph_markdown_out_path=graph_markdown_out,
        graph_html_out_path=graph_html_out,
        source_report_path=source_report,
        target_report_path=target_report,
        top_k=top_k,
        pool=pool,
        strength_sweep=strength_sweep,
        random_source_controls=random_source_controls,
        control_seed=control_seed,
        score_behavior=score_behavior,
        target_tokens=target_tokens,
        device=device,
        max_length=max_length,
        min_effect=min_effect,
        min_specificity=min_specificity,
        min_effect_control_ratio=min_effect_control_ratio,
        min_prompt_count=min_prompt_count,
        min_sign_consistency=min_sign_consistency,
        require_controls=require_controls,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    return HfSaePathValidation(
        selected_path_pairs=result.selected_path_pairs,
        path_records_path=result.path_records_path,
        validation_json_path=result.validation.json_path,
        validation_markdown_path=result.validation.markdown_path,
        validation_report=result.validation.report,
        annotated_graph_path=result.validation.annotated_graph_path,
        annotated_graph_markdown_path=result.validation.annotated_graph_markdown_path,
        annotated_graph_html_path=result.validation.annotated_graph_html_path,
    )


def publish_hf_artifact(
    *,
    repo_id: str,
    paths: list[str | Path],
    repo_type: str = "dataset",
    private: bool = False,
    path_in_repo: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload interp-lab artifact",
    card_title: str | None = None,
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> PublishResult:
    """Publish reports, records, or artifact folders to Hugging Face Hub."""
    return _publish_hf_artifact(
        repo_id=repo_id,
        paths=paths,
        repo_type=repo_type,
        private=private,
        path_in_repo=path_in_repo,
        revision=revision,
        commit_message=commit_message,
        card_title=card_title,
        tags=tags,
        dry_run=dry_run,
    )


def scale_plan(
    *,
    model_params: float,
    tokens: int,
    d_model: int,
    selected_layers: int = 1,
    latent_dim: int = 131072,
    dtype: str = "bf16",
    shards: int | None = None,
    profile: str = "auto",
    artifact_format: str = "auto",
    target_shard_size_bytes: int | None = None,
    top_k_active: int = 64,
    causal_features: int = 128,
    causal_prompts: int = 256,
    interventions_per_feature: int = 2,
    train_batch_size: int = 4096,
    env_profile: dict[str, Any] | str | Path | None = None,
    from_env: bool = False,
    env_path: str | Path = ".",
) -> dict[str, Any]:
    """Estimate activation storage and execution shape for a large run."""
    resolved_env_profile = None
    if isinstance(env_profile, (str, Path)):
        resolved_env_profile = load_environment_profile(env_profile)
    elif env_profile is not None:
        resolved_env_profile = env_profile
    elif from_env:
        resolved_env_profile = collect_environment_profile(path=env_path)
    return ScalePlan(
        model_params=model_params,
        tokens=tokens,
        d_model=d_model,
        selected_layers=selected_layers,
        latent_dim=latent_dim,
        dtype=dtype,
        shards=shards,
        profile=profile,
        artifact_format=artifact_format,
        target_shard_size_bytes=target_shard_size_bytes,
        top_k_active=top_k_active,
        causal_features=causal_features,
        causal_prompts=causal_prompts,
        interventions_per_feature=interventions_per_feature,
        train_batch_size=train_batch_size,
        environment_profile=resolved_env_profile,
    ).to_dict()


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
    goodfire_top_k: int,
    goodfire_api_key_env: str,
    scope_source: str | None,
    scope_release: str | None,
    scope_sae_id: str | None,
    scope_feature_indexes: str | None,
    scope_max_features: int,
    scope_device: str,
    scope_force_download: bool,
    scope_feature_metadata: str | Path | None,
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
    if backend == "goodfire":
        return GoodfireFeatureProvider(
            top_k=goodfire_top_k,
            api_key_env=goodfire_api_key_env,
        )
    if backend == "scope":
        if scope_source is None:
            raise ValueError("scope_source is required with backend='scope'")
        if scope_release is None:
            raise ValueError("scope_release is required with backend='scope'")
        if scope_sae_id is None:
            raise ValueError("scope_sae_id is required with backend='scope'")
        return ScopeFeatureProvider(
            source=scope_source,
            release=scope_release,
            sae_id=scope_sae_id,
            feature_indices=parse_feature_indices(scope_feature_indexes),
            max_features=scope_max_features,
            device=scope_device,
            force_download=scope_force_download,
            feature_metadata=load_saelens_feature_metadata(scope_feature_metadata),
        )
    raise ValueError("backend must be one of: toy, jsonl, records, neuronpedia, saelens, goodfire, scope")


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


def _as_optional_list(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


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
