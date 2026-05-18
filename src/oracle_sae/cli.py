from __future__ import annotations

import argparse
from pathlib import Path

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
from oracle_sae.doctor import collect_diagnostics, diagnostics_to_json, diagnostics_to_text
from oracle_sae.graphs import build_graph_export_parser, run_graph_export_from_args
from oracle_sae.hf_contrast import build_contrast_parser, run_contrast_from_args
from oracle_sae.hf_interventions import build_intervention_parser, run_interventions_from_args
from oracle_sae.hf_publish import build_hf_publish_parser, run_hf_publish_from_args
from oracle_sae.hf_records import build_export_parser, run_export_from_args
from oracle_sae.nnsight_records import build_nnsight_export_parser, run_nnsight_export_from_args
from oracle_sae.pipeline import inspect_model, match_reports
from oracle_sae.reporting import (
    load_inspection_report,
    write_inspection_report,
    write_match_markdown,
    write_match_report,
)
from oracle_sae.runs import RunOptions, run_config_file
from oracle_sae.sae_training import build_train_sae_parser, run_train_sae_from_args
from oracle_sae.scaling import build_scale_plan_parser, run_scale_plan_from_args
from oracle_sae.transformerlens_records import (
    build_transformerlens_export_parser,
    run_transformerlens_export_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interp-lab",
        description="Criterion-driven feature discovery and cross-model activation matching.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Rank and explain features for a criterion.")
    inspect.add_argument("--model", required=True, help="Model identifier.")
    inspect.add_argument("--criterion", required=True, help="Natural-language criterion.")
    inspect.add_argument(
        "--backend",
        default="toy",
        choices=["toy", "jsonl", "records", "neuronpedia", "saelens", "goodfire", "scope"],
        help="Feature backend.",
    )
    inspect.add_argument("--features", help="JSONL feature dump for --backend jsonl.")
    inspect.add_argument("--records", help="Activation records JSONL for --backend records.")
    inspect.add_argument(
        "--interventions",
        help="Optional intervention records JSONL with ablation/amplification/patching outcomes.",
    )
    inspect.add_argument(
        "--allow-intervention-criterion-mismatch",
        action="store_true",
        help="Use intervention rows even when their criterion text differs from --criterion.",
    )
    inspect.add_argument(
        "--require-interventions",
        action="store_true",
        help="When --interventions is set, give untested features zero criterion effect.",
    )
    inspect.add_argument(
        "--neuronpedia-feature",
        action="append",
        default=[],
        help="Neuronpedia feature ref or URL. Repeat for multiple features.",
    )
    inspect.add_argument(
        "--neuronpedia-features",
        help="Text or JSON file of Neuronpedia feature refs for --backend neuronpedia.",
    )
    inspect.add_argument(
        "--neuronpedia-base-url",
        default="https://www.neuronpedia.org",
        help="Neuronpedia base URL.",
    )
    inspect.add_argument("--saelens-release", help="SAELens release name or Hugging Face repo.")
    inspect.add_argument("--saelens-sae-id", help="SAELens SAE id inside the release.")
    inspect.add_argument(
        "--saelens-feature-indexes",
        help="Comma-separated feature indexes or inclusive ranges, e.g. 0,5,10-12.",
    )
    inspect.add_argument(
        "--saelens-max-features",
        type=int,
        default=32,
        help="Max SAELens features to import when indexes are omitted.",
    )
    inspect.add_argument("--saelens-device", default="cpu", help="Device passed to SAELens.")
    inspect.add_argument(
        "--saelens-force-download",
        action="store_true",
        help="Force SAELens to re-download the pretrained SAE.",
    )
    inspect.add_argument(
        "--saelens-feature-metadata",
        help="Optional JSON object keyed by feature index with labels/examples.",
    )
    inspect.add_argument(
        "--goodfire-top-k",
        type=int,
        default=32,
        help="Max Goodfire features to import.",
    )
    inspect.add_argument(
        "--goodfire-api-key-env",
        default="GOODFIRE_API_KEY",
        help="Environment variable containing a Goodfire API key.",
    )
    inspect.add_argument(
        "--scope-source",
        choices=["gemma-scope", "qwen-scope"],
        help="Named SAE suite for --backend scope.",
    )
    inspect.add_argument("--scope-release", help="SAELens release or HF repo for --backend scope.")
    inspect.add_argument("--scope-sae-id", help="SAE id inside --scope-release.")
    inspect.add_argument(
        "--scope-feature-indexes",
        help="Comma-separated feature indexes or inclusive ranges, e.g. 0,5,10-12.",
    )
    inspect.add_argument(
        "--scope-max-features",
        type=int,
        default=32,
        help="Max scope features to import when indexes are omitted.",
    )
    inspect.add_argument("--scope-device", default="cpu", help="Device passed to the scope loader.")
    inspect.add_argument(
        "--scope-force-download",
        action="store_true",
        help="Force the scope loader to re-download the pretrained SAE.",
    )
    inspect.add_argument(
        "--scope-feature-metadata",
        help="Optional JSON object keyed by feature index with labels/examples.",
    )
    inspect.add_argument("--top-k", type=int, default=8, help="Number of feature cards to keep.")
    inspect.add_argument("--out", default="reports/inspection", help="Output directory.")
    inspect.set_defaults(func=run_inspect)

    match = subparsers.add_parser("match", help="Match feature cards across two inspection reports.")
    match.add_argument("--left", required=True, help="Left report.json.")
    match.add_argument("--right", required=True, help="Right report.json.")
    match.add_argument("--top-k", type=int, default=10, help="Number of matches to keep.")
    match.add_argument("--out", default="reports/matches.json", help="Output JSON path.")
    match.set_defaults(func=run_match)

    demo = subparsers.add_parser("demo", help="Run two toy inspections and match their features.")
    demo.add_argument("--out", default="reports/demo", help="Output directory.")
    demo.set_defaults(func=run_demo)

    doctor = subparsers.add_parser("doctor", help="Check the local interp-lab environment.")
    doctor.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    doctor.set_defaults(func=run_doctor)

    run = subparsers.add_parser("run", help="Run a reproducible workflow from a JSON/TOML/YAML config.")
    run.add_argument("config", help="Run config path.")
    run.add_argument("--dry-run", action="store_true", help="Print the planned commands without running them.")
    run.add_argument(
        "--var",
        action="append",
        default=[],
        help="Template variable as KEY=VALUE. Usable in config strings as {KEY} or ${KEY}.",
    )
    run.set_defaults(func=run_config)

    export_hf = subparsers.add_parser(
        "export-hf-records",
        help="Export Hugging Face hidden states as activation records.",
        parents=[build_export_parser()],
        add_help=False,
    )
    export_hf.set_defaults(func=run_export_hf_records)

    export_tl = subparsers.add_parser(
        "export-transformerlens-records",
        help="Export TransformerLens hook activations as activation records.",
        parents=[build_transformerlens_export_parser()],
        add_help=False,
    )
    export_tl.set_defaults(func=run_export_transformerlens_records)

    export_nnsight = subparsers.add_parser(
        "export-nnsight-records",
        help="Export NNsight trace activations as activation records.",
        parents=[build_nnsight_export_parser()],
        add_help=False,
    )
    export_nnsight.set_defaults(func=run_export_nnsight_records)

    hf_interventions = subparsers.add_parser(
        "export-hf-interventions",
        help="Export HF hidden-dimension ablations as intervention records.",
        parents=[build_intervention_parser()],
        add_help=False,
    )
    hf_interventions.set_defaults(func=run_export_hf_interventions)

    hf_contrast = subparsers.add_parser(
        "export-hf-contrast",
        help="Export a contrast-direction feature and optional steering interventions.",
        parents=[build_contrast_parser()],
        add_help=False,
    )
    hf_contrast.set_defaults(func=run_export_hf_contrast)

    train_sae = subparsers.add_parser(
        "train-sae",
        help="Train an on-demand SAE from activation records or HF activations.",
        parents=[build_train_sae_parser()],
        add_help=False,
    )
    train_sae.set_defaults(func=run_train_sae)

    publish_hf = subparsers.add_parser(
        "publish-hf-artifact",
        help="Publish reports and artifacts to Hugging Face Hub.",
        parents=[build_hf_publish_parser()],
        add_help=False,
    )
    publish_hf.set_defaults(func=run_publish_hf_artifact)

    export_graph = subparsers.add_parser(
        "export-attribution-graph",
        help="Export a report as an attribution graph JSON.",
        parents=[build_graph_export_parser()],
        add_help=False,
    )
    export_graph.set_defaults(func=run_export_attribution_graph)

    scale_plan = subparsers.add_parser(
        "plan-scale",
        help="Estimate storage and execution shape for large model runs.",
        parents=[build_scale_plan_parser()],
        add_help=False,
    )
    scale_plan.set_defaults(func=run_plan_scale)

    return parser


def run_inspect(args: argparse.Namespace) -> int:
    provider = _provider_from_args(args)
    intervention_runner = _intervention_runner_from_args(args)
    report = inspect_model(
        model=args.model,
        criterion_text=args.criterion,
        feature_provider=provider,
        verbalizer=ToyVerbalizer(),
        intervention_runner=intervention_runner,
        top_k=args.top_k,
    )
    json_path, markdown_path = write_inspection_report(report, args.out)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


def run_match(args: argparse.Namespace) -> int:
    left = load_inspection_report(args.left)
    right = load_inspection_report(args.right)
    report = match_reports(left, right, top_k=args.top_k)
    path = write_match_report(report, args.out)
    markdown_path = write_match_markdown(report, _match_markdown_path(args.out))
    print(f"Wrote {path}")
    print(f"Wrote {markdown_path}")
    return 0


def run_demo(args: argparse.Namespace) -> int:
    base = Path(args.out)
    criterion = "the model is aware it is being evaluated"
    left = inspect_model(
        model="toy/model-a",
        criterion_text=criterion,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=8,
    )
    right = inspect_model(
        model="toy/model-b",
        criterion_text=criterion,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=8,
    )
    left_json, _ = write_inspection_report(left, base / "model-a")
    right_json, _ = write_inspection_report(right, base / "model-b")
    matches = match_reports(left, right, top_k=10)
    match_path = write_match_report(matches, base / "matches.json")
    match_markdown_path = write_match_markdown(matches, base / "matches.md")
    print(f"Wrote {left_json}")
    print(f"Wrote {right_json}")
    print(f"Wrote {match_path}")
    print(f"Wrote {match_markdown_path}")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    diagnostics = collect_diagnostics()
    if args.json:
        print(diagnostics_to_json(diagnostics))
    else:
        print(diagnostics_to_text(diagnostics))
    return 0 if diagnostics["ok"] else 1


def run_config(args: argparse.Namespace) -> int:
    return run_config_file(
        RunOptions(
            config_path=Path(args.config),
            dry_run=args.dry_run,
            variables=_parse_template_vars(args.var),
        ),
        command_runner=main,
    )


def run_export_hf_records(args: argparse.Namespace) -> int:
    path = run_export_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_export_transformerlens_records(args: argparse.Namespace) -> int:
    path = run_transformerlens_export_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_export_nnsight_records(args: argparse.Namespace) -> int:
    path = run_nnsight_export_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_export_hf_interventions(args: argparse.Namespace) -> int:
    path = run_interventions_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_export_hf_contrast(args: argparse.Namespace) -> int:
    records_path, interventions_path, feature_id = run_contrast_from_args(args)
    print(f"Wrote {records_path}")
    if interventions_path is not None:
        print(f"Wrote {interventions_path}")
    print(f"Feature {feature_id}")
    return 0


def run_train_sae(args: argparse.Namespace) -> int:
    artifact_path, records_path = run_train_sae_from_args(args)
    print(f"Wrote {artifact_path}")
    if records_path is not None:
        print(f"Wrote {records_path}")
    return 0


def run_publish_hf_artifact(args: argparse.Namespace) -> int:
    result = run_hf_publish_from_args(args)
    prefix = "Would upload" if result.dry_run else "Uploaded"
    print(f"{prefix} {len(result.uploaded)} artifact path(s) to {result.repo_type}/{result.repo_id}")
    for path in result.uploaded:
        print(f"- {path}")
    return 0


def run_export_attribution_graph(args: argparse.Namespace) -> int:
    path = run_graph_export_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_plan_scale(args: argparse.Namespace) -> int:
    run_scale_plan_from_args(args)
    return 0


def _provider_from_args(args: argparse.Namespace):
    if args.backend == "toy":
        return ToyFeatureProvider()
    if args.backend == "jsonl":
        if not args.features:
            raise SystemExit("--features is required with --backend jsonl")
        return JsonlFeatureProvider(args.features)
    if args.backend == "records":
        if not args.records:
            raise SystemExit("--records is required with --backend records")
        return ActivationRecordFeatureProvider(args.records)
    if args.backend == "neuronpedia":
        refs = list(args.neuronpedia_feature or [])
        if args.neuronpedia_features:
            refs.extend(load_neuronpedia_feature_refs(args.neuronpedia_features))
        if not refs:
            raise SystemExit(
                "--neuronpedia-feature or --neuronpedia-features is required with --backend neuronpedia"
            )
        client = NeuronpediaClient(base_url=args.neuronpedia_base_url)
        return NeuronpediaFeatureProvider(refs, client=client)
    if args.backend == "goodfire":
        return GoodfireFeatureProvider(
            top_k=args.goodfire_top_k,
            api_key_env=args.goodfire_api_key_env,
        )
    if args.backend == "scope":
        if not args.scope_source:
            raise SystemExit("--scope-source is required with --backend scope")
        if not args.scope_release:
            raise SystemExit("--scope-release is required with --backend scope")
        if not args.scope_sae_id:
            raise SystemExit("--scope-sae-id is required with --backend scope")
        try:
            feature_indices = parse_feature_indices(args.scope_feature_indexes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        metadata = load_saelens_feature_metadata(args.scope_feature_metadata)
        return ScopeFeatureProvider(
            source=args.scope_source,
            release=args.scope_release,
            sae_id=args.scope_sae_id,
            feature_indices=feature_indices,
            max_features=args.scope_max_features,
            device=args.scope_device,
            force_download=args.scope_force_download,
            feature_metadata=metadata,
        )
    if not args.saelens_release:
        raise SystemExit("--saelens-release is required with --backend saelens")
    if not args.saelens_sae_id:
        raise SystemExit("--saelens-sae-id is required with --backend saelens")
    try:
        feature_indices = parse_feature_indices(args.saelens_feature_indexes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    metadata = load_saelens_feature_metadata(args.saelens_feature_metadata)
    return SAELensFeatureProvider(
        release=args.saelens_release,
        sae_id=args.saelens_sae_id,
        feature_indices=feature_indices,
        max_features=args.saelens_max_features,
        device=args.saelens_device,
        force_download=args.saelens_force_download,
        feature_metadata=metadata,
    )


def _intervention_runner_from_args(args: argparse.Namespace):
    fallback = ToyInterventionRunner()
    if not args.interventions:
        return fallback
    return InterventionRecordRunner(
        args.interventions,
        fallback_runner=fallback,
        require_criterion_match=not args.allow_intervention_criterion_mismatch,
        require_records=args.require_interventions,
    )


def _match_markdown_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    if path.suffix:
        return path.with_suffix(".md")
    return path / "matches.md"


def _parse_template_vars(items: list[str]) -> dict[str, str]:
    variables = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--var values must be KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit("--var values must be KEY=VALUE")
        variables[key] = value
    return variables


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
