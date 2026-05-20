from __future__ import annotations

import argparse
import json
from pathlib import Path

from interp_lab.text_embedding import configure_text_embedder

from interp_lab.adapters.goodfire import GoodfireFeatureProvider
from interp_lab.adapters.interventions import InterventionRecordRunner
from interp_lab.adapters.jsonl import JsonlFeatureProvider
from interp_lab.adapters.neuronpedia import (
    NeuronpediaClient,
    NeuronpediaFeatureProvider,
    load_neuronpedia_feature_refs,
)
from interp_lab.adapters.nla import NlaVerbalizer
from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.adapters.saelens import (
    SAELensFeatureProvider,
    load_saelens_feature_metadata,
    parse_feature_indices,
)
from interp_lab.adapters.scope import ScopeFeatureProvider
from interp_lab.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from interp_lab.criterion_lab import (
    build_criterion_assay_validation_parser,
    build_criterion_lab_parser,
    format_available_presets,
    render_criterion_assay_validation_text,
    run_criterion_assay_validation_from_args,
    run_criterion_lab_from_args,
)
from interp_lab.demo_sweep import (
    build_demo_sweep_parser,
    render_demo_sweep_text,
    run_demo_sweep_from_args,
)
from interp_lab.doctor import collect_diagnostics, diagnostics_to_json, diagnostics_to_text
from interp_lab.env_profile import build_environment_profile_parser, run_environment_profile_from_args
from interp_lab.explanation_reports import (
    build_explanation_consistency_parser,
    build_feature_search_parser,
    build_model_family_parser,
    build_text_pivot_match_parser,
    run_explanation_consistency_from_args,
    run_feature_search_from_args,
    run_model_family_from_args,
    run_text_pivot_match_from_args,
)
from interp_lab.feature_interventions import build_intervene_parser, run_intervene_from_args
from interp_lab.graphs import (
    build_graph_export_parser,
    build_graph_summary_parser,
    export_attribution_graph,
    export_attribution_graph_summary,
    run_graph_export_from_args,
    run_graph_summary_from_args,
)
from interp_lab.graph_validation import build_graph_validation_parser, run_graph_validation_from_args
from interp_lab.hf_contrast import build_contrast_parser, run_contrast_from_args
from interp_lab.hf_interventions import build_intervention_parser, run_interventions_from_args
from interp_lab.hf_publish import build_hf_publish_parser, run_hf_publish_from_args
from interp_lab.hf_records import (
    build_export_parser,
    build_prompt_dataset_parser,
    build_prepare_sae_prompt_datasets_parser,
    run_build_prompt_dataset_from_args,
    run_export_from_args,
    run_prepare_sae_prompt_datasets_from_args,
)
from interp_lab.hf_sae_paths import build_hf_sae_paths_parser, run_hf_sae_paths_from_args
from interp_lab.hf_sae_validation import build_hf_sae_validation_parser, run_hf_sae_validation_from_args
from interp_lab.match_validation import (
    build_match_validation_parser,
    export_match_validation_report,
    run_match_validation_from_args,
)
from interp_lab.nnsight_records import build_nnsight_export_parser, run_nnsight_export_from_args
from interp_lab.pipeline import inspect_model, match_reports
from interp_lab.reporting import (
    load_inspection_report,
    write_inspection_html,
    write_inspection_report,
    write_match_markdown,
    write_match_report,
)
from interp_lab.release_check import (
    build_release_check_parser,
    render_release_check_text,
    run_release_check_from_args,
)
from interp_lab.runs import RunOptions, run_config_file
from interp_lab.sae_training import build_train_sae_parser, run_train_sae_from_args
from interp_lab.scaling import build_scale_plan_parser, run_scale_plan_from_args
from interp_lab.transformerlens_records import (
    build_transformerlens_export_parser,
    run_transformerlens_export_from_args,
)
from interp_lab.web_app import build_web_app_parser, command_specs_from_parser, write_web_app
from interp_lab.workflows import build_init_run_parser, run_init_run_from_args


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
    inspect.add_argument(
        "--verbalizer",
        choices=["toy", "nla"],
        default="toy",
        help="Explanation adapter. Use nla with --nla-explanations for external NLA/autointerp records.",
    )
    inspect.add_argument(
        "--nla-explanations",
        help="JSON or JSONL explanation records keyed by feature_id for --verbalizer nla.",
    )
    inspect.add_argument(
        "--nla-min-confidence",
        type=float,
        help="Ignore NLA explanation records below this confidence or score.",
    )
    inspect.add_argument(
        "--nla-no-fallback",
        action="store_true",
        help="When an NLA record is missing, use the feature label instead of the heuristic fallback.",
    )
    inspect.add_argument("--out", default="reports/inspection", help="Output directory.")
    inspect.add_argument("--html-out", help="Optional output self-contained HTML feature report path.")
    inspect.set_defaults(func=run_inspect)

    match = subparsers.add_parser("match", help="Match feature cards across two inspection reports.")
    match.add_argument("--left", required=True, help="Left report.json.")
    match.add_argument("--right", required=True, help="Right report.json.")
    match.add_argument("--top-k", type=int, default=10, help="Number of matches to keep.")
    match.add_argument("--out", default="reports/matches.json", help="Output JSON path.")
    match.set_defaults(func=run_match)

    validate_matches = subparsers.add_parser(
        "validate-matches",
        help="Validate cross-model candidate feature matches.",
        parents=[build_match_validation_parser()],
        add_help=False,
    )
    validate_matches.set_defaults(func=run_validate_matches)

    consistency = subparsers.add_parser(
        "check-explanation-consistency",
        help="Compare feature explanations across paraphrased inspection reports.",
        parents=[build_explanation_consistency_parser()],
        add_help=False,
    )
    consistency.set_defaults(func=run_check_explanation_consistency)

    search_features = subparsers.add_parser(
        "search-features",
        help="Search inspection reports for features matching a natural-language explanation.",
        parents=[build_feature_search_parser()],
        add_help=False,
    )
    search_features.set_defaults(func=run_search_features)

    family_report = subparsers.add_parser(
        "compare-model-families",
        help="Compare feature reports across model families.",
        parents=[build_model_family_parser()],
        add_help=False,
    )
    family_report.set_defaults(func=run_compare_model_families)

    text_pivot = subparsers.add_parser(
        "match-text-pivot",
        help="Match features across reports using explanations as the cross-model bridge.",
        parents=[build_text_pivot_match_parser()],
        add_help=False,
    )
    text_pivot.set_defaults(func=run_match_text_pivot)

    demo = subparsers.add_parser("demo", help="Run two toy inspections and match their features.")
    demo.add_argument("--out", default="reports/demo", help="Output directory.")
    demo.set_defaults(func=run_demo)

    demo_sweep = subparsers.add_parser(
        "demo-sweep",
        help="Verify or execute the real-model demo suite and archive the sweep report.",
        parents=[build_demo_sweep_parser()],
        add_help=False,
    )
    demo_sweep.set_defaults(func=run_demo_sweep)

    studio = subparsers.add_parser(
        "studio",
        aliases=["web-app"],
        help="Write the self-contained interp-lab Studio HTML command builder.",
        parents=[build_web_app_parser()],
        add_help=False,
    )
    studio.set_defaults(func=run_studio)

    doctor = subparsers.add_parser("doctor", help="Check the local interp-lab environment.")
    doctor.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    doctor.set_defaults(func=run_doctor)

    profile_env = subparsers.add_parser(
        "profile-env",
        help="Profile local compute, storage, and routing options.",
        parents=[build_environment_profile_parser()],
        add_help=False,
    )
    profile_env.set_defaults(func=run_profile_env)

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

    init_run = subparsers.add_parser(
        "init-run",
        help="Write an editable run config for common workflows.",
        parents=[build_init_run_parser()],
        add_help=False,
    )
    init_run.set_defaults(func=run_init_run)

    criterion_lab = subparsers.add_parser(
        "criterion-lab",
        help="Write a discovery-first Criterion Lab run config for a behavior.",
        parents=[build_criterion_lab_parser()],
        add_help=False,
    )
    criterion_lab.set_defaults(func=run_criterion_lab)

    validate_assay = subparsers.add_parser(
        "validate-assay",
        help="Validate a user-authored Criterion Lab assay JSON file.",
        parents=[build_criterion_assay_validation_parser()],
        add_help=False,
    )
    validate_assay.set_defaults(func=run_validate_assay)

    export_hf = subparsers.add_parser(
        "export-hf-records",
        help="Export Hugging Face hidden states as activation records.",
        parents=[build_export_parser()],
        add_help=False,
    )
    export_hf.set_defaults(func=run_export_hf_records)

    build_prompts = subparsers.add_parser(
        "build-prompts",
        help="Build a prompt JSONL from user-written prompts.",
        parents=[build_prompt_dataset_parser()],
        add_help=False,
    )
    build_prompts.set_defaults(func=run_build_prompts)

    prepare_sae_prompts = subparsers.add_parser(
        "prepare-sae-prompts",
        help="Split scored prompts into train, causal, and held-out SAE datasets.",
        parents=[build_prepare_sae_prompt_datasets_parser()],
        add_help=False,
    )
    prepare_sae_prompts.set_defaults(func=run_prepare_sae_prompts)

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

    intervene = subparsers.add_parser(
        "intervene",
        help="Amplify, suppress, or ablate selected features and write intervention records.",
        parents=[build_intervene_parser()],
        add_help=False,
    )
    intervene.set_defaults(func=run_intervene)

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

    hf_sae_paths = subparsers.add_parser(
        "export-hf-sae-paths",
        help="Patch source SAE latents and measure downstream SAE latent paths.",
        parents=[build_hf_sae_paths_parser()],
        add_help=False,
    )
    hf_sae_paths.set_defaults(func=run_export_hf_sae_paths)

    hf_sae_validation = subparsers.add_parser(
        "validate-hf-sae-paths",
        help="Rerun graph candidate SAE paths on held-out HF prompts and validate them.",
        parents=[build_hf_sae_validation_parser()],
        add_help=False,
    )
    hf_sae_validation.set_defaults(func=run_validate_hf_sae_paths)

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

    summarize_graph = subparsers.add_parser(
        "summarize-attribution-graph",
        help="Write a compact attribution graph summary JSON for agents.",
        parents=[build_graph_summary_parser()],
        add_help=False,
    )
    summarize_graph.set_defaults(func=run_summarize_attribution_graph)

    validate_graph = subparsers.add_parser(
        "validate-attribution-graph",
        help="Validate measured attribution graph paths with path records.",
        parents=[build_graph_validation_parser()],
        add_help=False,
    )
    validate_graph.set_defaults(func=run_validate_attribution_graph)

    scale_plan = subparsers.add_parser(
        "plan-scale",
        help="Estimate storage and execution shape for large model runs.",
        parents=[build_scale_plan_parser()],
        add_help=False,
    )
    scale_plan.set_defaults(func=run_plan_scale)

    release_check = subparsers.add_parser(
        "release-check",
        help="Assess whether interp-lab is ready for a stable public release.",
        parents=[build_release_check_parser()],
        add_help=False,
    )
    release_check.set_defaults(func=run_release_check)

    seen: set[int] = set()
    for subparser in subparsers.choices.values():
        if id(subparser) in seen:
            continue
        seen.add(id(subparser))
        subparser.add_argument(
            "--text-embedder",
            dest="text_embedder",
            default=None,
            metavar="NAME",
            help=(
                "Text embedder for fingerprints, matching, and search: 'hash' "
                "(default, lexical/offline), 'minilm' (local semantic, needs the "
                "[embeddings] extra), or 'st:<model>'. Overrides INTERP_LAB_TEXT_EMBEDDER."
            ),
        )

    return parser


def run_inspect(args: argparse.Namespace) -> int:
    provider = _provider_from_args(args)
    intervention_runner = _intervention_runner_from_args(args)
    report = inspect_model(
        model=args.model,
        criterion_text=args.criterion,
        feature_provider=provider,
        verbalizer=_verbalizer_from_args(args),
        intervention_runner=intervention_runner,
        top_k=args.top_k,
    )
    json_path, markdown_path = write_inspection_report(report, args.out)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if args.html_out:
        html_path = write_inspection_html(report, args.html_out)
        print(f"Wrote {html_path}")
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


def run_validate_matches(args: argparse.Namespace) -> int:
    result = run_match_validation_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.html_path is not None:
        print(f"Wrote {result.html_path}")
    return 0


def run_check_explanation_consistency(args: argparse.Namespace) -> int:
    result = run_explanation_consistency_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.html_path is not None:
        print(f"Wrote {result.html_path}")
    return 0


def run_search_features(args: argparse.Namespace) -> int:
    result = run_feature_search_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.html_path is not None:
        print(f"Wrote {result.html_path}")
    return 0


def run_compare_model_families(args: argparse.Namespace) -> int:
    result = run_model_family_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.html_path is not None:
        print(f"Wrote {result.html_path}")
    return 0


def run_match_text_pivot(args: argparse.Namespace) -> int:
    result = run_text_pivot_match_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.html_path is not None:
        print(f"Wrote {result.html_path}")
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
    left_json, left_markdown = write_inspection_report(left, base / "model-a")
    right_json, right_markdown = write_inspection_report(right, base / "model-b")
    left_html = write_inspection_html(left, base / "model-a" / "report.html")
    right_html = write_inspection_html(right, base / "model-b" / "report.html")
    matches = match_reports(left, right, top_k=10)
    match_path = write_match_report(matches, base / "matches.json")
    match_markdown_path = write_match_markdown(matches, base / "matches.md")
    match_validation = export_match_validation_report(
        matches_path=match_path,
        out_path=base / "match-validation.json",
        html_out_path=base / "match-validation.html",
    )
    graph_path = export_attribution_graph(
        report_path=[left_json, right_json],
        out_path=base / "graph.json",
        markdown_out_path=base / "graph.md",
        html_out_path=base / "graph.html",
        include_similarity_edges=True,
        similarity_threshold=0.75,
    )
    graph_summary_path = export_attribution_graph_summary(
        graph_path=graph_path,
        out_path=base / "graph-summary.json",
    )
    studio_path = write_web_app(
        base / "studio.html",
        command_specs=command_specs_from_parser(build_parser()),
    )
    print(f"Wrote {left_json}")
    print(f"Wrote {left_markdown}")
    print(f"Wrote {left_html}")
    print(f"Wrote {right_json}")
    print(f"Wrote {right_markdown}")
    print(f"Wrote {right_html}")
    print(f"Wrote {match_path}")
    print(f"Wrote {match_markdown_path}")
    print(f"Wrote {match_validation.json_path}")
    print(f"Wrote {match_validation.markdown_path}")
    if match_validation.html_path is not None:
        print(f"Wrote {match_validation.html_path}")
    print(f"Wrote {graph_path}")
    print(f"Wrote {graph_path.with_suffix('.md')}")
    print(f"Wrote {graph_path.with_suffix('.html')}")
    print(f"Wrote {graph_summary_path}")
    print(f"Wrote {studio_path}")
    return 0


def run_demo_sweep(args: argparse.Namespace) -> int:
    result = run_demo_sweep_from_args(args, command_runner=main)
    if args.json:
        print(json.dumps(result.report, indent=2, sort_keys=True))
    else:
        print(render_demo_sweep_text(result.report))
    if result.path is not None:
        print(f"Wrote {result.path}")
    if args.strict and result.report["status"] != "passed":
        return 1
    return 0


def run_studio(args: argparse.Namespace) -> int:
    if args.serve:
        from interp_lab.web_server import serve_web_app

        serve_web_app(
            host=args.host,
            port=args.port,
            reports_dir=args.reports_dir,
            command_specs=command_specs_from_parser(build_parser()),
            open_browser=args.open,
        )
        return 0
    path = write_web_app(args.out, command_specs=command_specs_from_parser(build_parser()))
    print(f"Wrote {path}")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    diagnostics = collect_diagnostics()
    if args.json:
        print(diagnostics_to_json(diagnostics))
    else:
        print(diagnostics_to_text(diagnostics))
    return 0 if diagnostics["ok"] else 1


def run_profile_env(args: argparse.Namespace) -> int:
    run_environment_profile_from_args(args)
    return 0


def run_config(args: argparse.Namespace) -> int:
    return run_config_file(
        RunOptions(
            config_path=Path(args.config),
            dry_run=args.dry_run,
            variables=_parse_template_vars(args.var),
        ),
        command_runner=main,
    )


def run_init_run(args: argparse.Namespace) -> int:
    result = run_init_run_from_args(args)
    print(f"Wrote {result.path}")
    print(f"Run with: interp-lab run {result.path}")
    return 0


def run_criterion_lab(args: argparse.Namespace) -> int:
    if args.list_presets:
        print(format_available_presets(preset_dirs=args.preset_dir))
        return 0
    result = run_criterion_lab_from_args(args)
    print(f"Wrote {result.path}")
    print(f"Criterion: {result.criterion}")
    if args.execute:
        return run_config_file(
            RunOptions(config_path=Path(result.path)),
            command_runner=main,
        )
    print(f"Run with: interp-lab run {result.path}")
    return 0


def run_validate_assay(args: argparse.Namespace) -> int:
    result = run_criterion_assay_validation_from_args(args)
    print(render_criterion_assay_validation_text(result.report))
    if result.json_path is not None:
        print(f"Wrote {result.json_path}")
    status = result.report["status"]
    if status == "fail" or (args.fail_on_warning and status == "warn"):
        return 1
    return 0


def run_export_hf_records(args: argparse.Namespace) -> int:
    path = run_export_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_build_prompts(args: argparse.Namespace) -> int:
    summary = run_build_prompt_dataset_from_args(args)
    print(f"Wrote {summary.path}")
    print(
        f"Prompts: {summary.record_count} total, "
        f"{summary.positive_count} positive, {summary.negative_count} negative"
    )
    return 0


def run_prepare_sae_prompts(args: argparse.Namespace) -> int:
    summary = run_prepare_sae_prompt_datasets_from_args(args)
    print(f"Wrote {summary.train_path}")
    print(f"Wrote {summary.causal_path}")
    print(f"Wrote {summary.validation_path}")
    print(f"Wrote {summary.manifest_path}")
    counts = summary.counts["splits"]
    print(
        "Prompts: "
        f"train={counts['train']['record_count']}, "
        f"causal={counts['causal']['record_count']}, "
        f"validation={counts['validation']['record_count']}"
    )
    for advisory in summary.advisories:
        print(f"Advisory: {advisory}")
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


def run_intervene(args: argparse.Namespace) -> int:
    result = run_intervene_from_args(args)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if result.dry_run:
        print("Planned feature interventions")
    if result.records_path is not None:
        print(f"Wrote {result.records_path}")
    if result.plan_path is not None:
        print(f"Wrote {result.plan_path}")
    if result.records_path is None and result.plan_path is None:
        print(json.dumps(result.plan, indent=2, sort_keys=True))
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


def run_export_hf_sae_paths(args: argparse.Namespace) -> int:
    path = run_hf_sae_paths_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_validate_hf_sae_paths(args: argparse.Namespace) -> int:
    result = run_hf_sae_validation_from_args(args)
    print(f"Validated {len(result.selected_path_pairs)} path pair(s)")
    print(f"Wrote {result.path_records_path}")
    print(f"Wrote {result.validation.json_path}")
    print(f"Wrote {result.validation.markdown_path}")
    if result.validation.annotated_graph_path is not None:
        print(f"Wrote {result.validation.annotated_graph_path}")
    if result.validation.annotated_graph_markdown_path is not None:
        print(f"Wrote {result.validation.annotated_graph_markdown_path}")
    if result.validation.annotated_graph_html_path is not None:
        print(f"Wrote {result.validation.annotated_graph_html_path}")
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
    if args.markdown_out:
        print(f"Wrote {args.markdown_out}")
    if args.html_out:
        print(f"Wrote {args.html_out}")
    return 0


def run_summarize_attribution_graph(args: argparse.Namespace) -> int:
    path = run_graph_summary_from_args(args)
    print(f"Wrote {path}")
    return 0


def run_validate_attribution_graph(args: argparse.Namespace) -> int:
    result = run_graph_validation_from_args(args)
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    if result.annotated_graph_path is not None:
        print(f"Wrote {result.annotated_graph_path}")
    if result.annotated_graph_markdown_path is not None:
        print(f"Wrote {result.annotated_graph_markdown_path}")
    if result.annotated_graph_html_path is not None:
        print(f"Wrote {result.annotated_graph_html_path}")
    return 0


def run_plan_scale(args: argparse.Namespace) -> int:
    run_scale_plan_from_args(args)
    return 0


def run_release_check(args: argparse.Namespace) -> int:
    result = run_release_check_from_args(args)
    if args.json:
        print(json.dumps(result.report, indent=2, sort_keys=True))
    else:
        print(render_release_check_text(result.report))
    if result.path is not None:
        print(f"Wrote {result.path}")
    if args.strict and not result.report["ready_for_stable_release"]:
        return 1
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


def _verbalizer_from_args(args: argparse.Namespace):
    fallback = None if args.nla_no_fallback else ToyVerbalizer()
    if args.verbalizer == "nla":
        return NlaVerbalizer(
            args.nla_explanations,
            min_confidence=args.nla_min_confidence,
            fallback=fallback,
        )
    return ToyVerbalizer()


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
    configure_text_embedder(getattr(args, "text_embedder", None))
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
