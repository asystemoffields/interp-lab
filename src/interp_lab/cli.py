from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from interp_lab import __version__
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
    write_inspection_csv,
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
from interp_lab.run_diff import build_compare_runs_parser, run_compare_runs_from_args
from interp_lab.runs import RunOptions, run_config_file
from interp_lab.sae_training import build_train_sae_parser, run_train_sae_from_args
from interp_lab.scaling import build_scale_plan_parser, run_scale_plan_from_args
from interp_lab.transformerlens_records import (
    build_transformerlens_export_parser,
    run_transformerlens_export_from_args,
)
from interp_lab.web_app import build_web_app_parser, command_specs_from_parser, write_web_app
from interp_lab.workflows import build_init_run_parser, run_init_run_from_args


_COMMAND_GROUPS_EPILOG = """\
New here? Run:  interp-lab demo --out reports/demo   (then open reports/demo/model-a/report.html)

Commands by purpose:

  Start here
    quickstart    a short guided walkthrough of the concepts
    doctor        check your local environment and optional extras
    demo          full toy tour: inspect, causally test, match, grade (no GPU, no downloads)
    studio        write/serve a point-and-click command builder

  Inspect & explain one model
    inspect, search-features, check-explanation-consistency, criterion-lab, init-run, run

  Compare across models
    match, validate-matches, compare-model-families, match-text-pivot

  Causal testing & attribution graphs
    intervene, export-attribution-graph, summarize-attribution-graph, validate-attribution-graph

  Bring your own model (optional extras: [hf] [transformerlens] [nnsight] [saelens])
    export-hf-records, export-transformerlens-records, export-nnsight-records,
    export-hf-interventions, export-hf-contrast, train-sae, export-hf-sae-paths,
    validate-hf-sae-paths, build-prompts, prepare-sae-prompts, publish-hf-artifact

  Agent integration
    capabilities  one-call discovery: commands, Python API, schemas, environment
    mcp           serve interp-lab tools over the Model Context Protocol (stdio)

  Utilities
    profile-env, plan-scale, validate-assay, release-check

Run `interp-lab <command> --help` for the options of any command.
"""


_QUICKSTART_TEXT = """\
interp-lab quickstart
=====================

interp-lab finds the internal features of a model that track a plain-language
criterion, explains them, tests whether they actually CAUSE the behavior, and
compares them across models -- grading how much evidence backs each claim.

1. Check your setup:
     interp-lab doctor

2. Run the no-download toy tour, then open the report it writes:
     interp-lab demo --out reports/demo
     # open reports/demo/model-a/report.html in a browser

3. Read the numbers (every report carries a "Metric notes" legend):
     Importance      overall rank score -- a heuristic blend; a ranking, not a probability
     Association     how strongly the feature co-activates with the criterion
     Causal effect   measured change from an intervention (shown only when records are
                     attached; otherwise the column reads "Criterion score" -- a
                     correlation, never a causal claim)
     Specificity     the criterion effect minus side effects on unrelated behavior
     Strong causal   the specificity-adjusted causal signal that flags real evidence

4. Run your own criterion (still toy, still no download):
     interp-lab inspect --model toy/demo --backend toy \\
       --criterion "the text gives cooking instructions" \\
       --out reports/cooking --html-out reports/cooking/report.html

5. Plug in a real model when you're ready (installs are optional extras):
     pip install "interp-lab[hf]"          # Hugging Face activations
     pip install "interp-lab[saelens]"     # pretrained SAEs
     pip install "interp-lab[embeddings]"  # semantic text matching (MiniLM)
   Then use --backend records / saelens / neuronpedia, or `interp-lab init-run`
   to scaffold an editable, reproducible run config.

The golden rule: correlation (association / criterion score) is a hypothesis;
only an intervention (causal effect) is evidence. interp-lab keeps the two apart
and grades every claim -- see `interp-lab validate-matches`.

Full command map:  interp-lab --help
"""


def run_quickstart(args: argparse.Namespace) -> int:
    print(_QUICKSTART_TEXT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interp-lab",
        description="Criterion-driven feature discovery and cross-model activation matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_COMMAND_GROUPS_EPILOG,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"interp-lab {__version__}",
        help="Show the installed interp-lab version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quickstart = subparsers.add_parser(
        "quickstart",
        aliases=["tutorial"],
        help="Print a short, guided getting-started walkthrough.",
    )
    quickstart.set_defaults(func=run_quickstart)

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
    inspect.add_argument(
        "--csv-out",
        help="Optional CSV of the ranked features and their scores (opens in any spreadsheet).",
    )
    inspect.set_defaults(func=run_inspect)

    match = subparsers.add_parser("match", help="Match feature cards across two inspection reports.")
    match.add_argument("--left", required=True, help="Left report.json.")
    match.add_argument("--right", required=True, help="Right report.json.")
    match.add_argument("--top-k", type=int, default=10, help="Number of matches to keep.")
    match.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop candidate matches scoring below this similarity (0..1).",
    )
    match.add_argument(
        "--weights",
        help=(
            "Override fingerprint component weights for sensitivity analysis, e.g. "
            "text=0.4,causal=0.3,activation=0.2,decoder=0.1 (keys: text, activation, decoder, causal)."
        ),
    )
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

    compare_runs = subparsers.add_parser(
        "compare-runs",
        help="Diff two inspection reports: rank drift, score deltas, added/dropped features.",
        parents=[build_compare_runs_parser()],
        add_help=False,
    )
    compare_runs.set_defaults(func=run_compare_runs)

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

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Print the machine-readable interp-lab capabilities payload for agents.",
    )
    capabilities.add_argument("--json", action="store_true", help="Print the payload as JSON on stdout.")
    capabilities.add_argument("--out", help="Optional output JSON file path.")
    capabilities.set_defaults(func=run_capabilities)

    mcp = subparsers.add_parser(
        "mcp",
        help="Serve interp-lab tools over the Model Context Protocol (stdio).",
    )
    mcp.set_defaults(func=run_mcp)

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
    if getattr(args, "csv_out", None):
        csv_path = write_inspection_csv(report, args.csv_out)
        print(f"Wrote {csv_path}")
    return 0


def run_match(args: argparse.Namespace) -> int:
    left = load_inspection_report(args.left)
    right = load_inspection_report(args.right)
    weights = _parse_match_weights(args.weights) if getattr(args, "weights", None) else None
    report = match_reports(
        left, right, top_k=args.top_k, min_score=getattr(args, "min_score", 0.0), weights=weights
    )
    if weights is not None and report.matches and not any(
        key in match.components for match in report.matches for key in weights
    ):
        # None of the --weights components the user named were ever actually comparable
        # (each was gated out: missing/length-mismatched signature, absent causal vector,
        # causal_provenance='none', or a text-embedder mismatch). The ranking then carries
        # no signal from the chosen weighting -- say so loudly instead of writing silent noise.
        print(
            "interp-lab: warning: none of the --weights component(s) you named "
            f"({', '.join(weights)}) are comparable for these reports "
            "(missing/mismatched signatures, absent causal vectors, causal_provenance='none', "
            "or a text-embedder mismatch). The ranking does not reflect them; add a comparable "
            "component (e.g. activation, decoder) or drop --weights.",
            file=sys.stderr,
        )
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


def run_compare_runs(args: argparse.Namespace) -> int:
    result = run_compare_runs_from_args(args)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.markdown_path}")
    return 0


def run_demo(args: argparse.Namespace) -> int:
    base = Path(args.out)
    criterion = "the model is aware it is being evaluated"
    # measured=True so the demo demonstrates a real causal claim (strong causal scores,
    # signed effects, intervention CIs/controls) -- not just a correlational ranking.
    left = inspect_model(
        model="toy/model-a",
        criterion_text=criterion,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(measured=True),
        top_k=8,
    )
    right = inspect_model(
        model="toy/model-b",
        criterion_text=criterion,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(measured=True),
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
    index_path = _write_demo_index(base, criterion)
    _ = (
        index_path,
        left_markdown,
        right_json,
        right_markdown,
        right_html,
        match_path,
        match_markdown_path,
        graph_summary_path,
        studio_path,
    )  # all written above; the summary highlights the few worth opening first
    _print_demo_summary(base, criterion)
    return 0


def _print_demo_summary(base: Path, criterion: str) -> None:
    base = Path(base)
    print()
    print("interp-lab demo complete -- a full toy tour, no GPU and no downloads.")
    print()
    print(f'It inspected two toy models for the criterion "{criterion}", ranked and')
    print("causally tested their features with interventions, matched them across models,")
    print("graded the evidence, and assembled an attribution graph.")
    print()
    print(f"Open this first:  {base / 'index.html'}")
    print("  (a one-page hub linking the feature reports, graded matches, graph, and studio)")
    print()
    print(f"Everything is under {base}/ (.json for machines, .md to read, .html to explore).")
    print()
    print("Next, try your own criterion (still no download):")
    print("  interp-lab inspect --model toy/demo --backend toy \\")
    print('    --criterion "the text gives cooking instructions" \\')
    print("    --out reports/cooking --html-out reports/cooking/report.html")
    print()
    print("New to mechanistic interpretability? `interp-lab quickstart` explains what the numbers mean.")


_DEMO_INDEX_ARTIFACTS = [
    ("Feature reports", [
        ("model-a/report.html", "Model A: ranked features for the criterion, with measured causal evidence"),
        ("model-b/report.html", "Model B: the same, for a second toy model"),
    ]),
    ("Cross-model comparison", [
        ("match-validation.html", "Candidate equivalent features across the two models, graded by evidence"),
        ("matches.md", "The raw candidate matches and their similarity components"),
    ]),
    ("Attribution graph", [
        ("graph.html", "Features and their cross-layer connections, spanning both models"),
    ]),
    ("Build your own runs", [
        ("studio.html", "A point-and-click command builder for every interp-lab command"),
    ]),
]


def _write_demo_index(base: Path, criterion: str) -> Path:
    """Write a tiny self-contained landing page linking the demo's artifacts."""
    import html as _html

    sections = []
    for title, items in _DEMO_INDEX_ARTIFACTS:
        rows = "\n".join(
            f'      <li><a href="{_html.escape(href, quote=True)}">{_html.escape(href)}</a>'
            f'<span>{_html.escape(desc)}</span></li>'
            for href, desc in items
        )
        sections.append(f"    <section>\n      <h2>{_html.escape(title)}</h2>\n      <ul>\n{rows}\n      </ul>\n    </section>")
    body = "\n".join(sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>interp-lab demo</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #f7f7f4; color: #1d2528;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5; }}
  main {{ width: min(820px, calc(100vw - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 6px; }}
  p.lede {{ color: #5d686e; margin: 0 0 28px; }}
  section {{ background: #fff; border: 1px solid #d9dedb; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; }}
  h2 {{ font-size: 15px; margin: 0 0 10px; color: #0f766e; }}
  ul {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }}
  li {{ display: grid; gap: 2px; }}
  a {{ color: #285e9e; font-weight: 650; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  span {{ color: #5d686e; font-size: 13px; }}
  footer {{ color: #5d686e; font-size: 13px; margin-top: 22px; }}
  code {{ background: #eef1ef; border-radius: 5px; padding: 1px 5px; }}
</style></head>
<body><main>
  <h1>interp-lab demo</h1>
  <p class="lede">A complete toy tour for the criterion &ldquo;{_html.escape(criterion)}&rdquo; &mdash; ranked features, measured causal evidence, graded cross-model matches, and an attribution graph. No GPU, no downloads.</p>
{body}
  <footer>New to the metrics? Run <code>interp-lab quickstart</code>. Build your own run with <code>interp-lab inspect --backend toy …</code>.</footer>
</main></body></html>
"""
    path = Path(base) / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def run_demo_sweep(args: argparse.Namespace) -> int:
    result = run_demo_sweep_from_args(args, command_runner=main)
    if args.json:
        print(json.dumps(result.report, indent=2, sort_keys=True))
    else:
        print(render_demo_sweep_text(result.report))
    if result.path is not None:
        # Keep stdout pure JSON under --json so `... --json | jq` works; the confirmation
        # goes to stderr instead.
        print(f"Wrote {result.path}", file=sys.stderr if args.json else sys.stdout)
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
        print(f"Wrote {result.path}", file=sys.stderr if args.json else sys.stdout)
    if args.strict and not result.report["ready_for_stable_release"]:
        return 1
    return 0


def run_capabilities(args: argparse.Namespace) -> int:
    from interp_lab.capabilities import build_capabilities, write_capabilities

    payload = build_capabilities()
    if args.json:
        # Keep stdout pure JSON so `interp-lab capabilities --json | jq` works.
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out:
        path = write_capabilities(args.out)
        print(f"Wrote {path}", file=sys.stderr if args.json else sys.stdout)
    if not args.json and not args.out:
        print(f"interp-lab {__version__} capabilities")
        print(f"- CLI commands: {len(payload['commands'])}")
        print(f"- Python API exports: {len(payload['python_api']['exports'])}")
        print(f"- Artifact schemas: {len(payload['python_api']['schemas'])}")
        print(f"- MCP server: {payload['conventions']['mcp']['command']} (stdio)")
        print()
        print("Full machine-readable payload: interp-lab capabilities --json")
    return 0


def run_mcp(args: argparse.Namespace) -> int:
    from interp_lab.mcp_server import run_mcp_server

    _ = args  # the MCP server takes no CLI options; stdin/stdout carry the protocol
    return run_mcp_server()


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
    # Always place the markdown next to the JSON with a .md suffix. The old branch
    # turned a suffixless --out (e.g. `--out reports/run`) into a directory, which then
    # collided with the JSON written at that same path (FileExistsError on real use).
    return Path(out_path).with_suffix(".md")


_MATCH_WEIGHT_KEYS = ("text", "activation", "decoder", "causal")


def _parse_match_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit("--weights entries must be KEY=VALUE, e.g. text=0.4,causal=0.3")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in _MATCH_WEIGHT_KEYS:
            raise SystemExit(f"--weights key {key!r} must be one of {', '.join(_MATCH_WEIGHT_KEYS)}")
        try:
            weights[key] = float(value)
        except ValueError as exc:
            raise SystemExit(f"--weights value for {key!r} must be a number") from exc
    if not weights:
        raise SystemExit("--weights must set at least one of: " + ", ".join(_MATCH_WEIGHT_KEYS))
    return weights


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
        # Inside the try so an unknown --text-embedder reports a clean error instead
        # of a traceback. OSError covers FileNotFoundError/PermissionError; ImportError
        # turns a missing optional extra into a one-line message instead of a stack.
        configure_text_embedder(getattr(args, "text_embedder", None))
        return int(args.func(args))
    except (RuntimeError, ValueError, OSError, ImportError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
