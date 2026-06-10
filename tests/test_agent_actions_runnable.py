"""Schema enforcement point for the unified `agent_next_actions` shape.

Every surface that emits `agent_next_actions` — inspection reports (report- and
card-level), run diffs, environment profiles, release checks, demo sweeps, and
the explanation reports (feature search, consistency, text pivot) — must emit
the canonical object shape:

    {id, title, command?, argv?, instruction?, requires?}

with id+title always present, EXACTLY ONE of command/instruction, command
always paired with argv (command == shlex-joined argv), and <angle-bracket>
placeholders for run-local values. Argv-bearing actions must parse against the
real CLI parser after placeholder substitution: a suggestion that exits 2 when
pasted back into the CLI is worse than no suggestion.

Since 3.0.0 the canonical keys are the ONLY keys: the legacy aliases that 2.3
kept for one release (release-check `next_action`, explanation reports
`description`, per-result `agent_next_action*` flat keys) must no longer be
emitted anywhere. This file pins both the canonical shape and the absence of
the legacy keys.
"""

import json
import re
import shlex
from pathlib import Path

import pytest

from interp_lab.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from interp_lab.cli import build_parser
from interp_lab.demo_sweep import build_demo_sweep_report
from interp_lab.env_profile import collect_environment_profile
from interp_lab.explanation_reports import (
    build_explanation_consistency_report,
    build_feature_search_report,
    build_text_pivot_match_report,
)
from interp_lab.pipeline import inspect_model
from interp_lab.release_check import build_release_readiness_report
from interp_lab.reporting import write_inspection_report
from interp_lab.run_diff import build_run_diff_report
from interp_lab.schema import Criterion, FeatureEvidence

CRITERION = "the model is aware it is being evaluated"
OTHER_CRITERION = "the text discusses cooking recipes"
PLACEHOLDER = re.compile(r"^<.+>$")

# Placeholders for values with constrained parse-time types/choices get a valid
# sample; everything path-like becomes a dummy path.
PLACEHOLDER_VALUES = {
    "<size>": "1B",
    "<tokens>": "1M",
    "<width>": "1024",
    "<backend>": "records",
    "<profile>": "local-cpu",
}


def test_every_agent_next_action_is_canonical_and_runnable(tmp_path: Path):
    surfaces = _collect_agent_next_actions(tmp_path)
    assert surfaces, "expected agent_next_actions to be emitted"
    seen_surfaces = {context.split(":")[0] for context, _ in surfaces}
    assert seen_surfaces >= {
        "inspection_report",
        "feature_card",
        "run_diff",
        "env_profile",
        "release_check",
        "demo_sweep",
        "demo_sweep_demo",
        "feature_search",
        "feature_search_result",
        "explanation_consistency",
        "text_pivot",
        "text_pivot_match",
    }, f"missing surfaces: {seen_surfaces}"
    parser = build_parser()
    for context, action in surfaces:
        _assert_canonical(context, action, parser)


def test_legacy_flat_keys_are_not_emitted(tmp_path: Path):
    # The 2.3-era flat keys were removed in 3.0.0: per-result/per-match payloads
    # carry only the canonical agent_next_actions list.
    left_path, right_path = _write_toy_reports(tmp_path)

    search = build_feature_search_report(reports=[left_path], query="evaluation awareness", top_k=3)
    assert search["results"]
    for result in search["results"]:
        assert "agent_next_action" not in result
        assert "agent_next_action_argv" not in result
        assert "agent_next_action_requires" not in result

    pivot = build_text_pivot_match_report(left_reports=[left_path], right_reports=[right_path], top_k=3, per_left=1)
    assert pivot["matches"]
    for match in pivot["matches"]:
        assert "agent_next_action" not in match
        assert "agent_next_action_argv" not in match
        assert "agent_next_action_requires" not in match


CANONICAL_ACTION_KEYS = {"id", "title", "command", "argv", "instruction", "requires"}


def _assert_canonical(context: str, action, parser) -> None:
    assert isinstance(action, dict), f"{context}: action is not an object: {action!r}"
    extra = set(action) - CANONICAL_ACTION_KEYS
    assert not extra, (
        f"{context}: action carries non-canonical keys {sorted(extra)} "
        f"(3.0.0 removed all legacy aliases): {action!r}"
    )
    assert isinstance(action.get("id"), str) and action["id"], f"{context}: missing id: {action!r}"
    assert isinstance(action.get("title"), str) and action["title"], f"{context}: missing title: {action!r}"
    has_command = "command" in action
    has_instruction = "instruction" in action
    assert has_command != has_instruction, (
        f"{context} [{action['id']}]: expected exactly one of command/instruction, got "
        f"command={has_command} instruction={has_instruction}"
    )
    if has_instruction:
        assert isinstance(action["instruction"], str) and action["instruction"], (
            f"{context} [{action['id']}]: empty instruction"
        )
        assert "argv" not in action, f"{context} [{action['id']}]: instruction action must not carry argv"
    if has_command:
        argv = action.get("argv")
        assert isinstance(argv, list) and argv and all(isinstance(item, str) and item for item in argv), (
            f"{context} [{action['id']}]: command requires a non-empty string argv"
        )
        expected = " ".join(shlex.quote(item) for item in argv)
        assert action["command"] == expected, (
            f"{context} [{action['id']}]: command is not the shlex-joined argv:\n"
            f"  command: {action['command']}\n  argv:    {expected}"
        )
        _assert_parses(parser, argv, f"{context} [{action['id']}]")
    if "requires" in action:
        requires = action["requires"]
        assert isinstance(requires, list) and requires and all(
            isinstance(item, str) and item for item in requires
        ), f"{context} [{action['id']}]: requires must be a non-empty list of strings"


def _assert_parses(parser, argv, context: str) -> None:
    filled = _substitute_placeholders(list(argv))
    try:
        parser.parse_args(filled)
    except SystemExit as exc:
        pytest.fail(f"agent action argv does not parse ({context}): {argv} -> exit code {exc.code}")


def _substitute_placeholders(argv: list[str]) -> list[str]:
    filled = [
        PLACEHOLDER_VALUES.get(str(token), "dummy-path") if PLACEHOLDER.match(str(token)) else str(token)
        for token in argv
    ]
    # Report-embedded argvs may carry the entry-point prefix; the parser sees what
    # follows it.
    if filled and filled[0] == "interp-lab":
        filled = filled[1:]
    return filled


def _collect_agent_next_actions(tmp_path: Path) -> list[tuple[str, dict]]:
    actions: list[tuple[str, dict]] = []

    # Inspection report + feature cards (hidden-dimension and SAE-latent ids get
    # per-card intervention plans).
    left = _toy_report("toy/model-a")
    for action in left.metadata.get("agent_next_actions", []):
        actions.append(("inspection_report", action))
    card_report = inspect_model(
        model="toy/model-a",
        criterion_text=CRITERION,
        feature_provider=_InterventionTargetProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(measured=True),
        top_k=2,
    )
    card_actions = [
        (f"feature_card:{card.feature_id}", action)
        for card in card_report.cards
        for action in card.metadata.get("agent_next_actions", [])
    ]
    assert card_actions, "expected per-card intervention actions"
    actions.extend(card_actions)

    # Run diff — different criteria force added/dropped so every action emits.
    other = _toy_report("toy/model-a", criterion=OTHER_CRITERION)
    diff = build_run_diff_report(left, other)
    for action in diff["agent_next_actions"]:
        actions.append(("run_diff", action))

    # Environment profile.
    profile = collect_environment_profile(path=tmp_path, env={}, probe_accelerators=False)
    for action in profile["agent_next_actions"]:
        actions.append(("env_profile", action))

    # Release check — an empty repo root produces warn/blocker next actions.
    release = build_release_readiness_report(tmp_path)
    assert release["agent_next_actions"]
    for action in release["agent_next_actions"]:
        actions.append(("release_check", action))

    # Demo sweep (verify-only payload, incomplete -> demo and sweep actions;
    # manifest-authored string actions must be coerced to canonical objects).
    _write_demo_manifest(tmp_path)
    sweep = build_demo_sweep_report(repo_root=tmp_path, manifest_dir="examples/real_model_demos")
    for action in sweep["agent_next_actions"]:
        actions.append(("demo_sweep", action))
    for demo in sweep["demos"]:
        for action in demo.get("agent_next_actions", []):
            actions.append((f"demo_sweep_demo:{demo.get('id')}", action))

    # Explanation reports.
    left_path, right_path = _write_toy_reports(tmp_path)
    search = build_feature_search_report(reports=[left_path], query="evaluation awareness", top_k=3)
    assert search["results"]
    for action in search["agent_next_actions"]:
        actions.append(("feature_search", action))
    for result in search["results"]:
        for action in result["agent_next_actions"]:
            actions.append((f"feature_search_result:{result['feature_id']}", action))

    consistency = build_explanation_consistency_report(reports=[left_path, right_path])
    for action in consistency["agent_next_actions"]:
        actions.append(("explanation_consistency", action))

    pivot = build_text_pivot_match_report(left_reports=[left_path], right_reports=[right_path], top_k=3, per_left=1)
    assert pivot["matches"]
    for action in pivot["agent_next_actions"]:
        actions.append(("text_pivot", action))
    for match in pivot["matches"]:
        for action in match["agent_next_actions"]:
            actions.append(
                (f"text_pivot_match:{match['left_feature_id']}->{match['right_feature_id']}", action)
            )

    return actions


def _write_toy_reports(tmp_path: Path) -> tuple[Path, Path]:
    left_path, _ = write_inspection_report(_toy_report("toy/model-a"), tmp_path / "left")
    right_path, _ = write_inspection_report(_toy_report("toy/model-b"), tmp_path / "right")
    return left_path, right_path


def _toy_report(model: str, criterion: str = CRITERION):
    return inspect_model(
        model=model,
        criterion_text=criterion,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(measured=True),
        top_k=4,
    )


class _InterventionTargetProvider:
    """One hidden dimension + one SAE latent, so both per-card action templates emit."""

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        return [
            FeatureEvidence(
                feature_id=feature_id,
                model=model,
                layer=6,
                label=label,
                examples=["p1: activation=1.0 | call the tool"],
                activation_signature=[1.0, 0.0],
                decoder_signature=[0.0, 1.0],
                causal_effects={"criterion": 0.2, "specificity": 0.1},
                source="activation-records",
            )
            for feature_id, label in [("L6:D12", "hidden dimension"), ("SAE:L6:F7", "sae latent")]
        ]


def _write_demo_manifest(root: Path) -> Path:
    doc = root / "docs" / "DEMO.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Demo\n", encoding="utf-8")
    manifest_dir = root / "examples" / "real_model_demos"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "interp-lab.real_model_demo.v1",
        "id": "sample-demo",
        "title": "Sample Demo",
        "model": "toy/model",
        "criterion": "sample behavior",
        "workflow": "sample-workflow",
        "doc": "docs/DEMO.md",
        "estimated_runtime": "quick",
        "commands": [
            {"name": f"step {index}", "argv": ["demo", "--out", f"reports/toy-{index}"]}
            for index in range(1, 4)
        ],
        "expected_artifacts": [
            {
                "path": f"reports/demo/{name}",
                "kind": "json",
                "why_it_matters": "Confirms an artifact exists.",
                "interpretation_notes": "Review before claiming evidence.",
            }
            for name in ["a.json", "b.html", "c.jsonl"]
        ],
        "evidence_checks": ["Check artifacts.", "Check limitations."],
        # Historical manifests carry plain strings; the sweep must coerce them.
        "agent_next_actions": ["Archive outputs.", "Repeat on held-out prompts."],
    }
    path = manifest_dir / "sample-demo.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
