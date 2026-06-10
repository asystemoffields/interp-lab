from __future__ import annotations

import re
import shlex
from dataclasses import replace
from typing import Any

from interp_lab.schema import FeatureCard, InspectionReport

HIDDEN_DIMENSION_PATTERN = re.compile(r"^L\d+:D\d+$")
SAE_LATENT_PATTERN = re.compile(r"^SAE:(?:L\d+:)?F\d+$")


def add_inspection_agent_actions(report: InspectionReport) -> InspectionReport:
    """Attach stable next-action templates to inspection reports.

    The commands intentionally use placeholders for run-local artifacts. Agents can
    fill these from the active run config while humans still get copyable guidance.
    """
    cards = [_with_card_agent_actions(card, report.criterion.text) for card in report.cards]
    metadata = dict(report.metadata)
    metadata.setdefault("agent_next_actions", _report_actions(report))
    return replace(report, cards=cards, metadata=metadata)


def _with_card_agent_actions(card: FeatureCard, criterion: str) -> FeatureCard:
    actions = _feature_actions(card, criterion)
    if not actions:
        return card
    metadata = dict(card.metadata)
    existing = list(metadata.get("agent_next_actions", []))
    metadata["agent_next_actions"] = [*existing, *actions]
    return replace(card, metadata=metadata)


def _feature_actions(card: FeatureCard, criterion: str) -> list[dict[str, Any]]:
    if HIDDEN_DIMENSION_PATTERN.match(card.feature_id):
        return [
            _action(
                action_id="plan_hidden_suppression",
                title="Plan a suppression test for this hidden dimension",
                argv=[
                    "interp-lab",
                    "intervene",
                    "--model",
                    card.model,
                    "--criterion",
                    criterion,
                    "--dataset",
                    "<causal-prompts.jsonl>",
                    "--feature",
                    card.feature_id,
                    "--mode",
                    "suppress",
                    "--target-token",
                    "auto",
                    "--out",
                    "<interventions.jsonl>",
                    "--plan-out",
                    "<intervention-plan.json>",
                    "--dry-run",
                    "--json",
                ],
                requires=["scored causal prompt JSONL"],
            ),
            _action(
                action_id="plan_hidden_ablation",
                title="Plan an ablation test for this hidden dimension",
                argv=[
                    "interp-lab",
                    "intervene",
                    "--model",
                    card.model,
                    "--criterion",
                    criterion,
                    "--dataset",
                    "<causal-prompts.jsonl>",
                    "--feature",
                    card.feature_id,
                    "--mode",
                    "ablate",
                    "--target-token",
                    "auto",
                    "--out",
                    "<interventions.jsonl>",
                    "--plan-out",
                    "<intervention-plan.json>",
                    "--dry-run",
                    "--json",
                ],
                requires=["scored causal prompt JSONL"],
            ),
        ]
    if SAE_LATENT_PATTERN.match(card.feature_id):
        return [
            _action(
                action_id="plan_sae_suppression",
                title="Plan a suppression test for this SAE latent",
                argv=[
                    "interp-lab",
                    "intervene",
                    "--model",
                    card.model,
                    "--criterion",
                    criterion,
                    "--dataset",
                    "<causal-prompts.jsonl>",
                    "--feature",
                    card.feature_id,
                    "--sae",
                    "<sae.json>",
                    "--mode",
                    "suppress",
                    "--target-token",
                    "auto",
                    "--out",
                    "<interventions.jsonl>",
                    "--plan-out",
                    "<intervention-plan.json>",
                    "--dry-run",
                    "--json",
                ],
                requires=["scored causal prompt JSONL", "matching interp-lab SAE artifact"],
            ),
            _action(
                action_id="plan_sae_amplification",
                title="Plan an amplification test for this SAE latent",
                argv=[
                    "interp-lab",
                    "intervene",
                    "--model",
                    card.model,
                    "--criterion",
                    criterion,
                    "--dataset",
                    "<causal-prompts.jsonl>",
                    "--feature",
                    card.feature_id,
                    "--sae",
                    "<sae.json>",
                    "--mode",
                    "amplify",
                    "--target-token",
                    "auto",
                    "--out",
                    "<interventions.jsonl>",
                    "--plan-out",
                    "<intervention-plan.json>",
                    "--dry-run",
                    "--json",
                ],
                requires=["scored causal prompt JSONL", "matching interp-lab SAE artifact"],
            ),
        ]
    return []


def _report_actions(report: InspectionReport) -> list[dict[str, Any]]:
    actions = [
        _action(
            action_id="plan_top_feature_interventions",
            title="Plan causal tests for the top report features",
            argv=[
                "interp-lab",
                "intervene",
                "--model",
                report.model,
                "--criterion",
                report.criterion.text,
                "--dataset",
                "<causal-prompts.jsonl>",
                "--report",
                "<report.json>",
                "--top-k",
                str(min(8, len(report.cards) or 8)),
                "--mode",
                "suppress",
                "--target-token",
                "auto",
                "--out",
                "<interventions.jsonl>",
                "--plan-out",
                "<intervention-plan.json>",
                "--dry-run",
                "--json",
            ],
            requires=["inspection report JSON", "scored causal prompt JSONL"],
        ),
        _action(
            action_id="reinspect_with_interventions",
            title="Rebuild the report with intervention evidence",
            argv=[
                "interp-lab",
                "inspect",
                "--model",
                report.model,
                "--criterion",
                report.criterion.text,
                "--backend",
                "records",
                "--records",
                "<activation-records.jsonl>",
                "--interventions",
                "<interventions.jsonl>",
                "--out",
                "<causal-report-dir>",
                "--html-out",
                "<causal-report-dir>/report.html",
            ],
            requires=["activation records", "intervention records"],
        ),
        _action(
            action_id="export_attribution_graph",
            title="Build a graph from the causal report",
            argv=[
                "interp-lab",
                "export-attribution-graph",
                "--report",
                "<causal-report-dir>/report.json",
                "--out",
                "<graph.json>",
                "--markdown-out",
                "<graph.md>",
                "--html-out",
                "<graph.html>",
            ],
            requires=["causal report JSON"],
        ),
    ]
    return actions


def next_action(
    *,
    action_id: str,
    title: str,
    argv: list[str] | None = None,
    instruction: str | None = None,
    requires: list[str] | None = None,
) -> dict[str, Any]:
    """Build a canonical ``agent_next_actions`` entry.

    The canonical shape, shared by every report surface that emits
    ``agent_next_actions``:

    - ``id`` and ``title`` are always present.
    - Runnable CLI suggestions carry BOTH ``command`` (a shlex-quoted string) and
      ``argv`` (the same tokens as a list). Run-local values the emitter cannot
      know use ``<angle-bracket>`` placeholders.
    - Prose-only guidance carries ``instruction`` instead of ``command``/``argv``.
    - ``requires`` (optional) lists prerequisite artifacts for the action.
    """
    if (argv is None) == (instruction is None):
        raise ValueError("next_action needs exactly one of argv or instruction")
    action: dict[str, Any] = {"id": action_id, "title": title}
    if argv is not None:
        action["argv"] = [str(item) for item in argv]
        action["command"] = format_command(argv)
    else:
        action["instruction"] = str(instruction)
    if requires:
        action["requires"] = list(requires)
    return action


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


# Backwards-compatible aliases for pre-2.3 internal callers.
_action = next_action
_format_command = format_command

