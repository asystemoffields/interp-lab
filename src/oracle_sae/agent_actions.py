from __future__ import annotations

import re
import shlex
from dataclasses import replace
from typing import Any

from oracle_sae.schema import FeatureCard, InspectionReport

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


def _action(
    *,
    action_id: str,
    title: str,
    argv: list[str],
    requires: list[str],
) -> dict[str, Any]:
    return {
        "id": action_id,
        "title": title,
        "argv": argv,
        "command": _format_command(argv),
        "requires": requires,
    }


def _format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)

