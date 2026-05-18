from __future__ import annotations

import json
from pathlib import Path

from oracle_sae.schema import InspectionReport, MatchReport


def write_inspection_report(report: InspectionReport, out_dir: str | Path) -> tuple[Path, Path]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "report.json"
    markdown_path = path / "report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_inspection_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def write_match_report(report: MatchReport, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_match_markdown(report: MatchReport, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_match_markdown(report), encoding="utf-8")
    return path


def load_inspection_report(path: str | Path) -> InspectionReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InspectionReport.from_dict(data)


def render_inspection_markdown(report: InspectionReport) -> str:
    lines = [
        f"# interp-lab Report: {report.model}",
        "",
        f"Criterion: {report.criterion.text}",
        "",
        "## Top Features",
        "",
    ]
    for index, card in enumerate(report.cards, start=1):
        layer = "unknown layer" if card.layer is None else f"layer {card.layer}"
        lines.extend(
            [
                f"### {index}. {card.feature_id} ({layer})",
                "",
                f"Label: {card.label}",
                "",
                f"Importance: {card.importance:.3f}",
                "",
                f"Association: {card.association:.3f} | Effect: {card.causal_effect:.3f} | "
                f"Specificity: {card.specificity:.3f} | Stability: {card.stability:.3f}",
                "",
            ]
        )
        evidence = _evidence_line(card)
        if evidence:
            lines.extend([evidence, ""])
        direction = _direction_line(
            card.causal_effects.get(
                "signed_causal_effect",
                card.causal_effects.get(
                    "signed_association",
                    card.metadata.get("signed_association"),
                ),
            )
        )
        if direction:
            lines.extend([direction, ""])
        strong = card.causal_effects.get("strong_causal_score")
        if strong is not None:
            lines.extend([f"Strong causal score: {float(strong):.3f}", ""])
        intervention = _intervention_lines(card.metadata.get("interventions"))
        if intervention:
            lines.extend(intervention)
        lines.extend([card.explanation, "", "Examples:"])
        for example in card.examples[:3]:
            lines.append(f"- {example}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_match_markdown(report: MatchReport) -> str:
    lines = [
        f"# interp-lab Match Report: {report.left_model} -> {report.right_model}",
        "",
        "## Candidate Equivalents",
        "",
    ]
    for index, match in enumerate(report.matches, start=1):
        lines.extend(
            [
                f"### {index}. {match.left_feature_id} -> {match.right_feature_id}",
                "",
                f"Score: {match.score:.3f}",
                "",
            ]
        )
        if match.left_label or match.right_label:
            lines.extend(
                [
                    f"Left: {match.left_label or match.left_feature_id}",
                    "",
                    f"Right: {match.right_label or match.right_feature_id}",
                    "",
                ]
            )
        if match.left_signed_effect is not None or match.right_signed_effect is not None:
            left = _format_optional_effect(match.left_signed_effect)
            right = _format_optional_effect(match.right_signed_effect)
            lines.extend([f"Signed effects: left={left}, right={right}", ""])
        if match.components:
            component_text = ", ".join(
                f"{name}={value:.3f}" for name, value in sorted(match.components.items())
            )
            lines.extend([f"Components: {component_text}", ""])
    return "\n".join(lines).strip() + "\n"


def _direction_line(raw_value: object) -> str:
    if raw_value is None:
        return ""
    value = float(raw_value)
    if value > 0.05:
        return f"Direction: promotes criterion (signed effect {value:.3f})"
    if value < -0.05:
        return f"Direction: suppresses criterion (signed effect {value:.3f})"
    return f"Direction: weak signed effect ({value:.3f})"


def _evidence_line(card) -> str:
    if isinstance(card.metadata.get("interventions"), dict):
        return "Evidence: causal intervention records"
    if card.source == "activation-records":
        return "Evidence: activation/criterion association"
    if card.source in {"activation-records", "hf-hidden-state"}:
        return "Evidence: activation/criterion association"
    if card.source in {"neuronpedia", "saelens", "goodfire", "gemma-scope", "qwen-scope"}:
        return f"Evidence: imported {card.source} feature evidence"
    return ""


def _intervention_lines(raw_value: object) -> list[str]:
    if not isinstance(raw_value, dict):
        return []
    count = raw_value.get("count")
    mean_directed = raw_value.get("mean_directed_effect")
    side_effect = raw_value.get("mean_side_effect")
    lines = [
        f"Interventions: n={count}, mean directed effect={float(mean_directed):.3f}"
    ]
    if side_effect is not None:
        lines[0] += f", mean side effect={float(side_effect):.3f}"
    ci_low = raw_value.get("criterion_ci_low")
    ci_high = raw_value.get("criterion_ci_high")
    if ci_low is not None and ci_high is not None:
        lines[0] += f", 95% CI=[{float(ci_low):.3f}, {float(ci_high):.3f}]"
    controls = raw_value.get("controls")
    if isinstance(controls, dict) and controls.get("count"):
        lines.append(
            "Controls: "
            f"n={controls.get('count')}, "
            f"mean abs effect={float(controls.get('mean_abs_directed_effect', 0.0)):.3f}"
        )
    examples = raw_value.get("examples", [])
    if examples:
        lines.append("")
        lines.append("Intervention examples:")
        for example in examples[:3]:
            lines.append(f"- {example}")
    lines.append("")
    return lines


def _format_optional_effect(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"
