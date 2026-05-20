from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interp_lab.reporting import load_match_report
from interp_lab.schema import CandidateMatch, MatchReport


DEFAULT_MIN_SCORE = 0.75
DEFAULT_MIN_COMPONENT = 0.65
DEFAULT_MIN_CAUSAL_COMPONENT = 0.65
DEFAULT_MAX_SIGNED_EFFECT_DELTA = 0.15
DEFAULT_MIN_ABS_SIGNED_EFFECT = 0.02


@dataclass(frozen=True)
class MatchValidationWriteResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    html_path: Path | None = None


def export_match_validation_report(
    *,
    matches_path: str | Path,
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    html_out_path: str | Path | None = None,
    top_k: int | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_component: float = DEFAULT_MIN_COMPONENT,
    min_causal_component: float = DEFAULT_MIN_CAUSAL_COMPONENT,
    max_signed_effect_delta: float = DEFAULT_MAX_SIGNED_EFFECT_DELTA,
    min_abs_signed_effect: float = DEFAULT_MIN_ABS_SIGNED_EFFECT,
) -> MatchValidationWriteResult:
    match_file = Path(matches_path)
    report = build_match_validation_report(
        load_match_report(match_file),
        match_path=str(match_file),
        top_k=top_k,
        min_score=min_score,
        min_component=min_component,
        min_causal_component=min_causal_component,
        max_signed_effect_delta=max_signed_effect_delta,
        min_abs_signed_effect=min_abs_signed_effect,
    )
    json_path = Path(out_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out_path) if markdown_out_path is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_match_validation_markdown(report), encoding="utf-8")
    html_path = None
    if html_out_path is not None:
        html_path = write_match_validation_html(report, html_out_path)
    return MatchValidationWriteResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        html_path=html_path,
    )


def build_match_validation_report(
    report: MatchReport,
    *,
    match_path: str | None = None,
    top_k: int | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_component: float = DEFAULT_MIN_COMPONENT,
    min_causal_component: float = DEFAULT_MIN_CAUSAL_COMPONENT,
    max_signed_effect_delta: float = DEFAULT_MAX_SIGNED_EFFECT_DELTA,
    min_abs_signed_effect: float = DEFAULT_MIN_ABS_SIGNED_EFFECT,
) -> dict[str, Any]:
    thresholds = {
        "min_score": float(min_score),
        "min_component": float(min_component),
        "min_causal_component": float(min_causal_component),
        "max_signed_effect_delta": float(max_signed_effect_delta),
        "min_abs_signed_effect": float(min_abs_signed_effect),
    }
    matches = report.matches[:top_k] if top_k is not None else list(report.matches)
    validations = [_validate_match(match, thresholds=thresholds) for match in matches]
    status_counts = _counts(validations, "status")
    claim_grade_counts = _counts(validations, "claim_grade")
    run_assessment = _validation_run_assessment(validations)
    return {
        "schema_version": "interp-lab.match_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_path": match_path,
        "left_model": report.left_model,
        "right_model": report.right_model,
        "thresholds": thresholds,
        "summary": {
            "match_count": len(matches),
            "validated_count": status_counts.get("validated", 0),
            "needs_causal_evidence_count": status_counts.get("needs_causal_evidence", 0),
            "plausible_count": status_counts.get("plausible", 0),
            "contradicted_count": status_counts.get("contradicted", 0),
            "weak_count": status_counts.get("weak", 0),
            "claim_grade_counts": dict(sorted(claim_grade_counts.items())),
            "overall_claim_grade": run_assessment["overall_claim_grade"],
            "recommended_next_action": run_assessment["recommended_next_action"],
            "status_counts": dict(sorted(status_counts.items())),
        },
        "run_assessment": run_assessment,
        "agent_next_actions": _validation_agent_next_actions(run_assessment),
        "validations": validations,
    }


def render_match_validation_markdown(report: dict[str, Any]) -> str:
    assessment = report.get("run_assessment", {})
    lines = [
        "# Cross-Model Match Validation",
        "",
        f"Left model: `{report.get('left_model', '')}`",
        f"Right model: `{report.get('right_model', '')}`",
        f"Matches checked: `{report['summary']['match_count']}`",
        f"Overall: `{assessment.get('overall_claim_grade', report['summary'].get('overall_claim_grade', ''))}`",
        f"Recommended next action: {assessment.get('recommended_next_action', report['summary'].get('recommended_next_action', ''))}",
        "",
        "| Status | Claim | Match | Score | Causal | Signed effect delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report.get("validations", []):
        lines.append(
            "| "
            f"{item['status']} | "
            f"{item['claim_grade']} | "
            f"`{item['left_feature_id']} -> {item['right_feature_id']}` | "
            f"{_markdown_number(item['score'])} | "
            f"{_markdown_number(item.get('causal_component'))} | "
            f"{_markdown_number(item.get('signed_effect_delta'))} |"
        )
    lines.extend(["", "## Notes", ""])
    for item in report.get("validations", []):
        lines.append(
            f"- `{item['left_feature_id']} -> {item['right_feature_id']}`: "
            f"{item['interpretation']} Next: {item['next_action']}"
        )
    lines.append("")
    actions = report.get("agent_next_actions", [])
    if actions:
        lines.extend(["## Agent Next Actions", ""])
        for action in actions:
            lines.append(f"- `{action['id']}`: {action['title']}: `{action['command']}`")
        lines.append("")
    return "\n".join(lines)


def write_match_validation_html(report: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_match_validation_html(report), encoding="utf-8")
    return path


def render_match_validation_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    assessment = report.get("run_assessment", {})
    validations = list(report.get("validations", []))
    status_options = "\n".join(
        f'<option value="{_attr(status)}">{_h(status)} ({count})</option>'
        for status, count in sorted(summary.get("status_counts", {}).items())
    )
    metric_cards = "\n".join(
        _summary_card(label, summary.get(key))
        for label, key in [
            ("Matches", "match_count"),
            ("Validated", "validated_count"),
            ("Need Causal Evidence", "needs_causal_evidence_count"),
            ("Contradicted", "contradicted_count"),
            ("Weak", "weak_count"),
        ]
    )
    table_rows = "\n".join(_validation_table_row(item) for item in validations)
    detail_cards = "\n".join(_validation_detail_card(item) for item in validations)
    actions = "\n".join(_agent_action_card(action) for action in report.get("agent_next_actions", []))
    action_section = (
        f"""
      <section class="panel">
        <h2>Agent Next Actions</h2>
        <div class="actions">{actions}</div>
      </section>
        """
        if actions
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>interp-lab Match Validation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #1d2528;
      --muted: #5c686d;
      --line: #d9dedb;
      --panel: #ffffff;
      --accent: #0f766e;
      --validated: #147a3f;
      --needs: #946200;
      --plausible: #285e9e;
      --contradicted: #b42318;
      --weak: #5f6670;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    header {{
      display: grid;
      gap: 18px;
      margin-bottom: 20px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 30px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    h3 {{ font-size: 16px; }}
    code {{
      padding: 1px 5px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #f2f4f3;
      font-size: 0.92em;
    }}
    .subhead {{ color: var(--muted); max-width: 920px; }}
    .model-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #fff;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
    }}
    .pill.status-validated {{ color: var(--validated); border-color: #9fd3b5; background: #eef9f2; }}
    .pill.status-needs-causal-evidence {{ color: var(--needs); border-color: #e5c06f; background: #fff7df; }}
    .pill.status-plausible {{ color: var(--plausible); border-color: #a9c7ed; background: #eef6ff; }}
    .pill.status-contradicted {{ color: var(--contradicted); border-color: #efa7a1; background: #fff1ef; }}
    .pill.status-weak {{ color: var(--weak); border-color: #cbd1d7; background: #f4f5f6; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 24px; font-weight: 750; }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
      margin-top: 14px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(170px, 240px) auto;
      gap: 10px;
      align-items: center;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .visible-count {{ color: var(--muted); font-size: 13px; text-align: right; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
      font-size: 14px;
    }}
    th {{
      text-align: left;
      color: var(--muted);
      font-size: 12px;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
    }}
    td {{
      padding: 11px 8px;
      border-bottom: 1px solid #ecefed;
      vertical-align: top;
    }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .match-ref {{ display: grid; gap: 4px; }}
    .label-pair {{ color: var(--muted); font-size: 12px; }}
    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
    }}
    .match-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 12px;
    }}
    .match-card h3 {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }}
    .reason-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .reason {{
      padding: 3px 7px;
      border-radius: 999px;
      background: #eef0ef;
      color: #3f494d;
      font-size: 12px;
    }}
    .meters {{ display: grid; gap: 7px; }}
    .meter {{
      display: grid;
      grid-template-columns: 90px 1fr 48px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .bar {{
      height: 7px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8ecea;
    }}
    .fill {{ height: 100%; background: var(--accent); }}
    .action-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: grid;
      gap: 6px;
    }}
    .command {{
      overflow-wrap: anywhere;
      color: #273033;
    }}
    [hidden] {{ display: none !important; }}
    @media (max-width: 760px) {{
      main {{ width: min(100vw - 20px, 1180px); padding-top: 20px; }}
      h1 {{ font-size: 24px; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .visible-count {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Cross-Model Match Validation</h1>
        <p class="subhead">{_h(assessment.get("summary", ""))}</p>
      </div>
      <div class="model-line">
        <span>Left <code>{_h(report.get("left_model", ""))}</code></span>
        <span>Right <code>{_h(report.get("right_model", ""))}</code></span>
        <span class="pill">{_h(summary.get("overall_claim_grade", ""))}</span>
      </div>
      <div class="summary-grid">{metric_cards}</div>
    </header>
    <section class="panel">
      <h2>Review</h2>
      <p class="subhead">{_h(summary.get("recommended_next_action", ""))}</p>
    </section>
    <section class="panel">
      <div class="toolbar">
        <input id="match-search" type="search" placeholder="Filter by feature, label, claim, or reason">
        <select id="status-filter" aria-label="Filter by status">
          <option value="">All statuses</option>
          {status_options}
        </select>
        <div id="visible-count" class="visible-count">{len(validations)} visible</div>
      </div>
    </section>
    <section class="panel">
      <h2>Candidate Matches</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Claim</th>
              <th>Match</th>
              <th class="num">Score</th>
              <th class="num">Causal</th>
              <th class="num">Signed Delta</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Evidence Details</h2>
      <div class="details">{detail_cards}</div>
    </section>
    {action_section}
  </main>
  <script>
    const search = document.getElementById("match-search");
    const statusFilter = document.getElementById("status-filter");
    const visibleCount = document.getElementById("visible-count");
    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const status = statusFilter.value;
      let visibleRows = 0;
      document.querySelectorAll("[data-match]").forEach((node) => {{
        const statusMatch = !status || node.dataset.status === status;
        const searchMatch = !query || (node.dataset.search || "").includes(query);
        const show = statusMatch && searchMatch;
        node.hidden = !show;
        if (show && node.dataset.match === "row") visibleRows += 1;
      }});
      visibleCount.textContent = `${{visibleRows}} visible`;
    }}
    search.addEventListener("input", applyFilters);
    statusFilter.addEventListener("change", applyFilters);
  </script>
</body>
</html>
"""


def build_match_validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate cross-model candidate feature matches.")
    parser.add_argument("--matches", required=True, help="Match report JSON from `interp-lab match`.")
    parser.add_argument("--out", required=True, help="Output validation JSON path.")
    parser.add_argument("--markdown-out", help="Output validation Markdown path. Defaults to --out with .md.")
    parser.add_argument("--html-out", help="Optional output self-contained HTML report path.")
    parser.add_argument("--top-k", type=int, help="Validate only the top K matches from the report.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-component", type=float, default=DEFAULT_MIN_COMPONENT)
    parser.add_argument("--min-causal-component", type=float, default=DEFAULT_MIN_CAUSAL_COMPONENT)
    parser.add_argument("--max-signed-effect-delta", type=float, default=DEFAULT_MAX_SIGNED_EFFECT_DELTA)
    parser.add_argument("--min-abs-signed-effect", type=float, default=DEFAULT_MIN_ABS_SIGNED_EFFECT)
    return parser


def run_match_validation_from_args(args: argparse.Namespace) -> MatchValidationWriteResult:
    return export_match_validation_report(
        matches_path=args.matches,
        out_path=args.out,
        markdown_out_path=args.markdown_out,
        html_out_path=args.html_out,
        top_k=args.top_k,
        min_score=args.min_score,
        min_component=args.min_component,
        min_causal_component=args.min_causal_component,
        max_signed_effect_delta=args.max_signed_effect_delta,
        min_abs_signed_effect=args.min_abs_signed_effect,
    )


def _validate_match(match: CandidateMatch, *, thresholds: dict[str, float]) -> dict[str, Any]:
    components = dict(match.components)
    structural_scores = [
        float(components[name])
        for name in ("text", "activation", "decoder")
        if name in components
    ]
    structural_pass_count = sum(
        1 for value in structural_scores if value >= thresholds["min_component"]
    )
    causal_component = _optional_float(components.get("causal"))
    signed_component = _optional_float(components.get("signed_effect"))
    signed_effect_delta = _signed_effect_delta(match)
    same_effect_direction = _same_effect_direction(match, thresholds["min_abs_signed_effect"])
    strong_signed_effects = _has_strong_signed_effects(match, thresholds["min_abs_signed_effect"])
    direction_conflict = same_effect_direction is False
    causal_component_pass = (
        causal_component is not None
        and causal_component >= thresholds["min_causal_component"]
    )
    score_pass = match.score >= thresholds["min_score"]
    signed_delta_ok = (
        signed_effect_delta is not None
        and signed_effect_delta <= thresholds["max_signed_effect_delta"]
    )
    status = _match_status(
        score_pass=score_pass,
        direction_conflict=direction_conflict,
        structural_pass_count=structural_pass_count,
        causal_component_pass=causal_component_pass,
        strong_signed_effects=strong_signed_effects,
        same_effect_direction=same_effect_direction,
        signed_delta_ok=signed_delta_ok,
    )
    reason_codes = _match_reason_codes(
        match=match,
        status=status,
        score_pass=score_pass,
        structural_pass_count=structural_pass_count,
        causal_component=causal_component,
        causal_component_pass=causal_component_pass,
        strong_signed_effects=strong_signed_effects,
        same_effect_direction=same_effect_direction,
        signed_delta_ok=signed_delta_ok,
        thresholds=thresholds,
    )
    claim_grade = _claim_grade(status)
    return {
        "left_feature_id": match.left_feature_id,
        "right_feature_id": match.right_feature_id,
        "left_model": match.left_model,
        "right_model": match.right_model,
        "left_label": match.left_label,
        "right_label": match.right_label,
        "score": round(float(match.score), 6),
        "status": status,
        "claim_grade": claim_grade,
        "reason_codes": reason_codes,
        "interpretation": _interpret_status(status, reason_codes),
        "next_action": _next_action(claim_grade, reason_codes),
        "components": {key: round(float(value), 6) for key, value in sorted(components.items())},
        "text_component": _optional_round(components.get("text")),
        "activation_component": _optional_round(components.get("activation")),
        "decoder_component": _optional_round(components.get("decoder")),
        "causal_component": _optional_round(causal_component),
        "signed_effect_component": _optional_round(signed_component),
        "left_signed_effect": _optional_round(match.left_signed_effect),
        "right_signed_effect": _optional_round(match.right_signed_effect),
        "signed_effect_delta": _optional_round(signed_effect_delta),
        "same_effect_direction": same_effect_direction,
        "strong_signed_effects": strong_signed_effects,
        "structural_pass_count": structural_pass_count,
    }


def _match_status(
    *,
    score_pass: bool,
    direction_conflict: bool,
    structural_pass_count: int,
    causal_component_pass: bool,
    strong_signed_effects: bool,
    same_effect_direction: bool | None,
    signed_delta_ok: bool,
) -> str:
    if direction_conflict:
        return "contradicted"
    if not score_pass:
        return "weak"
    if (
        causal_component_pass
        and strong_signed_effects
        and same_effect_direction is True
        and signed_delta_ok
    ):
        return "validated"
    if structural_pass_count >= 2:
        return "needs_causal_evidence"
    if structural_pass_count >= 1:
        return "plausible"
    return "weak"


def _match_reason_codes(
    *,
    match: CandidateMatch,
    status: str,
    score_pass: bool,
    structural_pass_count: int,
    causal_component: float | None,
    causal_component_pass: bool,
    strong_signed_effects: bool,
    same_effect_direction: bool | None,
    signed_delta_ok: bool,
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if not score_pass:
        reasons.append("score_below_threshold")
    if structural_pass_count < 2:
        reasons.append("structural_components_below_threshold")
    if causal_component is None:
        reasons.append("missing_causal_component")
    elif not causal_component_pass:
        if abs(causal_component - 0.5) <= 1e-9:
            reasons.append("causal_component_neutral")
        else:
            reasons.append("causal_component_below_threshold")
    if not strong_signed_effects:
        if match.left_signed_effect is None or match.right_signed_effect is None:
            reasons.append("missing_signed_effects")
        else:
            reasons.append("signed_effects_below_threshold")
    elif same_effect_direction is False:
        reasons.append("signed_effect_direction_conflict")
    elif not signed_delta_ok:
        reasons.append("signed_effect_delta_above_threshold")
    if reasons:
        return reasons
    if status == "validated":
        return ["passed_score_structural_causal_and_signed_effect_thresholds"]
    if status == "needs_causal_evidence":
        return ["passed_structural_thresholds_but_needs_causal_validation"]
    if status == "plausible":
        return ["passed_score_threshold_with_limited_component_support"]
    return [f"classified_{status}"]


def _claim_grade(status: str) -> str:
    if status == "validated":
        return "validated_equivalent"
    if status == "needs_causal_evidence":
        return "needs_more_evidence"
    if status == "plausible":
        return "plausible_equivalent"
    if status == "contradicted":
        return "contradicted_effect"
    return "weak_match"


def _interpret_status(status: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if status == "validated":
        return "The match preserves high fingerprint similarity and aligned measured signed effects under the current thresholds."
    if status == "needs_causal_evidence":
        return "The match has strong structural similarity, but needs aligned causal or signed-effect evidence before treating it as an equivalent feature."
    if status == "plausible":
        return "The match is plausible from the available fingerprints and should be prioritized after stronger candidates."
    if status == "contradicted":
        return "The features have opposite measured signed effects for this criterion."
    if "score_below_threshold" in reasons:
        return "The candidate does not clear the match-score threshold."
    return "The available fingerprint evidence is weak for this candidate."


def _next_action(claim_grade: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if claim_grade == "validated_equivalent":
        return "Replicate the match on held-out prompts, then include it in cross-model mechanism summaries."
    if claim_grade == "needs_more_evidence":
        if "missing_signed_effects" in reasons or "causal_component_neutral" in reasons:
            return "Run matched interventions or path-patching records for both features on the same criterion."
        if "signed_effect_delta_above_threshold" in reasons:
            return "Repeat interventions with more prompts and compare effect-size calibration."
        return "Collect causal evidence for this pair before using it as an equivalence claim."
    if claim_grade == "plausible_equivalent":
        return "Keep as a candidate and gather activation examples or interventions if the pair is scientifically useful."
    if claim_grade == "contradicted_effect":
        return "Inspect labels and intervention setup; treat this pair as a contrast unless new evidence resolves the direction conflict."
    return "Lower priority unless other evidence makes this pair important."


def _validation_run_assessment(validations: list[dict[str, Any]]) -> dict[str, str]:
    if not validations:
        return {
            "overall_claim_grade": "no_match_candidates",
            "summary": "No candidate matches were present in the match report.",
            "recommended_next_action": "Run `interp-lab match`, then validate the resulting matches.",
        }
    counts = _counts(validations, "status")
    if counts.get("validated"):
        return {
            "overall_claim_grade": "validated_matches_present",
            "summary": f"{counts['validated']} match claim(s) passed the validation thresholds.",
            "recommended_next_action": "Replicate validated matches on held-out prompts and include them in graph review.",
        }
    if counts.get("needs_causal_evidence"):
        return {
            "overall_claim_grade": "causal_evidence_needed",
            "summary": f"{counts['needs_causal_evidence']} match claim(s) have structural support and need causal evidence.",
            "recommended_next_action": "Run matched interventions or path patching for the highest-scoring pairs.",
        }
    if counts.get("contradicted"):
        return {
            "overall_claim_grade": "contradicted_matches_present",
            "summary": f"{counts['contradicted']} match claim(s) have opposite signed effects.",
            "recommended_next_action": "Review contradicted pairs as possible contrast features before publishing equivalence claims.",
        }
    if counts.get("plausible"):
        return {
            "overall_claim_grade": "plausible_matches_present",
            "summary": f"{counts['plausible']} match claim(s) have partial support.",
            "recommended_next_action": "Collect more examples, decoder evidence, or causal tests for the most useful pairs.",
        }
    return {
        "overall_claim_grade": "weak_matches_only",
        "summary": "The checked matches did not clear the current evidence thresholds.",
        "recommended_next_action": "Lower priority or rerun matching with richer feature fingerprints.",
    }


def _validation_agent_next_actions(run_assessment: dict[str, str]) -> list[dict[str, str]]:
    grade = run_assessment["overall_claim_grade"]
    common_review = {
        "id": "inspect_match_validation",
        "title": "Review match-level claim grades and reason codes",
        "command": "python -c \"from pathlib import Path; print(Path('<match-validation.md>').read_text(encoding='utf-8'))\"",
    }
    if grade == "validated_matches_present":
        return [
            common_review,
            {
                "id": "replicate_validated_matches",
                "title": "Replicate validated cross-model matches on held-out prompts",
                "command": "interp-lab validate-matches --matches <matches.json> --out <match-validation.json>",
            },
            {
                "id": "add_matches_to_graph_review",
                "title": "Use validated matches while reviewing attribution graphs",
                "command": "interp-lab export-attribution-graph --report <report.json> --out <graph.json> --html-out <graph.html>",
            },
        ]
    if grade == "causal_evidence_needed":
        return [
            common_review,
            {
                "id": "run_matched_interventions",
                "title": "Collect signed causal evidence for the highest-scoring feature pairs",
                "command": "interp-lab export-hf-interventions --model <model> --dataset <prompts.jsonl> --features <features.jsonl> --criterion <criterion> --out <interventions.jsonl>",
            },
        ]
    if grade == "contradicted_matches_present":
        return [
            common_review,
            {
                "id": "inspect_contrast_pairs",
                "title": "Inspect pairs with opposite signed effects before using them as equivalents",
                "command": "interp-lab validate-matches --matches <matches.json> --out <match-validation.json> --min-score 0.7",
            },
        ]
    return [
        common_review,
        {
            "id": "enrich_feature_fingerprints",
            "title": "Add examples, decoder signatures, or interventions before matching again",
            "command": "interp-lab inspect --model <model> --criterion <criterion> --backend records --records <records.jsonl> --interventions <interventions.jsonl> --out <report-dir>",
        },
    ]


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _validation_table_row(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    search = _search_text(item)
    return f"""
            <tr data-match="row" data-status="{_attr(status)}" data-search="{_attr(search)}">
              <td>{_status_pill(status)}</td>
              <td>{_h(item.get("claim_grade", ""))}</td>
              <td>
                <div class="match-ref">
                  <code>{_h(item.get("left_feature_id", ""))}</code>
                  <code>{_h(item.get("right_feature_id", ""))}</code>
                  <span class="label-pair">{_h(item.get("left_label", ""))} -> {_h(item.get("right_label", ""))}</span>
                </div>
              </td>
              <td class="num">{_markdown_number(item.get("score"))}</td>
              <td class="num">{_markdown_number(item.get("causal_component"))}</td>
              <td class="num">{_markdown_number(item.get("signed_effect_delta"))}</td>
            </tr>
"""


def _validation_detail_card(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    reasons = "".join(
        f'<span class="reason">{_h(reason)}</span>'
        for reason in item.get("reason_codes", [])
    )
    meters = "\n".join(
        _component_meter(label, item.get(key))
        for label, key in [
            ("score", "score"),
            ("text", "text_component"),
            ("activation", "activation_component"),
            ("decoder", "decoder_component"),
            ("causal", "causal_component"),
            ("signed", "signed_effect_component"),
        ]
    )
    return f"""
        <article class="match-card" data-match="card" data-status="{_attr(status)}" data-search="{_attr(_search_text(item))}">
          <h3>
            <span><code>{_h(item.get("left_feature_id", ""))}</code> -> <code>{_h(item.get("right_feature_id", ""))}</code></span>
            {_status_pill(status)}
          </h3>
          <p class="subhead">{_h(item.get("left_label", ""))} -> {_h(item.get("right_label", ""))}</p>
          <div class="meters">{meters}</div>
          <div class="reason-list">{reasons}</div>
          <p>{_h(item.get("interpretation", ""))}</p>
          <p class="subhead">Next: {_h(item.get("next_action", ""))}</p>
        </article>
"""


def _agent_action_card(action: dict[str, Any]) -> str:
    return f"""
          <article class="action-card">
            <h3>{_h(action.get("title", ""))}</h3>
            <p class="subhead">{_h(action.get("id", ""))}</p>
            <code class="command">{_h(action.get("command", ""))}</code>
          </article>
"""


def _summary_card(label: str, value: Any) -> str:
    return f"""
        <div class="metric">
          <div class="label">{_h(label)}</div>
          <div class="value">{_h(0 if value is None else value)}</div>
        </div>
"""


def _component_meter(label: str, value: Any) -> str:
    if value is None:
        display = ""
        width = 0.0
    else:
        numeric = max(0.0, min(1.0, float(value)))
        width = numeric * 100.0
        display = _markdown_number(value)
    return f"""
            <div class="meter">
              <span>{_h(label)}</span>
              <span class="bar"><span class="fill" style="width: {width:.1f}%"></span></span>
              <span>{_h(display)}</span>
            </div>
"""


def _status_pill(status: str) -> str:
    class_name = "status-" + status.replace("_", "-")
    return f'<span class="pill {_attr(class_name)}">{_h(status)}</span>'


def _search_text(item: dict[str, Any]) -> str:
    parts = [
        item.get("status"),
        item.get("claim_grade"),
        item.get("left_feature_id"),
        item.get("right_feature_id"),
        item.get("left_label"),
        item.get("right_label"),
        item.get("interpretation"),
        item.get("next_action"),
        " ".join(str(reason) for reason in item.get("reason_codes", [])),
    ]
    return " ".join(str(part) for part in parts if part is not None).lower()


def _has_strong_signed_effects(match: CandidateMatch, threshold: float) -> bool:
    if match.left_signed_effect is None or match.right_signed_effect is None:
        return False
    return abs(match.left_signed_effect) >= threshold and abs(match.right_signed_effect) >= threshold


def _same_effect_direction(match: CandidateMatch, threshold: float) -> bool | None:
    if not _has_strong_signed_effects(match, threshold):
        return None
    return (match.left_signed_effect or 0.0) * (match.right_signed_effect or 0.0) > 0.0


def _signed_effect_delta(match: CandidateMatch) -> float | None:
    if match.left_signed_effect is None or match.right_signed_effect is None:
        return None
    return abs(match.left_signed_effect - match.right_signed_effect)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_round(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _h(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _markdown_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"
