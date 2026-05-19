from __future__ import annotations

import html
import json
import re
from pathlib import Path

from oracle_sae.schema import InspectionReport, MatchReport

PROMOTING_THRESHOLD = 0.05
SMALL_EFFECT_THRESHOLD = 0.02
TOKEN_PATTERN = re.compile(r"token\[\d+\]=(['\"])(?P<token>.*?)(?<!\\)\1")
GENERIC_LABEL_PREFIXES = ("trained sae latent", "latent", "feature")


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


def write_inspection_html(report: InspectionReport, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_inspection_html(report), encoding="utf-8")
    return path


def load_inspection_report(path: str | Path) -> InspectionReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InspectionReport.from_dict(data)


def load_match_report(path: str | Path) -> MatchReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MatchReport.from_dict(data)


def render_inspection_markdown(report: InspectionReport) -> str:
    lines = [
        f"# interp-lab Report: {report.model}",
        "",
        f"Criterion: {report.criterion.text}",
        "",
        "Metric notes: Association is activation/criterion correlation in the evidence records. "
        "Effect is mean causal change from interventions. Specificity subtracts measured side effects. "
        "Strong causal score is the specificity-adjusted causal signal.",
        "",
    ]
    evidence_summary = _evidence_summary_lines(report.metadata.get("evidence"))
    if evidence_summary:
        lines.extend(evidence_summary)
    scope = _report_scope_line(report.metadata)
    if scope:
        lines.extend([scope, ""])
    mechanism = _mechanism_sketch_lines(report)
    if mechanism:
        lines.extend(mechanism)
    lines.extend(
        [
        "## Top Features",
        "",
        ]
    )
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
                _metric_line(card),
                "",
            ]
        )
        evidence = _evidence_line(card)
        if evidence:
            lines.extend([evidence, ""])
        direction = _direction_line(card)
        if direction:
            lines.extend([direction, ""])
        strong = card.causal_effects.get("strong_causal_score")
        if strong is not None:
            lines.extend([f"Strong causal score: {float(strong):.3f}", ""])
        sae_training = _sae_training_lines(card.metadata.get("sae_training"))
        if sae_training:
            lines.extend(sae_training)
        intervention = _intervention_lines(card.metadata.get("interventions"))
        if intervention:
            lines.extend(intervention)
        interpretation = _card_interpretation_lines(card)
        if interpretation:
            lines.extend(interpretation)
        lines.extend(["Examples:"])
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


def render_inspection_html(report: InspectionReport) -> str:
    cards = list(report.cards)
    evidence_summary = _html_evidence_summary(report.metadata.get("evidence"))
    scope = _report_scope_line(report.metadata)
    mechanism = _html_mechanism_summary(report)
    layer_options = "\n".join(
        f'<option value="{_attr(layer)}">{_h(layer)}</option>'
        for layer in _layer_filter_values(cards)
    )
    source_options = "\n".join(
        f'<option value="{_attr(source)}">{_h(source)}</option>'
        for source in sorted({str(card.source) for card in cards if card.source})
    )
    metric_cards = "\n".join(
        _summary_card(label, value)
        for label, value in [
            ("Features", len(cards)),
            ("Strong Causal", len(_top_causal_cards(report))),
            ("With Interventions", sum(1 for card in cards if _has_measured_intervention(card))),
            ("Layers", len({card.layer for card in cards if card.layer is not None})),
        ]
    )
    table_rows = "\n".join(_feature_table_row(card, index) for index, card in enumerate(cards, start=1))
    detail_cards = "\n".join(_feature_detail_card(card, index) for index, card in enumerate(cards, start=1))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>interp-lab Feature Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #1d2528;
      --muted: #5d686e;
      --line: #d9dedb;
      --panel: #ffffff;
      --accent: #0f766e;
      --good: #147a3f;
      --warn: #946200;
      --quiet: #5f6670;
      --blue: #285e9e;
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
      width: min(1200px, calc(100vw - 32px));
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
    .subhead {{ color: var(--muted); max-width: 960px; }}
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
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #fff;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
    }}
    .pill.good {{ color: var(--good); border-color: #9fd3b5; background: #eef9f2; }}
    .pill.warn {{ color: var(--warn); border-color: #e5c06f; background: #fff7df; }}
    .pill.blue {{ color: var(--blue); border-color: #a9c7ed; background: #eef6ff; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
    }}
    .metric, .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .metric {{ padding: 12px; }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 24px; font-weight: 750; }}
    .panel {{
      padding: 16px;
      margin-top: 14px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(140px, 210px) minmax(140px, 210px) auto;
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
    .notes {{
      display: grid;
      gap: 10px;
    }}
    .note {{
      border-left: 3px solid var(--accent);
      padding-left: 10px;
      color: var(--muted);
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 920px;
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
    .feature-ref {{ display: grid; gap: 4px; }}
    .label-line {{ color: var(--muted); font-size: 12px; }}
    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 12px;
    }}
    .feature-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 12px;
    }}
    .feature-card h3 {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }}
    .meters {{ display: grid; gap: 7px; }}
    .meter {{
      display: grid;
      grid-template-columns: 92px 1fr 54px;
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
    .examples {{
      display: grid;
      gap: 6px;
    }}
    .example {{
      padding: 8px;
      border-radius: 7px;
      background: #f6f8f7;
      color: #334044;
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    [hidden] {{ display: none !important; }}
    @media (max-width: 760px) {{
      main {{ width: min(100vw - 20px, 1200px); padding-top: 20px; }}
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
        <h1>Feature Report</h1>
        <p class="subhead">Criterion: {_h(report.criterion.text)}</p>
      </div>
      <div class="model-line">
        <span>Model <code>{_h(report.model)}</code></span>
        <span class="pill">{_h(report.created_at)}</span>
      </div>
      <div class="summary-grid">{metric_cards}</div>
    </header>
    <section class="panel">
      <h2>Run Context</h2>
      <div class="notes">
        {_html_note(scope)}
        {evidence_summary}
        {_html_note("Metric notes: Association is activation/criterion correlation. Effect is mean causal change from interventions. Specificity subtracts measured side effects. Strong causal score is the specificity-adjusted causal signal.")}
      </div>
    </section>
    <section class="panel">
      <h2>Mechanism Sketch</h2>
      <div class="notes">{mechanism}</div>
    </section>
    <section class="panel">
      <div class="toolbar">
        <input id="feature-search" type="search" placeholder="Filter by feature, label, examples, or evidence">
        <select id="layer-filter" aria-label="Filter by layer">
          <option value="">All layers</option>
          {layer_options}
        </select>
        <select id="source-filter" aria-label="Filter by source">
          <option value="">All sources</option>
          {source_options}
        </select>
        <div id="visible-count" class="visible-count">{len(cards)} visible</div>
      </div>
    </section>
    <section class="panel">
      <h2>Ranked Features</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Feature</th>
              <th>Evidence</th>
              <th class="num">Importance</th>
              <th class="num">Association</th>
              <th class="num">Effect</th>
              <th class="num">Specificity</th>
              <th class="num">Strong Causal</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Feature Details</h2>
      <div class="details">{detail_cards}</div>
    </section>
  </main>
  <script>
    const search = document.getElementById("feature-search");
    const layerFilter = document.getElementById("layer-filter");
    const sourceFilter = document.getElementById("source-filter");
    const visibleCount = document.getElementById("visible-count");
    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const layer = layerFilter.value;
      const source = sourceFilter.value;
      let visibleRows = 0;
      document.querySelectorAll("[data-feature]").forEach((node) => {{
        const layerMatch = !layer || node.dataset.layer === layer;
        const sourceMatch = !source || node.dataset.source === source;
        const searchMatch = !query || (node.dataset.search || "").includes(query);
        const show = layerMatch && sourceMatch && searchMatch;
        node.hidden = !show;
        if (show && node.dataset.feature === "row") visibleRows += 1;
      }});
      visibleCount.textContent = `${{visibleRows}} visible`;
    }}
    search.addEventListener("input", applyFilters);
    layerFilter.addEventListener("change", applyFilters);
    sourceFilter.addEventListener("change", applyFilters);
  </script>
</body>
</html>
"""


def _mechanism_sketch_lines(report: InspectionReport) -> list[str]:
    if not report.cards:
        return []
    lines = ["## Mechanism Sketch", ""]
    top_causal = _top_causal_cards(report)
    if top_causal:
        lines.append("Causal candidates:")
        for card in top_causal[:4]:
            lines.append(
                f"- {_feature_ref(card)} ({_display_label(card)}) changes the behavior score by "
                f"{_signed_effect(card):+.3f} on average "
                f"(strong causal score {float(card.causal_effects.get('strong_causal_score', 0.0)):.3f})."
            )
        lines.append("")
    else:
        lines.extend(
            [
                "Causal candidates: no tested feature crossed the current strong-effect threshold.",
                "",
            ]
        )
    themes = _activation_themes(report.cards)
    if themes:
        lines.append("Activation themes:")
        for theme, count in themes[:8]:
            lines.append(f"- `{theme}` appears in high-activation examples for {count} feature(s).")
        lines.append("")
    evidence_gaps = _evidence_gap_lines(report)
    if evidence_gaps:
        lines.extend(["Evidence gaps:", *[f"- {line}" for line in evidence_gaps], ""])
    return lines


def _evidence_summary_lines(raw_value: object) -> list[str]:
    if not isinstance(raw_value, dict):
        return []
    record_count = raw_value.get("record_count")
    feature_count = raw_value.get("feature_count")
    if record_count is None:
        return []
    parts = [f"{int(record_count)} activation rows"]
    if feature_count is not None:
        parts.append(f"{int(feature_count)} candidate features")
    mean_score = raw_value.get("criterion_score_mean")
    min_score = raw_value.get("criterion_score_min")
    max_score = raw_value.get("criterion_score_max")
    if mean_score is not None:
        score_text = f"criterion score mean={float(mean_score):.3f}"
        if min_score is not None and max_score is not None:
            score_text += f", range=[{float(min_score):.3f}, {float(max_score):.3f}]"
        parts.append(score_text)
    positive_count = raw_value.get("positive_record_count")
    negative_count = raw_value.get("negative_record_count")
    if positive_count is not None and negative_count is not None:
        parts.append(f"positive rows={int(positive_count)}, non-positive rows={int(negative_count)}")
    return ["Evidence summary: " + "; ".join(parts) + ".", ""]


def _metric_line(card) -> str:
    effect_label = "Causal effect" if _has_measured_intervention(card) else "Criterion score"
    return (
        f"Association: {card.association:.3f} | {effect_label}: {card.causal_effect:.3f} | "
        f"Specificity: {card.specificity:.3f} | Stability: {card.stability:.3f}"
    )


def _report_scope_line(metadata: dict) -> str:
    feature_count = metadata.get("feature_count")
    kept_count = metadata.get("kept_feature_count")
    if feature_count is None or kept_count is None:
        return ""
    return f"Report scope: ranked {int(kept_count)} kept feature(s) from {int(feature_count)} candidate feature(s)."


def _top_causal_cards(report: InspectionReport):
    return [
        card
        for card in sorted(
            report.cards,
            key=lambda item: float(item.causal_effects.get("strong_causal_score", 0.0)),
            reverse=True,
        )
        if float(card.causal_effects.get("strong_causal_score", 0.0)) >= PROMOTING_THRESHOLD
    ]


def _feature_ref(card) -> str:
    layer = "" if card.layer is None else f" layer {card.layer}"
    return f"{card.feature_id}{layer}"


def _signed_effect(card) -> float:
    value = card.causal_effects.get("signed_causal_effect")
    if value is None:
        value = card.causal_effects.get("signed_association", card.metadata.get("signed_association", 0.0))
    return float(value)


def _activation_themes(cards) -> list[tuple[str, int]]:
    counts: dict[str, set[str]] = {}
    for card in cards:
        for example in card.examples[:3]:
            token = _token_from_example(example)
            if token:
                counts.setdefault(token, set()).add(card.feature_id)
    ranked = [(token, len(feature_ids)) for token, feature_ids in counts.items()]
    ranked.sort(key=lambda item: (-item[1], item[0].lower()))
    return ranked


def _token_from_example(example: str) -> str:
    match = TOKEN_PATTERN.search(str(example))
    if not match:
        return ""
    token = match.group("token").replace("\\n", "\\n").strip()
    if not token or len(token) > 32:
        return ""
    return token


def _evidence_gap_lines(report: InspectionReport) -> list[str]:
    gaps = []
    if any(card.metadata.get("interventions") for card in report.cards):
        gaps.append(
            "Feature-level causal tests are present. Use export-attribution-graph, optionally with repeated "
            "--report arguments, to inspect candidate feature groups and cross-layer coactivation paths."
        )
    elif _attached_intervention_count(report.metadata) > 0:
        gaps.append(
            "Intervention records were attached, but none matched the kept features. Use --require-interventions "
            "to focus the report on tested features, increase --top-k, or add causal rows for the ranked features."
        )
    else:
        gaps.append("No intervention records were attached; causal claims are untested.")
    if not any(float(card.causal_effects.get("strong_causal_score", 0.0)) >= PROMOTING_THRESHOLD for card in report.cards):
        gaps.append("No feature currently meets the strong-effect threshold; broaden prompts, test more layers, or use graph attribution.")
    return gaps


def _attached_intervention_count(metadata: dict) -> int:
    value = metadata.get("interventions")
    if not isinstance(value, dict):
        return 0
    return int(value.get("record_count", 0) or 0)


def _card_interpretation_lines(card) -> list[str]:
    lines = []
    tokens = [token for token in (_token_from_example(example) for example in card.examples[:3]) if token]
    if tokens:
        unique_tokens = []
        for token in tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
        token_text = ", ".join(f"`{token}`" for token in unique_tokens[:4])
        if _has_generic_label(card):
            lines.append(f"Activation readout: high activations concentrate on tokens such as {token_text}.")
        else:
            lines.append(
                "Activation readout: "
                f"`{card.label}` is represented by high activations on tokens such as {token_text}."
            )
    elif _use_stored_explanation(card.explanation):
        lines.append(card.explanation)
    strong = float(card.causal_effects.get("strong_causal_score", 0.0))
    signed = _signed_effect(card)
    if strong >= PROMOTING_THRESHOLD:
        verb = "promoted" if signed > 0 else "suppressed"
        lines.append(
            f"Causal readout: steering or ablating this feature {verb} the criterion "
            f"with strong causal score {strong:.3f}."
        )
    elif _has_measured_intervention(card):
        lines.append(
            f"Causal readout: tested interventions produced a small or uncertain effect "
            f"(strong causal score {strong:.3f})."
        )
    if not lines:
        return []
    return [*lines, ""]


def _use_stored_explanation(explanation: str) -> bool:
    banned = [
        " ".join(["Treat", "this", "as", "a", "hypothesis"]),
        " ".join(["weak", "signed", "association"]),
    ]
    return bool(explanation) and not any(phrase in explanation for phrase in banned)


def _display_label(card) -> str:
    label = str(card.label).strip()
    if label and not _has_generic_label(card):
        return label
    tokens = [token for token in (_token_from_example(example) for example in card.examples[:3]) if token]
    unique_tokens = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    if unique_tokens:
        return "tokens: " + ", ".join(_display_token(token) for token in unique_tokens[:3])
    return label or card.feature_id


def _display_token(token: str) -> str:
    if len(token) == 1 and not token.isalnum():
        return repr(token)
    return token


def _has_generic_label(card) -> bool:
    return str(card.label).strip().lower().startswith(GENERIC_LABEL_PREFIXES)


def _has_measured_intervention(card) -> bool:
    if isinstance(card.metadata.get("interventions"), dict):
        return True
    return float(card.causal_effects.get("intervention_record_count", 0.0) or 0.0) > 0.0


def _direction_line(card) -> str:
    raw_value = card.causal_effects.get("signed_causal_effect")
    source = "Causal direction"
    if raw_value is None:
        raw_value = card.causal_effects.get("signed_association", card.metadata.get("signed_association"))
        source = "Activation association"
    if raw_value is None:
        return ""
    value = float(raw_value)
    if value > 0.05:
        return f"{source}: promotes criterion ({value:.3f})"
    if value < -0.05:
        return f"{source}: suppresses criterion ({value:.3f})"
    if abs(value) < SMALL_EFFECT_THRESHOLD:
        return f"{source}: near zero ({value:.3f})"
    return f"{source}: small directional effect ({value:.3f})"


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
    behavior_score = raw_value.get("behavior_score")
    if isinstance(behavior_score, dict):
        behavior_line = _behavior_score_line(behavior_score)
        if behavior_line:
            lines.append(behavior_line)
        advisory = behavior_score.get("advisory")
        if advisory:
            lines.append(f"Behavior note: {advisory}")
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


def _sae_training_lines(raw_value: object) -> list[str]:
    if not isinstance(raw_value, dict):
        return []
    latent_dim = raw_value.get("latent_dim")
    sample_count = raw_value.get("sample_count")
    active_fraction = raw_value.get("active_latent_fraction")
    dead_count = raw_value.get("dead_latent_count")
    if latent_dim is None and sample_count is None:
        return []
    parts = []
    if sample_count is not None:
        parts.append(f"rows={int(sample_count)}")
    if latent_dim is not None:
        parts.append(f"latents={int(latent_dim)}")
    if active_fraction is not None:
        parts.append(f"active={float(active_fraction):.3f}")
    if dead_count is not None:
        parts.append(f"dead={int(dead_count)}")
    mse = raw_value.get("validation_reconstruction_mse")
    if mse is not None:
        parts.append(f"val MSE={float(mse):.3f}")
    lines = [f"SAE training: {', '.join(parts)}"]
    advisories = raw_value.get("advisories")
    if isinstance(advisories, list):
        for advisory in advisories[:2]:
            lines.append(f"SAE training note: {advisory}")
    lines.append("")
    return lines


def _behavior_score_line(raw_value: dict) -> str:
    name = raw_value.get("name")
    baseline_mean = raw_value.get("baseline_mean")
    baseline_min = raw_value.get("baseline_min")
    baseline_max = raw_value.get("baseline_max")
    if baseline_mean is None:
        return ""
    line = f"Behavior score: {name or 'score'} baseline mean={float(baseline_mean):.3f}"
    if baseline_min is not None and baseline_max is not None:
        line += f", range=[{float(baseline_min):.3f}, {float(baseline_max):.3f}]"
    target_count = raw_value.get("target_token_count")
    strategy = raw_value.get("target_token_strategy")
    if target_count is not None:
        line += f", target tokens={target_count}"
        if strategy:
            line += f" ({strategy})"
        sample = _target_token_sample(raw_value)
        if sample:
            line += f", sample={sample}"
    elif strategy:
        line += f", target token strategy={strategy}"
    return line


def _target_token_sample(raw_value: dict) -> str:
    sample = raw_value.get("target_token_sample")
    if not isinstance(sample, list):
        return ""
    tokens = [str(token).replace("\n", "\\n") for token in sample[:4]]
    if not tokens:
        return ""
    return ", ".join(f"`{token}`" for token in tokens)


def _html_evidence_summary(raw_value: object) -> str:
    lines = _evidence_summary_lines(raw_value)
    if not lines:
        return ""
    return _html_note(" ".join(line for line in lines if line))


def _html_mechanism_summary(report: InspectionReport) -> str:
    notes: list[str] = []
    top_causal = _top_causal_cards(report)
    if top_causal:
        features = ", ".join(
            f"{card.feature_id} ({_display_label(card)}, strong causal {float(card.causal_effects.get('strong_causal_score', 0.0)):.3f})"
            for card in top_causal[:4]
        )
        notes.append(f"Causal candidates: {features}.")
    else:
        notes.append("Causal candidates: no tested feature crossed the current strong-effect threshold.")
    themes = _activation_themes(report.cards)
    if themes:
        notes.append(
            "Activation themes: "
            + ", ".join(f"{theme} ({count})" for theme, count in themes[:8])
            + "."
        )
    notes.extend(_evidence_gap_lines(report))
    return "\n".join(_html_note(note) for note in notes if note)


def _feature_table_row(card, index: int) -> str:
    layer = _layer_value(card)
    source = str(card.source or "unknown")
    strong = float(card.causal_effects.get("strong_causal_score", 0.0))
    return f"""
            <tr data-feature="row" data-layer="{_attr(layer)}" data-source="{_attr(source)}" data-search="{_attr(_feature_search_text(card))}">
              <td>{index}</td>
              <td>
                <div class="feature-ref">
                  <code>{_h(card.feature_id)}</code>
                  <span class="label-line">{_h(_display_label(card))}</span>
                </div>
              </td>
              <td>{_feature_evidence_pill(card)}</td>
              <td class="num">{card.importance:.3f}</td>
              <td class="num">{card.association:.3f}</td>
              <td class="num">{card.causal_effect:.3f}</td>
              <td class="num">{card.specificity:.3f}</td>
              <td class="num">{strong:.3f}</td>
            </tr>
"""


def _feature_detail_card(card, index: int) -> str:
    layer = _layer_value(card)
    source = str(card.source or "unknown")
    strong = float(card.causal_effects.get("strong_causal_score", 0.0))
    direction = _direction_line(card)
    evidence = _evidence_line(card)
    interpretation = _html_card_interpretation(card)
    examples = "\n".join(
        f'<div class="example">{_h(example)}</div>' for example in card.examples[:4]
    )
    if not examples:
        examples = '<div class="example">No activation examples were attached.</div>'
    training = _html_sae_training(card.metadata.get("sae_training"))
    interventions = _html_interventions(card.metadata.get("interventions"))
    chips = "\n".join(
        item
        for item in [
            _feature_evidence_pill(card),
            f'<span class="pill">layer {_h(layer)}</span>',
            f'<span class="pill">{_h(source)}</span>',
            _causal_strength_pill(strong),
        ]
        if item
    )
    return f"""
        <article class="feature-card" data-feature="card" data-layer="{_attr(layer)}" data-source="{_attr(source)}" data-search="{_attr(_feature_search_text(card))}">
          <h3>
            <span>{index}. <code>{_h(card.feature_id)}</code></span>
            <span class="pill blue">{_h(_display_label(card))}</span>
          </h3>
          <div class="chip-row">{chips}</div>
          <div class="meters">
            {_component_meter("importance", card.importance)}
            {_component_meter("association", card.association)}
            {_component_meter("effect", card.causal_effect)}
            {_component_meter("specificity", card.specificity)}
            {_component_meter("stability", card.stability)}
            {_component_meter("strong", strong)}
          </div>
          {_html_note(direction)}
          {_html_note(evidence)}
          {interpretation}
          {training}
          {interventions}
          <div class="examples">{examples}</div>
        </article>
"""


def _html_card_interpretation(card) -> str:
    lines = _card_interpretation_lines(card)
    notes = [line for line in lines if line.strip()]
    return "\n".join(_html_note(line) for line in notes)


def _html_sae_training(raw_value: object) -> str:
    lines = [line for line in _sae_training_lines(raw_value) if line.strip()]
    return "\n".join(_html_note(line) for line in lines)


def _html_interventions(raw_value: object) -> str:
    lines = [line for line in _intervention_lines(raw_value) if line.strip()]
    if not lines:
        return ""
    compact = []
    for line in lines:
        if line == "Intervention examples:":
            continue
        compact.append(line.removeprefix("- "))
    return "\n".join(_html_note(line) for line in compact[:6])


def _feature_evidence_pill(card) -> str:
    if _has_measured_intervention(card):
        return '<span class="pill good">causal records</span>'
    if card.source in {"activation-records", "hf-hidden-state"}:
        return '<span class="pill blue">activation records</span>'
    if card.source in {"neuronpedia", "saelens", "goodfire", "gemma-scope", "qwen-scope"}:
        return f'<span class="pill">{_h(card.source)}</span>'
    return '<span class="pill">feature evidence</span>'


def _causal_strength_pill(strong: float) -> str:
    if strong >= PROMOTING_THRESHOLD:
        return f'<span class="pill good">strong causal {strong:.3f}</span>'
    if strong >= SMALL_EFFECT_THRESHOLD:
        return f'<span class="pill warn">small causal {strong:.3f}</span>'
    return f'<span class="pill">causal {strong:.3f}</span>'


def _component_meter(label: str, value: float) -> str:
    display = f"{float(value):.3f}"
    width = min(1.0, abs(float(value))) * 100.0
    return f"""
            <div class="meter">
              <span>{_h(label)}</span>
              <span class="bar"><span class="fill" style="width: {width:.1f}%"></span></span>
              <span>{_h(display)}</span>
            </div>
"""


def _summary_card(label: str, value: object) -> str:
    return f"""
        <div class="metric">
          <div class="label">{_h(label)}</div>
          <div class="value">{_h(value)}</div>
        </div>
"""


def _html_note(value: object) -> str:
    if value is None or str(value).strip() == "":
        return ""
    return f'<p class="note">{_h(value)}</p>'


def _layer_filter_values(cards) -> list[str]:
    values = {_layer_value(card) for card in cards}
    numeric = sorted((value for value in values if value != "unknown"), key=lambda item: int(item))
    if "unknown" in values:
        numeric.append("unknown")
    return numeric


def _layer_value(card) -> str:
    return "unknown" if card.layer is None else str(card.layer)


def _feature_search_text(card) -> str:
    parts = [
        card.feature_id,
        card.label,
        card.explanation,
        card.source,
        _display_label(card),
        _direction_line(card),
        _evidence_line(card),
        " ".join(card.examples[:6]),
        " ".join(_card_interpretation_lines(card)),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def _h(value: object) -> str:
    return html.escape(str(value), quote=False)


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_optional_effect(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"
