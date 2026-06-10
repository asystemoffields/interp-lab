"""Criterion dossiers: cumulative evidence for one (model, criterion) across runs.

A dossier is a persistent JSON artifact keyed on (model, criterion). Each
``inspect``/``validate`` run appends a run entry; the dossier then recomputes a
rollup that tracks grade transitions, score drift, and contradictions across
runs. It makes long investigations stateful and resumable: an agent can reload
the dossier, read where every feature's evidence stands, and pick the next
action without replaying the whole history.

Provenance discipline (the same rule as matching/match_validation): sign flips
are only ever counted between signed effects of the SAME provenance. An
intervention-measured effect is never compared against a correlational
association -- such a pair is recorded as a provenance change, not a
contradiction. Only an intervention-vs-intervention sign flip raises the
contradiction flag.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interp_lab import __version__
from interp_lab.agent_actions import next_action
from interp_lab.matching import (
    SIGNED_EFFECT_DIRECTION_MIN,
    has_intervention_provenance,
    signed_effect_with_provenance,
)
from interp_lab.reporting import load_inspection_report, load_match_report
from interp_lab.schema import InspectionReport

DOSSIER_SCHEMA = "interp-lab.dossier.v1"
# Inline in their writer modules (match_validation.py / graph_validation.py),
# so mirrored here for attached-artifact validation.
_MATCH_VALIDATION_SCHEMA = "interp-lab.match_validation.v1"
_GRAPH_VALIDATION_SCHEMA = "interp-lab.graph_validation.v1"

# One honest evidence ladder for grade transitions. The five match_validation
# claim grades keep their relative order (validated > needs_more_evidence >
# plausible > weak > contradicted). Two dossier-derived grades cover features
# that have no match-validation claim yet: "causal_evidence" (the feature
# carries intervention-measured evidence) and "associational" (correlational
# only). "Upgraded"/"downgraded" compares ranks on this ladder.
GRADE_ORDER = {
    "contradicted_effect": 0,
    "weak_match": 1,
    "associational": 2,
    "plausible_equivalent": 3,
    "needs_more_evidence": 4,
    "causal_evidence": 5,
    "validated_equivalent": 6,
}

# Grades that mean "this feature still needs causal evidence" for the
# open_questions summary count.
_OPEN_QUESTION_GRADES = {"needs_more_evidence", "associational"}


def update_dossier(
    dossier_path: str | Path,
    report: InspectionReport | str | Path,
    *,
    matches: str | Path | None = None,
    match_validation: str | Path | None = None,
    graph_validation: str | Path | None = None,
    note: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append a run entry to the dossier at ``dossier_path`` and rewrite it atomically.

    Creates the dossier when the file is absent, keyed on the report's
    (model, criterion). On append, the report identity must match the dossier
    identity or a ValueError naming both identities is raised. ``report`` may be
    an ``InspectionReport`` or a path to a ``report.json`` (path inputs also get
    a sha256 recorded). ``now`` is injectable for tests; the default is
    ``datetime.now(timezone.utc)``, stored ISO-8601.
    """
    path = Path(dossier_path)
    report_obj, report_path = _resolve_report(report)
    timestamp = _timestamp(now)
    if path.exists():
        dossier = load_dossier(path)
        _check_identity(dossier, report_obj, path)
    else:
        dossier = {
            "schema_version": DOSSIER_SCHEMA,
            "model": report_obj.model,
            "criterion": report_obj.criterion.text,
            "created_at": timestamp,
            "runs": [],
        }
    attached, claim_grades = _attached_artifacts(
        matches=matches,
        match_validation=match_validation,
        graph_validation=graph_validation,
        model=str(dossier["model"]),
    )
    dossier["runs"].append(
        _run_entry(
            report_obj,
            report_path=report_path,
            timestamp=timestamp,
            note=note,
            attached=attached,
            claim_grades=claim_grades,
        )
    )
    dossier["updated_at"] = timestamp
    _recompute_rollup(dossier)
    _write_atomic(path, dossier)
    return dossier


def load_dossier(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a dossier JSON artifact."""
    dossier_path = Path(path)
    try:
        data = json.loads(dossier_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{dossier_path}: invalid dossier JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{dossier_path}: dossier must be a JSON object")
    schema = data.get("schema_version")
    if schema is not None and schema != DOSSIER_SCHEMA:
        raise ValueError(f"{dossier_path}: unsupported dossier schema_version {schema!r}")
    if "model" not in data or "criterion" not in data:
        raise ValueError(f"{dossier_path}: dossier must carry model and criterion")
    if not isinstance(data.get("runs", []), list):
        raise ValueError(f"{dossier_path}: dossier runs must be a list")
    return data


def dossier_summary(dossier: dict[str, Any]) -> dict[str, Any]:
    """Compact summary of the dossier: top-level rollup plus per-feature standing."""
    summary = dict(dossier.get("summary", {}))
    features = dossier.get("rollup", {}).get("features", {})
    summary["features"] = {
        feature_id: {
            "current_grade": item.get("current_grade"),
            "grade_transition": item.get("grade_transition"),
            "run_count": item.get("run_count"),
            "sign_flips": item.get("sign_flips"),
            "provenance_changes": item.get("provenance_changes"),
            "contradiction": item.get("contradiction"),
        }
        for feature_id, item in features.items()
        if isinstance(item, dict)
    }
    return summary


def write_dossier_markdown(dossier: dict[str, Any], path: str | Path) -> Path:
    """Render the dossier as Markdown and write it to ``path``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dossier_markdown(dossier), encoding="utf-8")
    return out_path


def render_dossier_markdown(dossier: dict[str, Any]) -> str:
    summary = dossier.get("summary", {})
    features = dossier.get("rollup", {}).get("features", {})
    span = summary.get("span", {})
    lines = [
        f"# interp-lab Criterion Dossier: {dossier.get('model', '')}",
        "",
        f"Criterion: {dossier.get('criterion', '')}",
        "",
        f"Runs: `{summary.get('run_count', 0)}` ({span.get('first', '')} -> {span.get('last', '')})",
        f"Features tracked: `{summary.get('features_total', 0)}`",
        f"Currently validated: `{summary.get('currently_validated', 0)}`"
        f"  |  Regressions: `{summary.get('regressions', 0)}`"
        f"  |  Open questions: `{summary.get('open_questions', 0)}`"
        f"  |  Contradictions: `{summary.get('contradictions', 0)}`",
        "",
        "## Feature Standing",
        "",
        "| Feature | Label | Runs | Grade | Transition | Sign flips | Provenance changes | Contradiction |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for feature_id, item in sorted(features.items()):
        lines.append(
            f"| `{feature_id}` | {item.get('label', '')} | {item.get('run_count', 0)} "
            f"| {item.get('current_grade', '')} | {item.get('grade_transition', '')} "
            f"| {item.get('sign_flips', 0)} | {item.get('provenance_changes', 0)} "
            f"| {'YES' if item.get('contradiction') else 'no'} |"
        )
    lines.extend(["", "## Run Log", ""])
    for index, run in enumerate(dossier.get("runs", []), start=1):
        attached = run.get("attached", {})
        parts = [f"### {index}. {run.get('timestamp', '')}", ""]
        if run.get("report_path"):
            parts.append(f"Report: `{run['report_path']}`")
        if run.get("note"):
            parts.append(f"Note: {run['note']}")
        parts.append(f"Features: `{len(run.get('features', {}))}`  |  tool `{run.get('tool_version', '')}`")
        for name in ("matches", "match_validation", "graph_validation"):
            info = attached.get(name)
            if isinstance(info, dict):
                parts.append(f"Attached {name}: `{info.get('path', '')}`")
        parts.append("")
        lines.extend(parts)
    actions = dossier.get("agent_next_actions", [])
    if actions:
        lines.extend(["## Agent Next Actions", ""])
        for action in actions:
            if isinstance(action, dict):
                title = action.get("title") or action.get("id") or "Next action"
                command = action.get("command") or action.get("instruction") or ""
                lines.append(f"- {title}: `{command}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# --- run entry construction -------------------------------------------------


def _resolve_report(report: InspectionReport | str | Path) -> tuple[InspectionReport, Path | None]:
    if isinstance(report, InspectionReport):
        return report, None
    report_path = Path(report)
    return load_inspection_report(report_path), report_path


def _timestamp(now: datetime | None) -> str:
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.isoformat()


def _check_identity(dossier: dict[str, Any], report: InspectionReport, path: Path) -> None:
    dossier_model = str(dossier.get("model", ""))
    dossier_criterion = str(dossier.get("criterion", ""))
    if dossier_model != report.model or dossier_criterion.strip() != report.criterion.text.strip():
        raise ValueError(
            f"{path}: dossier identity mismatch: dossier is for "
            f"model={dossier_model!r} criterion={dossier_criterion!r}; "
            f"report is for model={report.model!r} criterion={report.criterion.text!r}"
        )


def _run_entry(
    report: InspectionReport,
    *,
    report_path: Path | None,
    timestamp: str,
    note: str | None,
    attached: dict[str, Any],
    claim_grades: dict[str, str],
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for card in report.cards:
        signed, provenance = signed_effect_with_provenance(card.causal_effects, card.metadata)
        features[card.feature_id] = {
            "label": card.label,
            "importance": round(float(card.importance), 6),
            "association": round(float(card.association), 6),
            "causal_effect": round(float(card.causal_effect), 6),
            "signed_effect": None if signed is None else round(float(signed), 6),
            "signed_effect_provenance": provenance,
            "intervention_record_count": _intervention_record_count(card),
            "has_intervention_provenance": has_intervention_provenance(
                card.causal_effects, card.metadata
            ),
            "claim_grade": claim_grades.get(card.feature_id),
        }
    return {
        "timestamp": timestamp,
        "tool_version": __version__,
        "report_path": str(report_path) if report_path is not None else None,
        "report_hash": _sha256(report_path) if report_path is not None else None,
        "report_created_at": report.created_at,
        "note": note,
        "features": features,
        "attached": attached,
    }


def _intervention_record_count(card: Any) -> int:
    count = float(card.causal_effects.get("intervention_record_count", 0.0) or 0.0)
    if count > 0:
        return int(count)
    interventions = card.metadata.get("interventions")
    if isinstance(interventions, dict):
        try:
            return int(float(interventions.get("count", 0) or 0))
        except (TypeError, ValueError):
            return 0
    return 0


# --- attached artifacts -----------------------------------------------------


def _attached_artifacts(
    *,
    matches: str | Path | None,
    match_validation: str | Path | None,
    graph_validation: str | Path | None,
    model: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    attached: dict[str, Any] = {}
    claim_grades: dict[str, str] = {}
    if matches is not None:
        matches_path = Path(matches)
        match_report = load_match_report(matches_path)
        attached["matches"] = {
            "path": str(matches_path),
            "sha256": _sha256(matches_path),
            "left_model": match_report.left_model,
            "right_model": match_report.right_model,
            "match_count": len(match_report.matches),
        }
    if match_validation is not None:
        validation_path = Path(match_validation)
        data = _load_json_artifact(validation_path, _MATCH_VALIDATION_SCHEMA, "match validation")
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        attached["match_validation"] = {
            "path": str(validation_path),
            "sha256": _sha256(validation_path),
            "match_count": summary.get("match_count"),
            "validated_count": summary.get("validated_count"),
            "contradicted_count": summary.get("contradicted_count"),
            "needs_causal_evidence_count": summary.get("needs_causal_evidence_count"),
            "overall_claim_grade": summary.get("overall_claim_grade"),
        }
        claim_grades = _claim_grades_for_model(data, model)
    if graph_validation is not None:
        graph_path = Path(graph_validation)
        data = _load_json_artifact(graph_path, _GRAPH_VALIDATION_SCHEMA, "graph validation")
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        attached["graph_validation"] = {
            "path": str(graph_path),
            "sha256": _sha256(graph_path),
            "candidate_count": summary.get("candidate_count"),
            "path_record_count": summary.get("path_record_count"),
            "status_counts": summary.get("status_counts", {}),
            "overall_claim_grade": summary.get("overall_claim_grade"),
        }
    return attached, claim_grades


def _load_json_artifact(path: Path, expected_schema: str, kind: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid {kind} JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: {kind} report must be a JSON object")
    schema = data.get("schema_version")
    if schema is not None and schema != expected_schema:
        raise ValueError(f"{path}: unsupported {kind} schema_version {schema!r}")
    return data


def _claim_grades_for_model(validation_data: dict[str, Any], model: str) -> dict[str, str]:
    """Per-feature claim grade for the dossier's model side of each validated match.

    A feature can appear in several matches; the dossier keeps the best-supported
    grade ("does this feature have a validated equivalent anywhere?"). Direction
    conflicts are still surfaced through the dossier's own intervention-provenance
    sign-flip tracking, so a contradicted weaker match cannot hide a flip.
    """
    grades: dict[str, str] = {}
    for item in validation_data.get("validations", []):
        if not isinstance(item, dict):
            continue
        grade = item.get("claim_grade")
        if not grade:
            continue
        for side in ("left", "right"):
            if str(item.get(f"{side}_model", "")) != model:
                continue
            feature_id = str(item.get(f"{side}_feature_id", ""))
            if not feature_id:
                continue
            current = grades.get(feature_id)
            if current is None or GRADE_ORDER.get(str(grade), -1) > GRADE_ORDER.get(current, -1):
                grades[feature_id] = str(grade)
    return grades


# --- rollup -----------------------------------------------------------------


def _effective_grade(feature_entry: dict[str, Any]) -> str:
    claim = feature_entry.get("claim_grade")
    if claim:
        return str(claim)
    if (
        feature_entry.get("has_intervention_provenance")
        or int(feature_entry.get("intervention_record_count") or 0) > 0
        or feature_entry.get("signed_effect_provenance") == "intervention"
    ):
        return "causal_evidence"
    return "associational"


def _recompute_rollup(dossier: dict[str, Any]) -> None:
    """Recompute the per-feature rollup and top-level summary from the full run list.

    Recomputing from scratch (instead of patching incrementally) keeps the rollup
    consistent even when historical entries were written by older tool versions.
    """
    runs = [run for run in dossier.get("runs", []) if isinstance(run, dict)]
    history: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for run in runs:
        timestamp = str(run.get("timestamp", ""))
        for feature_id, entry in run.get("features", {}).items():
            if isinstance(entry, dict):
                history.setdefault(str(feature_id), []).append((timestamp, entry))

    features: dict[str, Any] = {}
    for feature_id, appearances in history.items():
        features[feature_id] = _feature_rollup(appearances)

    current_grades = {fid: item["current_grade"] for fid, item in features.items()}
    validated_rank = GRADE_ORDER["validated_equivalent"]
    regressions = 0
    for feature_id, item in features.items():
        past_grades = [grade for _ts, grade in item["grade_history"][:-1]]
        current_rank = GRADE_ORDER.get(item["current_grade"], -1)
        if "validated_equivalent" in past_grades and current_rank < validated_rank:
            regressions += 1
    summary = {
        "model": dossier.get("model"),
        "criterion": dossier.get("criterion"),
        "run_count": len(runs),
        "span": {
            "first": runs[0].get("timestamp") if runs else None,
            "last": runs[-1].get("timestamp") if runs else None,
        },
        "features_total": len(features),
        "currently_validated": sum(
            1 for grade in current_grades.values() if grade == "validated_equivalent"
        ),
        "regressions": regressions,
        "open_questions": sum(
            1 for grade in current_grades.values() if grade in _OPEN_QUESTION_GRADES
        ),
        "contradictions": sum(1 for item in features.values() if item["contradiction"]),
    }
    dossier["rollup"] = {"features": features}
    dossier["summary"] = summary
    dossier["agent_next_actions"] = _dossier_next_actions(dossier, runs)


def _feature_rollup(appearances: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    grade_history = [[timestamp, _effective_grade(entry)] for timestamp, entry in appearances]
    current_grade = grade_history[-1][1]
    previous_grade = grade_history[-2][1] if len(grade_history) >= 2 else None
    sign_flips = 0
    provenance_changes = 0
    contradiction = False
    # Compare each appearance against the feature's PREVIOUS appearance (which is
    # the previous run whenever the feature was kept in both; otherwise the most
    # recent run that still ranked it -- the natural reading for resumable
    # investigations where a feature can drop out of the top-k and return).
    for (_prev_ts, prev), (_cur_ts, cur) in zip(appearances, appearances[1:]):
        prev_signed = prev.get("signed_effect")
        cur_signed = cur.get("signed_effect")
        prev_provenance = str(prev.get("signed_effect_provenance") or "none")
        cur_provenance = str(cur.get("signed_effect_provenance") or "none")
        if prev_signed is None or cur_signed is None:
            continue
        if prev_provenance != cur_provenance:
            # Never a contradiction: an intervention-vs-association change is a
            # provenance change, not a sign flip.
            if prev_provenance != "none" and cur_provenance != "none":
                provenance_changes += 1
            continue
        if prev_provenance == "none":
            continue
        if (
            abs(float(prev_signed)) >= SIGNED_EFFECT_DIRECTION_MIN
            and abs(float(cur_signed)) >= SIGNED_EFFECT_DIRECTION_MIN
            and float(prev_signed) * float(cur_signed) < 0.0
        ):
            sign_flips += 1
            if prev_provenance == "intervention":
                contradiction = True
    return {
        "label": appearances[-1][1].get("label", ""),
        "first_seen": appearances[0][0],
        "last_seen": appearances[-1][0],
        "run_count": len(appearances),
        "grade_history": grade_history,
        "current_grade": current_grade,
        "previous_grade": previous_grade,
        "grade_transition": _grade_transition(current_grade, previous_grade),
        "score_trajectory": [
            {
                "timestamp": timestamp,
                "importance": entry.get("importance"),
                "association": entry.get("association"),
                "causal_effect": entry.get("causal_effect"),
            }
            for timestamp, entry in appearances
        ],
        "sign_flips": sign_flips,
        "provenance_changes": provenance_changes,
        "contradiction": contradiction,
    }


def _grade_transition(current: str, previous: str | None) -> str:
    if previous is None:
        return "new"
    current_rank = GRADE_ORDER.get(current)
    previous_rank = GRADE_ORDER.get(previous)
    if current_rank is None or previous_rank is None:
        return "unknown"
    if current_rank > previous_rank:
        return "upgraded"
    if current_rank < previous_rank:
        return "downgraded"
    return "stable"


# --- agent next actions -----------------------------------------------------


def _dossier_next_actions(dossier: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = runs[-1] if runs else {}
    report_ref = latest.get("report_path") or "<report.json>"
    model = str(dossier.get("model", ""))
    criterion = str(dossier.get("criterion", ""))
    actions = [
        # plan-evidence ships in the same release wave as the dossier; the
        # integration pass wires the subcommand, so the dossier emits it now.
        next_action(
            action_id="plan_evidence_latest_report",
            title="Plan the next evidence-gathering step from the latest report",
            argv=["interp-lab", "plan-evidence", "--report", str(report_ref)],
            requires=["inspection report JSON"],
        ),
        next_action(
            action_id="reinspect_criterion",
            title="Re-inspect the criterion and append the fresh report to this dossier",
            argv=[
                "interp-lab",
                "inspect",
                "--model",
                model,
                "--criterion",
                criterion,
                "--backend",
                "records",
                "--records",
                "<activation-records.jsonl>",
                "--out",
                "<report-dir>",
            ],
            requires=["activation records"],
        ),
    ]
    attached = latest.get("attached", {}) if isinstance(latest.get("attached"), dict) else {}
    matches_info = attached.get("matches")
    if isinstance(matches_info, dict) and "match_validation" not in attached:
        actions.append(
            next_action(
                action_id="validate_attached_matches",
                title="Validate the attached matches so claim grades feed the dossier",
                argv=[
                    "interp-lab",
                    "validate-matches",
                    "--matches",
                    str(matches_info.get("path", "<matches.json>")),
                    "--out",
                    "<match-validation.json>",
                ],
                requires=["match report JSON"],
            )
        )
    return actions


# --- file helpers -----------------------------------------------------------


def _write_atomic(path: Path, dossier: dict[str, Any]) -> None:
    # House atomic-write pattern (web_server._persist_jobs_locked): write a
    # sibling tmp file, then rename over the target so a crash mid-write never
    # leaves a truncated dossier behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_path.replace(path)


def _sha256(path: Path) -> str:
    # Same local helper runs.py and demo_sweep.py keep privately.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
