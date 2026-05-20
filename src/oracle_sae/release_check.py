from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10 in CI.
    import tomli as tomllib

from oracle_sae import __version__ as oracle_version
from oracle_sae.explanation_reports import (
    EXPLANATION_CONSISTENCY_SCHEMA,
    FEATURE_SEARCH_SCHEMA,
    MODEL_FAMILY_COMPARISON_SCHEMA,
)
from oracle_sae.feature_interventions import INTERVENTION_SCHEMA, PLAN_SCHEMA
from oracle_sae.schema import INSPECTION_REPORT_SCHEMA, MATCH_REPORT_SCHEMA

RELEASE_CHECK_SCHEMA = "interp-lab.release_check.v1"
REAL_MODEL_DEMO_SCHEMA = "interp-lab.real_model_demo.v1"


@dataclass(frozen=True)
class ReleaseCheckResult:
    report: dict[str, Any]
    path: Path | None = None


def build_release_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess whether interp-lab is ready for a stable public release.")
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect.")
    parser.add_argument("--out", help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any blocker is present. Use this as the final stable-release gate.",
    )
    return parser


def run_release_check_from_args(args: argparse.Namespace) -> ReleaseCheckResult:
    root = Path(args.repo_root).resolve()
    report = build_release_readiness_report(root)
    path = None
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReleaseCheckResult(report=report, path=path)


def build_release_readiness_report(root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(root).resolve()
    pyproject = _load_pyproject(repo_root / "pyproject.toml")
    checks = [
        _check_version_sync(pyproject),
        _check_development_classifier(pyproject),
        _check_required_docs(repo_root),
        _check_stable_release_doc(repo_root),
        _check_known_stable_blockers(repo_root),
        _check_golden_demo_doc(repo_root),
        _check_real_model_demo_coverage(repo_root),
        _check_real_model_demo_sweep(repo_root),
        _check_browser_app(repo_root),
        _check_ci_matrix(repo_root),
        _check_publish_workflow(repo_root),
        _check_schema_contracts(repo_root),
        _check_worktree_clean(repo_root),
    ]
    counts = {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "warn": sum(1 for check in checks if check["status"] == "warn"),
        "blocker": sum(1 for check in checks if check["status"] == "blocker"),
    }
    ready = counts["blocker"] == 0
    next_actions = [
        {
            "id": check["id"],
            "title": check["title"],
            "next_action": check["next_action"],
        }
        for check in checks
        if check["status"] in {"warn", "blocker"} and check.get("next_action")
    ]
    return {
        "schema_version": RELEASE_CHECK_SCHEMA,
        "ready_for_stable_release": ready,
        "summary": counts,
        "checks": checks,
        "agent_next_actions": next_actions,
    }


def render_release_check_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    verdict = "READY" if report.get("ready_for_stable_release") else "NOT READY"
    lines = [
        f"interp-lab stable release check: {verdict}",
        f"pass={summary.get('pass', 0)} warn={summary.get('warn', 0)} blocker={summary.get('blocker', 0)}",
        "",
    ]
    for check in report.get("checks", []):
        status = str(check.get("status", "")).upper()
        lines.append(f"[{status}] {check.get('title', check.get('id', 'check'))}")
        detail = check.get("detail")
        if detail:
            lines.append(f"  {detail}")
        next_action = check.get("next_action")
        if next_action:
            lines.append(f"  Next: {next_action}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_pyproject(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _check_version_sync(pyproject: dict[str, Any]) -> dict[str, str]:
    project_version = str(pyproject.get("project", {}).get("version", ""))
    interp_version = _interp_lab_version()
    versions = {
        "pyproject": project_version,
        "oracle_sae": oracle_version,
        "interp_lab": interp_version,
    }
    ok = bool(project_version) and len(set(versions.values())) == 1
    return _check(
        "version_sync",
        "Package versions are synchronized",
        "pass" if ok else "blocker",
        f"pyproject={project_version}, oracle_sae={oracle_version}, interp_lab={interp_version}",
        "Synchronize pyproject.toml, oracle_sae.__version__, and interp_lab.__version__ before release.",
    )


def _interp_lab_version() -> str:
    try:
        import interp_lab
    except Exception:  # pragma: no cover - import failure is surfaced in the check detail.
        return "<unavailable>"
    return str(getattr(interp_lab, "__version__", "<missing>"))


def _check_development_classifier(pyproject: dict[str, Any]) -> dict[str, str]:
    classifiers = [str(item) for item in pyproject.get("project", {}).get("classifiers", [])]
    status = "warn"
    detail = "No production/stable development classifier found."
    if any("Development Status :: 3 - Alpha" in item or "Development Status :: 4 - Beta" in item for item in classifiers):
        status = "blocker"
        detail = "Project is still classified as alpha/beta."
    if any("Development Status :: 5 - Production/Stable" in item or "Development Status :: 6 - Mature" in item for item in classifiers):
        status = "pass"
        detail = "Project advertises a stable development classifier."
    return _check(
        "development_classifier",
        "PyPI classifier matches stable-release intent",
        status,
        detail,
        "Keep the alpha classifier until release blockers are resolved; switch to Production/Stable only at the stable release.",
    )


def _check_required_docs(root: Path) -> dict[str, str]:
    required = [
        "README.md",
        "docs/PRODUCTION.md",
        "docs/PYTHON_API.md",
        "docs/RELEASE.md",
        "docs/SCALING.md",
        "docs/GOLDEN_REAL_MODEL_DEMO.md",
        "docs/REAL_MODEL_DEMOS.md",
    ]
    missing = [path for path in required if not (root / path).exists()]
    return _check(
        "required_docs",
        "Core user and release docs exist",
        "pass" if not missing else "blocker",
        "All required docs are present." if not missing else f"Missing: {', '.join(missing)}",
        "Add missing release, production, API, scaling, or golden-demo docs.",
    )


def _check_stable_release_doc(root: Path) -> dict[str, str]:
    path = root / "docs/STABLE_RELEASE.md"
    if not path.exists():
        return _check(
            "stable_release_criteria",
            "Stable release criteria are documented",
            "blocker",
            "docs/STABLE_RELEASE.md is missing.",
            "Document the non-alpha release bar and keep this checklist tied to it.",
        )
    text = path.read_text(encoding="utf-8")
    required_terms = ["real-model", "browser", "schema", "causal", "cross-platform", "release"]
    missing = [term for term in required_terms if term not in text.lower()]
    return _check(
        "stable_release_criteria",
        "Stable release criteria are documented",
        "pass" if not missing else "warn",
        "Stable release criteria doc is present." if not missing else f"Missing topics: {', '.join(missing)}",
        "Expand docs/STABLE_RELEASE.md so it covers the full non-alpha release bar.",
    )


def _check_known_stable_blockers(root: Path) -> dict[str, str]:
    path = root / "docs/STABLE_RELEASE.md"
    if not path.exists():
        return _check(
            "known_stable_blockers",
            "Known stable-release blockers are tracked",
            "blocker",
            "Stable release doc is missing, so known blockers cannot be audited.",
            "Add docs/STABLE_RELEASE.md with a Current Known Blockers section.",
        )
    blockers = _stable_release_blockers(path.read_text(encoding="utf-8"))
    return _check(
        "known_stable_blockers",
        "Known stable-release blockers are resolved",
        "pass" if not blockers else "blocker",
        "No known stable-release blockers are listed."
        if not blockers
        else f"{len(blockers)} blocker(s) listed in docs/STABLE_RELEASE.md.",
        "Resolve or intentionally remove every item in docs/STABLE_RELEASE.md Current Known Blockers before stable release.",
    )


def _stable_release_blockers(text: str) -> list[str]:
    lines = text.splitlines()
    in_section = False
    blockers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lower() == "## current known blockers"
            continue
        if in_section and stripped.startswith("- "):
            blockers.append(stripped[2:].strip())
    return blockers


def _check_golden_demo_doc(root: Path) -> dict[str, str]:
    path = root / "docs/GOLDEN_REAL_MODEL_DEMO.md"
    if not path.exists():
        return _check(
            "golden_demo",
            "Golden real-model demo is documented",
            "blocker",
            "docs/GOLDEN_REAL_MODEL_DEMO.md is missing.",
            "Add a reproducible end-to-end real-model demo.",
        )
    text = path.read_text(encoding="utf-8")
    required = ["prepare-sae-prompts", "train-sae", "inspect", "intervene", "export-attribution-graph"]
    missing = [item for item in required if item not in text]
    return _check(
        "golden_demo",
        "Golden real-model demo is documented",
        "pass" if not missing else "blocker",
        "Golden demo includes prompt prep, SAE training, inspection, intervention, and graph export."
        if not missing
        else f"Missing commands: {', '.join(missing)}",
        "Complete the golden demo workflow so new users and agents can reproduce it.",
    )


def _check_real_model_demo_coverage(root: Path) -> dict[str, str]:
    demo_dir = root / "examples/real_model_demos"
    manifests = sorted(demo_dir.glob("*.json")) if demo_dir.exists() else []
    valid: list[str] = []
    invalid: list[str] = []
    docs: set[str] = set()
    workflows: set[str] = set()
    for path in manifests:
        ok, detail = _validate_real_model_demo_manifest(path, root)
        if ok:
            valid.append(path.name)
            payload = json.loads(path.read_text(encoding="utf-8"))
            docs.add(str(payload.get("doc", "")))
            workflows.add(str(payload.get("workflow", "")))
        else:
            invalid.append(f"{path.name}: {detail}")
    doc_files = [root / "docs/GOLDEN_REAL_MODEL_DEMO.md", root / "docs/REAL_MODEL_SMOKE_TEST.md", root / "docs/GEMMA4_WALKTHROUGH.md"]
    documented = [path.name for path in doc_files if path.exists()]
    enough = len(valid) >= 3 and len(docs) >= 3 and len(workflows) >= 3 and not invalid
    status = "pass" if enough else "warn"
    detail = (
        f"Found {len(valid)} valid real-model demo manifest(s): {', '.join(valid)}"
        if enough
        else (
            f"Valid manifests={len(valid)}, docs={len(documented)}, workflows={len(workflows)}. "
            + ("Invalid: " + "; ".join(invalid) if invalid else "Add or complete real-model demo manifests.")
        )
    )
    return _check(
        "real_model_demo_coverage",
        "Multiple real-model workflows are documented",
        status,
        detail,
        "Add at least three reproducible real-model demo manifests with expected artifacts and interpretation notes.",
    )


def _check_real_model_demo_sweep(root: Path) -> dict[str, str]:
    path = root / "reports/real-model-demo-sweep.json"
    if not path.exists():
        return _check(
            "real_model_demo_sweep",
            "Real-model demo sweep is archived",
            "blocker",
            "reports/real-model-demo-sweep.json is missing.",
            "Run interp-lab demo-sweep --run --out reports/real-model-demo-sweep.json after installing demo dependencies.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _check(
            "real_model_demo_sweep",
            "Real-model demo sweep is archived",
            "blocker",
            f"Invalid sweep JSON: {exc.msg}",
            "Regenerate reports/real-model-demo-sweep.json with interp-lab demo-sweep --run.",
        )
    problems = []
    if payload.get("schema_version") != "interp-lab.real_model_demo_sweep.v1":
        problems.append("schema_version")
    if payload.get("status") != "passed":
        problems.append(f"status={payload.get('status', '<missing>')}")
    if not payload.get("run_commands"):
        problems.append("run_commands=false")
    try:
        selected_demo_count = int(payload.get("selected_demo_count", 0) or 0)
    except (TypeError, ValueError):
        selected_demo_count = 0
    if selected_demo_count < 3:
        problems.append("selected_demo_count<3")
    return _check(
        "real_model_demo_sweep",
        "Real-model demo sweep is archived",
        "pass" if not problems else "blocker",
        "Full demo sweep passed with command execution evidence."
        if not problems
        else "Sweep report is incomplete: " + ", ".join(problems),
        "Run the full real-model demo sweep with command execution and archive the generated report before stable release.",
    )


def _validate_real_model_demo_manifest(path: Path, root: Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return False, "manifest must be a JSON object"
    required_scalars = ["schema_version", "id", "title", "model", "criterion", "workflow", "doc", "estimated_runtime"]
    missing = [key for key in required_scalars if not payload.get(key)]
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    if payload["schema_version"] != REAL_MODEL_DEMO_SCHEMA:
        return False, f"schema_version must be {REAL_MODEL_DEMO_SCHEMA}"
    doc_path = root / str(payload["doc"])
    if not doc_path.exists():
        return False, f"doc path does not exist: {payload['doc']}"
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) < 3:
        return False, "commands must include at least three runnable steps"
    for command in commands:
        if not isinstance(command, dict) or not command.get("name") or not command.get("argv"):
            return False, "each command needs name and argv"
        argv = command["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            return False, "command argv must be a non-empty string list"
    artifacts = payload.get("expected_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 3:
        return False, "expected_artifacts must include at least three artifacts"
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False, "expected_artifacts entries must be objects"
        missing_artifact = [
            key
            for key in ["path", "kind", "why_it_matters", "interpretation_notes"]
            if not artifact.get(key)
        ]
        if missing_artifact:
            return False, f"artifact missing fields: {', '.join(missing_artifact)}"
    checks = payload.get("evidence_checks")
    if not isinstance(checks, list) or len(checks) < 2:
        return False, "evidence_checks must include at least two checks"
    inputs = payload.get("required_inputs", [])
    if inputs and not isinstance(inputs, list):
        return False, "required_inputs must be a list when present"
    for item in inputs:
        if isinstance(item, str):
            if not item:
                return False, "required_inputs string entries cannot be empty"
            continue
        if not isinstance(item, dict) or not item.get("path"):
            return False, "required_inputs entries must be strings or objects with path"
    return True, "ok"


def _check_browser_app(root: Path) -> dict[str, str]:
    files = ["src/oracle_sae/web_app.py", "src/oracle_sae/web_server.py"]
    missing = [path for path in files if not (root / path).exists()]
    web_app = (root / "src/oracle_sae/web_app.py").read_text(encoding="utf-8") if not missing else ""
    web_server = (root / "src/oracle_sae/web_server.py").read_text(encoding="utf-8") if not missing else ""
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    documented = "studio --serve" in readme and "reports-dir" in readme and "persistent job history" in readme.lower()
    required_surfaces = {
        "run-config import": "run-config-import" in web_app and "startImportedConfig" in web_app,
        "persistent history schema": "STUDIO_HISTORY_SCHEMA" in web_server,
        "history path API": "history_path" in web_server,
        "artifact API": "/api/artifacts" in web_server,
    }
    missing_surfaces = [name for name, present in required_surfaces.items() if not present]
    status = "pass" if not missing and documented and not missing_surfaces else "warn"
    detail = (
        "Studio supports served jobs, persistent history, run-config import, and artifact browsing."
        if status == "pass"
        else "Missing browser-app polish: " + ", ".join(missing + missing_surfaces + ([] if documented else ["README served-mode docs"]))
    )
    return _check(
        "browser_app_workflow",
        "Browser app workflow is present",
        status,
        detail,
        "Polish served Studio until a less-technical user can run core workflows and inspect artifacts from the browser.",
    )


def _check_ci_matrix(root: Path) -> dict[str, str]:
    path = root / ".github/workflows/ci.yml"
    if not path.exists():
        return _check("ci_matrix", "Cross-platform CI is configured", "blocker", "CI workflow missing.", "Add CI before stable release.")
    text = path.read_text(encoding="utf-8")
    required = ["ubuntu-latest", "macos-latest", "windows-latest", "3.10", "3.11", "3.12"]
    missing = [item for item in required if item not in text]
    return _check(
        "ci_matrix",
        "Cross-platform CI is configured",
        "pass" if not missing else "blocker",
        "CI covers Ubuntu, macOS, Windows, and Python 3.10-3.12." if not missing else f"Missing: {', '.join(missing)}",
        "Restore the full OS and Python CI matrix.",
    )


def _check_publish_workflow(root: Path) -> dict[str, str]:
    path = root / ".github/workflows/publish.yml"
    if not path.exists():
        return _check("publish_workflow", "Trusted publishing workflow exists", "blocker", "Publish workflow missing.", "Add trusted publishing workflow.")
    text = path.read_text(encoding="utf-8")
    required = ["pypa/gh-action-pypi-publish", "id-token: write", "release:"]
    missing = [item for item in required if item not in text]
    return _check(
        "publish_workflow",
        "Trusted publishing workflow exists",
        "pass" if not missing else "blocker",
        "Publish workflow is configured for GitHub release-triggered PyPI publishing."
        if not missing
        else f"Missing: {', '.join(missing)}",
        "Fix trusted publishing before stable release.",
    )


def _check_schema_contracts(root: Path) -> dict[str, str]:
    required = {
        "inspection_report": INSPECTION_REPORT_SCHEMA,
        "match_report": MATCH_REPORT_SCHEMA,
        "intervention_result": INTERVENTION_SCHEMA,
        "intervention_plan": PLAN_SCHEMA,
        "attribution_graph": "interp-lab.attribution_graph.v1",
        "attribution_graph_summary": "interp-lab.attribution_graph_summary.v1",
        "run_manifest": "interp-lab.run.v1",
        "path_patch": "interp-lab.path_patch.v1",
        "match_validation": "interp-lab.match_validation.v1",
        "graph_validation": "interp-lab.graph_validation.v1",
        "environment_profile": "interp-lab.env_profile.v1",
        "explanation_consistency": EXPLANATION_CONSISTENCY_SCHEMA,
        "feature_search": FEATURE_SEARCH_SCHEMA,
        "public_api_contract": "interp-lab.public_api_contract.v1",
        "release_check": RELEASE_CHECK_SCHEMA,
        "real_model_demo": REAL_MODEL_DEMO_SCHEMA,
        "real_model_demo_sweep": "interp-lab.real_model_demo_sweep.v1",
        "model_family_comparison": MODEL_FAMILY_COMPARISON_SCHEMA,
    }
    contract_tests = (root / "tests/test_contracts.py").exists()
    missing = [key for key, value in required.items() if not value.endswith(".v1")]
    if not contract_tests:
        missing.append("contract_tests")
    return _check(
        "schema_contracts",
        "Core machine-readable schemas are versioned",
        "pass" if not missing else "blocker",
        "Inspection reports, match reports, graph summaries, intervention plans, run manifests, public API contracts, and release checks use versioned schemas."
        if not missing
        else f"Missing v1 schema constants: {', '.join(missing)}",
        "Add explicit schema_version fields and contract tests for stable machine-readable outputs.",
    )


def _check_worktree_clean(root: Path) -> dict[str, str]:
    if not (root / ".git").exists():
        return _check(
            "worktree_clean",
            "Git worktree is clean",
            "warn",
            "No .git directory found at repo root.",
            "Run release checks from a Git checkout before tagging.",
        )
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - platform/git environment issue.
        return _check("worktree_clean", "Git worktree is clean", "warn", f"Could not inspect git status: {exc}", "Check git status manually before release.")
    dirty = result.stdout.strip()
    return _check(
        "worktree_clean",
        "Git worktree is clean",
        "pass" if not dirty else "blocker",
        "Working tree is clean." if not dirty else "Uncommitted changes are present.",
        "Commit, stash, or intentionally discard local changes before tagging.",
    )


def _check(id: str, title: str, status: str, detail: str, next_action: str) -> dict[str, str]:
    payload = {
        "id": id,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if status != "pass":
        payload["next_action"] = next_action
    return payload
