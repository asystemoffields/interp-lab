# Stable Release Bar

interp-lab should only be released as stable when it is dependable for a serious external user bringing a local or open model plus a natural-language criterion.

## Required Capabilities

- A real-model golden path runs end to end: prompt preparation, activation collection or SAE training, inspection, intervention, causal re-inspection, attribution graph export, and compact machine-readable summary.
- At least three real-model walkthroughs cover different model families, criteria, and workflows, including one small CPU-friendly run.
- The browser app supports the core workflow for less-technical users: guided command setup, local job execution, persistent run history, artifact browsing, report preview, graph preview, clear errors, and configuration export/import.
- Causal validation defaults encourage held-out prompts, control prompts or control paths, repeated interventions, and honest evidence-strength language.
- Cross-model and path-patching reports distinguish association, measured causal effects, validated paths, controls, and unvalidated hypotheses.
- Public CLI, Python API, and JSON schemas are versioned and covered by compatibility tests.
- Optional adapters fail clearly when dependencies, credentials, network access, model trust settings, or local files are missing.
- The package has cross-platform validation across Windows, macOS, Linux, and Python 3.10-3.12.
- Release docs cover installation, first run, production workflow, scaling, PyPI publishing, and artifact publishing.

## Stable Release Gate

Run:

```bash
interp-lab release-check --strict --out reports/release-check.json
```

The command must report no blockers before changing the PyPI classifier to `Development Status :: 5 - Production/Stable`, tagging a stable GitHub release, or publishing the matching PyPI release.

## Current Known Blockers

- Keep the package classifier as alpha until the release gate passes.
- Expand the real-model demo suite beyond smoke tests into reproducible walkthroughs with expected artifacts and interpretation notes.
- Continue hardening the browser app until the guided path feels complete for a new user.
- Keep adding schema and API compatibility tests as public surfaces settle.
