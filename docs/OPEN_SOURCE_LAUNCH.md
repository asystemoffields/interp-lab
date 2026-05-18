# Open-Source Launch Checklist

This checklist is for publishing Oracle SAE / Interp Lab as a useful public tool.

## Repository

- Initialize git and set the remote to `https://github.com/asystemoffields/interp-lab`.
- Keep source, tests, examples, docs, and small fixtures in git.
- Keep generated reports, model caches, and large SAE artifacts out of git.
- Add a release tag after CI is green.

## Quality Bar

- `oracle-sae doctor` succeeds on a clean environment.
- `oracle-sae run examples/run_records.json` writes a manifest and report.
- Unit tests pass on Ubuntu, macOS, and Windows.
- Package build succeeds for Python 3.10, 3.11, and 3.12.
- README quickstart works from a fresh clone.

## Public Docs

- README: quickstart, commands, real-model example, adapter overview.
- `docs/PRODUCTION.md`: run configs, manifests, platform support.
- `docs/REAL_MODEL_SMOKE_TEST.md`: validated small-model path.
- `docs/ARCHITECTURE.md`: adapter contracts and core loop.
- `docs/ROADMAP.md`: next research and engineering milestones.

## Demo Packet

For outreach, send:

- repository link;
- one-page technical summary;
- command transcript or manifest from a real small-model run;
- generated causal report;
- short description of the feedback or collaboration wanted.

## First Issues To File

- Add TransformerLens activation cache adapter.
- Add nnsight activation hook adapter.
- Add HTML feature-card report.
- Add criterion dataset generator.
- Add causal validation suite with confidence intervals.
- Add cross-model transfer validation.
