# Open-Source Launch Checklist

This checklist is for publishing interp-lab as a useful public tool.

## Repository

- Initialize git and set the remote to `https://github.com/asystemoffields/interp-lab`.
- Keep source, tests, examples, docs, and small fixtures in git.
- Keep generated reports, model caches, and large SAE artifacts out of git.
- Add a release tag after CI is green.

## Quality Bar

- `interp-lab doctor` succeeds on a clean environment.
- `interp-lab run examples/run_records.json` writes a manifest and report.
- `from interp_lab import inspect, compare, train_sae` works from an installed wheel.
- Unit tests pass on Ubuntu, macOS, and Windows.
- Package build succeeds for Python 3.10, 3.11, and 3.12.
- README quickstart works from a fresh clone.

## Public Docs

- README: quickstart, commands, real-model example, adapter overview.
- `docs/PRODUCTION.md`: run configs, manifests, platform support.
- `docs/RELEASE.md`: PyPI trusted publishing and release process.
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
