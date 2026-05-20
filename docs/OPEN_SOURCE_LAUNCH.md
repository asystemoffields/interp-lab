# Post-1.0 Outreach Checklist

This checklist is for sharing interp-lab with researchers, builders, and interpretability teams.

## Current Public Assets

- Repository: `https://github.com/asystemoffields/interp-lab`.
- Stable release: `v1.0.0`.
- PyPI package: `interp-lab==1.0.0`.
- Keep the release-check report, demo-sweep report, CI matrix, and built distributions attached to the release packet.

## Repository Hygiene

- Keep source, tests, examples, docs, and small fixtures in git.
- Keep generated reports, model caches, and large SAE artifacts out of git.
- Run `interp-lab demo-sweep --run --out reports/real-model-demo-sweep.json` and archive the sweep packet before major outreach.
- Run `interp-lab release-check --strict --out reports/release-check.json` before each stable release tag.

## Quality Bar

- `interp-lab doctor` succeeds on a clean environment.
- `interp-lab run examples/run_records.json` writes a manifest and report.
- `from interp_lab import inspect, compare, validate_matches, train_sae` works from an installed wheel.
- Unit tests pass on Ubuntu, macOS, and Windows.
- Package build succeeds for Python 3.10, 3.11, and 3.12.
- README quickstart works from a fresh clone.

## Public Docs

- README: quickstart, commands, real-model example, adapter overview.
- `docs/PRODUCTION.md`: run configs, manifests, platform support.
- `docs/RELEASE.md`: PyPI trusted publishing and release process.
- `docs/REAL_MODEL_SMOKE_TEST.md`: validated small-model path.
- `docs/ARCHITECTURE.md`: adapter contracts and core loop.
- `docs/SCALING.md`: sharded activation and 1T+ execution shape.
- `docs/ROADMAP.md`: next research and engineering milestones.

## Demo Packet

For outreach, send:

- repository link;
- one-page technical summary;
- command transcript or manifest from a real small-model run;
- `reports/real-model-demo-sweep.json`;
- generated causal report;
- generated attribution graph, validation report, and compact graph summary;
- short description of the feedback or collaboration wanted.

## Good First Follow-Ups

- Add held-out transfer calibration datasets for cross-model validation.
- Add distributed SAE training manifests.
- Add remote causal validation workers.
- Add Goodfire activation-record export once the desired API shape is settled.
