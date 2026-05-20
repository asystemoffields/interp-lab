# Release Guide

interp-lab publishes to PyPI with trusted publishing. The package name is `interp-lab`, and the import package is `interp_lab`.

## Current Stable Release

- GitHub release: `v1.0.0`
- PyPI package: `interp-lab==1.0.0`
- Verification: install from PyPI in a clean environment, run `interp-lab doctor`, and confirm `interp_lab.__version__ == "1.0.0"`.

## One-Time PyPI Setup

The PyPI trusted-publisher setup is complete for this repository. These steps are for maintainers recreating the project setup or moving ownership.

Create the PyPI project and trusted publisher:

1. Sign in to PyPI with an account that can own `interp-lab`.
2. Create or reserve the `interp-lab` project.
3. Add a trusted publisher with:
   - PyPI project name: `interp-lab`
   - Owner: `asystemoffields`
   - Repository: `interp-lab`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

The GitHub workflow uses OpenID Connect, so no PyPI token is stored in GitHub.

## Pre-Release Checks

Run locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall src tests
python -c "import shutil, pathlib; [shutil.rmtree(path) if path.is_dir() else path.unlink() for pattern in ('dist', 'build', '*.egg-info') for path in pathlib.Path('.').glob(pattern)]"
python -m build
python -m twine check dist/*
interp-lab doctor
interp-lab run examples/run_records.json
interp-lab demo-sweep --run --out reports/real-model-demo-sweep.json
interp-lab release-check --strict --out reports/release-check.json
```

Archive the generated demo-sweep and release-check reports with the GitHub release notes.

Confirm the package imports:

```bash
python -c "from interp_lab import inspect, compare, train_sae, doctor; print(doctor()['tool'])"
```

## Release

1. Update `version` in `pyproject.toml`.
2. Update `__version__` in `src/oracle_sae/__init__.py` and `src/interp_lab/__init__.py`.
3. Commit the version bump.
4. Tag the release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

5. Create a GitHub Release for that tag.
6. The `Publish` workflow will build the package, check metadata, and upload to PyPI.

Manual publish is also available from GitHub Actions through the `Publish` workflow dispatch.

## Post-Release Verification

After PyPI publishes a new version:

```bash
python -m pipx run --spec interp-lab interp-lab doctor
python -m pip install interp-lab==1.0.0
python -c "from interp_lab import inspect; print(inspect('toy/a', 'benchmark awareness').cards[0].feature_id)"
```

## Current Release Artifact

The wheel should include:

- `interp_lab`: public Python API
- `oracle_sae`: internal engine package
- `interp-lab`: console script
