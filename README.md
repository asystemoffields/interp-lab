# interp-lab

interp-lab is an open-source toolkit for criterion-driven mechanistic interpretability.

Give it a model, a plain-language criterion, and feature evidence. It ranks the internal features that track the criterion, explains them, tests their causal impact with interventions, and searches for equivalent features in other models, then grades how much each claim is supported by evidence.

```bash
python -m pip install interp-lab
interp-lab doctor
interp-lab quickstart        # a short guided walkthrough of the workflow and metrics

# A complete tour on toy models in one command — no GPU, no downloads:
interp-lab demo --out reports/demo   # then open reports/demo/index.html
```

```bash
interp-lab inspect \
  --model google/gemma-2-2b \
  --criterion "the model is aware it is being evaluated" \
  --backend toy \
  --out reports/eval-awareness
```

## Features

- **Correlational vs. causal evidence are kept separate.** Association comes from activation/criterion statistics; causal effect comes from ablation, amplification, clamp, patch, and steering runs. A feature that merely co-activates is not treated like one that *moves* the behavior.
- **Claims are graded, not asserted.** `validate-matches` and `validate-attribution-graph` mark each result as `validated`, `needs_causal_evidence`, `plausible`, `contradicted`, or `weak`, with reason codes.
- **Controls and uncertainty are first-class.** Intervention runs support `random_feature`, `matched_frequency`, and `placebo` controls, side-effect checks, sign-consistency, and confidence intervals.
- **Everything is reproducible and agent-friendly.** Runs emit manifests with the tool version, platform, and input hashes; reports include `agent_next_actions` with exact follow-up commands; `interp_lab.public_api_contract()` exposes the stable surface as data.

## The workflow

1. Compile a natural-language criterion into examples and scores.
2. Collect candidate features from SAEs, NLA explanations, or feature dumps — or feed any latents (crosscoders included) through the model-agnostic activation-records path.
3. Rank features by criterion association, specificity, causal evidence, and stability.
4. Build a feature fingerprint that can be compared across models.
5. Validate cross-model equivalents with interventions.

```python
from interp_lab import compare, inspect, validate_matches

left = inspect("toy/model-a", "the model is aware it is being evaluated", backend="toy", out="reports/model-a")
right = inspect("toy/model-b", "the model is aware it is being evaluated", backend="toy", out="reports/model-b")
matches = compare(left.report, right.report, out="reports/matches.json")
validation = validate_matches(matches.report, out="reports/match-validation.json")
```

## Evidence sources

interp-lab keeps portable JSONL evidence formats stable in the base package; heavier model tooling lives behind optional extras. Supported paths include toy, JSONL feature dumps, activation records, Neuronpedia, SAE Lens, Goodfire, Gemma Scope / Qwen-Scope, Hugging Face, TransformerLens, NNsight, contrast-direction, and on-demand SAE training. Each integration is an optional bridge (`pip install "interp-lab[saelens]"`, `[hf]`, `[transformerlens]`, `[nnsight]`, `[goodfire]`, `[modal]`, `[publish]`, …).

## Architecture

The core object is a `FeatureFingerprint`:

```text
activation signature
+ text explanation embedding
+ decoder signature
+ causal effect vector
+ examples
```

Cross-model equivalence is scored by fingerprint similarity; `validate-matches` turns candidates into explicit evidence grades. The pipeline is built around four small adapter interfaces, so new backends are easy to add:

- `FeatureProvider` — returns candidate features.
- `Verbalizer` — adds NLA-style text explanations.
- `InterventionRunner` — ablates, amplifies, patches, or estimates causal effects.
- `CriterionCompiler` — turns natural-language criteria into examples and scoring hints.

### Text matching: lexical by default, semantic when you want it

The text component of a fingerprint defaults to a dependency-free **lexical** vector (token hashing) — deterministic, offline, and comparable across versions, but it matches shared *words*, not meaning. For real cross-model and cross-vocabulary matching, opt into a **semantic** embedder:

```bash
pip install "interp-lab[embeddings]"

# Local MiniLM (sentence-transformers): free, offline, no API key.
interp-lab inspect ... --text-embedder minilm
# or set once for a whole pipeline:
export INTERP_LAB_TEXT_EMBEDDER=minilm
```

Each fingerprint records the embedder that produced it, and matching refuses to compare vectors from different embedders (it drops the text component and renormalizes rather than silently cosine-ing across incompatible axes). `interp-lab doctor` shows the active embedder and whether the extra is installed.

> Note: ranking importance weights are heuristic — treat scores as evidence-weighted rankings, not probabilities.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Documentation

- **[Full command reference](docs/COMMANDS.md)** — every CLI command and the JSONL data formats (feature dumps, activation records, intervention records).
- [`docs/PYTHON_API.md`](docs/PYTHON_API.md) — the Python API.
- [`docs/GOLDEN_REAL_MODEL_DEMO.md`](docs/GOLDEN_REAL_MODEL_DEMO.md) — a compact real-model walkthrough (trains a small DistilGPT-2 SAE, suppresses latents, re-inspects with causal evidence, exports an attribution graph).
- **[Archived DistilGPT-2 run](examples/real_model_demos/golden-distilgpt2-unit/)** — real committed artifacts from that walkthrough: a measured criterion-promoting SAE latent, an authentic suppression dose-response, and semantic (MiniLM) fingerprints. Open `inspect-causal/report.html` to see the numbers.
- [`docs/REAL_MODEL_DEMOS.md`](docs/REAL_MODEL_DEMOS.md) and [`examples/real_model_demos/`](examples/real_model_demos) — the broader real-model suite.
- [`docs/GEMMA4_WALKTHROUGH.md`](docs/GEMMA4_WALKTHROUGH.md) and [`docs/SCALING.md`](docs/SCALING.md) — large-model and 1T+ paths.

Common entry points:

```bash
interp-lab demo --out reports/demo            # full toy tour (open reports/demo/index.html)
interp-lab quickstart                         # guided getting-started walkthrough
interp-lab inspect ... --csv-out features.csv # ranked features as a spreadsheet
interp-lab compare-runs --left a/report.json --right b/report.json --out diff.json  # rank/score drift
interp-lab studio --serve --reports-dir reports   # local browser command-builder + runner
interp-lab release-check --strict             # stable-release readiness
```

## Roadmap

- Richer Natural Language Autoencoder explanation audits.
- Crosscoder training and import.
- Distributed SAE training manifests.
- Remote causal validation workers.
- Feature transfer tests across model families.
- Public example gallery with archived real-model reports (started — see [`examples/real_model_demos/`](examples/real_model_demos)).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

MIT licensed. Contributions welcome — see the [issue tracker](https://github.com/asystemoffields/interp-lab/issues).
