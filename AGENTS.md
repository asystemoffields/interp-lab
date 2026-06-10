# interp-lab for AI agents

This is the operating manual for coding agents (Claude Code, Codex, and friends)
driving interp-lab on a user's machine. Read this before running commands; it
tells you how to discover the surface, how to interpret evidence, and what you
are allowed to claim from each artifact.

## What interp-lab is

interp-lab is a criterion-driven mechanistic interpretability toolkit. You give
it a model, a plain-language criterion ("the model is aware it is being
evaluated"), and feature evidence (SAE latents, activation records, feature
dumps); it ranks the internal features that track the criterion, explains them,
tests their causal impact with interventions, and matches equivalent features
across models — grading every claim by the evidence behind it.

## The evidence philosophy (do not skip)

- Correlational and causal evidence are kept strictly separate. `association`
  comes from activation/criterion statistics; `causal_effect` comes from
  ablation, amplification, clamp, patch, and steering interventions.
- Claims are graded, not asserted: `validated`, `needs_causal_evidence`,
  `plausible`, `contradicted`, `weak` — with machine-readable reason codes.
- Provenance fields tell you how a signed effect was obtained:
  `signed_causal_effect` was measured via interventions;
  `signed_association` is a correlational proxy. Never conflate them, and
  never compare one against the other — interp-lab itself refuses to.
- The `contradicted` grade requires intervention provenance on both sides.
  Two opposite-signed *correlations* are not a contradiction — they are two
  untested hypotheses pointing different ways. Such pairs grade
  `needs_causal_evidence` with the reason code
  `opposite_associations_lack_intervention_provenance`. Only an
  intervention-vs-intervention sign conflict earns `contradicted`.

## Discovery: start here

```bash
interp-lab capabilities --json   # ONE payload: full command specs, Python API
                                 # contract, environment, and house conventions
interp-lab doctor --json         # environment / optional-extras health check
```

From Python, `interp_lab.public_api_contract()` returns the stable API surface
as data (exports, schema versions, signatures).

## MCP server

`interp-lab mcp` serves the workflow — including the full investigation loop —
as Model Context Protocol tools over stdio: `capabilities`, `doctor`,
`inspect`, `compare`, `validate_matches`, `search_features`, `compare_runs`,
`check_explanation_consistency`, `attribution_graph`,
`validate_attribution_graph`, `plan_evidence`, `dossier_update`,
`dossier_show`, `quant_diff`, `calibrate`, `migrate_report`, `export_steering`,
`intervene` (dry-run by default: it returns the plan without loading a model;
without the `[hf]` extra, execution fails cleanly with an install hint as an
`isError` result), and `train_sae` (`method=auto` falls back to the stdlib
trainer when torch is absent). `apply-steering` is deliberately **not** served:
generating text from arbitrary models is a decision the host agent (you) must
make explicitly — use the CLI once you have. Example client config:

```json
{
  "mcpServers": {
    "interp-lab": { "command": "interp-lab", "args": ["mcp"] }
  }
}
```

## Conventions

- **JSON-first artifacts.** Every machine-readable artifact carries a versioned
  `schema_version` (for example `interp-lab.inspection_report.v1`).
- **`--json` keeps stdout pure.** With `--json`, stdout is exactly one JSON
  payload; confirmations like `Wrote ...` go to stderr. Parse stdout directly.
- **Errors are uniform.** Failures print `interp-lab: error: ...` and exit 2.
- **Placeholders.** Suggested commands use `<angle-bracket>` tokens
  (`<causal-prompts.jsonl>`, `<sae.json>`) for run-local artifacts you must
  substitute before running.
- **`agent_next_actions` is the canonical follow-up channel.** Reports embed
  next actions in one shape everywhere:

  ```json
  {
    "id": "plan_sae_suppression",
    "title": "Plan a suppression test for this SAE latent",
    "command": "interp-lab intervene --model m ... --dry-run --json",
    "argv": ["interp-lab", "intervene", "--model", "m", "..."],
    "requires": ["scored causal prompt JSONL"]
  }
  ```

  `id` and `title` are always present. Runnable suggestions carry both
  `command` (shlex-quoted string) and `argv` (same tokens as a list — prefer
  `argv` for execution). Prose-only guidance carries `instruction` instead of
  `command`/`argv`. `requires` lists artifacts you need before the action can
  run. The 2.x legacy keys (`release-check`'s `next_action`,
  `check-explanation-consistency`'s `description` and flat
  `agent_next_action*` keys) were **removed in 3.0.0** — consumers must read
  the canonical shape above; there is no fallback.

## The core loop

Run the toy demo first — it exercises the whole pipeline with no GPU and no
downloads:

```bash
interp-lab demo --out reports/demo
```

Then the real loop:

```bash
# 1. Inspect: rank and explain features for a criterion.
interp-lab inspect --model toy/model-a \
  --criterion "the model is aware it is being evaluated" \
  --backend toy --out reports/eval-awareness

# 2. Read reports/eval-awareness/report.json. Check each card's association,
#    causal_effect, stability, and claim grades; then follow the embedded
#    agent_next_actions (they are exact, parseable command templates).

# 3. Intervene: ALWAYS plan with --dry-run first, then run for real.
interp-lab intervene --model <model> --criterion "<criterion>" \
  --dataset <causal-prompts.jsonl> --report reports/eval-awareness/report.json \
  --top-k 8 --mode suppress --target-token auto \
  --out <interventions.jsonl> --plan-out <intervention-plan.json> --dry-run --json
# ...inspect the plan, drop --dry-run to execute.

# 4. Re-inspect with intervention evidence attached.
interp-lab inspect --model <model> --criterion "<criterion>" \
  --backend records --records <activation-records.jsonl> \
  --interventions <interventions.jsonl> --out reports/causal

# 5. Compare across models, then validate the candidate matches.
interp-lab match --left reports/model-a/report.json --right reports/model-b/report.json \
  --out reports/matches.json
interp-lab validate-matches --matches reports/matches.json --out reports/match-validation.json

# 6. Export and validate an attribution graph from the causal report.
interp-lab export-attribution-graph --report reports/causal/report.json --out reports/graph.json
interp-lab validate-attribution-graph --graph reports/graph.json --records <path-patches.jsonl> \
  --out reports/graph-validation.json
```

Useful side loops:

```bash
interp-lab compare-runs --left a/report.json --right b/report.json --out diff.json
                                  # rank/score drift between two runs
interp-lab search-features --report reports/**/report.json --query "tool calls"
interp-lab check-explanation-consistency --report a.json --report b.json
interp-lab profile-env --json     # compute/storage routing before big runs
```

Reproducible runs: `interp-lab init-run` writes an editable JSON/TOML/YAML
config, `interp-lab run --config run.json` executes it and writes a
`manifest.json` recording tool version, platform, and input hashes. Prefer this
for anything you may need to rerun or hand off.

## Investigation loop, fully driven

Since 3.0.0 the loop above has a brain, a memory, and a trust anchor — you can
run an entire investigation without inventing a single command yourself.

**`plan-evidence` is the brain.** Point it at any report and it diagnoses each
card's evidence gaps (`no_causal_evidence`, `no_signed_effect`,
`sign_inconsistency`, `insufficient_power`, `no_controls`), computes the
intervention sample size that would close them (same Student-t machinery as
every reported CI), and ranks the cheapest grade-moving runs — each entry
carrying ready-to-run `intervene` next actions. When a gap is labeled
`effect_size_source: "association_prior"`, the power estimate is seeded by a
correlation, not a measurement. When it says `effect_likely_too_small`, stop:
more prompts will not rescue the claim.

```bash
interp-lab plan-evidence --report reports/run-1/report.json --out reports/run-1/plan.json
```

**Dossiers are the memory.** One JSON artifact per (model, criterion),
appended every round, atomically rewritten, identity-checked. The rollup
tracks each feature's grade transitions, score drift, sign flips, and
contradictions across runs — so round 5 knows what round 1 measured. An
intervention-measured effect replacing a correlational one is recorded as a
*provenance change*, never a contradiction.

```bash
interp-lab dossier-update --dossier reports/dossier.json --report reports/run-1/report.json \
  --match-validation reports/match-validation.json --note "after control interventions"
interp-lab dossier-show --dossier reports/dossier.json
```

**`calibrate` is the trust anchor.** It plants synthetic ground truth (truly
causal features, equally-correlated decoys, noise), runs the REAL pipeline
blind, and reports what the grades actually mean: discovery precision/recall,
decoy resistance, P(truly causal | evidence tier) with Wilson CIs, and a
verdict (`well_calibrated` / `overclaims_causality` /
`underpowered_or_misranked`). Cite its numbers when you report findings — and
remember its own caveat: synthetic worlds certify the grading machinery, not
behavior on messy real activations.

**`quant-diff` compares precision variants.** Baseline report vs quantized
variant of the same criterion: which intervention-validated features were
`preserved`, `degraded`, `lost`, or `emerged`. `degraded_validated` is the
list that matters — correlational-only pairs cannot land in it by
construction. See `docs/QUANT_DIFF.md` for the GGUF/PMRA walkthrough.

**Steering artifacts are the deliverable.** `export-steering` turns a
validated feature into a reusable steering-vector JSON — and it enforces the
provenance gate: cards without intervention-measured evidence are refused.
`--allow-unvalidated` exports anyway but stamps `provenance: "unvalidated"`
with a top-level warning, so the artifact can never masquerade as a validated
direction. Applying an artifact (`apply-steering`) generates text from a model
— that call is yours to make deliberately; it is CLI/API only, not an MCP tool.

**Old artifacts: `migrate-report`.** Pre-2.3 reports counted correlational
evidence on the causal axis; migration re-scores under current semantics,
re-ranks, and records exactly what changed under `metadata.migration`. Migrate
before diffing old runs against new ones.

**CPU-only labs: the GGUF bridge.** `export-gguf-records` (final layer, via
llama-cpp-python) and `convert-hidden-dump` (any layers, stdlib-only) emit the
standard activation-records JSONL, so llama.cpp models feed
`inspect --backend records` — and the whole loop above — without torch. See
`docs/GGUF_BRIDGE.md`.

The round, end to end: `inspect` → `plan-evidence` → run the suggested
`intervene --dry-run`, inspect the plan, execute → re-`inspect` with
`--interventions` → `dossier-update`. Repeat until validated, or until the
planner says the effect is too small to chase.

## What NOT to claim

- Do not present association-backed features as causal. A high `association`
  with no intervention evidence supports "correlates with", nothing stronger.
- Check the claim grade before asserting: only `validated` matches and paths
  have intervention-backed support; `needs_causal_evidence` and `plausible`
  are hypotheses.
- Check provenance before comparing effects: `signed_causal_effect` (measured)
  and `signed_association` (correlational) are not interchangeable, and two
  aligned correlations are still not causal support.
- Do not call opposite correlations a contradiction. `contradicted` requires
  intervention provenance on both sides; association-only opposite pairs grade
  `needs_causal_evidence` with reason
  `opposite_associations_lack_intervention_provenance` — the right response is
  to run the interventions, not to report a conflict.
- Do not hand anyone a steering artifact stamped `provenance: "unvalidated"`
  without saying so — the stamp exists because the direction was never causally
  tested.
- Importance scores are heuristic, evidence-weighted rankings — not
  probabilities.

## Pointers

- [`docs/COMMANDS.md`](docs/COMMANDS.md) — every CLI command and the JSONL data
  formats.
- [`docs/PYTHON_API.md`](docs/PYTHON_API.md) — the stable Python API.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fingerprints, adapters, and
  the evidence model.
