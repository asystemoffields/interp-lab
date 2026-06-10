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

## Discovery: start here

```bash
interp-lab capabilities --json   # ONE payload: full command specs, Python API
                                 # contract, environment, and house conventions
interp-lab doctor --json         # environment / optional-extras health check
```

From Python, `interp_lab.public_api_contract()` returns the stable API surface
as data (exports, schema versions, signatures).

## MCP server

`interp-lab mcp` serves the core workflow as Model Context Protocol tools over
stdio: `capabilities`, `doctor`, `inspect`, `compare`, `validate_matches`,
`search_features`, `compare_runs`, `check_explanation_consistency`,
`attribution_graph`, `validate_attribution_graph`. Example client config:

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
  run. Some surfaces additionally carry deprecated legacy keys
  (`next_action`, `description`, per-result `agent_next_action*` flat keys)
  scheduled for removal; prefer the canonical keys.

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

## What NOT to claim

- Do not present association-backed features as causal. A high `association`
  with no intervention evidence supports "correlates with", nothing stronger.
- Check the claim grade before asserting: only `validated` matches and paths
  have intervention-backed support; `needs_causal_evidence` and `plausible`
  are hypotheses.
- Check provenance before comparing effects: `signed_causal_effect` (measured)
  and `signed_association` (correlational) are not interchangeable, and two
  aligned correlations are still not causal support.
- Importance scores are heuristic, evidence-weighted rankings — not
  probabilities.

## Pointers

- [`docs/COMMANDS.md`](docs/COMMANDS.md) — every CLI command and the JSONL data
  formats.
- [`docs/PYTHON_API.md`](docs/PYTHON_API.md) — the stable Python API.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fingerprints, adapters, and
  the evidence model.
