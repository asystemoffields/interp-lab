---
name: interp-lab-investigate
description: Run a full interp-lab evidence investigation for a natural-language criterion against a model — inspect, plan evidence, intervene, validate, track in a dossier. Use when the user asks to investigate, find, or causally test what a model represents/tracks/encodes for some criterion or behavior (e.g. "does this model represent evaluation awareness?", "find the features for X and test them").
---

# interp-lab investigation loop

You are running an evidence investigation: find the model features that track
a criterion, then *prove or kill* the causal claim. Correlation is a
hypothesis; only an intervention is evidence. Never skip the planning steps.

Discovery first: `interp-lab capabilities --json` returns every command spec,
the Python API contract, and artifact schemas in one payload. `interp-lab
doctor --json` tells you which optional extras (`[hf]`, `[gguf]`, ...) are
installed. Read `AGENTS.md` in the repo root for the evidence rules.

## 0. Set up the run

Pick a working directory, e.g. `reports/<slug>/`. You need:
- a model id (HF name, `toy/*` for dry runs, or activation records from any runtime),
- the criterion as one plain-language sentence — keep it byte-identical across every round,
- evidence: `--backend records --records <records.jsonl>` for real models
  (`export-hf-records`, `export-gguf-records`, or `convert-hidden-dump` produce these),
  or `--backend toy` to rehearse the loop with zero downloads.

## 1. Inspect

```bash
interp-lab inspect --model <model> --criterion "<criterion>" \
  --backend records --records <records.jsonl> --out reports/<slug>/round-1 --json
```

Read `report.json`. Per card: `association` (correlational), `causal_effect`
(zero until interventions are attached), and `causal_effects` provenance keys —
`signed_causal_effect` is measured, `signed_association` is a correlational
proxy. At this point you have hypotheses, nothing more.

## 2. Plan evidence — the decision point

```bash
interp-lab plan-evidence --report reports/<slug>/round-1/report.json \
  --out reports/<slug>/round-1/plan.json --json
```

The plan names each card's gaps (`no_causal_evidence`, `no_signed_effect`,
`sign_inconsistency`, `insufficient_power`, `no_controls`), the recommended
intervention count, and ready-to-run next actions. Decide per feature:
- gaps present and affordable → run the suggested intervention (step 3);
- `effect_likely_too_small` → STOP chasing this feature; record the negative in the dossier;
- `effect_size_source: "association_prior"` → the power estimate is seeded by a
  correlation; treat the sample size as a floor, not a promise;
- no gaps → the card is already at target grade; move on.

## 3. Intervene — always dry-run first

Take the plan's `agent_next_actions` entry (prefer `argv`; substitute
`<angle-bracket>` placeholders). It looks like:

```bash
interp-lab intervene --model <model> --criterion "<criterion>" \
  --dataset <causal-prompts.jsonl> --report reports/<slug>/round-1/report.json \
  --feature <feature-id> --mode suppress --target-token auto \
  --out reports/<slug>/interventions.jsonl --plan-out reports/<slug>/intervene-plan.json \
  --dry-run --json
```

Inspect the plan (features, expected forward passes, advisories). Only then
drop `--dry-run` to spend model time. Requires `pip install "interp-lab[hf]"`;
without it the command fails cleanly with that install hint.

## 4. Re-inspect with causal evidence, then update the dossier

```bash
interp-lab inspect --model <model> --criterion "<criterion>" \
  --backend records --records <records.jsonl> \
  --interventions reports/<slug>/interventions.jsonl --out reports/<slug>/round-2

interp-lab dossier-update --dossier reports/<slug>/dossier.json \
  --report reports/<slug>/round-2/report.json --note "round 2: suppression on <feature-id>"
interp-lab dossier-show --dossier reports/<slug>/dossier.json
```

Update the dossier EVERY round — it is the investigation's memory: grade
transitions, sign flips, provenance changes, contradictions across runs.

## 5. Loop or stop

Go back to step 2 with the newest report. Stop when:
- the features you care about reach `validated` (intervention-backed) — then
  optionally `interp-lab export-steering --report ... --feature <id> --out steer.json`
  as the deliverable (it refuses unvalidated cards; that refusal is correct), or
- `plan-evidence` flags `effect_likely_too_small` for everything left — report
  the honest negative.

## Reading grades and provenance

- Grades: `validated` > `plausible` / `needs_causal_evidence` (hypotheses) > `weak`;
  `contradicted` means intervention-vs-intervention sign conflict.
- Opposite correlations are NOT a contradiction: they grade
  `needs_causal_evidence` with reason `opposite_associations_lack_intervention_provenance`.
  The fix is to run interventions, not to report a conflict.
- Cite `interp-lab calibrate --out reports/calibration.json` when asked how much
  to trust the grades: it scores the pipeline against planted ground truth
  (decoy resistance, P(truly causal | tier)).

## What NOT to claim

- Never present association-backed features as causal — "correlates with" is the ceiling.
- Never compare `signed_causal_effect` against `signed_association`.
- Importance is a heuristic ranking, not a probability.
- Never hand over a `provenance: "unvalidated"` steering artifact without saying so.
