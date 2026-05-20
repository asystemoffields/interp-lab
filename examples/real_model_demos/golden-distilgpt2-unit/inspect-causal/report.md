# interp-lab Report: distilgpt2

Criterion: the next token should be a physical measurement unit

Metric notes: Association is activation/criterion correlation in the evidence records. Effect is mean causal change from interventions. Specificity subtracts measured side effects. Strong causal score is the specificity-adjusted causal signal.

Evidence summary: 8 activation rows; 32 candidate features; criterion score mean=0.500, range=[0.000, 1.000]; positive rows=4, non-positive rows=4.

Report scope: ranked 3 kept feature(s) from 32 candidate feature(s).

## Mechanism Sketch

Causal candidates:
- SAE:L6:F10 layer 6 (trained SAE latent 10) changes the behavior score by +0.226 on average (strong causal score 0.224).

Evidence gaps:
- Feature-level causal tests are present. Use export-attribution-graph, optionally with repeated --report arguments, to inspect candidate feature groups and cross-layer coactivation paths.

## Agent Next Actions

- Plan causal tests for the top report features: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --report '<report.json>' --top-k 3 --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: inspection report JSON, scored causal prompt JSONL.
- Rebuild the report with intervention evidence: `interp-lab inspect --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --backend records --records '<activation-records.jsonl>' --interventions '<interventions.jsonl>' --out '<causal-report-dir>' --html-out '<causal-report-dir>/report.html'`. Requires: activation records, intervention records.
- Build a graph from the causal report: `interp-lab export-attribution-graph --report '<causal-report-dir>/report.json' --out '<graph.json>' --markdown-out '<graph.md>' --html-out '<graph.html>'`. Requires: causal report JSON.

## Top Features

### 1. SAE:L6:F10 (layer 6)

Label: trained SAE latent 10

Importance: 0.437

Association: 0.765 | Causal effect: 0.226 | Specificity: 0.224 | Stability: 1.000

Evidence: causal intervention records

Causal direction: promotes criterion (0.226)

Strong causal score: 0.224

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Interventions: n=1, mean directed effect=0.226, mean side effect=0.001, 95% CI=[0.226, 0.226]
Behavior score: target_token_probability_mass baseline mean=0.890, range=[0.890, 0.890], target tokens=16 (auto), sample=` feet`, ` ft`, ` meters`, ` metres`
Behavior note: Baseline score is already high; use a narrower target-token set, harder positive prompts, or a more specific behavior scorer.

Intervention examples:
- unit-1: suppress baseline=0.890, intervention=0.664, directed_effect=0.226

Activation summary: trained SAE latent 10. Representative high-activation contexts include unit-4: activation=3.139, criterion_score=1.000 | The room is 14; unit-6: activation=2.445, criterion_score=1.000 | The recipe calls for 250.
Causal readout: steering or ablating this feature promoted the criterion with strong causal score 0.224.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F10 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F10 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-4: activation=3.139, criterion_score=1.000 | The room is 14
- unit-6: activation=2.445, criterion_score=1.000 | The recipe calls for 250
- unit-3: activation=2.418, criterion_score=1.000 | The trail climbs 600

### 2. SAE:L6:F30 (layer 6)

Label: trained SAE latent 30

Importance: 0.351

Association: 0.962 | Causal effect: 0.016 | Specificity: 0.016 | Stability: 1.000

Evidence: causal intervention records

Causal direction: near zero (-0.016)

Strong causal score: 0.016

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Interventions: n=1, mean directed effect=-0.016, mean side effect=0.000, 95% CI=[-0.016, -0.016]
Behavior score: target_token_probability_mass baseline mean=0.890, range=[0.890, 0.890], target tokens=16 (auto), sample=` feet`, ` ft`, ` meters`, ` metres`
Behavior note: Baseline score is already high; use a narrower target-token set, harder positive prompts, or a more specific behavior scorer.

Intervention examples:
- unit-1: suppress baseline=0.890, intervention=0.906, directed_effect=-0.016

Activation summary: trained SAE latent 30. Representative high-activation contexts include ordinary-prefix-4: activation=6.328, criterion_score=0.000 | A good onboarding message should; ordinary-prefix-1: activation=5.984, criterion_score=0.000 | Please write a friendly.
Causal readout: tested interventions produced a small or uncertain effect (strong causal score 0.016).

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F30 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F30 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- ordinary-prefix-4: activation=6.328, criterion_score=0.000 | A good onboarding message should
- ordinary-prefix-1: activation=5.984, criterion_score=0.000 | Please write a friendly
- ordinary-prefix-3: activation=4.661, criterion_score=0.000 | Summarize the feedback into

### 3. SAE:L6:F13 (layer 6)

Label: trained SAE latent 13

Importance: 0.346

Association: 0.924 | Causal effect: 0.023 | Specificity: 0.023 | Stability: 1.000

Evidence: causal intervention records

Causal direction: small directional effect (-0.023)

Strong causal score: 0.023

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Interventions: n=1, mean directed effect=-0.023, mean side effect=0.000, 95% CI=[-0.023, -0.023]
Behavior score: target_token_probability_mass baseline mean=0.890, range=[0.890, 0.890], target tokens=16 (auto), sample=` feet`, ` ft`, ` meters`, ` metres`
Behavior note: Baseline score is already high; use a narrower target-token set, harder positive prompts, or a more specific behavior scorer.

Intervention examples:
- unit-1: suppress baseline=0.890, intervention=0.913, directed_effect=-0.023

Activation summary: trained SAE latent 13. Representative high-activation contexts include ordinary-prefix-4: activation=7.192, criterion_score=0.000 | A good onboarding message should; ordinary-prefix-1: activation=6.169, criterion_score=0.000 | Please write a friendly.
Causal readout: tested interventions produced a small or uncertain effect (strong causal score 0.023).

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F13 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F13 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- ordinary-prefix-4: activation=7.192, criterion_score=0.000 | A good onboarding message should
- ordinary-prefix-1: activation=6.169, criterion_score=0.000 | Please write a friendly
- ordinary-prefix-5: activation=4.528, criterion_score=0.000 | The assistant should answer
