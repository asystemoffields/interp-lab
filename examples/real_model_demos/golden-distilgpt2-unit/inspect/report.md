# interp-lab Report: distilgpt2

Criterion: the next token should be a physical measurement unit

Metric notes: Association is activation/criterion correlation in the evidence records. Effect is mean causal change from interventions. Specificity subtracts measured side effects. Strong causal score is the specificity-adjusted causal signal.

Evidence summary: 8 activation rows; 32 candidate features; criterion score mean=0.500, range=[0.000, 1.000]; positive rows=4, non-positive rows=4.

Report scope: ranked 32 kept feature(s) from 32 candidate feature(s).

## Mechanism Sketch

Causal candidates: no tested feature crossed the current strong-effect threshold.

Evidence gaps:
- No intervention records were attached; causal claims are untested.
- No feature currently meets the strong-effect threshold; broaden prompts, test more layers, or use graph attribution.

## Agent Next Actions

- Plan causal tests for the top report features: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --report '<report.json>' --top-k 8 --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: inspection report JSON, scored causal prompt JSONL.
- Rebuild the report with intervention evidence: `interp-lab inspect --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --backend records --records '<activation-records.jsonl>' --interventions '<interventions.jsonl>' --out '<causal-report-dir>' --html-out '<causal-report-dir>/report.html'`. Requires: activation records, intervention records.
- Build a graph from the causal report: `interp-lab export-attribution-graph --report '<causal-report-dir>/report.json' --out '<graph.json>' --markdown-out '<graph.md>' --html-out '<graph.html>'`. Requires: causal report JSON.

## Top Features

### 1. SAE:L6:F30 (layer 6)

Label: trained SAE latent 30

Importance: 0.944

Association: 0.962 | Criterion score: 0.962 | Specificity: 0.818 | Stability: 1.000

Activation association: suppresses criterion (-0.962)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 30. Representative high-activation contexts include ordinary-prefix-4: activation=6.328, criterion_score=0.000 | A good onboarding message should; ordinary-prefix-1: activation=5.984, criterion_score=0.000 | Please write a friendly.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F30 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F30 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- ordinary-prefix-4: activation=6.328, criterion_score=0.000 | A good onboarding message should
- ordinary-prefix-1: activation=5.984, criterion_score=0.000 | Please write a friendly
- ordinary-prefix-3: activation=4.661, criterion_score=0.000 | Summarize the feedback into

### 2. SAE:L6:F13 (layer 6)

Label: trained SAE latent 13

Importance: 0.903

Association: 0.924 | Criterion score: 0.924 | Specificity: 0.732 | Stability: 1.000

Activation association: suppresses criterion (-0.924)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 13. Representative high-activation contexts include ordinary-prefix-4: activation=7.192, criterion_score=0.000 | A good onboarding message should; ordinary-prefix-1: activation=6.169, criterion_score=0.000 | Please write a friendly.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F13 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F13 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- ordinary-prefix-4: activation=7.192, criterion_score=0.000 | A good onboarding message should
- ordinary-prefix-1: activation=6.169, criterion_score=0.000 | Please write a friendly
- ordinary-prefix-5: activation=4.528, criterion_score=0.000 | The assistant should answer

### 3. SAE:L6:F10 (layer 6)

Label: trained SAE latent 10

Importance: 0.769

Association: 0.765 | Criterion score: 0.765 | Specificity: 0.637 | Stability: 1.000

Activation association: promotes criterion (0.765)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 10. Representative high-activation contexts include unit-4: activation=3.139, criterion_score=1.000 | The room is 14; unit-6: activation=2.445, criterion_score=1.000 | The recipe calls for 250.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F10 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F10 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-4: activation=3.139, criterion_score=1.000 | The room is 14
- unit-6: activation=2.445, criterion_score=1.000 | The recipe calls for 250
- unit-3: activation=2.418, criterion_score=1.000 | The trail climbs 600

### 4. SAE:L6:F16 (layer 6)

Label: trained SAE latent 16

Importance: 0.729

Association: 0.731 | Criterion score: 0.731 | Specificity: 0.540 | Stability: 1.000

Activation association: promotes criterion (0.731)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 16. Representative high-activation contexts include unit-3: activation=8.210, criterion_score=1.000 | The trail climbs 600; unit-4: activation=5.021, criterion_score=1.000 | The room is 14.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F16 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F16 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-3: activation=8.210, criterion_score=1.000 | The trail climbs 600
- unit-4: activation=5.021, criterion_score=1.000 | The room is 14
- unit-6: activation=4.498, criterion_score=1.000 | The recipe calls for 250

### 5. SAE:L6:F26 (layer 6)

Label: trained SAE latent 26

Importance: 0.705

Association: 0.710 | Criterion score: 0.710 | Specificity: 0.483 | Stability: 1.000

Activation association: promotes criterion (0.710)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 26. Representative high-activation contexts include unit-2: activation=9.252, criterion_score=1.000 | The package weighs 3.2; unit-6: activation=5.267, criterion_score=1.000 | The recipe calls for 250.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F26 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F26 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-2: activation=9.252, criterion_score=1.000 | The package weighs 3.2
- unit-6: activation=5.267, criterion_score=1.000 | The recipe calls for 250
- unit-4: activation=1.751, criterion_score=1.000 | The room is 14

### 6. SAE:L6:F9 (layer 6)

Label: trained SAE latent 9

Importance: 0.692

Association: 0.689 | Criterion score: 0.689 | Specificity: 0.505 | Stability: 1.000

Activation association: promotes criterion (0.689)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 9. Representative high-activation contexts include unit-6: activation=8.127, criterion_score=1.000 | The recipe calls for 250; unit-3: activation=5.627, criterion_score=1.000 | The trail climbs 600.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F9 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F9 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-6: activation=8.127, criterion_score=1.000 | The recipe calls for 250
- unit-3: activation=5.627, criterion_score=1.000 | The trail climbs 600
- unit-4: activation=2.678, criterion_score=1.000 | The room is 14

### 7. SAE:L6:F0 (layer 6)

Label: trained SAE latent 0

Importance: 0.653

Association: 0.651 | Criterion score: 0.651 | Specificity: 0.436 | Stability: 1.000

Activation association: suppresses criterion (-0.651)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 0. Representative high-activation contexts include ordinary-prefix-1: activation=7.703, criterion_score=0.000 | Please write a friendly; ordinary-prefix-5: activation=2.962, criterion_score=0.000 | The assistant should answer.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F0 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F0 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- ordinary-prefix-1: activation=7.703, criterion_score=0.000 | Please write a friendly
- ordinary-prefix-5: activation=2.962, criterion_score=0.000 | The assistant should answer
- ordinary-prefix-4: activation=2.761, criterion_score=0.000 | A good onboarding message should

### 8. SAE:L6:F11 (layer 6)

Label: trained SAE latent 11

Importance: 0.647

Association: 0.639 | Criterion score: 0.639 | Specificity: 0.453 | Stability: 1.000

Activation association: promotes criterion (0.639)

SAE training: rows=8, latents=32, active=1.000, dead=0, val MSE=1.187
SAE training note: Training rows are fewer than latents; collect more activations or reduce latent_dim.
SAE training note: Validation reconstruction is much worse than train; broaden training prompts and keep a separate held-out eval set.

Activation summary: trained SAE latent 11. Representative high-activation contexts include unit-6: activation=8.391, criterion_score=1.000 | The recipe calls for 250; unit-3: activation=5.123, criterion_score=1.000 | The trail climbs 600.

Next actions:

- Plan a suppression test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F11 --sae '<sae.json>' --mode suppress --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.
- Plan an amplification test for this SAE latent: `interp-lab intervene --model distilgpt2 --criterion 'the next token should be a physical measurement unit' --dataset '<causal-prompts.jsonl>' --feature SAE:L6:F11 --sae '<sae.json>' --mode amplify --target-token auto --out '<interventions.jsonl>' --plan-out '<intervention-plan.json>' --dry-run --json`. Requires: scored causal prompt JSONL, matching interp-lab SAE artifact.

Examples:
- unit-6: activation=8.391, criterion_score=1.000 | The recipe calls for 250
- unit-3: activation=5.123, criterion_score=1.000 | The trail climbs 600
- unit-4: activation=1.678, criterion_score=1.000 | The room is 14
