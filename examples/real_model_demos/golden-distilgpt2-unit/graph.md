# Attribution Graph

Model: `distilgpt2`
Criterion: the next token should be a physical measurement unit
Nodes: `6`
Edges: `11`

## Strong Causal Features

| Feature | Label | Layer | Role | Signed | Strong |
| --- | --- | ---: | --- | ---: | ---: |
| `SAE:L6:F10` | SAE:L6:F10 | 6 | criterion_promoter | 0.2258 | 0.2244 |

## Candidate Paths

No candidate paths are currently present.

## Candidate Feature Groups

| Group | Label | Members | Mean Strong |
| --- | --- | ---: | ---: |
| `supernode:layer-6:criterion_promoter` | layer 6 criterion promoter: SAE:L6:F10 | 1 | 0.2244 |
| `supernode:layer-6:associated_detector` | layer 6 associated detector: SAE:L6:F30 | 2 | 0.0193 |

## Validation Plan

- Sweep steering strengths for SAE:L6:F10; it currently promotes the criterion with strong causal score 0.224.
