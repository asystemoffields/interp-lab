# Scaling interp-lab

interp-lab is designed to separate model execution from interpretability evidence.

For small and medium runs, the built-in Hugging Face, TransformerLens, and NNsight exporters can collect activations directly. For frontier-scale or 1T+ models, run activation harvesting wherever the model already lives, then hand interp-lab sharded records, SAE metadata, intervention records, and manifests.

## Large-Model Contract

The stable boundary is the activation-record JSONL format:

```json
{
  "model": "lab/model",
  "prompt_id": "batch-0001/token-1042",
  "text": "optional prompt or token window",
  "criterion_score": 1.0,
  "features": [
    {
      "feature_id": "L42:F104921",
      "activation": 0.91,
      "label": "feature label when known",
      "layer": 42
    }
  ]
}
```

The records backend streams this format while keeping per-feature sufficient statistics, top examples, and compact activation signatures.

## 1T+ Execution Shape

Recommended shape for very large models:

1. Use a colocated activation harvester near the model runtime.
2. Capture a small number of hook points per run.
3. Write activation shards with deterministic names.
4. Train SAEs against streamed shards or import existing SAE artifacts.
5. Run causal validation as resumable batches.
6. Merge intervention JSONL shards.
7. Generate reports and attribution graphs from the merged evidence.
8. Publish artifacts to Hugging Face Hub or an internal object store.

The model runtime can be a hosted Goodfire/NNsight-style service, an internal inference cluster, or a local PyTorch stack. interp-lab only needs the records and metadata.

## Planning Storage

First, profile the environment that might run the job:

```bash
interp-lab profile-env \
  --path /data/interp-lab-artifacts \
  --out reports/env-profile.json \
  --json
```

The profile includes CPU count, RAM, disk space for the inspected path, local CUDA/MPS accelerators when visible, optional adapter packages, sanitized API-key presence checks, route options, alerts, and agent next actions.

Use:

```bash
interp-lab plan-scale \
  --model-params 1T \
  --tokens 1B \
  --d-model 16384 \
  --selected-layers 8 \
  --latent-dim 1M \
  --dtype bf16 \
  --from-env \
  --target-shard-size 64GB \
  --out reports/scale-plan.json
```

This estimates:

- dense activation storage;
- sparse feature-record storage after SAE or feature extraction;
- recommended profile and artifact format;
- shard count and per-shard size;
- SAE parameter storage and training memory floor;
- causal validation forward-pass count;
- risk flags, recommendations, and agent next actions.

Profiles:

- `local-cpu`: quick pilots and tiny model-backed runs.
- `single-gpu`: local GPU harvesting or SAE training.
- `cluster`: distributed harvesting and training.
- `remote-api`: model execution through an API or external service.
- `frontier-lab`: colocated activation harvesting for very large models.

Use `--profile auto` for default selection, or choose a profile explicitly. Use `--from-env` to profile the current environment during planning, and `--env-profile other-machine.json` when the target environment was profiled elsewhere. The environment advisory can suggest a starting route and still leave the researcher free to choose local CPU, GPU, cluster, remote API, or frontier-lab execution with `--profile`. Use `--json` for stdout JSON and `--out` for a reusable plan file.

## Environment Routing

The route advisory is intentionally structured for both humans and agents:

```json
{
  "schema_version": "interp-lab.env_profile.v1",
  "routing": {
    "suggested_profile": "single-gpu",
    "suggested_reason": "a local accelerator was detected",
    "options": [
      {"profile": "local-cpu", "status": "available"},
      {"profile": "single-gpu", "status": "available"},
      {"profile": "cluster", "status": "candidate"},
      {"profile": "remote-api", "status": "candidate"},
      {"profile": "frontier-lab", "status": "candidate"}
    ]
  }
}
```

This lets an agent alert a user with context like: "Looks like this is a CPU-only laptop with 32 GB RAM and 80 GB free disk; for this run, consider a remote API or cluster route." The saved JSON can then be passed directly into `plan-scale`.

## Robustness Rules

- Prefer sharded append-only records.
- Store run manifests with input hashes and tool versions.
- Keep causal validation outputs separate from activation records.
- Use controls in intervention records through `metadata.control_type`.
- Publish reports, graphs, and manifests alongside raw records.
- Treat adapter code as replaceable and evidence schemas as stable.
