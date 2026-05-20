"""Run the golden real-model demo on Modal (CPU) and archive the artifacts.

This reproduces ``docs/GOLDEN_REAL_MODEL_DEMO.md`` end to end on a small open
model (DistilGPT-2) so the run is cheap and CPU-only, while exercising the real
loop: train an SAE from Hugging Face activations, inspect learned latents for a
criterion, suppress top features, re-inspect with causal evidence, and export an
attribution graph. Semantic text embeddings (MiniLM) are enabled so the
fingerprints in the archived report demonstrate the ``[embeddings]`` path.

Artifacts are written to a Modal Volume (so a ``--detach`` run survives a client
disconnect) and also returned to the local entrypoint, which writes them under
``examples/real_model_demos/golden-distilgpt2-unit/``.

Usage (detached, background-friendly):

    modal run --detach examples/modal_golden_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "interp-lab-golden-demo"
MODEL = "distilgpt2"
CRITERION = "the next token should be a physical measurement unit"
WORK = "/app/out/distilgpt2-unit"
RESULTS_DIR = "/results/golden-distilgpt2-unit"
LOCAL_DATASET = ROOT / "examples" / "hf_prompts_unit_prediction.jsonl"
LOCAL_OUT = ROOT / "examples" / "real_model_demos" / "golden-distilgpt2-unit"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("interp-lab-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("interp-lab-golden-demo", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=5.5.0",
        "accelerate>=1.0",
        "safetensors>=0.4",
        "huggingface-hub>=1.0",
        "sentence-transformers>=3.0",
        "PyYAML>=6.0",
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "INTERP_LAB_TEXT_EMBEDDER": "minilm",
        }
    )
    .workdir("/app")
    .add_local_dir(ROOT / "src", remote_path="/app/src")
)


def _golden_commands() -> list[list[str]]:
    cli = [sys.executable, "-m", "interp_lab"]
    records = f"{WORK}/sae/records.jsonl"
    return [
        cli + [
            "prepare-sae-prompts",
            "--dataset", f"{WORK}/prompts.jsonl",
            "--out-dir", f"{WORK}/prompt-pack",
            "--latent-dim", "32",
            "--max-length", "64",
        ],
        cli + [
            "train-sae",
            "--preset", "minimal",
            "--hf-model", MODEL,
            "--dataset", f"{WORK}/prompt-pack/train.jsonl",
            "--layer", "6",
            "--latent-dim", "32",
            "--epochs", "20",
            "--batch-size", "6",
            "--lr", "0.003",
            "--l1", "0.001",
            "--out", f"{WORK}/sae/sae.json",
            "--records-out", records,
            "--max-length", "64",
        ],
        cli + [
            "inspect",
            "--model", MODEL,
            "--criterion", CRITERION,
            "--backend", "records",
            "--records", records,
            "--out", f"{WORK}/inspect",
            "--html-out", f"{WORK}/inspect/report.html",
        ],
        cli + [
            "intervene",
            "--model", MODEL,
            "--dataset", f"{WORK}/prompt-pack/causal.jsonl",
            "--criterion", CRITERION,
            "--report", f"{WORK}/inspect/report.json",
            "--records", records,
            "--top-k", "3",
            "--sae", f"{WORK}/sae/sae.json",
            "--mode", "suppress",
            "--strength-sweep", "1,3,10",
            "--target-token", "auto",
            "--max-length", "64",
            "--out", f"{WORK}/interventions.jsonl",
            "--plan-out", f"{WORK}/intervention-plan.json",
        ],
        cli + [
            "inspect",
            "--model", MODEL,
            "--criterion", CRITERION,
            "--backend", "records",
            "--records", records,
            "--interventions", f"{WORK}/interventions.jsonl",
            "--require-interventions",
            "--out", f"{WORK}/inspect-causal",
            "--html-out", f"{WORK}/inspect-causal/report.html",
        ],
        cli + [
            "export-attribution-graph",
            "--report", f"{WORK}/inspect-causal/report.json",
            "--out", f"{WORK}/graph.json",
            "--markdown-out", f"{WORK}/graph.md",
            "--html-out", f"{WORK}/graph.html",
        ],
        cli + [
            "summarize-attribution-graph",
            "--graph", f"{WORK}/graph.json",
            "--out", f"{WORK}/graph-summary.json",
        ],
    ]


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/results": results},
)
def run_golden(dataset_text: str) -> dict:
    work = Path(WORK)
    work.mkdir(parents=True, exist_ok=True)
    (work / "prompts.jsonl").write_text(dataset_text, encoding="utf-8")

    logs: list[dict] = []
    for command in _golden_commands():
        completed = subprocess.run(
            command, cwd="/app", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        logs.append({"command": command, "returncode": completed.returncode, "output": completed.stdout})
        if completed.returncode != 0:
            return {"ok": False, "logs": logs, "files": {}}

    # Collect every text artifact under the work dir, persist to the results
    # Volume (survives detached disconnects), and return it to the caller.
    files: dict[str, str] = {}
    results_root = Path(RESULTS_DIR)
    for path in sorted(work.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(work))
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files[relative] = content
        dest = results_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    results.commit()
    hf_cache.commit()
    return {"ok": True, "logs": logs, "files": files}


@app.local_entrypoint()
def main() -> None:
    dataset_text = LOCAL_DATASET.read_text(encoding="utf-8")
    result = run_golden.remote(dataset_text)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    for relative, content in result.get("files", {}).items():
        dest = LOCAL_OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    (LOCAL_OUT / "modal-run.json").write_text(
        json.dumps(
            {"ok": result.get("ok"), "logs": result.get("logs", []), "results_volume": "interp-lab-golden-demo"},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not result.get("ok"):
        raise SystemExit(f"Golden demo failed on Modal; see {LOCAL_OUT / 'modal-run.json'}")
    print(f"Wrote golden demo artifacts to {LOCAL_OUT}")
