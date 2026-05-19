from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalGemmaWorkflow:
    model: str = "google/gemma-4-E2B-it"
    criterion: str = "code-oriented completions"
    workflow: str = "contrast"
    dataset_path: str = "/tmp/interp-lab/dataset.jsonl"
    work_dir: str = "/tmp/interp-lab/gemma4"
    layer: str = ""
    layers: str = "20,28,35"
    features_per_layer: int = 32
    top_k: int = 16
    group_top_k: int = 8
    target_token: str = "auto"
    strength_sweep: str = "-4,-2,2,4"
    max_length: int = 64
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    model_class: str = "gemma4-conditional"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ModalSaeWorkflow:
    model: str = "HuggingFaceTB/SmolLM3-3B"
    criterion: str = "code-oriented completions"
    dataset_path: str = "/tmp/interp-lab/dataset.jsonl"
    causal_dataset_path: str = "/tmp/interp-lab/causal-dataset.jsonl"
    work_dir: str = "/tmp/interp-lab/sae"
    layers: str = "12,24"
    preset: str = "production"
    latent_dim: int = 512
    epochs: int = 40
    batch_size: int = 128
    sparsity: str = ""
    l1: float = 0.0
    lr: float = 0.0
    max_records: int = 512
    max_length: int = 96
    top_k: int = 16
    top_k_features: int = 32
    report_top_k: int = 8
    causal_top_k: int = 8
    causal_strength_sweep: str = "-2,2,4"
    target_token: str = "auto"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    model_class: str = "auto-causal-lm"
    trust_remote_code: bool = False


def build_modal_gemma_commands(config: ModalGemmaWorkflow) -> list[list[str]]:
    if config.workflow not in {"contrast", "hidden"}:
        raise ValueError("workflow must be 'contrast' or 'hidden'")
    if config.workflow == "contrast":
        return _contrast_commands(config)
    return _hidden_commands(config)


def build_modal_sae_commands(config: ModalSaeWorkflow) -> list[list[str]]:
    commands: list[list[str]] = []
    for layer in _parse_layers(config.layers):
        layer_dir = f"{config.work_dir}/layer-{layer}"
        artifact = f"{layer_dir}/sae.json"
        records = f"{layer_dir}/records.jsonl"
        interventions = f"{layer_dir}/interventions.jsonl"
        report = f"{layer_dir}/report"
        train_command = [
            "python",
            "-m",
            "oracle_sae",
            "train-sae",
            "--preset",
            config.preset,
            "--hf-model",
            config.model,
            "--dataset",
            config.dataset_path,
            "--causal-dataset",
            config.causal_dataset_path,
            "--layer",
            str(layer),
            "--latent-dim",
            str(config.latent_dim),
            "--epochs",
            str(config.epochs),
            "--batch-size",
            str(config.batch_size),
            "--max-records",
            str(config.max_records),
            "--max-length",
            str(config.max_length),
            "--top-k",
            str(config.top_k),
            "--top-k-features",
            str(config.top_k_features),
            "--out",
            artifact,
            "--records-out",
            records,
            "--causal-out",
            interventions,
            "--criterion",
            config.criterion,
            "--causal-top-k",
            str(config.causal_top_k),
            f"--causal-strength-sweep={config.causal_strength_sweep}",
            "--target-token",
            config.target_token,
            "--device",
            config.device,
            "--torch-dtype",
            config.torch_dtype,
            "--model-class",
            config.model_class,
        ]
        if config.trust_remote_code:
            train_command.append("--trust-remote-code")
        if config.sparsity:
            train_command.extend(["--sparsity", config.sparsity])
        if config.l1 > 0:
            train_command.extend(["--l1", str(config.l1)])
        if config.lr > 0:
            train_command.extend(["--lr", str(config.lr)])
        commands.extend(
            [
                train_command,
                [
                    "python",
                    "-m",
                    "oracle_sae",
                    "inspect",
                    "--model",
                    config.model,
                    "--criterion",
                    config.criterion,
                    "--backend",
                    "records",
                    "--records",
                    records,
                    "--interventions",
                    interventions,
                    "--require-interventions",
                    "--top-k",
                    str(config.report_top_k),
                    "--out",
                    report,
                ],
            ]
        )
    return commands


def expected_modal_gemma_outputs(workflow: str) -> list[str]:
    if workflow == "contrast":
        return [
            "contrast-records.jsonl",
            "contrast-interventions.jsonl",
            "contrast-report/report.json",
            "contrast-report/report.md",
        ]
    if workflow == "hidden":
        return [
            "hidden-records.jsonl",
            "hidden-associated/report.json",
            "hidden-associated/report.md",
            "hidden-interventions.jsonl",
            "hidden-group-records.jsonl",
            "hidden-causal/report.json",
            "hidden-causal/report.md",
        ]
    raise ValueError("workflow must be 'contrast' or 'hidden'")


def expected_modal_sae_outputs(config: ModalSaeWorkflow) -> list[str]:
    outputs: list[str] = []
    for layer in _parse_layers(config.layers):
        prefix = f"layer-{layer}"
        outputs.extend(
            [
                f"{prefix}/sae.json",
                f"{prefix}/records.jsonl",
                f"{prefix}/interventions.jsonl",
                f"{prefix}/report/report.json",
                f"{prefix}/report/report.md",
            ]
        )
    return outputs


def summarize_modal_result(result: dict) -> dict:
    summary = {key: value for key, value in result.items() if key != "files"}
    files = result.get("files")
    if isinstance(files, dict):
        summary["file_manifest"] = {
            str(path): {"bytes": len(str(content).encode("utf-8"))}
            for path, content in sorted(files.items())
        }
        summary["file_count"] = len(files)
    return summary


def _parse_layers(value: str) -> list[int]:
    layers = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if stripped:
            layers.append(int(stripped))
    if not layers:
        raise ValueError("layers must contain at least one layer")
    return layers


def _contrast_commands(config: ModalGemmaWorkflow) -> list[list[str]]:
    records = f"{config.work_dir}/contrast-records.jsonl"
    interventions = f"{config.work_dir}/contrast-interventions.jsonl"
    report = f"{config.work_dir}/contrast-report"
    export_command = [
        "python",
        "-m",
        "oracle_sae",
        "export-hf-contrast",
        "--model",
        config.model,
        "--dataset",
        config.dataset_path,
        "--records-out",
        records,
        "--interventions-out",
        interventions,
        "--criterion",
        config.criterion,
        "--pool",
        "last",
        "--device",
        config.device,
        "--max-length",
        str(config.max_length),
        "--target-token",
        config.target_token,
        f"--strength-sweep={config.strength_sweep}",
        "--torch-dtype",
        config.torch_dtype,
        "--model-class",
        config.model_class,
    ]
    if config.layer:
        export_command.extend(["--layer", config.layer])
    if config.trust_remote_code:
        export_command.append("--trust-remote-code")
    return [
        export_command,
        [
            "python",
            "-m",
            "oracle_sae",
            "inspect",
            "--model",
            config.model,
            "--criterion",
            config.criterion,
            "--backend",
            "records",
            "--records",
            records,
            "--interventions",
            interventions,
            "--require-interventions",
            "--top-k",
            str(config.top_k),
            "--out",
            report,
        ],
    ]


def _hidden_commands(config: ModalGemmaWorkflow) -> list[list[str]]:
    records = f"{config.work_dir}/hidden-records.jsonl"
    associated = f"{config.work_dir}/hidden-associated"
    interventions = f"{config.work_dir}/hidden-interventions.jsonl"
    group_records = f"{config.work_dir}/hidden-group-records.jsonl"
    causal = f"{config.work_dir}/hidden-causal"
    export_records_command = [
        "python",
        "-m",
        "oracle_sae",
        "export-hf-records",
        "--model",
        config.model,
        "--dataset",
        config.dataset_path,
        "--out",
        records,
        "--layers",
        config.layers,
        "--features-per-layer",
        str(config.features_per_layer),
        "--pool",
        "last",
        "--device",
        config.device,
        "--max-length",
        str(config.max_length),
        "--torch-dtype",
        config.torch_dtype,
        "--model-class",
        config.model_class,
    ]
    export_interventions_command = [
        "python",
        "-m",
        "oracle_sae",
        "export-hf-interventions",
        "--model",
        config.model,
        "--report",
        f"{associated}/report.json",
        "--dataset",
        config.dataset_path,
        "--out",
        interventions,
        "--criterion",
        config.criterion,
        "--top-k",
        str(config.top_k),
        "--group-top-k",
        str(config.group_top_k),
        "--records",
        records,
        "--append-group-records",
        group_records,
        "--target-token",
        config.target_token,
        "--device",
        config.device,
        "--max-length",
        str(config.max_length),
        "--torch-dtype",
        config.torch_dtype,
        "--model-class",
        config.model_class,
    ]
    if config.trust_remote_code:
        export_records_command.append("--trust-remote-code")
        export_interventions_command.append("--trust-remote-code")
    return [
        export_records_command,
        [
            "python",
            "-m",
            "oracle_sae",
            "inspect",
            "--model",
            config.model,
            "--criterion",
            config.criterion,
            "--backend",
            "records",
            "--records",
            records,
            "--top-k",
            str(config.top_k),
            "--out",
            associated,
        ],
        export_interventions_command,
        [
            "python",
            "-m",
            "oracle_sae",
            "inspect",
            "--model",
            config.model,
            "--criterion",
            config.criterion,
            "--backend",
            "records",
            "--records",
            group_records,
            "--interventions",
            interventions,
            "--require-interventions",
            "--top-k",
            str(config.top_k),
            "--out",
            causal,
        ],
    ]
