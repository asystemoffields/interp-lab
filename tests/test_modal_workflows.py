from oracle_sae.modal_workflows import (
    ModalGemmaWorkflow,
    ModalSaeWorkflow,
    build_modal_gemma_commands,
    build_modal_sae_commands,
    expected_modal_gemma_outputs,
    expected_modal_sae_outputs,
    summarize_modal_result,
)


def test_modal_gemma_contrast_commands_are_agent_friendly():
    config = ModalGemmaWorkflow(workflow="contrast", layer="35")

    commands = build_modal_gemma_commands(config)

    assert len(commands) == 2
    assert commands[0][:4] == ["python", "-m", "oracle_sae", "export-hf-contrast"]
    assert "--model-class" in commands[0]
    assert "gemma4-conditional" in commands[0]
    assert "--layer" in commands[0]
    assert commands[1][:4] == ["python", "-m", "oracle_sae", "inspect"]
    assert "contrast-report/report.md" in expected_modal_gemma_outputs("contrast")


def test_modal_gemma_hidden_commands_include_causal_validation():
    config = ModalGemmaWorkflow(workflow="hidden", layers="28,35", top_k=8, group_top_k=4)

    commands = build_modal_gemma_commands(config)

    assert len(commands) == 4
    assert commands[0][:4] == ["python", "-m", "oracle_sae", "export-hf-records"]
    assert commands[2][:4] == ["python", "-m", "oracle_sae", "export-hf-interventions"]
    assert "--append-group-records" in commands[2]
    assert "--require-interventions" in commands[3]
    assert "hidden-causal/report.md" in expected_modal_gemma_outputs("hidden")


def test_modal_sae_commands_train_and_inspect_each_layer():
    config = ModalSaeWorkflow(
        layers="12,24",
        epochs=5,
        latent_dim=128,
        causal_top_k=3,
        sparsity="relu-l1",
        l1=0.0002,
    )

    commands = build_modal_sae_commands(config)

    assert len(commands) == 4
    assert commands[0][:4] == ["python", "-m", "oracle_sae", "train-sae"]
    assert "--preset" in commands[0]
    assert "production" in commands[0]
    assert "--causal-dataset" in commands[0]
    assert "--causal-out" in commands[0]
    assert "--target-token" in commands[0]
    assert "auto" in commands[0]
    assert "--top-k-features" in commands[0]
    assert commands[0][commands[0].index("--top-k-features") + 1] == "32"
    assert "--sparsity" in commands[0]
    assert "relu-l1" in commands[0]
    assert "--l1" in commands[0]
    assert commands[0][commands[0].index("--layer") + 1] == "12"
    assert commands[2][commands[2].index("--layer") + 1] == "24"
    assert commands[1][:4] == ["python", "-m", "oracle_sae", "inspect"]
    assert "--require-interventions" in commands[1]
    assert "layer-12/sae.json" in expected_modal_sae_outputs(config)
    assert "layer-24/report/report.md" in expected_modal_sae_outputs(config)


def test_modal_result_summary_omits_inline_file_contents():
    summary = summarize_modal_result(
        {
            "ok": True,
            "logs": [{"returncode": 0, "output": "done"}],
            "files": {
                "layer-12/sae.json": "{\"large\": true}",
                "layer-12/report/report.md": "# Report\n",
            },
        }
    )

    assert "files" not in summary
    assert summary["file_count"] == 2
    assert summary["file_manifest"]["layer-12/sae.json"]["bytes"] == 15
