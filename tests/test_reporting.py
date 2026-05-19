from oracle_sae.reporting import render_inspection_markdown
from oracle_sae.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport


def test_inspection_markdown_includes_mechanism_sketch_without_boilerplate():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="code-oriented completions"),
        cards=[
            FeatureCard(
                feature_id="SAE:L24:F8",
                model="m",
                layer=24,
                label="code planning latent",
                explanation="This feature appears to represent something. Treat this as a hypothesis, then check intervention results.",
                importance=0.5,
                association=0.42,
                specificity=0.07,
                causal_effect=0.07,
                stability=0.95,
                examples=[
                    "p1: activation=3.0 | Write a Python function | token[2]='Python'",
                    "p2: activation=2.8 | Return JSON | token[1]='JSON'",
                ],
                source="trained-sae",
                fingerprint=_fingerprint(),
                causal_effects={
                    "signed_causal_effect": 0.073,
                    "strong_causal_score": 0.072,
                },
                metadata={
                    "interventions": {
                        "count": 2,
                        "mean_directed_effect": 0.073,
                        "mean_side_effect": 0.001,
                    }
                },
            )
        ],
    )

    markdown = render_inspection_markdown(report)

    assert "## Mechanism Sketch" in markdown
    assert "Metric notes: Association is activation/criterion correlation" in markdown
    assert "SAE:L24:F8 layer 24 (code planning latent) changes the behavior score by +0.073" in markdown
    assert "`Python` appears in high-activation examples" in markdown
    assert "Causal direction: promotes criterion (0.073)" in markdown
    assert "Activation readout: `code planning latent`" in markdown
    assert "Causal readout: steering or ablating this feature promoted the criterion" in markdown
    assert "Treat this as a hypothesis" not in markdown
    assert "weak signed association" not in markdown


def test_inspection_markdown_avoids_repeated_token_readout_for_generic_labels():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="code-oriented completions"),
        cards=[
            FeatureCard(
                feature_id="SAE:L24:F8",
                model="m",
                layer=24,
                label="trained SAE latent 8",
                explanation="",
                importance=0.5,
                association=0.42,
                specificity=0.07,
                causal_effect=0.07,
                stability=0.95,
                examples=[
                    "p1: activation=3.0 | A report about an election | token[2]='report'",
                    "p2: activation=2.8 | The election shifted | token[1]='election'",
                ],
                source="trained-sae",
                fingerprint=_fingerprint(),
                causal_effects={"signed_causal_effect": 0.073, "strong_causal_score": 0.072},
                metadata={"interventions": {"count": 2, "mean_directed_effect": 0.073}},
            )
        ],
    )

    markdown = render_inspection_markdown(report)

    assert "Activation readout: high activations concentrate on tokens such as `report`, `election`." in markdown
    assert "tokens: report, election` is represented by high activations" not in markdown


def _fingerprint():
    return FeatureFingerprint(
        feature_id="SAE:L24:F8",
        model="m",
        layer=24,
        text="code planning latent",
        text_vector=[],
        activation_signature=[],
        decoder_signature=[],
        causal_vector=[],
    )
