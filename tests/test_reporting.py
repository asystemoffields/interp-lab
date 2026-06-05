from interp_lab.reporting import render_inspection_html, render_inspection_markdown, write_inspection_html
from interp_lab.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport


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
                    "agent_next_actions": [
                        {
                            "id": "plan_sae_suppression",
                            "title": "Plan a suppression test",
                            "command": "interp-lab intervene --feature SAE:L24:F8 --dry-run --json",
                            "requires": ["causal prompts", "SAE artifact"],
                        }
                    ],
                    "interventions": {
                        "count": 2,
                        "mean_directed_effect": 0.073,
                        "mean_side_effect": 0.001,
                    }
                },
            )
        ],
        metadata={
            "agent_next_actions": [
                {
                    "id": "plan_top_feature_interventions",
                    "title": "Plan causal tests",
                    "command": "interp-lab intervene --report report.json --dry-run --json",
                    "requires": ["report JSON", "causal prompts"],
                }
            ]
        },
    )

    markdown = render_inspection_markdown(report)

    assert "## Mechanism Sketch" in markdown
    assert "Metric notes: Importance is the overall rank score" in markdown
    assert "Strong causal score is the specificity-adjusted causal signal" in markdown
    assert "SAE:L24:F8 layer 24 (code planning latent) changes the behavior score by +0.073" in markdown
    assert "`Python` appears in high-activation examples" in markdown
    assert "Causal direction: promotes criterion (0.073)" in markdown
    assert "Activation readout: `code planning latent`" in markdown
    assert "Causal readout: steering or ablating this feature promoted the criterion" in markdown
    assert "## Agent Next Actions" in markdown
    assert "interp-lab intervene --report report.json --dry-run --json" in markdown
    assert "Next actions:" in markdown
    assert "interp-lab intervene --feature SAE:L24:F8 --dry-run --json" in markdown
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


def test_inspection_markdown_distinguishes_attached_unmatched_interventions():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="unit prediction"),
        cards=[
            FeatureCard(
                feature_id="SAE:L5:F1",
                model="m",
                layer=5,
                label="trained SAE latent 1",
                explanation="",
                importance=0.5,
                association=0.62,
                specificity=0.4,
                causal_effect=0.62,
                stability=0.8,
                examples=["p1: activation=1.0 | The answer is measured in meters."],
                source="activation-records",
                fingerprint=_fingerprint(),
                causal_effects={"signed_association": 0.62},
                metadata={},
            )
        ],
        metadata={"interventions": {"record_count": 4, "feature_count": 2}},
    )

    markdown = render_inspection_markdown(report)

    assert "Criterion score: 0.620" in markdown
    assert "Intervention records were attached, but none matched the kept features." in markdown
    assert "No intervention records were attached" not in markdown


def test_empty_intervention_metadata_is_not_rendered_as_measured_causal_evidence():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="unit prediction"),
        cards=[
            FeatureCard(
                feature_id="SAE:L5:F1",
                model="m",
                layer=5,
                label="trained SAE latent 1",
                explanation="",
                importance=0.5,
                association=0.62,
                specificity=0.4,
                causal_effect=0.62,
                stability=0.8,
                examples=[],
                source="activation-records",
                fingerprint=_fingerprint(),
                causal_effects={"signed_association": 0.62},
                metadata={"interventions": {}},
            )
        ],
    )

    markdown = render_inspection_markdown(report)

    assert "Criterion score: 0.620" in markdown
    assert "Evidence: causal intervention records" not in markdown
    assert "Interventions: n=" not in markdown


def test_inspection_html_renders_searchable_feature_cards(tmp_path):
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="code-oriented completions"),
        cards=[
            FeatureCard(
                feature_id="SAE:L24:F8",
                model="m",
                layer=24,
                label="code planning latent",
                explanation="plans code",
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
                causal_effects={"signed_causal_effect": 0.073, "strong_causal_score": 0.072},
                metadata={"interventions": {"count": 2, "mean_directed_effect": 0.073}},
            )
        ],
        metadata={
            "evidence": {"record_count": 12, "feature_count": 1},
            "feature_count": 1,
            "kept_feature_count": 1,
            "agent_next_actions": [
                {
                    "id": "plan_top_feature_interventions",
                    "title": "Plan causal tests",
                    "command": "interp-lab intervene --report report.json --dry-run --json",
                    "requires": ["report JSON", "causal prompts"],
                }
            ],
        },
    )

    html = render_inspection_html(report)

    assert "Feature Report" in html
    assert "feature-search" in html
    assert "layer-filter" in html
    assert "source-filter" in html
    assert "evidence-filter" in html
    assert "SAE:L24:F8" in html
    assert "code planning latent" in html
    assert "strong causal 0.072" in html
    assert "Agent Next Actions" in html
    assert "interp-lab intervene --report report.json --dry-run --json" in html
    assert "copy-command" in html
    assert "copyCommand" in html
    assert "visibleRows" in html
    assert 'data-evidence="causal_records"' in html

    path = write_inspection_html(report, tmp_path / "report.html")
    assert path == tmp_path / "report.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


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
