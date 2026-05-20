# Real-Model Demo Suite

The stable release bar includes three reproducible real-model walkthroughs. Each walkthrough has a machine-readable manifest in `examples/real_model_demos/` so agents can locate commands, expected artifacts, and interpretation checks without scraping prose.

## Manifests

- `examples/real_model_demos/distilgpt2-unit-golden.json`: the CPU-friendly golden path from prompt pack to trained SAE, causal intervention records, causal report, and attribution graph.
- `examples/real_model_demos/tiny-gpt2-sae-paths.json`: the tiny GPT-2 SAE path-patching sanity run, including source and target SAE latents, measured path records, controls, and graph fusion.
- `examples/real_model_demos/gemma4-tool-calls-modal.json`: the Gemma 4 successful tool-call workflow for Modal, from assay validation through hidden-feature discovery and behavior SAE training.

## How To Use Them

Open the manifest for the model family and workflow you want, then run the `commands[].argv` entries in order by prefixing each array with `interp-lab` unless the command is already an external launcher such as `modal`.

After a run, compare the produced files with `expected_artifacts`. The `why_it_matters` field explains what each artifact proves about the workflow. The `interpretation_notes` field is the first-pass review checklist for deciding whether a result is a publishable claim, a useful pilot signal, or only a plumbing check.

Agents and release maintainers can verify the whole suite with:

```bash
interp-lab demo-sweep --out reports/real-model-demo-sweep.json
```

That default mode validates manifests and checks whether expected artifacts exist. To execute commands as part of the sweep, add `--run`. External launchers such as Modal are skipped unless `--allow-external` is present:

```bash
interp-lab demo-sweep --run --allow-external --out reports/real-model-demo-sweep.json
```

Use `--demo <id>` to run one manifest at a time, and `--strict` when the sweep is acting as a release gate.

## Evidence Standard

For stable-release demos, a run should preserve:

- the run manifest, prompt-pack manifest, or remote-run summary;
- feature reports in JSON and HTML;
- causal or path records when the walkthrough claims causal evidence;
- graph JSON plus graph HTML or compact graph summary;
- notes on limitations, including weak intervention effects, control-path failures, saturated behavior scores, validation drift, or prompt coverage gaps.

The release gate validates that the manifest set exists, uses `interp-lab.real_model_demo.v1`, points at real docs, includes runnable command arrays, and names expected artifacts with interpretation notes.
