# EWALD Data Training

This directory is an isolated workspace for generating and training against
synthetic, labeled GIWAXS/WAXS datasets. It is intentionally separate from the
desktop application so large sweeps, cluster jobs, and future ML experiments can
evolve without making the GUI harder to install.

## Goals

- Build auditable catalogs of CIF, MCIF, POSCAR, and VASP structure inputs.
- Simulate many GIWAXS/WAXS images per structure under controlled detector,
  orientation, texture, and peak-width conditions.
- Export structured truth for every image: structure id, condition id, `(hkl)`,
  `qxy`, `qz`, Bragg intensity, rendered peak amplitude, and artifact seed.
- Add randomized detector and sample artifacts so models learn from images that
  resemble real beamline data instead of ideal simulations only.
- Train and validate a staged recognition pipeline: peak detection, peak
  indexing, structure-family retrieval, and physics-aware reranking.
- Run production sweeps on external SLURM clusters such as Alpine while keeping
  large image shards on scratch and syncing compact manifests/results back.

## Layout

```text
data_training/
  catalog/                 Structure catalog examples and notes
  configs/                 Local generation plans and artifact recipes
  src/ewald_data_training/  Isolated Python package for generation utilities
  scripts/                 Entrypoints that set PYTHONPATH for local/cluster use
  cluster/                 Alpine/SLURM staging and job templates
  deploy/                  One-command trigger folders for each pipeline stage
  reports/                 Design reports and mathematical ranking notes
  tests/                   Smoke tests for the isolated package
```

## Config-First Planning Scaffold

The top-level Python modules provide a small, standard-library-first planning
surface that can coexist with the package under `src/ewald_data_training/`:

- `catalog.py`: parses structure catalog records.
- `conditions.py`: expands simulation-condition sweeps.
- `artifacts.py`: parses detector artifact profiles.
- `manifest.py`: defines generated-image and peak-label manifest records.
- `runtime.py`: describes cluster scratch, dataset, and repo-link paths.
- `orchestration.py`: previews plans, manifests, and Slurm array scripts.

Preview the config-first plan:

```bash
python -m data_training.orchestration plan \
  --catalog data_training/configs/structure_catalog.yaml \
  --sweep data_training/configs/simulation_sweep.yaml \
  --artifacts data_training/configs/artifact_profiles.yaml
```

The configs reference EWALD simulation functions by import path, for example
`ewald.simulation.giwaxs:simulate_giwaxs_image`, but planning does not import
the simulator or any UI code.

## Detector-Aware Artifacts

The scaffold includes named detector footprints in
`src/ewald_data_training/detectors.py`. Current presets cover PILATUS 1M,
legacy PSI PILATUS 1M geometry, EIGER2 X 1M/1M-W/4M, and a continuous
PerkinElmer-style flat panel. Artifact profiles can request one detector
explicitly or use `random_common` to draw a randomized footprint per augmented
image.

Diffuse scattering is generated as broad q-rings tied to the simulated Bragg
spacing distribution. Detector effects are layered separately: module gaps,
direct/specular beam artifacts, Yoneda bands, substrate horizon shadowing,
critical-angle peak splitting, dead pixels, dead-pixel clusters, beamstop
shadow, flat-field response, counting noise, read noise, and saturation. Clean
simulations now default to lower-information detector windows around
`qmax = 2.8 A^-1`, with automatically recommended `hkl_extent` values,
q-dependent peak broadening, and an optional Ewald-sphere flat-detector
solid-angle response. This 2-3 A^-1 bias is intentional: it forces the ranker
and future solver stages to work with realistic GIWAXS information content
instead of relying on a wide 4 A^-1 WAXS frame. Wider high-q shards can still be
requested explicitly for ablation studies or detector-specific production
runs. When detector geometry includes incident angle and wavelength, the clean
simulator applies a pyFAI-style fiber/GI missing-wedge correction in qIP/qOOP
space before writing images and peak labels. The substrate horizon model uses
substrate dimensions, beam width/height, and incident angle to estimate
footprint spillover, then uses that spillover to tune horizon shadowing and
additional qz peak broadening. Each artifacted label file now also contains an
`artifact_assessment` block that converts direct beam/specular intensity,
Yoneda bands, substrate horizon/footprint spillage, critical-angle splitting,
beamstop shadows, and detector-module gaps into compact q-space or pixel-space
regions. Feedback evaluation and structure-guess export use those regions as
training weights in a blended overlap score, so the baseline ranker can work
around non-Bragg aberrations without discarding the clean Bragg retrieval
signal. A separate `quality_assessment` block estimates signal-to-noise,
retrievable-signal fraction, usable artifact-weighted area, saturation, and
clean/artifact overlap; this keeps the normal training distribution solvable
while reserving especially harsh profiles as a bounded stress-test tail.

## Local Smoke Test

Plan a run:

```bash
python data_training/scripts/generate_dataset.py \
  --plan data_training/configs/simulation_sweep.example.yaml \
  --output-root data_training/runs/demo \
  --manifest data_training/runs/demo/manifest.jsonl \
  --dry-run
```

Validate the resulting manifest:

```bash
python data_training/scripts/validate_manifest.py \
  data_training/runs/demo/manifest.jsonl \
  --root data_training/runs/demo
```

Remove `--dry-run` when the EWALD runtime dependencies are installed and the
example CIF paths are available. Production jobs should use the SLURM templates
under `data_training/cluster/`.

## Training Contract

Every generated sample should write:

- `artifact.tiff`: the image the model sees.
- `clean.tiff`: the artifact-free forward simulation.
- `peaks.json`: the labeled Bragg peak table with `(hkl)`, q-space metadata,
  and artifact-overlap labels for Bragg-peak assessment.
- `labels.json`: structure, condition, artifact, provenance metadata, and an
  `artifact_assessment` payload for artifact-aware indexing/ranking plus
  `quality_assessment` metrics for filtering unrealistic failed simulations.
- `manifest.jsonl`: one compact row per sample for training and validation.

The manifest is the boundary between simulation and ML training. Downstream
models can use image tensors, peak-table tensors, or both without re-running the
forward model.

The Alpine-oriented clean simulation plan
`configs/simulation_alpine_fibril_training.example.yaml` uses a `detectors:`
list to sweep PILATUS, EIGER2, and PerkinElmer-style detector windows and
incident angles before artifact augmentation. This is the recommended starting
point for cluster-scale HybriD3 runs once fixture and low-limit live tests pass.

## Reports

- [Training Framework](reports/training_framework_report.md)
- [Config-First Training Framework](reports/training_framework.md)
- [Dirac Structure Ranking](reports/dirac_structure_ranking.md)
- [Surface Artifacts And Training Model](../docs/development/surface-artifacts-training-model.md)
- [Trigger Pipeline Design](reports/trigger_pipeline_design.md)
- [Remote Alpine Workflow](cluster/REMOTE_ALPINE.md)
- [Workflow And Execution Report](../docs/development/workflow-execution-report.md)

## Trigger Folders

Each folder under `deploy/` has a single `run.sh`:

```bash
bash data_training/deploy/00_fetch_hybrid3_structures/run.sh
bash data_training/deploy/01_generate_simulations/run.sh
bash data_training/deploy/02_apply_artifacts/run.sh
bash data_training/deploy/03_train_ranker/run.sh
bash data_training/deploy/04_feedback_evaluate/run.sh
bash data_training/deploy/05_export_structure_guesses/run.sh
```

The defaults are small local checks. On Alpine, point `EWALD_RUN_BASE` at
scratch and set the stage-specific environment variables in the corresponding
`run.env.example` file.

## Local HybriD3 Scaffold Study

Run the ten-structure 2D lead-iodide smoke study and regenerate its PDF report:

```bash
python data_training/scripts/run_local_hybrid3_scaffold_test.py --use-default-ids
```

The run pulls live HybriD3 structure files, writes an enriched catalog, produces
clean and artifact-augmented GIWAXS samples, trains/evaluates the baseline
ranker, exports top-k structure-file guesses, and updates
`docs/development/hybrid3-local-training-scaffold-test.pdf`.
Use `--q-max`, `--qxy-max`, or `--qz-max` to deliberately generate narrower
or wider detector windows.
