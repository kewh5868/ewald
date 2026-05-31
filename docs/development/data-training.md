# Data Training

EWALD now includes an isolated `data_training/` scaffold for synthetic GIWAXS
training data, structure-recognition experiments, and cluster execution.

The first target is fibril-textured scattering because it gives a practical path
from known structure files to labeled detector-like images:

1. catalog known CIF/POSCAR structures,
2. sweep texture/orientation/detector conditions,
3. simulate clean GIWAXS maps and labeled `(hkl)` peak tables,
4. add reproducible detector artifacts and detector-footprint masks,
5. train peak detection, indexing, retrieval, and reranking models,
6. run large production sweeps on SLURM scratch storage.

The design lives outside the Qt application so generation and training jobs can
run on a workstation, Alpine-style SLURM node, or future container without
importing GUI dependencies.

## Detector-Aware Augmentation

The artifact generator now separates sample scattering from detector effects.
Diffuse scattering is synthesized as q-rings tied to the simulated Bragg
spacing distribution, while detector realism is added through named footprints:
`pilatus1m_pyfai`, `pilatus1m_legacy_psi`, `eiger2_x_1m`,
`eiger2_x_1m_w`, `eiger2_x_4m`, and `perkin_elmer_xrd_1621`.

The clean simulation grid now defaults to a lower-information detector window:
`qxy = [-2.8, 2.8] A^-1` and `qz = [0, 2.8] A^-1` for the local HybriD3
smoke runner. This 2-3 A^-1 bias better reflects many practical GIWAXS
measurements and tests whether the solver can rank structures with less
high-q information. The range is still configurable with `--q-max`,
`--qxy-max`, and `--qz-max`, and high-q shards can be generated explicitly
when a detector geometry or ablation study needs them.

The local HybriD3 smoke runner selects the maximum recommended hkl search
across the pulled structures for the requested q-range because long-axis 2D
perovskites can otherwise truncate the detector plane. A first-order
flat-detector solid-angle response is available from the Ewald-sphere relation
`q = 4 pi sin(theta) / lambda`, and the simulator supports q-dependent
in-plane and out-of-plane peak broadening. Clean simulations also apply a
pyFAI-style fiber/GI missing-wedge correction when an incident angle and
wavelength are present: the qIP/qOOP bins below the sample horizon are zeroed,
and inaccessible Bragg peaks are excluded from the labeled `(hkl)` table. The
artifact profiles then add randomized detector masks, direct/specular beam
artifacts, Yoneda bands, substrate horizon shadowing, critical-angle peak
splitting, hot/dead pixels, dead-pixel clusters, beamstop shadows, flat-field
variation, Poisson counting statistics, and read noise. Surface-scattering
artifacts use the same incident angle and wavelength stored in the detector
geometry; critical angle is estimated from the structure electron density when
a structure file is available. The horizon model computes the illuminated beam
footprint length from beam height and incident angle, compares that footprint
with the configured substrate size, and uses the resulting spillover fraction
to tune horizon intensity, below-horizon leakage, and added qz peak broadening.
The training labels now include `artifact_assessment` regions for direct beam
and specular spots, Yoneda bands, substrate horizon/footprint spillage,
critical-angle peak splitting, detector masks, and beamstop shadows. The
baseline feedback and structure-guess scripts rasterize these regions into
training weights and blend artifact-weighted overlap with clean image overlap,
so artifact-dominated pixels are tracked without discarding Bragg-rich regions
needed for indexing. Labels also include `quality_assessment` metrics for
signal-to-noise, retrievable-signal fraction, usable weighted area, saturation,
and clean/artifact overlap, so production runs can filter augmentations that
are too degraded to be realistically solvable.

For Alpine staging, `data_training/configs/simulation_alpine_fibril_training.example.yaml`
extends the clean generation step to a `detectors:` sweep. The current example
crosses PILATUS 1M, EIGER2 1M-W, and PerkinElmer-style detector windows with
different incident angles, q-ranges, texture widths, and seeds before the
artifact stage expands each clean image across surface and detector profiles.

## Repository Section

- [`data_training/`](https://github.com/kewh5868/ewald/tree/main/data_training)
  describes the scaffold and local
  smoke-test commands.
- [`data_training/reports/training_framework_report.md`](https://github.com/kewh5868/ewald/blob/main/data_training/reports/training_framework_report.md)
  explains the dataset-generation and training stages.
- [`data_training/reports/dirac_structure_ranking.md`](https://github.com/kewh5868/ewald/blob/main/data_training/reports/dirac_structure_ranking.md)
  documents the normalized vector-overlap ranking method using Dirac notation.
- [`data_training/cluster/`](https://github.com/kewh5868/ewald/tree/main/data_training/cluster)
  contains Alpine/SLURM
  staging scripts and array-job templates.
- [`data_training/cluster/REMOTE_ALPINE.md`](https://github.com/kewh5868/ewald/blob/main/data_training/cluster/REMOTE_ALPINE.md)
  describes the remote session workflow for staging code, opening a persistent
  Alpine shell, submitting generation jobs, and syncing compact outputs.
- [`data_training/reports/trigger_pipeline_design.md`](https://github.com/kewh5868/ewald/blob/main/data_training/reports/trigger_pipeline_design.md)
  defines the five one-command triggers for HybriD3 ingestion, simulation,
  artifact augmentation, ranker deployment, and feedback evaluation.
- [Workflow And Execution Report](workflow-execution-report.md) provides the
  consolidated operating plan, local commands, Alpine execution path, manifest
  contract, and validation gates.
- [Surface Artifacts And Training Model](surface-artifacts-training-model.md)
  documents the q-space surface-artifact model, detector footprints, artifact
  labels, quality gates, and artifact-aware ranking/training equations with
  literature citations.
- [HybriD3 Local Training Scaffold Test](hybrid3-local-training-scaffold-test.md)
  reports a ten-structure local smoke study with simulation previews, artifact
  examples, baseline ranking metrics, and a PDF version.

## Alpine Ascent Shortcut

After staging the repository and configuring `data_training/cluster/alpine.paths.env`,
the full compute-node pipeline can be submitted with:

```bash
bash data_training/cluster/scripts/submit_ascent_pipeline.sh \
  data_training/cluster/alpine.paths.env
```

Set `EWALD_ALPINE_ACCOUNT` to the Slurm account assigned to the Ascent
allocation. The submitter chains HybriD3 extraction, GIWAXS simulation, artifact
augmentation, ranker construction, feedback evaluation, and top-k structure-file
export with Slurm dependencies.

## Ranking Model

The initial physics baseline treats an experimental image as a normalized state
`\ket{\psi}` and each simulated standard as `\ket{\phi_{j,c}}`. The structure
score is the weighted overlap:

```math
s(j,c) = \braket{\phi_{j,c} | \psi}.
```

Future learned models should improve robustness to missing peaks, artifacts,
mixtures, and calibration drift, but the vector-overlap baseline remains an
auditable reference.
