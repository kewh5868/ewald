# EWALD Training Cluster Templates

This directory is an isolated execution scaffold for running EWALD training data
generation and model training on an external Slurm cluster such as Alpine. It is
intended to sit beside the future `data_training` code without requiring local
workstations to generate large detector-image datasets.

The templates here separate three concerns:

- **Runtime directory**: a staged copy of the EWALD source tree, structure
  catalogs, simulation plans, and cluster job scripts.
- **Scratch directory**: high-volume generated TIFF/NPZ/Zarr/HDF5 files,
  intermediate peak tables, and per-task manifests.
- **Sync directory**: compact manifests, metrics, and trained-model metadata
  copied back to the workstation or an online dataset store.

## Files

```text
data_training/cluster/
  README.md
  STAGING.md
  alpine.paths.example.env
  scripts/
    bootstrap_remote_runtime.sh
    cluster_status.sh
    offline_local_dry_run.sh
    remote_dev_session.sh
    remote_exec.sh
    stage_to_cluster.sh
    submit_generation_array.sh
    submit_deploy_trigger.sh
    sync_manifests_from_cluster.sh
    tail_cluster_logs.sh
  slurm/
    generate_dataset_array.sbatch
    run_deploy_trigger.sbatch
    merge_generation_manifests.sbatch
    train_structure_ranker.sbatch
    validate_indexing_pipeline.sbatch
  templates/
    generation_plan.example.jsonl
    training_manifest.schema.json
```

## Expected Workflow

1. Build or update a structure catalog locally.
2. Create a generation plan. YAML plans may define one `detector` or a
   `detectors` list for multi-geometry sweeps; JSONL plans may define one row
   per structure, orientation family, detector geometry, artifact recipe, and
   random seed.
3. Stage the repository and plan to the cluster runtime directory.
4. Submit the generation Slurm array. Each array task writes images and metadata
   to scratch, then emits a compact manifest row.
5. Merge generated manifests into a training manifest.
6. Submit training and validation jobs against the manifest.
7. Sync manifests, metrics, and model cards back from the cluster.

The Slurm templates call command entry points through environment variables such
as `EWALD_GENERATE_CMD` and `EWALD_TRAIN_CMD`. `EWALD_GENERATE_CMD` defaults to
`python data_training/scripts/generate_dataset.py`, while training and
validation commands remain overridable hooks for the next ML implementation
phase.

## Quick Start

Copy the example environment and edit it for your account and project paths:

```bash
cp data_training/cluster/alpine.paths.example.env data_training/cluster/alpine.paths.env
$EDITOR data_training/cluster/alpine.paths.env
```

Stage the repo and optional manifests:

```bash
bash data_training/cluster/scripts/stage_to_cluster.sh data_training/cluster/alpine.paths.env
```

Bootstrap/check runtime directories and the configured Python environment:

```bash
bash data_training/cluster/scripts/bootstrap_remote_runtime.sh data_training/cluster/alpine.paths.env
```

Open a persistent remote working shell in the staged checkout:

```bash
bash data_training/cluster/scripts/remote_dev_session.sh data_training/cluster/alpine.paths.env
```

On the cluster login node:

```bash
source /path/to/runtime/cluster/alpine.paths.env
sbatch /path/to/runtime/repo/data_training/cluster/slurm/generate_dataset_array.sbatch
sbatch /path/to/runtime/repo/data_training/cluster/slurm/merge_generation_manifests.sbatch
```

After the array finishes, sync compact outputs back:

```bash
bash data_training/cluster/scripts/sync_manifests_from_cluster.sh data_training/cluster/alpine.paths.env
```

For a local-to-remote generation submission without manually opening an Alpine
shell:

```bash
bash data_training/cluster/scripts/submit_generation_array.sh data_training/cluster/alpine.paths.env
```

For any trigger folder:

```bash
bash data_training/cluster/scripts/submit_deploy_trigger.sh \
  data_training/cluster/alpine.paths.env \
  data_training/deploy/01_generate_simulations
```

For the allocation-aware Ascent workflow, set `EWALD_ALPINE_ACCOUNT` in the
environment file and submit the full dependent pipeline:

```bash
bash data_training/cluster/scripts/submit_ascent_pipeline.sh \
  data_training/cluster/alpine.paths.env
```

This submits compute-node jobs for HybriD3 live ingestion, clean multi-detector
GIWAXS simulation, detector/surface-artifact augmentation, baseline ranker
construction, feedback evaluation, and top-k structure-file export. Jobs are
chained with Slurm `afterok` dependencies and write deterministic outputs under
`$EWALD_SCRATCH_DIR`.

Check jobs and recent logs:

```bash
bash data_training/cluster/scripts/cluster_status.sh data_training/cluster/alpine.paths.env
bash data_training/cluster/scripts/tail_cluster_logs.sh data_training/cluster/alpine.paths.env
```

## Runtime Contract

The generation job expects one JSON object per plan line. A minimal plan row is
shown in `templates/generation_plan.example.jsonl`. The generator should write a
manifest JSONL row for each produced detector image. Each row should include:

- `sample_id`
- `structure_id`
- `source_structure_file`
- `condition_id`
- `orientation_family`
- `image_uri`
- `q_map_uri`
- `hkl_table_uri`
- `peak_table_uri`
- `bragg_intensity_uri`
- `artifact_recipe`
- `simulator_version`
- `random_seed`

The training job should consume the merged manifest and emit:

- model checkpoints
- ranking metrics
- indexing metrics
- confusion tables by structure family, orientation family, and artifact recipe
- a compact model card describing training data provenance

## Design Choices

- Jobs write large data to scratch first, then sync only manifests and metrics
  by default.
- Slurm array tasks are indexed by line number in a plan JSONL file so failed
  tasks can be re-run deterministically.
- Paths are passed through environment variables rather than hard-coded cluster
  directories.
- The templates do not assume network access from compute nodes.
- The staging scripts use `rsync` so large catalogs can be updated without
  repeatedly copying unchanged files.
- `REMOTE_ALPINE.md` documents the remote workflow and explicitly keeps model
  training disabled until the training implementation is requested.
