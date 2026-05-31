# Remote Alpine Workflow

This workflow prepares Alpine as the execution environment for EWALD
data-training work without launching model training.

## One-Time Local Setup

Copy the environment template and edit user/account paths:

```bash
cp data_training/cluster/alpine.paths.example.env data_training/cluster/alpine.paths.env
$EDITOR data_training/cluster/alpine.paths.env
```

Keep credentials out of this file. Use your normal SSH key or CU Boulder
authentication flow.

## Stage and Bootstrap

Stage the current EWALD checkout, cluster scripts, and generation plan:

```bash
bash data_training/cluster/scripts/stage_to_cluster.sh data_training/cluster/alpine.paths.env
```

Create/check runtime and scratch directories:

```bash
bash data_training/cluster/scripts/bootstrap_remote_runtime.sh data_training/cluster/alpine.paths.env
```

By default this only checks for the configured conda environment. To create the
environment after staging, set `EWALD_BOOTSTRAP_CREATE_ENV=1` in
`alpine.paths.env` and rerun the bootstrap script.

## Open a Remote Working Session

Open or reattach a remote `tmux` session in the staged checkout:

```bash
bash data_training/cluster/scripts/remote_dev_session.sh data_training/cluster/alpine.paths.env
```

This gives a persistent Alpine shell at:

```text
$EWALD_RUNTIME_DIR/repo
```

When Codex is asked to operate remotely later, use the same staged checkout and
the helper below to run noninteractive commands through SSH.

## Remote Commands

Run a single command in the staged checkout:

```bash
bash data_training/cluster/scripts/remote_exec.sh \
  data_training/cluster/alpine.paths.env \
  python data_training/scripts/generate_dataset.py --help
```

This is the command path Codex can use after you approve SSH access.

## Generation Jobs Only

Submit the dataset-generation SLURM array:

```bash
bash data_training/cluster/scripts/submit_generation_array.sh data_training/cluster/alpine.paths.env
```

This does not train a model. It only runs `EWALD_GENERATE_CMD`, which defaults to
`python data_training/scripts/generate_dataset.py`.

## Ascent Allocation Pipeline

CURC treats allocations as Slurm accounts. Set the exact account string from
your Ascent allocation email:

```bash
export EWALD_ALPINE_ACCOUNT="your_ascent_slurm_account"
export EWALD_ALPINE_PARTITION=amilan
export EWALD_ALPINE_QOS=normal
```

Then submit the dependent compute-node pipeline:

```bash
bash data_training/cluster/scripts/submit_ascent_pipeline.sh \
  data_training/cluster/alpine.paths.env
```

The pipeline submits:

1. HybriD3 online extraction and enriched structure-library catalog creation.
2. Clean multi-detector GIWAXS simulation dataset construction.
3. Detector and surface-scattering artifact augmentation with artifact labels.
4. Baseline structure-ranker checkpoint construction.
5. Feedback evaluation on artifact images.
6. Top-k structure-file guess export from known training structures.

Start with `EWALD_HYBRID3_LIMIT=25` or lower. Set
`EWALD_HYBRID3_LIMIT=0` only after the environment, storage paths, and logs
look healthy.

Submit any single trigger folder through the generic trigger wrapper:

```bash
bash data_training/cluster/scripts/submit_deploy_trigger.sh \
  data_training/cluster/alpine.paths.env \
  data_training/deploy/00_fetch_hybrid3_structures
```

Check status:

```bash
bash data_training/cluster/scripts/cluster_status.sh data_training/cluster/alpine.paths.env
```

Tail recent logs:

```bash
bash data_training/cluster/scripts/tail_cluster_logs.sh data_training/cluster/alpine.paths.env
```

Sync compact outputs back:

```bash
bash data_training/cluster/scripts/sync_manifests_from_cluster.sh data_training/cluster/alpine.paths.env
```

Large detector images remain on Alpine scratch unless
`EWALD_SYNC_LARGE_OUTPUTS=1` is set.

## Training Guardrail

The standalone deep-training and validation SLURM templates are placeholders for
future work. The dependent pipeline's `03_train_ranker` trigger is safe: it
builds the current baseline vector-ranker checkpoint. Do not submit:

```text
data_training/cluster/slurm/train_structure_ranker.sbatch
data_training/cluster/slurm/validate_indexing_pipeline.sbatch
```

until the training package is implemented and explicitly requested.
