# Runtime, Scratch, and Manifest Staging

This note describes the intended data movement model for EWALD training jobs on
an external compute node or Slurm cluster.

## Directory Roles

### Local workstation

The local workstation holds the active EWALD checkout and small planning files:

```text
ewald/
  data_training/
    cluster/
    catalogs/
    plans/
```

Generated detector images should not be committed to the source repository.
Large datasets should live in cluster scratch, object storage, or a separate
versioned dataset release.

### Cluster runtime

The runtime directory contains files needed to launch jobs:

```text
$EWALD_RUNTIME_DIR/
  repo/
  catalogs/
  plans/
  manifests/
  cluster/
  logs/
```

Runtime is the right place for source code, CIF/PDB/JSON structure inputs,
generation-plan JSONL files, and Slurm logs. It should be backed up or easy to
recreate.

### Cluster scratch

Scratch contains high-volume job outputs:

```text
$EWALD_SCRATCH_DIR/
  generated/
    task-000001/
    task-000002/
  manifests/
  training-runs/
  validation-runs/
```

Each generation task should write all raw arrays, detector TIFFs, q-map files,
peak labels, and Bragg intensity tables under its own task directory. The final
manifest row should use stable relative paths rooted at `$EWALD_SCRATCH_DIR` or
a later object-store URI.

## Manifest Strategy

Use JSONL for append-friendly per-task output. A generation task should write a
local manifest first:

```text
$EWALD_SCRATCH_DIR/generated/task-000123/manifest.jsonl
```

Then copy or append a compact row into:

```text
$EWALD_SCRATCH_DIR/manifests/generate-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID.jsonl
```

After the array finishes, concatenate and validate those rows into a dataset
manifest:

```text
$EWALD_SCRATCH_DIR/manifests/training-manifest.jsonl
```

The synced manifest should be small enough to track in a release, upload to an
artifact store, or inspect locally. The raw images remain on scratch or in an
external dataset bucket.

## Artifact Recipes

Image artifact recipes should be serialized into each manifest row. Candidate
artifact fields include:

- detector dark current and pedestal offsets
- Poisson shot noise
- read noise and gain drift
- hot/dead pixels
- panel masks and beamstop masks
- saturation and blooming
- parasitic streaks or rings
- diffuse background families
- q-space warping and detector tilt perturbations
- missing wedges, crop windows, and bad-row/bad-column masks

Keeping artifact settings in the manifest makes the training set auditable and
allows validation metrics to be grouped by artifact severity.

## Sync Policy

Default sync should copy back:

- manifests
- metrics
- model cards
- small preview images
- failed-task logs

Default sync should not copy back:

- full detector TIFF sets
- dense q-map arrays
- large model checkpoints unless explicitly requested
- temporary scratch intermediates

Use the `EWALD_SYNC_LARGE_OUTPUTS=1` environment variable only when you really
intend to mirror high-volume outputs to the workstation.
