# Structure Recognition Training Framework

## Goal

The data-training section should produce large, labeled GIWAXS image corpora
with known structure identity, known `(h, k, l)` peak labels, known q-space
positions, and known Bragg intensities. The first target domain is fibril
textured scattering because the orientation distribution can be parameterized
cleanly, but the same manifest and sweep model should extend to powders,
single-crystal-like spots, mixed phases, and partially oriented films.

## Generation Pipeline

1. Parse the structure catalog.
2. Expand simulation sweeps over structures, orientation, broadening, q-space
   ranges, texture parameters, and artifact profiles.
3. Run the configured EWALD simulator for each condition.
4. Export a clean peak table with `(h, k, l)`, `qxy`, `qz`, intensity, and
   amplitude before image artifacts are applied.
5. Apply randomized detector artifacts from the selected profile.
6. Write image files, label files, peak tables, and the top-level manifest.
7. Train and evaluate stage-wise models against held-out structures and
   artifact regimes.

## Artifact Families

The initial profiles model common GIWAXS detector issues: Poisson counting
noise, Gaussian read noise, diffuse air scatter, beamstop shadows, detector
gaps, missing wedges, hot pixels, saturation, flat-field gradients, and
cosmic-ray streaks. The artifact generator should store both sampled parameters
and random seeds so every image can be reproduced exactly.

## Learning Stages

The training workflow should be stage-wise rather than one monolithic model:

- Peak detection: image to peak candidates with uncertainty ellipses.
- Peak-family grouping: candidates to q-series or projected reciprocal-family
  groups.
- Indexing: grouped peaks to plausible `(h, k, l)` assignments.
- Structure guessing: indexed observations to ranked catalog candidates.
- Solver refinement: candidate structure and texture parameters to residual
  scoring against the observed image.

Each stage should write intermediate predictions so errors can be attributed to
missed peaks, incorrect grouping, wrong indexing, or poor structure ranking.

## Vector Ranking Model

The ranking layer can be written in the vector language used in Zihan-style
derivations. Represent an observed image or peak table as a normalized feature
state:

```text
|I_obs> = normalize([p_1, p_2, ..., p_n, b_1, ..., b_m])
```

Represent each simulated candidate under a structure and texture condition as:

```text
|S_j(theta)> = normalize([s_1(theta), ..., s_n(theta), c_1(theta), ..., c_m(theta)])
```

The simplest score is an overlap:

```text
score(j, theta) = <S_j(theta)|I_obs>
```

Practical scoring should combine multiple channels:

```text
score =
  w_peak * <P_sim|P_obs>
  + w_family * <F_sim|F_obs>
  + w_image * <I_sim|I_obs>
  - w_residual * ||I_obs - alpha I_sim - beta||_2
```

where `P` encodes peak positions/intensities, `F` encodes family-group
relationships, and `I` encodes the normalized image. The generated manifest
provides supervised labels for every term in this expression.

## Evaluation Splits

Use deterministic hash-based splits at the sample level. For stronger tests,
add future split modes that hold out entire structures, chemistry families, or
artifact profiles. Fibril-texture training can then be evaluated on:

- Same structures with unseen orientations.
- Same families with harsher detector artifacts.
- Held-out structures from the same chemistry family.
- Non-fibril or mixed-orientation images to test generalization.

## Cluster Execution

Cluster execution should start from a serialized training plan. A Slurm array
job receives one plan index, resolves the structure path, calls the configured
EWALD simulator, applies artifacts, writes labels, and appends or shards the
manifest. The runtime config keeps repository paths separate from scratch and
dataset roots so Alpine or another cluster can generate large datasets without
requiring local storage.
