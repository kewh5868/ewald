# EWALD Synthetic Structure-Recognition Training Framework

Date: 2026-05-29

## Scope

The `data_training` section is designed to turn known structures into labeled
detector images for training structure-recognition algorithms. The first target
domain is fibril-textured GIWAXS because the structure/orientation space is rich
but still controllable. The same contracts should later support powders, single
crystals, mixed phases, grazing-incidence detector geometries, and broader 2D
diffraction modalities.

The repository section is intentionally isolated. EWALD's current simulator is
used as a backend when available, but catalogs, manifests, artifact recipes, and
cluster jobs do not import Qt or mutate project files.

## Relevant Prior Art

- Zihan Zhang's [`2D_diffraction`](https://github.com/ZihanZhang-1996/2D_diffraction)
  contains the closest collaborator code reference: POSCAR parsing,
  reciprocal-lattice construction, rotated Bragg peak generation, atomic form
  factor intensity weighting, Gaussian broadening, diffuse variants, and a
  simple `scipy.ndimage` peak finder.
- Zihan's defense slides frame the inverse search as normalized vector overlap:
  an experimental image state is compared to simulated standards, the highest
  overlap is selected, and mixture components can be searched by subtracting the
  best basis state from the residual.
- Zihan's [`Poisson2DSolver_ZZ`](https://github.com/ZihanZhang-1996/Poisson2DSolver_ZZ)
  and [`Poisson2DSolver_doc`](https://github.com/ZihanZhang-1996/Poisson2DSolver_doc)
  are not diffraction tools, but they model a useful documentation pattern:
  clean separation of input schema, numerical assembly, solver, tests, and
  math-first docs.
- The JACS CrystalX paper, published online April 20, 2026, reports a deep
  learning workflow for routine single-crystal XRD structure analysis using more
  than 50,000 authentic experimental measurements and temporal validation:
  <https://pubs.acs.org/doi/10.1021/jacs.5c21832>.
- The public CrystalX model card describes a two-stage geometric deep learning
  pipeline with an Equivariant Transformer/TorchMD-NET backbone and confidence
  outputs: <https://huggingface.co/Kaipengm2/CrystalX>.
- `pygidSIM` is a strong reference for CIF-to-GIWAXS peak simulation with
  `(qxy, qz)` positions and intensities: <https://pypi.org/project/pygidsim/>.
- `pyFAI` detector docs are useful for masks, flat fields, detector gaps, and
  distortion correction: <https://pyfai.readthedocs.io/en/stable/>.
- `xrayutilities` is useful for reciprocal-space conversion and diffraction
  simulation concepts: <https://xrayutilities.sourceforge.io/>.
- Yager et al.'s synthetic X-ray scattering dataset explicitly includes masks,
  parasitic streaks, and Poisson noise as training artifacts:
  <https://acdc.alcf.anl.gov/mdf/detail/pub_94_yager_synthetic_v1.2/>.
- Recent GIWAXS peak-detection benchmarking emphasizes reciprocal-space mapping,
  annotated peak data, detector gaps, and physics-aware metrics:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11957406/>.

No collaborator code has been copied into EWALD in this scaffold. The current
implementation reuses EWALD's existing `src/ewald/simulation/giwaxs.py`
interfaces and records the external repositories as design references.

## Dataset Generation Pipeline

1. Structure catalog

   Catalog records live in `data_training/catalog/`. Each record stores a stable
   `structure_id`, file path, family, phase class, tags, source, and license.
   Production catalogs should include all redistribution constraints because
   the generated dataset may be stored online.

2. Condition sweep

   A generation config expands detector geometry, texture/orientation settings,
   line-shape widths, HKL extent, artifact profile, and random seed into
   deterministic `SimulationCondition` records. The first supported texture
   family is `fiber_gaussian`, matching fibril texture, but the schema leaves
   room for powder, single-crystal, tilted-fiber, and mixed-phase variants.

3. Forward simulation

   For each structure/condition pair, the generator calls EWALD's
   `simulate_giwaxs_image()` and `calculate_giwaxs_peak_rows()`. The clean image
   is written as `clean.tiff`, while the peak table records `(h, k, l)`, `qxy`,
   `qz`, Bragg intensity, and rendered amplitude.

4. Artifact augmentation

   `ArtifactProfile` controls deterministic image corruption. The first scaffold
   includes Poisson noise, Gaussian read noise, background gradients, diffuse
   rings, parasitic streaks, detector gaps, beamstop shadow, direct/specular
   beam intensity, Yoneda bands, substrate horizon and footprint-spillage
   effects, critical-angle peak splitting, flat-field variation, hot pixels,
   dead pixels, and saturation. Every artifact seed and operation is written to
   `labels.json`. The label also includes an `artifact_assessment` block with
   compact q-space/pixel-space regions and scalar features so peak assessment,
   indexing, and ranker feedback can downweight non-Bragg aberrations while
   learning to recognize them. A paired `quality_assessment` block estimates
   signal-to-noise, retrievable-signal fraction, usable weighted area,
   saturation, and clean/artifact overlap so especially harsh augmentations can
   be bounded or filtered instead of teaching the model unrealistic failure
   cases.

5. Manifest boundary

   Each sample directory contains `artifact.tiff`, `clean.tiff`, `peaks.json`,
   and `labels.json`. The shard manifest is JSONL so failed cluster tasks can be
   retried and merged without opening every TIFF.

## Training Stages

Stage 1: peak detection

Train a detector or segmentation model to localize Bragg spots, arcs, and
families under artifact conditions. Truth comes from simulated peak rows and
rendered clean images. Metrics should include precision/recall in q-space,
center error in `A^-1`, missed weak peak rate, and false peaks in masked areas.

Stage 2: peak indexing

Train a model that maps detected peaks to candidate `(hkl)` assignments or
reciprocal basis vectors. Inputs can be peak clouds, local image crops, or both.
Metrics should include top-k index accuracy, lattice basis error, and reflection
family confusion.

Stage 3: structure retrieval

Use the indexed/weighted peak set to rank catalog structures. The initial
baseline should be the normalized vector-overlap ranker described in
`dirac_structure_ranking.md`. The baseline accepts artifact-derived weight maps
from `artifact_assessment` and blends artifact-weighted overlap with the clean
image-overlap score, so direct beam, Yoneda, horizon, critical-angle splitting,
beamstop, and detector-gap regions are tracked without wiping out Bragg-rich
retrieval signal. Learned retrieval models can then be evaluated against that
physics baseline.

Stage 4: solve and rerank

For the top candidates, rerun higher-resolution simulations over a narrower
condition sweep, fit peak widths/backgrounds, compare residual maps, and return
a ranked list with confidence, expected orientation, and failure annotations.

## Cluster Workflow

The `data_training/cluster/` section provides Alpine-style SLURM templates:

1. Stage EWALD, catalogs, and generation plans into a runtime directory.
2. Submit a line-indexed SLURM array. Each task writes high-volume images to
   scratch and emits a compact manifest row.
3. Merge per-task manifests into a training manifest.
4. Train and validate on the cluster using scratch-resident tensors.
5. Sync only manifests, metrics, model cards, and selected examples back unless
   a dataset publication job explicitly uploads the image shards.

For online storage, prefer chunked Zarr/HDF5/NPZ or compressed TIFF shards plus
a manifest. The manifest should remain small enough to browse on GitHub, while
large image tensors belong in a dataset service such as Hugging Face Datasets,
Zenodo, OSF, Globus, or institutional object storage.

## Next Implementation Steps

- Add a real trainer package that consumes manifests and implements peak
  detection, peak indexing, retrieval, and reranking baselines.
- Add explicit q-map exports for physical detector coordinates, not only
  reciprocal-space grids.
- Add structure-family balanced train/validation/test splitting.
- Add real experimental calibration-artifact profiles from representative
  beamlines.
- Add a license/provenance gate before online dataset publication.
- Add benchmark reports that compare current EWALD peak grouping against the
  synthetic truth tables.
