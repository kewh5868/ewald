# EWALD

**Experimental WAXS Analysis for Lattice Determination**

**Documentation:** https://kewh5868.github.io/ewald/

EWALD is a Qt6 scientific software package for GIWAXS/WAXS analysis. It brings
detector-image import, calibration-aware q-space correction, ROI and peak
fitting workflows, lattice and structure-candidate ranking, and GIWAXS
simulation into one project-based desktop interface. The simulation workflow is
designed to compare experimental scattering targets against generated solved CIF
structures, including residual difference maps for judging fit quality.

Use the docs links below for the maintained setup and workflow details:

- Full docs: https://kewh5868.github.io/ewald/
- Installation and setup: https://kewh5868.github.io/ewald/getting-started/installation/
- Quickstart: https://kewh5868.github.io/ewald/getting-started/quickstart/
- Developer notes: https://kewh5868.github.io/ewald/developer/notes/

## What EWALD does

- Imports detector images and stores analysis state in `.ewld` projects.
- Applies calibration/correction context for GIWAXS/WAXS q-space workflows.
- Provides ROI, peak identification, peak fitting, and structure-analysis tools.
- Generates draft CIF candidates and compares simulated GIWAXS patterns against
  experimental q-space targets with residual difference maps.
- Exports project, simulation, and analysis products for downstream reuse.

## Quick start workflow

1. Install and activate the repository environment.
2. Install EWALD in editable mode.
3. Start EWALD and create a new `.ewld` project.
4. Load an image and confirm correction inputs.
5. Open Data Viewer and begin ROI / peak workflows.

## Install

Prerequisites:

- Git.
- A conda-compatible environment manager such as Miniconda, Anaconda, or
  Mambaforge.
- Python 3.12 is recommended; the package metadata supports Python
  `>=3.11, <3.14`.

From a terminal:

```bash
git clone https://github.com/kewh5868/ewald.git
cd ewald
conda env create -f requirements/ewald-py312.yml
conda activate ewald-py312
python -m pip install -e .
ewald
```

Equivalent launch command:

```bash
python -m ewald.app
```

If the app command is not found, confirm that the conda environment is active and
installation completed successfully.

## Development checks

Install test and hook dependencies, then validate the checkout:

```bash
python -m pip install -r requirements/tests.txt
pre-commit install
pre-commit run --all-files
pytest
```

The repository intentionally ignores local example datasets, generated
simulation outputs, and private prompt notes (`/example/`, `/output/`, and
`/PROMPTS.txt`). Tests that require local example data are skipped when that
folder is absent.

## Build the docs locally

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

Then open the local URL (typically `http://127.0.0.1:8000`).

For a static build:

```bash
mkdocs build
```

Output appears in `site/`.

## Documentation map

- Getting started:
  - [Installation](https://kewh5868.github.io/ewald/getting-started/installation/)
  - [Quickstart](https://kewh5868.github.io/ewald/getting-started/quickstart/)
- User guide:
  - [User Interface Overview](https://kewh5868.github.io/ewald/user-interface/overview/)
  - [Data Loading](https://kewh5868.github.io/ewald/guides/data-loading/)
  - [Peak Fitting](https://kewh5868.github.io/ewald/guides/peak-fitting/)
  - [Structure Analysis](https://kewh5868.github.io/ewald/guides/structure-analysis/)
  - [Simulation](https://kewh5868.github.io/ewald/guides/simulation/)
- Tutorials:
  - [Loading data and viewing a GIWAXS image](https://kewh5868.github.io/ewald/tutorials/loading-data-and-viewing/)
  - [Creating and editing ROIs](https://kewh5868.github.io/ewald/tutorials/create-edit-rois/)
  - [Using Structure Analysis](https://kewh5868.github.io/ewald/tutorials/structure-analysis/)

## Project structure

```text
src/ewald/
  app/               CLI and application startup
  data/              Project and state models
  io/                Import/export and metadata helpers
  processing/        Calibration, fitting, and helper workflows
  crystallography/    Lattice, structure, and CIF utilities
  simulation/        GIWAXS simulation and refinement
  ui/                Qt6 application interface
tests/               Test coverage for processing and workflows
requirements/        Runtime dependency manifests
docs/                MkDocs documentation sources
```

- Active code is under `src/ewald/`.
- `src/ewald/legacy/` contains historical references and is not the default
  implementation path.
