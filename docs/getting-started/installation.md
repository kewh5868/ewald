# Installation

!!! warning "Documentation notice"
This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.

This page is the source of truth for EWALD setup.

## Supported systems and Python versions

- Operating systems: macOS, Windows, and Linux are supported by the repository workflow.
- Python: `>=3.11, <3.14` (recommended: Python 3.12).
- EWALD depends on Qt and scientific/visualization packages, so a native Python
  installation is preferred over constrained container images.

## Clone the repository

Get the current source tree before installing:

```bash
git clone https://github.com/kewh5868/ewald.git
cd ewald
```

Local datasets and generated outputs are intentionally not part of the source
tree. Keep private data, example working copies, and generated simulations under
local `example/projects/` folders as needed; Git ignores that tree.

## Environment setup

From the repository root:

EWALD ships a pinned conda environment file for development:

```bash
conda env create -f requirements/ewald-py312.yml
conda activate ewald-py312
```

## Package installation

Install EWALD in editable mode:

```bash
python -m pip install -e .
```

If you prefer a virtualenv workflow, create the environment and install the runtime
requirements from `requirements/pip.txt` before installing EWALD in editable mode.

Validate the command-line entry point:

```bash
ewald --help
```

## Development installation

For development and local extension of features:

```bash
conda env create -f requirements/ewald-py312.yml
conda activate ewald-py312
python -m pip install -e .
python -m pip install -r requirements/tests.txt
pre-commit install
```

You can also install documentation dependencies in the same environment:

```bash
python -m pip install -r docs/requirements.txt
```

Before pushing a branch, run:

```bash
pre-commit run --all-files
pytest
```

Tests that depend on local source-checkout example data are skipped when the
ignored `example/` folder is not present.

## Running EWALD

Start the Qt app:

```bash
ewald
```

Equivalent command:

```bash
python -m ewald.app
```

## Optional dependencies

- `legacy` extra in `pyproject.toml` installs `PyHyperScattering`.
- `docs/requirements.txt` contains only documentation dependencies.

## Common install issues

- **`No module named PySide6` or Qt errors:** verify the environment was created from
  `requirements/ewald-py312.yml` and activated.
- **`ewald` command not found:** confirm editable install completed after activation.
- **Example data tests are skipped:** populate a local `example/` folder if you
  want to run source-checkout sample-data tests.
- **Missing native runtime libraries on Linux:** use a native Python/conda stack that
  supports Qt6 and Matplotlib backends.

If you need help, file an issue with your OS, Python version, and the install log.

## Build the docs locally

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

Open the URL reported by `mkdocs serve` (typically `http://127.0.0.1:8000`).

For a production build:

```bash
mkdocs build
```
