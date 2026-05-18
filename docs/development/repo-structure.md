# Repository Structure

EWALD uses a scikit-package-style `src/` layout.

```text
src/ewald/
  app/              Application entry points
  data/             Project models and xarray/dask datasets
  io/               Detector, metadata, and .ewld project I/O
  processing/       Calibration, peak detection, and fitting
  crystallography/  Lattice, structure, and CIF helpers
  simulation/       GIWAXS simulation and minimization interfaces
  ui/               Qt6 main window and workflow tabs
  legacy/           Reference-only pre-rebuild code
```

The legacy exploratory files are reference-only. New UI and backend code should
not import from `src/ewald/legacy/`; migrate useful ideas into the active
package modules instead.

The active UI renders loaded files from `ProjectState.data_groups` in the left
`Experimental Data` dock. Filename metadata inference lives in
`src/ewald/io/metadata.py`; import helpers that turn parsed files into project
groups live in `src/ewald/io/importers.py`.
