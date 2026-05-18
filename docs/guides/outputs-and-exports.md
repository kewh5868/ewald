# Outputs and Exports

EWALD stores most results in the active `.ewld` project and selected tool outputs.

## Integrated traces

- ROI integration traces are kept in fit records and displayed in Peak Fitting.

## Fit results

- 1D and 2D fit summaries include quality metrics and fit centers.
- Fit metadata is synchronized into structure analysis for downstream ranking.

## ROI metadata

- ROI geometry, tags, and coupling state are saved with project state.
- ROI edits update linked metadata records.

## Peak tables

- Peak points, tags, metadata, and fit references are stored in project tables.

## Pole figures

- CSV/PNG exports and metadata write-back are supported in the Pole Figure Generator.

## CIF files

- Candidate CIF records can be generated and tracked in structure analysis outputs.

## Simulated GIWAXS patterns

- Simulation datasets are written as project outputs and can be reloaded from the
  simulation result list.

## Recommended file organization

- Keep source images separate from `.ewld` project files.
- Keep structure files and masks in a version-controlled project folder when possible.
- Keep exported screenshots and tables in an `outputs/` directory next to the project.

