# UI Workflow Regression Checklist

!!! warning "Documentation notice"
This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.

Use this checklist for workflows that are too GUI-heavy for stable headless
coverage. Run it after ROI, peak, structure-analysis, pole-figure, or
simulation workflow changes.

## Data And Calibration

- Launch `pyFAI-calib2` from the calibration controls twice. Confirm only one
  process/window is started and the existing launch is reused or foregrounded.
- Add a film stoichiometry/density memory item, edit it, delete it, save the
  project, and reload. Confirm the memory list matches the final state.
- Load a structure/reference file and estimate density/optical constants.
  Confirm stoichiometry, density, refractive-index delta, and critical angle
  populate the correction state and can be saved/reloaded.

## ROI And Peak Identification

- Create a box ROI and an arch ROI. Edit ROI table geometry values and confirm
  the plotted ROI moves/resizes immediately.
- Move a plotted ROI and confirm the ROI table updates without stale values.
- Add temporary integration-channel markers, push them to Peak Identification,
  move the parent ROI, and confirm temporary channel markers clear while the
  committed peak remains in Peak Identification.
- Add, move, and delete a Peak Identification point. Exercise undo and redo.
- Run "snap all" and confirm all eligible committed peak points move to local
  maxima.
- Run the symmetry check with peaks on unequal positive/negative `q<sub>xy</sub>` ranges.
  Confirm it reports unmatched peaks instead of assuming symmetric extents.
- Create a gap-estimated peak. Confirm it is visibly distinct and saved with
  `point_kind=gap-estimated-peak`, `gap_estimated=true`, and an estimate method.

## Fits, Tags, And Structure Analysis

- Fit one ROI in Peak Fit. Confirm the fitted center and fit metrics appear in
  the Structure Analysis table.
- Edit a Structure Analysis peak center and confirm the Structure Analysis plot
  updates immediately and the edited center is not overwritten by a later fit.
- Phase-tag peaks and run candidate search. Confirm rejected/secondary/gap
  phases affect candidate selection and phase filters.
- Add or edit hkl tags on peaks/ROIs. Confirm pole-figure labels and crystal
  overlay labels update.
- Create/refine a structure candidate, then save and reload the project.
  Confirm candidates, refinements, families, and selected phase state persist.
- Fill Wyckoff setup inputs and generate CIF records. Confirm generated CIF
  references and displayed paths persist after reload, and missing external
  files show a recoverable status instead of crashing.

## Pole Figures

- Generate a pole figure from an hkl-tagged ROI. Confirm the ROI table shows
  "Current" and the saved output path appears in project metadata.
- Move the parent ROI or its coupled partner. Confirm both linked pole-figure
  metadata entries become "Stale" and the stale reason is preserved on reload.
- Re-run pole figure generation and confirm stale state returns to "Current".

## Simulation And Animation

- Change the GIWAXS simulation orientation preset and confirm the visualization
  updates.
- Run the same simulation twice. Confirm the second run reuses the cached result
  when parameters and inputs match.
- Change one simulation parameter and confirm the cache is invalidated.
- Start the Ewald sphere sweep animation/video preview, stop it, and confirm the
  previous orientation/selection is restored.

## Performance Expectations

- Long-running tasks must not block the main Qt event loop: pyFAI calibration,
  3D ROI visualization, pole-figure generation, structure candidate search,
  Wyckoff/CIF generation, GIWAXS simulation, orientation-distribution
  visualization, and Ewald sweep export.
- Slow operations should show status/progress text or a progress bar.
- Repeated pole-figure, candidate-search, simulation, and visualization work
  should reuse valid cached results.
- Large visualization outputs should be released or replaced when inputs change
  so memory does not grow across repeated runs.
