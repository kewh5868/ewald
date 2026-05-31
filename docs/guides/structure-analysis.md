# Structure Analysis

!!! warning "Documentation notice"
This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.

Structure Analysis turns peak measurements into candidate structures.
The reciprocal-lattice and Bragg-condition derivations are collected in
[Mathematical Foundations](mathematical-foundations.md).

## Importing peaks

- Import peaks from peak fit/peak identification state for the active data target.

## Editing peak centers

- Edit peak centers in table rows.
- User edits are preserved across fit updates until explicitly changed.

## Phase tagging and hkl assignment

- Tag peaks as main, secondary, unassigned, or gap/excluded.
- Assign `(hkl)` labels in table cells.

## Candidate families

- Build candidate families from filtered peaks and configured tolerances.
- Compare grouped families for consistency.

## Structure approximation and ranking

- Candidate search and best-guess refinement produce ranked candidates.
- Scores include match counts and quality metrics.

## Lattice refinement

- Run best-guess or full candidate refinements.
- Review matched peaks and outlier counts.

## Secondary phase handling

- Tag and isolate secondary/unassigned peaks.
- Use phase tags and family suggestions to reduce cross-phase interference.

## Wyckoff setup

- Wyckoff and space-group utilities are available from structure analysis helpers.
- If data is incomplete, options may be disabled until inputs are provided.

## CIF generation

- Candidate CIF generation is implemented through the backend and available through the UI controls.
- Review generated records in project analysis tables.

## Planned

- Full Wyckoff workflow depth beyond core generation is partially **Experimental** and evolving.

![Structure Analysis screen capture](../assets/screenshots/tutorials/structure-analysis.png)
