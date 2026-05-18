# Documentation Style Guide

This guide keeps user-facing docs consistent across EWALD pages.

## Terms

- Use **EWALD** (all caps).
- Use these UI names exactly:
  - **Data Viewer**
  - **Peak Identification**
  - **Peak Fit**
  - **Structure Analysis**
  - **GIWAXS simulation**
- Use `q<sub>xy</sub>` and `q<sub>z</sub>` in markdown and captions.
- Refer to detector-space outputs with units as `Å<sup>-1</sup>`.

## Acronyms

- Explain acronyms on first use:
  - PONI (for detector geometry)
  - ROI (region of interest)
- Prefer full term first, acronym second.

## Language

- Use user-facing language.
- Prefer active voice and short steps.
- Avoid claiming features as complete unless implemented in current UI.

## Planned vs implemented

- Mark feature states clearly:
  - **Implemented**: currently wired and available.
  - **Experimental**: available but subject to change.
  - **Planned**: not yet implemented in the current release.

## Units and numbers

- Use SI or common scientific units in context.
- Include numeric units where they affect reproducibility.

## Files and paths

- Prefer repository-relative paths in examples.
- Use fenced code blocks for commands.
