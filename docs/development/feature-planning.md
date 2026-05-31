# EWALD Feature Planning

!!! warning "Documentation notice"
This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.

This document collects planned development work for EWALD's GIWAXS/WAXS
analysis workflow. It is intentionally project-facing: each note should be
specific enough to turn into issues, design docs, or implementation milestones.

## Geometry And Low-Q Corrections

### Critical-angle calculations

- Expand the current refractive-index delta calculation into a material-aware
  critical-angle model. The first version should accept user-entered film and
  substrate composition, density, and X-ray energy, then calculate film and
  substrate critical angles with uncertainty metadata.
- Store both measured and calculated critical-angle values in the project
  correction state. The UI should show which value drives the current q-space
  markers and preserve the alternative values for auditability.
- Add critical-angle fitting helpers for low-q data. Candidate inputs include
  Yoneda-band intensity, specular reflectivity, and a user-selected low-q ROI.
- Support multi-layer critical-angle notes rather than a single scalar when a
  sample has film, interlayer, and substrate contrast.
- Expose derived markers in q-space: film critical q, substrate critical q,
  Yoneda band, horizon, specular reflection, and an effective-beam cut center.

### Direct-beam and reflected-beam calculations

- Consolidate direct-beam definitions across PONI calibration, pixel-space
  image review, q-space transforms, and peak-indexing overlays. The direct beam
  should be treated as a calibrated geometry object with pixel coordinates,
  q-space coordinates, provenance, and uncertainty.
- Improve reflected/specular beam estimation when the primary beam is hidden by
  a beamstop. Combine centroid finding in a user-selected ROI, the incident
  angle, detector geometry, sample orientation, and the PONI beam center.
- Add a consistency check that compares the direct-beam center implied by PONI
  geometry with the observed specular/reflected beam and sample horizon.
- Propagate direct-beam and reflected-beam uncertainty into low-q markers,
  q-range generation, ROI defaults, and peak-indexing confidence.

### Peak splitting near the critical angle

- Add a feature to diagnose and correct apparent peak splitting caused by
  multiple reflection paths and oscillations below/above the film or substrate
  critical angle.
- Model the dynamic-regime case where incident, refracted, and reflected beams
  all contribute to the measured signal. When the incident angle is near or
  between relevant critical angles, EWALD should generate multiple candidate
  q-ranges or transform branches for the same detector region.
- Let peak fitting and indexing assign observed split peaks to candidate
  branches, then report whether a split is likely structural, geometric, or
  caused by grazing-incidence multiple-reflection effects.
- Keep the original detector-space ROI linked to all generated q-ranges so a
  user can compare branch-specific integrations without losing provenance.

## GIXSGUI-Inspired Features

Reference publication: Z. Jiang, "GIXSGUI: a MATLAB toolbox for
grazing-incidence X-ray scattering data visualization and reduction, and
indexing of buried three-dimensional periodic nanostructured films", Journal of
Applied Crystallography 48, 917-926 (2015),
https://doi.org/10.1107/S1600576715004434.

GIXSGUI features to evaluate for EWALD:

- GUI plus scriptable backend: keep EWALD's Qt workflow backed by reusable
  Python APIs so workflows can be repeated in notebooks or batch jobs.
- Broad image input support: TIFF, CBF, EDF, FITS, MAT-style arrays, and
  detector-specific metadata sidecars.
- Correction stack: detector flat field, efficiency, air-path absorption,
  polarization, Lorentz, solid-angle, and user-defined correction maps.
- Q and angle map generation: q, qx/qy/qz, qr/qz, alpha_f, 2theta, chi, and
  detector azimuth maps stored with units and provenance.
- 2D reshaping: detector-pixel images transformed to q-space or angle-space
  images, including GI missing-wedge handling.
- 1D profile extraction: vertical, horizontal, radial, azimuthal, box, and arch
  linecuts with reusable ROI definitions.
- ROI scan/counting: extract intensity traces over folders, time series,
  temperature ramps, and other 2D image scans.
- Calibration helpers: sample-to-detector distance calibration using standards
  such as silver behenate and specular reflection from thin films or wafers.
- Curve fitting workflow: linecut fitting with reusable model definitions,
  result tables, and fit provenance.
- 3D nanostructure indexing: search allowed grazing-incidence diffraction
  positions for a given space group or unit-cell construction and overlay
  labels directly on experimental data.
- Batch automation: project-level scripted execution for large in situ and
  time-resolved data sets.

## INSIGHT-Inspired Features

Reference publication: M. A. Reus, L. K. Reb, D. P. Kosbahn, S. V. Roth and
P. Mueller-Buschbaum, "INSIGHT: in situ heuristic tool for the efficient
reduction of grazing-incidence X-ray scattering data", Journal of Applied
Crystallography 57, 509-528 (2024),
https://doi.org/10.1107/S1600576723011159.

INSIGHT features to evaluate for EWALD:

- Object-oriented per-frame processing: mirror the set -> frame -> reduced data
  -> interpretation workflow with durable project records.
- Efficient batch processing: vectorized and parallel processing for large
  time-resolved GIWAXS/GISAXS data sets.
- Frame-dependent parameters: support incident-angle, sample-to-detector
  distance, detector-position, footprint, flatfield, and mask values that change
  per frame.
- Arbitrary detector rotations: represent detector rotations in three
  dimensions and carry them through q-space mapping and plotting.
- Intensity corrections: solid angle, angular pixel sensitivity, air
  attenuation, polarization, mask, gap mask, flatfield, and user-defined maps.
- Cutting tools: tube cuts, cake cuts, pseudo-XRD cuts, vertical and horizontal
  GISAXS cuts, radial cuts, azimuthal cuts, and unbinned q-space cuts.
- Visualization modes: qr/qz plots, chi/q plots, raw detector images,
  correction maps, count distributions, and publication-ready exports.
- Data cleaning: hot-pixel removal, tilt correction, smoothing, upsampling, and
  local background subtraction for weak tube-cut signals.
- Fit tracking: fit isolated q-regions and track fitted center, width,
  intensity, and background values across time, temperature, or process axes.
- SDD normalization: estimate or correct sample-to-detector distance per image
  when experimental geometry drifts.
- GIWAXS simulation and indexing: simulate Bragg spots for candidate phases and
  orientations, handle missing-wedge overlays, and compare candidate indexing
  within minutes.
- Cross-software export: save raw, corrected, binned, unbinned, cut, fitted,
  plotted, and simulated data in reusable formats such as NetCDF/Zarr/CSV.

## PyHyperScattering Backend

- Add an optional PyHyperScattering-backed processing path for time- and
  temperature-dependent data sets. This backend should sit behind an EWALD
  processing interface so users can choose between native pyFAI wrappers and a
  PyHyperScattering workflow without changing project semantics.
- Use PyHyperScattering's xarray-centered model to load raw frame stacks,
  preserve coordinates, and return labeled q-space products for downstream
  EWALD tools.
- Map EWALD project metadata into PyHyperScattering loader conventions,
  including frame timestamp, exposure time, incident angle, temperature,
  solution/process metadata, mask, and PONI calibration.
- Support lazy output storage for large experiments through NetCDF or Zarr,
  with links retained in `.ewld` project archives.
- Add parity tests that compare native EWALD/pyFAI q-space products against
  PyHyperScattering outputs for shared example data.

## Simulation, Machine Learning, And Phase Intelligence

- Add bulk GIWAXS simulation generation from existing structure files
  (`.cif`, POSCAR/CONTCAR, and project-generated reference structures). The
  simulation runner should sweep orientation, texture, lattice perturbations,
  incident angle, detector geometry, and broadening parameters.
- Use the bulk simulation corpus to train machine-learning models for rapid
  GIWAXS simulation, phase classification, peak-indexing suggestions, and
  candidate-structure ranking.
- Preserve labels for phase, structure source, space group, lattice parameters,
  orientation, texture model, simulation parameters, and known processing
  conditions so training data stays scientifically inspectable.
- Couple machine-learning predictions to physics-based refinement. The model
  should propose likely phases or indexing assignments, while EWALD verifies
  candidates against calibrated q-space data and simulation residuals.
- Add active-learning hooks so uncertain experimental patterns can trigger new
  targeted simulations that improve the model over time.

## Beamline And Live Data Development

- Eventually add a bridge to NSLS-II JupyterHub or an equivalent beamline
  JupyterHub environment so EWALD can transfer data, update project state, and
  process detector frames as they are produced.
- Design this as a live-data adapter with authentication, read-only dry runs,
  throttling, explicit user approval for writes, and robust provenance for each
  imported or processed frame.
- Support on-the-fly q-space conversion, correction review, ROI integration,
  time/temperature trend plotting, and AI-integrated peak indexing during data
  acquisition.
- Base AI indexing on known and classified phase information from the project,
  published structure files, user-supplied phase libraries, and EWALD's bulk
  simulation corpus.
- Keep the live workflow resumable: every streamed frame, correction decision,
  model prediction, and user override should be recoverable from the `.ewld`
  project history.

## Source Notes

- GIXSGUI publication record: https://www.osti.gov/biblio/1393983
- GIXSGUI APS software page:
  https://www.aps.anl.gov/Sector-8/8-ID/Operations-and-Schedules/Useful-Links/Sector-8-GIXSGUI
- GIXSGUI documentation:
  https://www.aps.anl.gov/files/APS-Uploads/SECTOR8/8-ID/doc.pdf
- INSIGHT publication:
  https://journals.iucr.org/j/issues/2024/02/00/jl5080/
- INSIGHT project page:
  https://www.ph.nat.tum.de/en/functmat/forschung-research/insight/
- PyHyperScattering documentation:
  https://pages.nist.gov/PyHyperScattering/en/main/source/getting_started/idx_getting_started.html
