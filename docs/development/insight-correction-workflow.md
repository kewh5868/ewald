# INSIGHT-Style Correction Gate

!!! warning "Documentation notice"
    This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.


EWALD's raw detector workflow follows the INSIGHT analysis order: load a raw
frame, verify geometry and correction inputs, apply intensity/mask corrections,
then proceed to reciprocal-space viewing, integration, peak work, structure
analysis, and GIWAXS simulation.

The INSIGHT paper describes the `SingleImage` object as the per-frame handler
for reciprocal-space transformation, intensity corrections, plotting, and export.
It stores reciprocal-space maps as image attributes and applies corrections such
as solid angle, angular pixel sensitivity, air attenuation, polarization,
pixel/gap masks, and flatfields before analysis. Its geometry parameter table
tracks direct-beam coordinates, specular/reflected beam coordinates, sample
incidence angle, wavelength, sample-to-detector distance, detector rotations,
and correction parameters.

In EWALD, a selected raw data file remains locked to a single
`Apply Image Corrections` tab until the user confirms:

- selected MASK asset
- selected PONI calibrant
- reflected/specular beam position estimate
- critical angle estimate
- user-boxed artifact regions such as beamstop shadows, detector gaps, hot
  pixels, or scattering artifacts

Confirmation stores an `ImageCorrectionState` under the data file ID. Once this
state is confirmed, the correction controls are disabled for that file and the
analysis tabs become available. The first unlocked tab is `Data Viewer`, where
the corrected $q_{xy}$/$q_{z}$ image is displayed and box or arch ROIs are drawn for
later integration. Corrections are intentionally treated as permanent for the
current imported data target; reprocessing from raw data should start by
re-importing the data file or folder.

The public INSIGHT paper and TUM project page state that the current software,
demo scripts, and full documentation are available by request from the authors.
No public GitHub repository was found during implementation.

References:

- https://journals.iucr.org/j/issues/2024/02/00/jl5080/index.html
- https://www.ph.nat.tum.de/en/functmat/forschung-research/insight/
