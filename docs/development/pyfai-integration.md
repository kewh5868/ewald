# pyFAI Integration Notes

!!! warning "Documentation notice"
    This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.


EWALD uses `pyFAI` as the geometry and integration engine for calibrated
detector images. The active boundary for this lives in
`ewald.processing.qspace`, where pyFAI results are converted into `xarray`
objects before they move deeper into the application.

## Core Roles

- `AzimuthalIntegrator` is the standard pyFAI object for detector geometry,
  PONI calibration, 1D azimuthal integration, and 2D caking.
- `FiberIntegrator` is the grazing-incidence/fiber diffraction integrator. It
  inherits from `AzimuthalIntegrator` and adds qIP/qOOP, exit-angle, and polar
  transforms that account for incident angle and the Ewald-sphere geometry.
- PONI files are the persistent geometry object. In EWALD, PONI assets should
  stay project-level calibrants that can be shared by multiple data files or
  data folders.
- MASK assets should be loaded or generated once, then applied to one or more
  data targets and passed to pyFAI as mask arrays during integration.

## Caking

Use `AzimuthalIntegrator.integrate2d` for caking:

```python
from ewald.processing.qspace import CakingConfig, cake_image

caked = cake_image(
    image,
    "calib.poni",
    config=CakingConfig(
        npt_radial=1000,
        npt_azimuthal=360,
        radial_unit="q_nm^-1",
    ),
    mask=mask,
)
```

The returned `xarray.DataArray` has dimensions `("azimuth", "radial")` and
stores pyFAI unit metadata in attributes. Prefer direct output in the desired
unit, such as `q_nm^-1`, instead of integrating in one unit and interpolating
later.

## GIWAXS Reciprocal-Space Mapping

Use `FiberIntegrator.integrate2d_grazing_incidence` for qIP/qOOP maps:

```python
from ewald.processing.qspace import GrazingIncidenceConfig
from ewald.processing.qspace import map_grazing_incidence_qspace

qmap = map_grazing_incidence_qspace(
    image,
    "calib.poni",
    config=GrazingIncidenceConfig(
        xray_energy_kev=12.7,
        incident_angle_deg=0.3,
        tilt_angle_deg=0.0,
        sample_orientation=6,
        unit_ip="qip_A^-1",
        unit_oop="qoop_A^-1",
        correct_solid_angle=True,
        polarization_factor=0.95,
        normalization_factor=1.0,
    ),
    mask=mask,
)
```

The returned `xarray.DataArray` has dimensions `("q_oop", "q_ip")`. The wrapper
passes incident and tilt angles in degrees because EWALD metadata and UI labels
will use degrees; pyFAI itself defaults to radians. EWALD uses Å^{-1} GI map
units by default (`qip_A^-1` and `qoop_A^-1`) so the Data Viewer $q_{xy}$ and
$q_{z}$ axes match the displayed Å^{-1} labels. Using pyFAI's default
inverse-nanometer units would make the plotted coordinates appear 10 times too
large.

The correction UI lets the user enter X-ray energy in keV. EWALD converts that
energy to a pyFAI wavelength before mapping, while still preserving the
selected energy in the `.ewld` project state. The same confirmation step also
stores pyFAI correction controls for solid angle, polarization factor,
normalization factor, and optional dummy/delta-dummy rejection values.

## Orientation

Raw detector images may need to be rotated before reciprocal-space mapping so
the pixel-space preview matches the qIP/qOOP orientation seen after image
corrections. EWALD stores that choice in `ImageCorrectionState` as both a pixel
array rotation and a pyFAI `sample_orientation`.

For non-mirrored rotations, EWALD maps the correction UI rotation to pyFAI's
EXIF-style orientation values as:

```text
0 deg clockwise   -> sample_orientation 1
90 deg clockwise  -> sample_orientation 6
180 deg           -> sample_orientation 3
270 deg clockwise -> sample_orientation 8
```

When the raw preview is mirrored over its active y-axis, the same rotations map
to pyFAI's mirrored orientation values:

```text
0 deg clockwise   + mirror -> sample_orientation 2
90 deg clockwise  + mirror -> sample_orientation 5
180 deg           + mirror -> sample_orientation 4
270 deg clockwise + mirror -> sample_orientation 7
```

The same rotation is applied to the MASK array before it is passed to pyFAI.
The same mirror transform is applied after rotation, so the raw pixel preview,
mask, and qIP/qOOP map use one shared orientation state rather than changing
the PONI geometry.

## Practical Defaults

- Ordinary caking uses pyFAI's CSR backend with bounding-box pixel splitting:
  `("bbox", "csr", "cython")`.
- Grazing-incidence qIP/qOOP mapping defaults to no pixel splitting:
  `("no", "histogram", "cython")`. The pyFAI documentation warns that pixel
  splitting can populate the GIWAXS missing wedge unless the experimental
  missing-wedge mask path is used deliberately.
- Synthetic GIWAXS training data applies the same qIP/qOOP missing-wedge
  concept analytically when `missing_wedge_correction` is enabled: bins below
  the incident-angle sample horizon are zeroed, and generated `(hkl)` labels are
  filtered to the accessible reciprocal-space region before artifact
  augmentation.
- `correctSolidAngle=True` should remain the default for calibrated maps.
- `polarization_factor=0.95` should remain the default for GI maps and may be
  `None` when no polarization correction is desired.
- `normalization_factor` defaults to `1.0`.
- `dummy` and `delta_dummy` should stay unset unless the user has an explicit
  bad-pixel sentinel value to reject during pyFAI integration.
- Store incident angle per data target from metadata inference, but allow the
  user to override it before q-space mapping.

## References

- pyFAI stable documentation: https://pyfai.readthedocs.io/en/stable/
- AzimuthalIntegrator design:
  https://pyfai.readthedocs.io/en/stable/design/ai.html
- 2D caking tutorial:
  https://pyfai.readthedocs.io/en/stable/usage/tutorial/Introduction/introduction.html#d-integration-or-caking
- 2D non-azimuthal/qx-qy integration:
  https://pyfai.readthedocs.io/en/latest/usage/tutorial/integrate2d.html
- Grazing-incidence and FiberIntegrator tutorial:
  https://pyfai.readthedocs.io/en/stable/usage/tutorial/FiberGrazingIncidence.html
