# Troubleshooting

!!! warning "Documentation notice"
    This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.


## Installation problems

- Activate the conda environment before running `ewald`.
- Reinstall runtime dependencies if `PySide6` or Qt plugins are missing.

## pyFAI launch problems

- Confirm `pyfai-calib2` is on `PATH`.
- Use one active instance; EWALD blocks duplicate launches.

## Calibration problems

- Verify calibrant path and rotation/mirror settings.
- Confirm corrections before proceeding.

## Missing dependencies

- Install docs or runtime dependencies from `requirements/pip.txt` and `docs/requirements.txt`.

## Slow startup

- Large projects or slow filesystems can delay startup.
- Use a focused dataset while testing.

## UI scaling problems

- Adjust OS scaling before launching when high-DPI scaling behaves unexpectedly.

## Plot aspect ratio issues

- EWALD locks image aspect for q-space plots; test with standard detector dimensions.

## ROI/table synchronization issues

- Refresh target selection after ROI edits.
- Check that ROI row edits are committed and project changes are saved.

## Simulation performance issues

- Lower resolution and smaller `hkl_extent` for faster iteration.
- Use cache-aware workflows where settings match prior runs.

## File parsing errors

- Ensure image file extensions are supported.
- Validate metadata sidecar YAML syntax.
