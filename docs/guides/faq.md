# FAQ

## What file formats does EWALD support?

- Detector images: `.tif`, `.tiff`
- Structure inputs for optics/simulation: `.cif`, `.mcif`, `POSCAR*`, `CONTCAR*`, `*.vasp`

## How do I calibrate detector geometry?

- Run the managed `pyfai-calib2` tool from EWALD.
- Save a `.poni` file and load it as the active calibrant.
- Apply and confirm corrections before downstream analysis.

## What is `q<sub>xy</sub>`?

- In-plane reciprocal-space axis used for GIWAXS lineouts.

## What is `q<sub>z</sub>`?

- Out-of-plane reciprocal-space axis used with `q<sub>xy</sub>`.

## What is an arch ROI?

- A circular-sector ROI defined by radius and `chi` bounds for azimuthal extraction.

## How do I export fitted peak positions?

- Save the project and use table/export tools in active pages.
- Pole figures and simulation outputs include explicit export paths in their windows.

## How do I simulate a candidate structure?

- Use the **GIWAXS Simulation** tool.
- Load a structure file, choose preset or manual orientation values, and run simulation.

## How do I report bugs?

- Open an issue in the repository with:
  - EWALD version
  - OS/Python version
  - Reproduction steps
  - error logs
