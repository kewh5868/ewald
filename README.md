# EWALD  
**Experimental WAXS Analysis of Lattice Diffraction**

EWALD is a modular, Python-based toolkit that takes you from raw GIWAXS/WAXS images all the way through to structural insights. By combining industry-standard libraries (pyFAI, LMFit, pymatgen/diffpy) with interactive ROI tools and automated workflows, EWALD makes it easy to go from calibration to peak fitting to Bragg-peak overlay in a reproducible, scriptable environment.

## Key Features

- **Multi-format I/O & Calibration**  
  – Read raw detector frames (TIFF)  
  – Import EDF/JSON mask-files and .poni calibration files  
  – Apply geometric corrections, polarization and solid-angle factors  

- **Reciprocal-Space Mapping & ROI Definition**  
  – Convert 2D images into (qₓᵧ, q_z) or χ–q grids  
  – Draw or load arbitrary regions of interest via Matplotlib widgets or napari  

- **1D/2D Integration & Automated Peak Fitting**  
  – Radial, χ, qₓᵧ and q_z integrations within each ROI  
  – Convoluted Gaussian/Voigt fits with interactive parameter-guessing (LMFit)  

- **Crystallographic Simulation & Overlay**  
  – Import CIFs or define custom unit cells (a, b, c, α, β, γ)  
  – Compute and plot predicted Bragg reflections atop corrected GIWAXS images  
  – Interactive Miller-index assignment  

- **Advanced Analysis**  
  – Williamson–Hall and pole-figure plotting  
  – Peak-width vs. Q analysis for microstrain/crystallite‐size insights  

- **Project Management & Persistence**  
  – Encapsulate images, masks, ROIs, fits and structure models in a single “.ewld” project  
  – Save and reload full analysis sessions via HDF5 or JSON+NumPy archives  

---

EWALD empowers beamline scientists and data analysts to build reproducible GIWAXS/WAXS workflows, whether you’re exploring new materials or automating large‐scale experiments.  

*Ready to get started? Check out the [Installation](#installation) and [Quickstart](#quickstart) sections below.*  
