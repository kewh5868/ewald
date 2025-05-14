# EWALD  
**Experimental WAXS Analysis of Lattice Diffraction**

EWALD is a modular, Python-based toolkit that takes you from raw GIWAXS images all the way through to structural insights. By combining data reduction packages (pyHyperScattering, pyFAI) with interactive ROI tools and automated workflows, EWALD makes it easy to go from calibration to peak fitting to Bragg-peak overlay in a reproducible, scriptable environment.

## Planned Features

- **Multi-format I/O & Calibration**  
  – Read raw detector frames (TIFF)  
  – Import EDF/JSON mask-files and .poni calibration files  
  – Apply geometric corrections, polarization and solid-angle factors  

- **Reciprocal-Space Mapping & ROI Definition**  
  – Convert 2D images into (qₓᵧ, q_z) or χ–q grids  
  – Draw or load arbitrary regions of interest via Matplotlib widgets or napari  

- **1D/2D Integration & Automated Peak Fitting**  
  – Radial, χ, qxy and q_z integrations within each ROI  
  – Convoluted Gaussian/Voigt fits with interactive parameter-guessing (LMFit)  

- **Crystallographic Simulation & Overlay**  
  – Import CIFs or define custom unit cells (a, b, c, α, β, γ)  
  – Compute and plot predicted Bragg reflections atop corrected GIWAXS images  
  – Interactive Miller-index assignment  

- **Advanced Analysis**  
  – Williamson–Hall and pole-figure plotting  
  – Peak-width analysis

- **Project Management**  
  – Encapsulate images, masks, ROIs, fits and structure models in a single “.ewld” project  
  – Save and reload full analysis sessions via HDF5 or JSON+NumPy archives  