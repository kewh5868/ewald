# Surface Artifacts And Training Model

EWALD's synthetic training scaffold treats surface scattering artifacts as
metadata-bearing physics operators, not just random image noise. The goal is to
train peak detection, indexing, and structure retrieval models on images that
look like difficult GIWAXS detector frames while still preserving enough Bragg
information for a scientifically retrievable answer.

The implementation lives in:

- `data_training/src/ewald_data_training/artifacts.py`
- `data_training/src/ewald_data_training/artifact_features.py`
- `data_training/src/ewald_data_training/ranking.py`
- `data_training/scripts/train_ranker.py`
- `data_training/scripts/feedback_evaluate.py`
- `data_training/scripts/export_structure_guesses.py`

## Geometry Contract

The clean simulation is generated in reciprocal-space coordinates and then
augmented with detector and sample-surface effects. The GIWAXS coordinate
system follows the same practical qIP/qOOP convention used by pyFAI's
Fiber/Grazing-Incidence integrator. pyFAI's documentation warns that
grazing-incidence pixel splitting can smear intensity into the missing wedge
unless a missing-wedge mask is used deliberately; EWALD therefore applies an
analytical missing-wedge correction before writing clean synthetic labels.
The broader detector-regrouping and calibration foundation follows the pyFAI
software paper, while the GIWAXS indexing and refraction-aware interpretation
path is aligned with fiber-texture indexing work and GIWAXS-SIIRkit.

For the first-order surface labels, EWALD uses

```math
k_0 = \frac{2\pi}{\lambda}
```

with incident angle `alpha_i`, critical angle `alpha_c`, and exit angle
`alpha_f`. The out-of-plane positions used by the artifact generator are:

```math
q_z^{\mathrm{direct}}
  = k_0[\sin(\alpha_i) + \sin(-\alpha_i)] \approx 0
```

```math
q_z^{\mathrm{specular}}
  = 2k_0\sin(\alpha_i)
```

```math
q_z^{\mathrm{Yoneda}}
  = k_0[\sin(\alpha_i) + \sin(\alpha_c)]
```

```math
q_z^{\mathrm{horizon}}
  = k_0\sin(\alpha_i)
```

The critical-angle peak-splitting offset is recorded as

```math
\Delta q_c
  = k_0|\sin(\alpha_c) - \sin(\alpha_i)|.
```

When the structure parser can estimate electron density, the scaffold uses the
small-angle X-ray optics approximation

```math
\alpha_c \approx \sqrt{2\delta},
\qquad
\delta \approx \frac{r_e \lambda^2 \rho_e}{2\pi},
```

where `r_e` is the classical electron radius and `rho_e` is electron density.
Profiles may override this when a measured material/substrate value is known.

## Surface Operators

The surface-scattering augmentation is controlled by `ArtifactProfile` and is
applied after the clean Bragg image is normalized. The important operators are:

- **Direct beam and specular reflection** place compact q-space lobes near
  `qz = 0` and `qz = 2 k0 sin(alpha_i)`.
- **Yoneda band** places a horizon-parallel enhancement at the exit critical
  angle. This follows the classical Yoneda effect: surface-scattered intensity
  is enhanced when the exit angle is close to the critical angle. Future
  higher-fidelity versions should move from this labeled approximation toward a
  distorted-wave Born approximation treatment for samples where rough-surface
  scattering dominates the low-q signal.
- **Critical-angle peak splitting** creates weaker shifted copies of bright
  Bragg features by moving high-intensity components along `qz` by
  `Delta q_c`.
- **Substrate horizon** attenuates and brightens the detector near the sample
  horizon with optional slope and roughness.
- **Footprint spillage broadening** broadens peaks in `qz` when the projected
  grazing-incidence footprint is larger than the substrate.
- **Structure-correlated diffuse rings** add broad q-rings centered near the
  simulated Bragg spacing distribution, rather than unphysical random vertical
  streaks.

The footprint model is intentionally simple and explicit:

```math
L_{\mathrm{footprint}}
  =
  \frac{h_{\mathrm{beam}}}{\sin(\alpha_i)}
```

```math
A_{\mathrm{footprint}}
  =
  L_{\mathrm{footprint}} w_{\mathrm{beam}}
```

```math
A_{\mathrm{covered}}
  =
  \min(L_{\mathrm{footprint}}, L_{\mathrm{sub}})
  \min(w_{\mathrm{beam}}, w_{\mathrm{sub}})
```

```math
f_{\mathrm{spill}}
  =
  1 -
  \frac{A_{\mathrm{covered}}}{A_{\mathrm{footprint}}}.
```

`f_spill` increases substrate-horizon width, below-horizon leakage, horizon
roughness, and vertical peak broadening. This gives the training model explicit
examples of sample-size and beam-footprint artifacts instead of treating the
horizon as a fixed mask.

## Detector Footprints

Detector masks are added separately from sample-surface artifacts. The current
presets include PILATUS 1M, legacy PSI PILATUS 1M, EIGER2 X 1M/1M-W/4M, and a
continuous PerkinElmer-style flat panel. The PILATUS 1M reference mask follows
published modular-detector geometry: 172 um pixels, a 981 x 1043 pixel array,
a 2 x 5 module layout, and intermodule gaps of 7 horizontal and 17 vertical
pixels in the DECTRIS PILATUS3 R 1M specification. pyFAI also exposes detector
definitions and masks for PILATUS, EIGER, PerkinElmer, and other beamline
detectors, which makes those footprints a good target for future calibration
parity.

The detector footprint is stored in artifact metadata and converted into
`artifact_assessment` regions so downstream models can know which pixels were
structurally unavailable, which were only downweighted, and which were still
usable for Bragg information.

## Label Contract

Every artifacted sample writes a `labels.json` payload with:

- `artifact`: the selected profile, random seed, and operation metadata.
- `artifact_assessment`: compact q-space and pixel-space regions for direct
  beam, specular reflection, Yoneda band, substrate horizon, footprint
  spillage, critical-angle splitting, detector gaps, beamstop, and hot/dead
  pixels.
- `quality_assessment`: signal-to-noise, retrievable-signal fraction, usable
  weighted area, saturation fraction, and clean/artifact overlap.

The current quality gate marks an augmented image as solvable when it keeps:

- signal-to-noise at or above 2.5,
- retrievable-signal fraction at or above 0.45,
- usable weighted area at or above 0.65,
- saturation fraction at or below 0.18,
- weighted clean/artifact overlap at or above 0.25.

These are scaffold thresholds, not final scientific limits. They keep normal
training shards realistic and reserve harsh profiles for bounded stress tests.

## Ranking Model

The baseline ranker is a transparent vector-overlap model. An observed artifact
image is converted into a normalized state

```math
\left|\psi\right\rangle
  =
  \frac{M W_\alpha I_{\mathrm{obs}}}
       {\lVert M W_\alpha I_{\mathrm{obs}}\rVert_2},
```

where `M` is the valid-pixel mask and `W_alpha` is the artifact-derived weight
map. A clean simulation for structure `j` under condition `c` is converted to

```math
\left|\phi_{j,c}\right\rangle
  =
  \frac{M W_\alpha I_{\mathrm{sim}}(S_j, c)}
       {\lVert M W_\alpha I_{\mathrm{sim}}(S_j, c)\rVert_2}.
```

The score is the normalized overlap:

```math
s(j,c) =
\left\langle \phi_{j,c} \middle| \psi \right\rangle.
```

EWALD also stores the unweighted overlap and blends it with the artifact-aware
overlap:

```math
s_\gamma(j,c)
  =
  (1-\gamma)
  \left\langle \phi_{j,c} \middle| \psi \right\rangle_{\mathrm{full}}
  +
  \gamma
  \left\langle \phi_{j,c} \middle| \psi \right\rangle_{W_\alpha}.
```

The default `gamma` is exposed as `--artifact-weight-fraction`. This prevents
the ranker from becoming blind to Bragg-rich regions that happen to overlap
moderate surface artifacts, while still teaching the evaluation path to route
around beamstop, detector gaps, direct beam, and horizon-dominated pixels.

## Learned Model Plan

The future learned model should keep the same manifest and label contract while
replacing the baseline overlap score with a staged architecture:

1. **Artifact segmentation and quality head**
   Learns direct/specular beam, Yoneda, horizon, footprint, detector-mask, and
   hot/dead-pixel regions from `artifact_assessment`.
2. **Peak detection head**
   Detects Bragg spots/arcs in q-space while using artifact maps as auxiliary
   supervision.
3. **Peak indexing head**
   Predicts `(hkl)` assignments, reflection families, or reciprocal basis
   vectors from detected peak clouds and image crops.
4. **Structure retrieval head**
   Ranks catalog structures using image embeddings, peak-table embeddings, and
   artifact-aware confidence.
5. **Rerank/solve head**
   Runs high-resolution candidate simulations and reports residuals, score
   gaps, likely mixtures, and confidence.

A useful training objective is:

```math
\mathcal{L}
  =
  \mathcal{L}_{\mathrm{rank}}
  + \lambda_p \mathcal{L}_{\mathrm{peak}}
  + \lambda_h \mathcal{L}_{\mathrm{hkl}}
  + \lambda_a \mathcal{L}_{\mathrm{artifact}}
  + \lambda_c \mathcal{L}_{\mathrm{calibration}}
  + \lambda_q \mathcal{L}_{\mathrm{quality}}.
```

The artifact term uses the known synthetic artifact labels, while the quality
term discourages confident predictions on images that the scaffold marks as
unrecoverable. The rank term should be reported with top-1/top-5 accuracy,
score gaps, structure-family confusion, and artifact-stratified metrics.

## Practical Guidance

- Keep most artifact profiles physically recoverable; use harsh profiles as a
  small, labeled stress-test tail.
- Store incident angle, wavelength, critical-angle source, detector footprint,
  footprint spillage, and quality metrics with every sample.
- Evaluate on both clean and artifacted images, then stratify by Yoneda,
  horizon, detector mask, and direct-beam severity.
- Preserve the baseline vector ranker as an audit tool even after adding neural
  retrieval. It gives a simple sanity check for whether a learned model is
  improving the physics problem or only learning dataset shortcuts.

## References

- pyFAI Fiber/Grazing-Incidence tutorial, including qIP/qOOP mapping and the
  missing-wedge warning:
  <https://pyfai.readthedocs.io/en/stable/usage/tutorial/FiberGrazingIncidence.html>
- Ashiotis, G. et al. "The fast azimuthal integration Python library: pyFAI."
  Journal of Applied Crystallography 48, 510-519 (2015). DOI:
  <https://doi.org/10.1107/S1600576715004306>
- Steele, J. A. et al. "How to GIWAXS: Grazing Incidence Wide Angle X-Ray
  Scattering Applied to Metal Halide Perovskite Thin Films." Advanced Energy
  Materials 13, 2300760 (2023). DOI:
  <https://doi.org/10.1002/aenm.202300760>
- Yoneda, Y. "Anomalous Surface Reflection of X Rays." Physical Review 131,
  2010-2013 (1963). DOI:
  <https://doi.org/10.1103/PhysRev.131.2010>
- Vineyard, G. H. "Grazing-incidence diffraction and the distorted-wave
  approximation for the study of surfaces." Physical Review B 26, 4146-4159
  (1982). DOI: <https://doi.org/10.1103/PhysRevB.26.4146>
- Sinha, S. K., Sirota, E. B., Garoff, S., and Stanley, H. B. "X-ray and
  neutron scattering from rough surfaces." Physical Review B 38, 2297-2311
  (1988). DOI: <https://doi.org/10.1103/PhysRevB.38.2297>
- Baker, J. L. et al. "Quantification of Thin Film Crystallographic Orientation
  Using X-ray Diffraction with an Area Detector." Langmuir 26, 9146-9151
  (2010). DOI: <https://doi.org/10.1021/la904840q>
- Savikhin, V. et al. "GIWAXS-SIIRkit: scattering intensity, indexing and
  refraction calculation toolkit for grazing-incidence wide-angle X-ray
  scattering of organic materials." Journal of Applied Crystallography 53
  (2020). DOI: <https://doi.org/10.1107/S1600576720005476>
- Simbrunner, J. et al. "Indexing of grazing-incidence X-ray diffraction
  patterns: the case of fibre-textured thin films." Acta Crystallographica
  Section A 74, 373-387 (2018). DOI:
  <https://doi.org/10.1107/S2053273318006629>
- Hailey, A. K. et al. "The Diffraction Pattern Calculator (DPC) toolkit: a
  user-friendly approach to unit-cell lattice parameter identification of
  two-dimensional grazing-incidence wide-angle X-ray scattering data." Journal
  of Applied Crystallography 47, 2090-2099 (2014). DOI:
  <https://doi.org/10.1107/S1600576714022006>
- DECTRIS PILATUS3 R 1M Technical Specification, detector pixel size, module
  layout, and intermodule gaps:
  <https://media.dectris.com/filer_public/8f/e6/8fe610b7-327a-4cd6-ade2-992b835af1eb/technicalspecification_pilatus3_r_1m.pdf>
- pyFAI detector definitions, including PILATUS detector masks and module gaps:
  <https://www.silx.org/doc/pyFAI/latest/api/detectors/index.html>
- Broennimann, C. et al. "The PILATUS 1M detector." Journal of Synchrotron
  Radiation 13, 120-130 (2006):
  <https://journals.iucr.org/s/issues/2006/02/00/gf0003/>
- Zheng, K. et al. "CrystalX: High-Accuracy Crystal Structure Analysis Using
  Deep Learning." Journal of the American Chemical Society, published online
  April 20, 2026. DOI: <https://doi.org/10.1021/jacs.5c21832>
