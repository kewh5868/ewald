# Mathematical Foundations

EWALD is built around a small set of reciprocal-space conventions. This page
collects the derivations used to interpret detector coordinates, ROI
integrations, peak fits, structure candidates, film-optics inputs, and GIWAXS
simulation outputs.

The equations below document the working model and notation. Exact numerical
results still depend on the calibration file, detector geometry, image
orientation, masks, and correction choices stored in the active `.ewld`
project.

## Scattering Vector

For elastic X-ray scattering, the incident and exit wavevectors have the same
magnitude:

\[
  |\mathbf{k_i}| = |\mathbf{k_f}| = k = \frac{2\pi}{\lambda}
\]

where \(\lambda\) is the X-ray wavelength. The momentum transfer is

\[
  \mathbf{q} = \mathbf{k_f} - \mathbf{k_i}.
\]

If \(2\theta\) is the angle between incident and scattered beams, the magnitude
is

\[
  q = |\mathbf{q}| = 2k\sin\theta
    = \frac{4\pi}{\lambda}\sin\theta.
\]

This is the source of the common conversion between scattering angle and
reciprocal-space radius. EWALD displays reciprocal-space axes in
\(\text{\AA}^{-1}\) when the calibration supplies length and wavelength
metadata.

## Detector Pixels To Reciprocal Space

A detector pixel is first converted into a ray direction in the laboratory
frame. For an idealized flat detector with pixel coordinate \((u, v)\), beam
center \((u_0, v_0)\), pixel sizes \(p_u, p_v\), and sample-detector distance
\(D\), an unrotated detector ray can be written as

\[
  \mathbf{r}(u, v) =
  \begin{bmatrix}
    (u - u_0)p_u \\
    (v - v_0)p_v \\
    D
  \end{bmatrix}.
\]

Detector tilts and pyFAI orientation metadata are applied as rotations to this
vector. The normalized exit direction is

\[
  \hat{\mathbf{s}}_f =
  \frac{\mathbf{R}_{det}\mathbf{r}}{|\mathbf{R}_{det}\mathbf{r}|},
  \qquad
  \mathbf{k_f} = \frac{2\pi}{\lambda}\hat{\mathbf{s}}_f.
\]

With the incident beam direction \(\hat{\mathbf{s}}_i\),

\[
  \mathbf{q}(u, v) =
  \frac{2\pi}{\lambda}
  \left(\hat{\mathbf{s}}_f(u, v) - \hat{\mathbf{s}}_i\right).
\]

EWALD keeps image rotation and mirror settings separate from detector geometry
so users can correct storage/display orientation before accepting the calibrated
mapping.

## GIWAXS Axes

In grazing-incidence workflows, the laboratory axes are usually chosen so the
sample surface defines the in-plane directions and the surface normal defines
the out-of-plane direction. EWALD uses:

- \(q_z\): out-of-plane component along the sample normal.
- \(q_{\parallel}\): in-plane magnitude,

\[
  q_{\parallel} = \sqrt{q_x^2 + q_y^2}.
\]

For image display and left/right peak separation, EWALD often presents a signed
in-plane coordinate \(q_{\mathrm{xy}}\). Conceptually this is an in-plane
projection that preserves detector-side sign while retaining the physical scale
of \(q_{\parallel}\):

\[
  q_{\mathrm{xy}} \approx \operatorname{sgn}(q_x)q_{\parallel}.
\]

The exact sign convention is controlled by detector orientation, mirroring, and
the selected pyFAI sample orientation. The important practical rule is that
equal-magnitude peaks on opposite detector sides should appear at opposite
signs of \(q_{\mathrm{xy}}\) after orientation is correct.

## Reciprocal Lattice

Let the direct lattice basis vectors be columns of

\[
  \mathbf{A} =
  \begin{bmatrix}
    \mathbf{a} & \mathbf{b} & \mathbf{c}
  \end{bmatrix}.
\]

The physics reciprocal basis, including the \(2\pi\) factor, is

\[
  \mathbf{B} = 2\pi \mathbf{A}^{-T}.
\]

For Miller index vector \(\mathbf{h} = [h, k, l]^T\), the reciprocal-lattice
vector is

\[
  \mathbf{G}_{hkl} = \mathbf{B}\mathbf{h}.
\]

The plane spacing follows from

\[
  d_{hkl} = \frac{2\pi}{|\mathbf{G}_{hkl}|}.
\]

Combining this with the scattering-vector magnitude gives Bragg's law:

\[
  |\mathbf{q}| = |\mathbf{G}_{hkl}| = \frac{2\pi}{d_{hkl}},
  \qquad
  2d_{hkl}\sin\theta = \lambda.
\]

Structure Analysis uses this relationship to compare fitted peak centers
against candidate reciprocal-lattice vectors.

## Crystal Orientation

Candidate structures are rotated into the sample/laboratory frame before
comparison with observed peaks. If \(\mathbf{R}\) is the orientation matrix,
then the predicted reciprocal vector is

\[
  \mathbf{q}_{pred}(h,k,l) = \mathbf{R}\mathbf{G}_{hkl}.
\]

The displayed coordinates are projected from \(\mathbf{q}_{pred}\) to
\((q_{\mathrm{xy}}, q_z)\). A simple residual for one assigned peak is

\[
  \Delta_i =
  \begin{bmatrix}
    q_{\mathrm{xy},i}^{obs} - q_{\mathrm{xy},i}^{pred} \\
    q_{z,i}^{obs} - q_{z,i}^{pred}
  \end{bmatrix}.
\]

Weighted candidate scores are variations on

\[
  \chi^2 =
  \sum_i
  \Delta_i^T
  \mathbf{W}_i
  \Delta_i,
\]

where \(\mathbf{W}_i\) can encode user tolerances, peak confidence, phase tags,
or fit uncertainty.

## ROI Integration

Rectangular ROIs integrate intensity over a bounded region in
\((q_{\mathrm{xy}}, q_z)\). For a horizontal lineout,

\[
  I(q_{\mathrm{xy}}) =
  \int_{q_{z,min}}^{q_{z,max}}
  I(q_{\mathrm{xy}}, q_z)\,dq_z.
\]

For a vertical lineout,

\[
  I(q_z) =
  \int_{q_{\mathrm{xy},min}}^{q_{\mathrm{xy},max}}
  I(q_{\mathrm{xy}}, q_z)\,dq_{\mathrm{xy}}.
\]

Arch ROIs use polar coordinates in the \(q_{\mathrm{xy}}, q_z\) plane:

\[
  q_r = \sqrt{q_{\mathrm{xy}}^2 + q_z^2},
  \qquad
  \chi = \operatorname{atan2}(q_z, q_{\mathrm{xy}}).
\]

An azimuthal trace over an annulus is

\[
  I(\chi) =
  \int_{q_{r,min}}^{q_{r,max}}
  I(q_r, \chi)\,q_r\,dq_r.
\]

The factor \(q_r\) is the polar-coordinate Jacobian. In discrete detector data,
EWALD approximates these integrals by summing finite pixels that fall inside the
ROI bounds after masking and correction.

## Peak Fitting

For one-dimensional traces, a common local model is a Gaussian peak with a
smooth baseline:

\[
  I(x) =
  A\exp\left[-\frac{(x - x_0)^2}{2\sigma^2}\right]
  + b_0 + b_1x.
\]

Here \(x_0\) is the fitted peak center and \(\sigma\) controls width. The full
width at half maximum is

\[
  \operatorname{FWHM} = 2\sqrt{2\ln 2}\,\sigma.
\]

For two-dimensional peak fitting in reciprocal space, the separable Gaussian
form is

\[
  I(q_{\mathrm{xy}}, q_z) =
  A\exp\left[
    -\frac{1}{2}
    \left(
      \frac{(q_{\mathrm{xy}}-\mu_{\mathrm{xy}})^2}{\sigma_{\mathrm{xy}}^2}
      +
      \frac{(q_z-\mu_z)^2}{\sigma_z^2}
    \right)
  \right]
  + B(q_{\mathrm{xy}}, q_z).
\]

The fitted center \((\mu_{\mathrm{xy}}, \mu_z)\) becomes the peak coordinate
passed into Structure Analysis.

## Film Optics

For X-rays in matter, the refractive index is commonly written as

\[
  n = 1 - \delta + i\beta.
\]

For small \(\delta\), the critical angle for total external reflection is

\[
  \alpha_c \approx \sqrt{2\delta}.
\]

The corresponding vertical momentum transfer is

\[
  q_c =
  \frac{4\pi}{\lambda}\sin\alpha_c
  \approx
  \frac{4\pi}{\lambda}\alpha_c.
\]

EWALD stores stoichiometry, density, refractive-index delta, and critical angle
in the correction state so those values can be reviewed alongside the
reciprocal-space transform.

## Structure Factors And Simulation

For a crystal structure with atoms at fractional positions \(\mathbf{r}_j\),
the structure factor for reflection \(hkl\) is

\[
  F_{hkl} =
  \sum_j
  f_j(|\mathbf{G}_{hkl}|)
  \exp\left(2\pi i\,\mathbf{h}\cdot\mathbf{r}_j\right)
  \exp\left(-\frac{B_j|\mathbf{G}_{hkl}|^2}{16\pi^2}\right).
\]

Reflection intensity is proportional to

\[
  I_{hkl} \propto |F_{hkl}|^2
\]

with additional experimental weights from orientation spread, footprint,
polarization, detector response, and masking. In simulation, a rotated
reflection contributes near its predicted detector coordinate:

\[
  I_{sim}(\mathbf{q}) =
  \sum_{hkl}
  |F_{hkl}|^2
  W(\mathbf{R})
  K_{\sigma}\left(\mathbf{q} - \mathbf{R}\mathbf{G}_{hkl}\right),
\]

where \(K_{\sigma}\) is a peak-shape kernel and \(W(\mathbf{R})\) represents the
orientation distribution.

The Ewald-sphere condition can be written as

\[
  |\mathbf{k_i} + \mathbf{G}_{hkl}| = |\mathbf{k_i}|.
\]

Equivalently, an excitation error can be defined by

\[
  \epsilon_{hkl} =
  |\mathbf{k_i} + \mathbf{G}_{hkl}| - |\mathbf{k_i}|.
\]

Reflections with small \(|\epsilon_{hkl}|\) are close to the Ewald sphere and
are more likely to appear in the simulated image.

## Difference Maps

EWALD simulation comparison uses normalized experimental and simulated images.
A simple scale/offset fit is

\[
  I_{exp}(\mathbf{q}) \approx a I_{sim}(\mathbf{q}) + b.
\]

The residual image is

\[
  R(\mathbf{q}) =
  I_{exp}(\mathbf{q}) - \left(a I_{sim}(\mathbf{q}) + b\right).
\]

The root-mean-square residual is

\[
  \operatorname{RMSE} =
  \sqrt{
    \frac{1}{N}
    \sum_{\mathbf{q}\in\Omega}
    R(\mathbf{q})^2
  },
\]

where \(\Omega\) is the valid, finite, unmasked comparison region. A lower RMSE
means the simulated pattern explains more of the corrected experimental target
after normalization.

## Practical Reading Guide

- Use detector and correction settings to make \(q_{\mathrm{xy}}\) symmetry and
  \(q_z\) placement physically plausible before fitting.
- Use ROIs to extract stable local traces rather than fitting broad image
  regions with mixed phases.
- Use hkl labels and phase tags to constrain Structure Analysis to peaks that
  should share one reciprocal lattice.
- Use simulation residuals as an image-level check after lattice-level
  candidate ranking.
