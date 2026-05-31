# Dirac-Notation Structure Ranking

This note translates Zihan Zhang's slide-deck vector matching idea into an
EWALD ranking contract.

## Image States

Let an experimental GIWAXS image be a normalized state

```math
\ket{\psi}
  =
  \frac{M W I_{\mathrm{exp}}}
       {\lVert M W I_{\mathrm{exp}} \rVert_2}
```

where `I_exp` is the corrected reciprocal-space image, `M` is the valid-pixel
mask, and `W` is an uncertainty or physics weight. In the training scaffold,
`W(q_{xy}, q_z)` is generated from `artifact_assessment` labels so direct beam,
specular reflection, Yoneda, substrate horizon, footprint-spillage,
critical-angle, beamstop, and detector-gap regions are downweighted before the
image is compared with clean Bragg simulations. For a catalog structure `S_j`
under simulation condition `c`, define a simulated standard

```math
\ket{\phi_{j,c}}
  =
  \frac{M W I_{\mathrm{sim}}(S_j, c)}
       {\lVert M W I_{\mathrm{sim}}(S_j, c) \rVert_2}.
```

The simplest recognition score is the overlap

```math
s(j,c) = \braket{\phi_{j,c} | \psi}.
```

For artifact-aware feedback, EWALD also computes a weighted overlap and blends
the two scores:

```math
s_\alpha(j,c)
  =
  (1-\gamma)\braket{\phi_{j,c} | \psi}
  +
  \gamma\braket{W_\alpha\phi_{j,c} | W_\alpha\psi},
```

where `0 <= gamma <= 1` is `--artifact-weight-fraction`. The default keeps the
baseline anchored to full-image Bragg similarity while still exposing
artifact-aware residuals and peak-assessment weights.

The best structure is

```math
j^\star = \arg\max_j \max_c s(j,c).
```

This is implemented in `ewald_data_training.ranking.rank_image_candidates()`.

## Forward Model

For a structure state `\ket{S_j}`, a GIWAXS image can be written as

```math
\ket{I_{j,c,\alpha}}
  =
  A_\alpha D_g P T_c \ket{S_j}.
```

The operators are:

- `T_c`: orientation and texture operator, such as fibril Gaussian mosaicity.
- `P`: projection from reciprocal-lattice Bragg peaks onto a 2D GIWAXS map.
- `D_g`: detector geometry, q-space sampling, masks, and corrections.
- `A_alpha`: stochastic artifact operator, including direct/specular beam
  intensity, Yoneda scattering, substrate horizon/footprint effects,
  critical-angle peak splitting, noise, beamstop, gaps, backgrounds,
  hot/dead pixels, and saturation.
- `W_alpha`: artifact-assessment weight operator derived from the known
  simulation metadata for `A_alpha`.

During dataset generation, EWALD stores both the clean state
`D_g P T_c \ket{S_j}` and the observed training state
`A_alpha D_g P T_c \ket{S_j}`. It also stores the compact
`artifact_assessment` metadata needed to construct `W_alpha`, allowing the
baseline ranker and future peak-indexing models to learn which pixels or peaks
are likely non-Bragg aberrations.

## Peak-Table States

Images are not the only useful states. A labeled peak table can be rasterized or
embedded as

```math
\ket{p}
  =
  \sum_n a_n \ket{q_{xy,n}, q_{z,n}, h_n, k_n, l_n},
```

where `a_n` may be Bragg intensity, rendered amplitude, fitted peak area, or a
confidence-weighted intensity. Peak-table states are useful for fast retrieval
because they ignore most detector pixels and focus on indexed reflections.

The scaffold includes `peak_table_vector()` as a simple raster baseline. A
future learned model should compare:

- image-only ranking,
- detected-peak ranking,
- true-peak oracle ranking,
- hybrid image-plus-peak ranking.

## Mixtures and Residual Search

For mixed or multilayer samples, rank the strongest component first, subtract
its projection, then search the residual:

```math
c_j = \braket{\phi_j | \psi}
```

```math
\ket{\psi_1}
  =
  \ket{\psi} - c_j \ket{\phi_j}.
```

Repeat ranking on `\ket{\psi_1}` until the residual norm or confidence falls
below a threshold. This gives a physics-readable route to mixture decomposition:
each accepted component has a structure id, condition id, coefficient, and
residual contribution.

## Training Losses

For supervised learning, a neural ranker can be trained to preserve this overlap
ordering while becoming robust to missing peaks and artifacts:

```math
\mathcal{L}
  =
  \mathcal{L}_{\mathrm{contrastive}}
  +
  \lambda_1 \mathcal{L}_{\mathrm{peak}}
  +
  \lambda_2 \mathcal{L}_{\mathrm{hkl}}
  +
  \lambda_3 \mathcal{L}_{\mathrm{calibration}}.
```

Suggested terms:

- `L_contrastive`: correct structure/condition states rank above negatives.
- `L_peak`: peak center and segmentation loss in q-space.
- `L_hkl`: reflection-index classification or set-matching loss.
- `L_calibration`: penalty for q-space drift and detector geometry mismatch.

## Confidence

Return top-k structures with score gaps, not only the top hit:

```math
\Delta_k = s_1 - s_k.
```

A confident prediction has high absolute overlap, a large score gap, stable
rank under artifact dropout, and a low residual after high-resolution reranking.
Low-confidence outputs should name likely causes: weak peaks, strong masking,
ambiguous related structures, mixed phases, or calibration drift.
