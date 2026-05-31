"""Vector-space ranking utilities for structure recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(slots=True)
class CandidateScore:
    """A ranked structure/condition candidate."""

    candidate_id: str
    score: float
    coefficient: float
    rank: int = 0
    metadata: dict[str, object] | None = None


def normalize_state(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    center: bool = True,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return a flattened normalized image ket ``|psi>``."""

    state = np.asarray(image, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError("normalize_state expects a 2D image.")
    if mask is not None:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != state.shape:
            raise ValueError("mask shape must match image shape.")
        state = np.where(mask_values, state, 0.0)
    if weights is not None:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != state.shape:
            raise ValueError("weights shape must match image shape.")
        state = state * np.clip(
            np.nan_to_num(weight_values, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            None,
        )
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    vector = state.ravel()
    if center:
        vector = vector - float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector)
    return vector / norm


def overlap_score(
    observed: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Return ``<candidate|observed>`` for normalized image states."""

    obs = normalize_state(observed, mask=mask, weights=weights)
    ref = normalize_state(candidate, mask=mask, weights=weights)
    return float(np.dot(ref, obs))


def rank_image_candidates(
    observed: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    artifact_weight_fraction: float = 1.0,
) -> list[CandidateScore]:
    """Rank candidate images by normalized vector overlap."""

    weight_fraction = float(np.clip(artifact_weight_fraction, 0.0, 1.0))
    observed_state = normalize_state(observed, mask=mask, weights=weights)
    observed_unweighted_state = None
    if weights is not None and weight_fraction < 1.0:
        observed_unweighted_state = normalize_state(observed, mask=mask)
    rows: list[CandidateScore] = []
    for candidate_id, image in candidates.items():
        candidate_state = normalize_state(image, mask=mask, weights=weights)
        weighted_coefficient = float(np.dot(candidate_state, observed_state))
        coefficient = weighted_coefficient
        metadata: dict[str, object] | None = None
        if observed_unweighted_state is not None:
            candidate_unweighted_state = normalize_state(image, mask=mask)
            unweighted_coefficient = float(
                np.dot(candidate_unweighted_state, observed_unweighted_state)
            )
            coefficient = (
                weight_fraction * weighted_coefficient
                + (1.0 - weight_fraction) * unweighted_coefficient
            )
            metadata = {
                "artifact_weight_fraction": weight_fraction,
                "weighted_coefficient": weighted_coefficient,
                "unweighted_coefficient": unweighted_coefficient,
            }
        rows.append(
            CandidateScore(
                candidate_id=candidate_id,
                score=coefficient,
                coefficient=coefficient,
                metadata=metadata,
            )
        )
    rows.sort(key=lambda item: item.score, reverse=True)
    for index, row in enumerate(rows, start=1):
        row.rank = index
    return rows


def subtract_component(
    observed: np.ndarray,
    basis: np.ndarray,
    *,
    coefficient: float | None = None,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract ``c_j |phi_j>`` from an observed state image."""

    observed_state = normalize_state(observed, mask=mask, weights=weights)
    basis_state = normalize_state(basis, mask=mask, weights=weights)
    coeff = float(np.dot(basis_state, observed_state))
    if coefficient is not None:
        coeff = float(coefficient)
    residual = observed_state - coeff * basis_state
    return residual.reshape(np.asarray(observed).shape)


def peak_table_vector(
    peaks: Iterable[Mapping[str, object]],
    *,
    qxy_range: tuple[float, float],
    qz_range: tuple[float, float],
    bins: tuple[int, int] = (256, 128),
    intensity_key: str = "amplitude",
) -> np.ndarray:
    """Rasterize labeled peaks into a sparse reciprocal-space state."""

    qxy_min, qxy_max = qxy_range
    qz_min, qz_max = qz_range
    x_bins, z_bins = bins
    image = np.zeros((z_bins, x_bins), dtype=np.float32)
    for peak in peaks:
        if _peak_excluded_from_indexing(peak):
            continue
        try:
            qxy = float(peak["qxy"])
            qz = float(peak["qz"])
        except (KeyError, TypeError, ValueError):
            continue
        if qxy < qxy_min or qxy > qxy_max:
            continue
        if qz < qz_min or qz > qz_max:
            continue
        x = int((qxy - qxy_min) / (qxy_max - qxy_min) * (x_bins - 1))
        z = int((qz - qz_min) / (qz_max - qz_min) * (z_bins - 1))
        intensity = float(peak.get(intensity_key, peak.get("intensity", 1.0)))
        image[z, x] += max(0.0, intensity)
    return image


def _peak_excluded_from_indexing(peak: Mapping[str, object]) -> bool:
    return bool(
        peak.get("forbidden_reflection")
        or peak.get("excluded_from_indexing")
        or str(peak.get("reflection_status", "")).lower() == "forbidden"
    )
