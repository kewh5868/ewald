"""Tests for automated peak detection helpers."""

import numpy as np

from ewald.processing.peak_detection import (
    LocalMaxPeakFinderConfig,
    find_local_maxima_peaks,
)


def test_local_maxima_peak_finder_uses_axes_and_valid_mask():
    image = np.zeros((7, 7), dtype=float)
    image[2, 2] = 10.0
    image[2, 3] = 9.5
    image[5, 5] = 20.0
    valid_mask = np.ones_like(image, dtype=bool)
    valid_mask[5, 5] = False
    x_axis = np.linspace(-1.0, 1.0, image.shape[1])
    y_axis = np.linspace(-2.0, 2.0, image.shape[0])

    peaks = find_local_maxima_peaks(
        image,
        x_axis=x_axis,
        y_axis=y_axis,
        valid_mask=valid_mask,
        config=LocalMaxPeakFinderConfig(
            threshold_percentile=50.0,
            max_peaks=10,
            min_distance_px=2,
            neighborhood_radius_px=1,
        ),
    )

    assert len(peaks) == 1
    assert peaks[0].x == x_axis[2]
    assert peaks[0].y == y_axis[2]
    assert peaks[0].intensity == 10.0


def test_adaptive_peak_finder_recovers_weaker_local_peaks():
    rng = np.random.default_rng(42)
    y_grid, x_grid = np.mgrid[:80, :80]
    background = 40.0 + 0.22 * x_grid + 0.12 * y_grid
    image = background + rng.normal(0.0, 1.0, background.shape)
    image[20, 20] += 180.0
    image[58, 55] += 8.0

    global_only = find_local_maxima_peaks(
        image,
        config=LocalMaxPeakFinderConfig(
            threshold_percentile=99.9,
            adaptive_threshold=False,
            max_peaks=10,
            min_distance_px=5,
            neighborhood_radius_px=1,
        ),
    )
    adaptive = find_local_maxima_peaks(
        image,
        config=LocalMaxPeakFinderConfig(
            threshold_percentile=99.9,
            adaptive_threshold=True,
            adaptive_floor_percentile=90.0,
            min_snr=3.0,
            background_radius_px=7,
            max_peaks=10,
            min_distance_px=5,
            neighborhood_radius_px=1,
        ),
    )

    global_positions = {(round(peak.x), round(peak.y)) for peak in global_only}
    adaptive_positions = {(round(peak.x), round(peak.y)) for peak in adaptive}

    assert (20, 20) in global_positions
    assert (55, 58) not in global_positions
    assert {(20, 20), (55, 58)} <= adaptive_positions
    weak_peak = next(
        peak
        for peak in adaptive
        if round(peak.x) == 55 and round(peak.y) == 58
    )
    assert weak_peak.snr is not None
    assert weak_peak.snr > 3.0
