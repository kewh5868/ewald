"""Image orientation helpers shared by preview and correction panes."""

from __future__ import annotations


def normalize_rotation_degrees(rotation_deg: int) -> int:
    """Return the nearest supported clockwise rotation in degrees."""

    return (round(rotation_deg / 90) * 90) % 360


def sample_orientation_for_image_transform(
    rotation_deg: int,
    *,
    mirrored_y: bool = False,
) -> int:
    """Return the pyFAI sample orientation for the preview transform."""

    rotation = normalize_rotation_degrees(rotation_deg)
    if mirrored_y:
        return {
            0: 2,
            90: 5,
            180: 4,
            270: 7,
        }.get(rotation, 2)
    return {
        0: 1,
        90: 8,
        180: 3,
        270: 6,
    }.get(rotation, 1)
