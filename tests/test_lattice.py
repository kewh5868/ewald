"""Crystallography framework tests."""

import numpy as np
import pytest

from ewald.crystallography import (
    CrystalOverlayCalculator,
    CrystalOverlayParameters,
    Lattice,
)
from ewald.crystallography.overlay import (
    compose_quaternions,
    euler_angles_from_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_euler_angles,
)


def test_cubic_lattice_reciprocal_magnitude():
    reciprocal = Lattice(a=10.0, b=10.0, c=10.0).reciprocal()

    assert np.isclose(reciprocal.q_magnitude((1, 0, 0)), 2 * np.pi / 10.0)


def test_crystal_overlay_projects_hkl_with_quaternion_rotation():
    calculator = CrystalOverlayCalculator()
    base = CrystalOverlayParameters(
        crystal_system="Cubic",
        a=10.0,
        b=10.0,
        c=10.0,
        h_max=1,
        k_max=0,
        l_max=0,
        positive_qz_only=False,
    )

    unrotated = calculator.project(base)
    assert len(unrotated.hkl) == 2
    assert np.any(np.isclose(np.abs(unrotated.qxy), 2 * np.pi / 10.0))

    rotated = calculator.project(
        CrystalOverlayParameters.from_dict(
            {
                **base.as_dict(),
                "orientation_quaternion": quaternion_from_axis_angle(
                    (0.0, 1.0, 0.0),
                    90.0,
                ),
            }
        )
    )

    assert np.any(np.isclose(np.abs(rotated.qz), 2 * np.pi / 10.0))


def test_quaternion_composition_stays_normalized():
    quaternion = compose_quaternions(
        quaternion_from_axis_angle((1.0, 0.0, 0.0), 25.0),
        quaternion_from_axis_angle((0.0, 1.0, 0.0), 15.0),
    )

    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_euler_orientation_helpers_round_trip():
    angles = (25.0, -40.0, 75.0)
    quaternion = quaternion_from_euler_angles(*angles)

    assert euler_angles_from_quaternion(quaternion) == pytest.approx(angles)
