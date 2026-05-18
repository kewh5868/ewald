"""Crystal overlay calculations for q-space peak identification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from ewald.crystallography.lattice import Lattice

CRYSTAL_SYSTEMS: dict[str, dict[str, Any]] = {
    "Triclinic": {"disabled": []},
    "Monoclinic": {
        "disabled": ["alpha", "gamma"],
        "alpha": 90.0,
        "gamma": 90.0,
    },
    "Orthorhombic": {
        "disabled": ["alpha", "beta", "gamma"],
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    },
    "Tetragonal": {
        "disabled": ["b", "alpha", "beta", "gamma"],
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    },
    "Trigonal": {
        "disabled": ["b", "alpha", "beta", "gamma"],
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
    },
    "Hexagonal": {
        "disabled": ["b", "alpha", "beta", "gamma"],
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
    },
    "Cubic": {
        "disabled": ["b", "c", "alpha", "beta", "gamma"],
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    },
}


@dataclass(slots=True)
class CrystalOverlayParameters:
    """Parameters used to project reciprocal-lattice peaks onto
    q_{xy}/q_{z}."""

    crystal_system: str = "Triclinic"
    a: float = 6.3
    b: float = 6.3
    c: float = 6.3
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    h_max: int = 3
    k_max: int = 3
    l_max: int = 3
    orientation_quaternion: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    positive_qz_only: bool = True

    def constrained(self) -> "CrystalOverlayParameters":
        """Return a copy with crystal-system constraints applied."""

        values = self.as_dict()
        apply_crystal_system_constraints(values)
        values["orientation_quaternion"] = tuple(
            normalize_quaternion(values["orientation_quaternion"])
        )
        return CrystalOverlayParameters.from_dict(values)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["orientation_quaternion"] = list(self.orientation_quaternion)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CrystalOverlayParameters":
        quaternion = payload.get(
            "orientation_quaternion", (0.0, 0.0, 0.0, 1.0)
        )
        return cls(
            crystal_system=str(payload.get("crystal_system", "Triclinic")),
            a=float(payload.get("a", 6.3)),
            b=float(payload.get("b", payload.get("a", 6.3))),
            c=float(payload.get("c", payload.get("a", 6.3))),
            alpha=float(payload.get("alpha", 90.0)),
            beta=float(payload.get("beta", 90.0)),
            gamma=float(payload.get("gamma", 90.0)),
            h_max=int(payload.get("h_max", 3)),
            k_max=int(payload.get("k_max", 3)),
            l_max=int(payload.get("l_max", 3)),
            orientation_quaternion=tuple(
                float(value) for value in normalize_quaternion(quaternion)
            ),
            positive_qz_only=bool(payload.get("positive_qz_only", True)),
        )


@dataclass(slots=True)
class CrystalOverlayResult:
    """Projected Bragg peak and unit-cell geometry for an overlay
    state."""

    qxy: np.ndarray
    qz: np.ndarray
    hkl: np.ndarray
    q_vectors: np.ndarray
    cell_corners: np.ndarray
    cell_edges: list[tuple[int, int]]

    def peak_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        count = len(self.hkl) if limit is None else min(len(self.hkl), limit)
        for index in range(count):
            h, k, ell = (int(value) for value in self.hkl[index])
            rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": ell,
                    "qxy": float(self.qxy[index]),
                    "qz": float(self.qz[index]),
                }
            )
        return rows


class CrystalOverlayCalculator:
    """Cache direct and reciprocal lattice arrays for responsive
    overlays."""

    def __init__(
        self, parameters: CrystalOverlayParameters | None = None
    ) -> None:
        self._signature: tuple[Any, ...] | None = None
        self._hkl = np.empty((0, 3), dtype=int)
        self._q_vectors = np.empty((0, 3), dtype=float)
        self._cell_corners = np.empty((0, 3), dtype=float)
        if parameters is not None:
            self.set_parameters(parameters)

    def set_parameters(self, parameters: CrystalOverlayParameters) -> None:
        params = parameters.constrained()
        signature = (
            params.a,
            params.b,
            params.c,
            params.alpha,
            params.beta,
            params.gamma,
            params.h_max,
            params.k_max,
            params.l_max,
        )
        if signature == self._signature:
            return
        self._signature = signature
        lattice = Lattice(
            a=params.a,
            b=params.b,
            c=params.c,
            alpha=params.alpha,
            beta=params.beta,
            gamma=params.gamma,
        )
        reciprocal = lattice.reciprocal()
        self._hkl = _hkl_grid(params.h_max, params.k_max, params.l_max)
        if len(self._hkl):
            self._q_vectors = np.vstack(
                [reciprocal.q_vector(row) for row in self._hkl]
            )
        else:
            self._q_vectors = np.empty((0, 3), dtype=float)
        self._cell_corners = _cell_corners(lattice.vectors())

    def project(
        self, parameters: CrystalOverlayParameters
    ) -> CrystalOverlayResult:
        params = parameters.constrained()
        self.set_parameters(params)
        quaternion = normalize_quaternion(params.orientation_quaternion)
        rotated_q = rotate_vectors_by_quaternion(self._q_vectors, quaternion)
        rotated_cell = rotate_vectors_by_quaternion(
            self._cell_corners, quaternion
        )
        qxy = rotated_q[:, 0]
        qz = rotated_q[:, 2]
        hkl = self._hkl
        q_vectors = rotated_q
        if params.positive_qz_only:
            keep = qz >= 0.0
            qxy = qxy[keep]
            qz = qz[keep]
            hkl = hkl[keep]
            q_vectors = q_vectors[keep]
        return CrystalOverlayResult(
            qxy=qxy,
            qz=qz,
            hkl=hkl,
            q_vectors=q_vectors,
            cell_corners=rotated_cell,
            cell_edges=_cell_edges(),
        )


def apply_crystal_system_constraints(values: dict[str, Any]) -> dict[str, Any]:
    """Mutate a lattice-parameter mapping according to its crystal
    system."""

    system = str(values.get("crystal_system", "Triclinic"))
    rules = CRYSTAL_SYSTEMS.get(system, CRYSTAL_SYSTEMS["Triclinic"])
    if "b" in rules.get("disabled", []):
        values["b"] = values.get("a", values.get("b", 1.0))
    if "c" in rules.get("disabled", []):
        values["c"] = values.get("a", values.get("c", 1.0))
    for angle in ("alpha", "beta", "gamma"):
        if angle in rules:
            values[angle] = rules[angle]
    return values


def quaternion_from_axis_angle(
    axis: Iterable[float],
    angle_degrees: float,
) -> tuple[float, float, float, float]:
    """Return an ``x, y, z, w`` quaternion for an axis-angle
    rotation."""

    vector = np.asarray(tuple(axis), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    unit = vector / norm
    half_angle = np.deg2rad(angle_degrees) / 2.0
    xyz = unit * np.sin(half_angle)
    return normalize_quaternion((*xyz, np.cos(half_angle)))


def compose_quaternions(
    current_xyzw: Iterable[float],
    delta_xyzw: Iterable[float],
) -> tuple[float, float, float, float]:
    """Apply ``delta`` after ``current`` and return a normalized
    quaternion."""

    x1, y1, z1, w1 = normalize_quaternion(delta_xyzw)
    x2, y2, z2, w2 = normalize_quaternion(current_xyzw)
    return normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def quaternion_from_euler_angles(
    x_degrees: float,
    y_degrees: float,
    z_degrees: float,
) -> tuple[float, float, float, float]:
    """Return an ``x, y, z, w`` quaternion from fixed-axis XYZ
    angles."""

    quaternion = (0.0, 0.0, 0.0, 1.0)
    for axis, angle in (
        ((1.0, 0.0, 0.0), x_degrees),
        ((0.0, 1.0, 0.0), y_degrees),
        ((0.0, 0.0, 1.0), z_degrees),
    ):
        quaternion = compose_quaternions(
            quaternion,
            quaternion_from_axis_angle(axis, angle),
        )
    return quaternion


def euler_angles_from_quaternion(
    quaternion_xyzw: Iterable[float],
) -> tuple[float, float, float]:
    """Return fixed-axis XYZ angles in degrees for a quaternion."""

    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )
    y_angle = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    cos_y = np.cos(y_angle)
    if abs(cos_y) > 1.0e-8:
        x_angle = np.arctan2(matrix[2, 1], matrix[2, 2])
        z_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x_angle = 0.0
        z_angle = np.arctan2(-matrix[0, 1], matrix[1, 1])
    return tuple(
        float(np.degrees(angle)) for angle in (x_angle, y_angle, z_angle)
    )


def normalize_quaternion(
    quaternion: Iterable[float],
) -> tuple[float, float, float, float]:
    values = np.asarray(tuple(quaternion), dtype=float)
    if values.size != 4 or not np.isfinite(values).all():
        return 0.0, 0.0, 0.0, 1.0
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    values = values / norm
    return tuple(float(value) for value in values)


def rotate_vectors_by_quaternion(
    vectors: np.ndarray,
    quaternion_xyzw: Iterable[float],
) -> np.ndarray:
    """Rotate row-vector coordinates without Euler-angle
    singularities."""

    array = np.asarray(vectors, dtype=float)
    if array.size == 0:
        return array.reshape((-1, 3))
    qx, qy, qz, qw = normalize_quaternion(quaternion_xyzw)
    qvec = np.array([qx, qy, qz], dtype=float)
    uv = np.cross(qvec, array)
    uuv = np.cross(qvec, uv)
    return array + 2.0 * (qw * uv + uuv)


def _hkl_grid(h_max: int, k_max: int, l_max: int) -> np.ndarray:
    h_range = np.arange(-max(0, int(h_max)), max(0, int(h_max)) + 1)
    k_range = np.arange(-max(0, int(k_max)), max(0, int(k_max)) + 1)
    l_range = np.arange(-max(0, int(l_max)), max(0, int(l_max)) + 1)
    h_grid, k_grid, l_grid = np.meshgrid(
        h_range,
        k_range,
        l_range,
        indexing="ij",
    )
    hkl = np.column_stack(
        (h_grid.ravel(), k_grid.ravel(), l_grid.ravel())
    ).astype(int)
    return hkl[np.any(hkl != 0, axis=1)]


def _cell_corners(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    a_vec, b_vec, c_vec = vectors
    origin = np.zeros(3)
    return np.vstack(
        [
            origin,
            a_vec,
            b_vec,
            c_vec,
            a_vec + b_vec,
            a_vec + c_vec,
            b_vec + c_vec,
            a_vec + b_vec + c_vec,
        ]
    )


def _cell_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
