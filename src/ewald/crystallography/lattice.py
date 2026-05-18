"""Lattice and reciprocal-space calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Lattice:
    """Crystallographic unit-cell parameters."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = np.deg2rad(self.alpha)
        beta = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)
        volume = (
            self.a
            * self.b
            * self.c
            * np.sqrt(
                1
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
        )
        a_vec = np.array([self.a, 0.0, 0.0])
        b_vec = np.array([self.b * np.cos(gamma), self.b * np.sin(gamma), 0.0])
        c_vec = np.array(
            [
                self.c * np.cos(beta),
                self.c
                * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                / np.sin(gamma),
                volume / (self.a * self.b * np.sin(gamma)),
            ]
        )
        return a_vec, b_vec, c_vec

    def reciprocal(self) -> "ReciprocalLattice":
        return ReciprocalLattice.from_lattice(self)


@dataclass(frozen=True, slots=True)
class ReciprocalLattice:
    """Reciprocal-space lattice vectors."""

    a_star: np.ndarray
    b_star: np.ndarray
    c_star: np.ndarray

    @classmethod
    def from_lattice(cls, lattice: Lattice) -> "ReciprocalLattice":
        a_vec, b_vec, c_vec = lattice.vectors()
        volume = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
        factor = 2 * np.pi / volume
        return cls(
            a_star=factor * np.cross(b_vec, c_vec),
            b_star=factor * np.cross(c_vec, a_vec),
            c_star=factor * np.cross(a_vec, b_vec),
        )

    def q_vector(self, hkl: Iterable[int]) -> np.ndarray:
        h, k, ell = hkl
        return h * self.a_star + k * self.b_star + ell * self.c_star

    def q_magnitude(self, hkl: Iterable[int]) -> float:
        return float(np.linalg.norm(self.q_vector(hkl)))
