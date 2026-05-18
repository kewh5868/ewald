"""Refinement planning for experimental/simulated GIWAXS matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RefinementPlan:
    """Description of a simulation minimization problem."""

    experimental_data_id: str
    simulation_data_id: str
    objective: str = "least-squares"
    variables: dict[str, tuple[float, float]] = field(default_factory=dict)
    fixed_parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experimental_data_id": self.experimental_data_id,
            "simulation_data_id": self.simulation_data_id,
            "objective": self.objective,
            "variables": self.variables,
            "fixed_parameters": self.fixed_parameters,
        }
