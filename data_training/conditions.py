"""Simulation condition sweeps for structure-recognition training."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .catalog import StructureCatalog, StructureCatalogRecord
from .config_io import load_config, require_mapping, tuple_of_strings


@dataclass(frozen=True, slots=True)
class SweepAxis:
    """One named sweep axis that writes values to a dotted target
    path."""

    name: str
    target: str
    values: tuple[Any, ...]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SweepAxis":
        data = require_mapping(payload, context="sweep axis")
        values = tuple(data.get("values", ()))
        if not data.get("name"):
            raise ValueError("sweep axis is missing name.")
        if not data.get("target"):
            raise ValueError("sweep axis is missing target.")
        if not values:
            raise ValueError(f"sweep axis {data['name']!r} has no values.")
        return cls(
            name=str(data["name"]),
            target=str(data["target"]),
            values=values,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class SimulationCondition:
    """A fully expanded structure, simulator, texture, and artifact
    request."""

    condition_id: str
    structure_id: str
    simulator: str
    peak_table_exporter: str
    parameters: dict[str, Any]
    texture: dict[str, Any] = field(default_factory=dict)
    artifact_profile_id: str = "clean"
    replicate_index: int = 0
    axis_values: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "structure_id": self.structure_id,
            "simulator": self.simulator,
            "peak_table_exporter": self.peak_table_exporter,
            "parameters": dict(self.parameters),
            "texture": dict(self.texture),
            "artifact_profile_id": self.artifact_profile_id,
            "replicate_index": self.replicate_index,
            "axis_values": dict(self.axis_values),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SimulationSweepSpec:
    """A configurable sweep over structures and GIWAXS conditions."""

    sweep_id: str
    structures: tuple[str, ...] | str
    simulator: str
    peak_table_exporter: str
    base_parameters: dict[str, Any]
    base_texture: dict[str, Any] = field(default_factory=dict)
    axes: tuple[SweepAxis, ...] = ()
    artifact_profiles: tuple[str, ...] = ("clean",)
    repetitions: int = 1
    split_policy: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SimulationSweepSpec":
        data = require_mapping(payload, context="simulation sweep")
        structures_value = data.get("structures", "all")
        structures: tuple[str, ...] | str
        structures = (
            "all"
            if structures_value == "all"
            else tuple_of_strings(structures_value)
        )
        hooks = require_mapping(
            data.get("ewald_hooks", {}), context="ewald_hooks"
        )
        return cls(
            sweep_id=str(data.get("sweep_id") or "simulation_sweep"),
            structures=structures,
            simulator=str(
                hooks.get(
                    "simulator",
                    "ewald.simulation.giwaxs:simulate_giwaxs_image",
                )
            ),
            peak_table_exporter=str(
                hooks.get(
                    "peak_table_exporter",
                    "ewald.simulation.giwaxs:calculate_giwaxs_peak_rows",
                )
            ),
            base_parameters=dict(data.get("base_parameters") or {}),
            base_texture=dict(data.get("base_texture") or {}),
            axes=tuple(
                SweepAxis.from_mapping(item) for item in data.get("axes", [])
            ),
            artifact_profiles=tuple_of_strings(
                data.get("artifact_profiles") or ("clean",)
            ),
            repetitions=int(data.get("repetitions", 1)),
            split_policy=dict(data.get("split_policy") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationSweepSpec":
        return cls.from_mapping(load_config(path))

    def selected_structures(
        self, catalog: StructureCatalog
    ) -> tuple[StructureCatalogRecord, ...]:
        return catalog.select(self.structures)

    def expand(
        self, catalog: StructureCatalog
    ) -> tuple[SimulationCondition, ...]:
        """Expand this sweep into deterministic simulation
        conditions."""

        structures = self.selected_structures(catalog)
        axis_products = tuple(
            itertools.product(*(axis.values for axis in self.axes))
        )
        conditions: list[SimulationCondition] = []
        for structure in structures:
            for values in axis_products:
                axis_values = {
                    axis.name: value for axis, value in zip(self.axes, values)
                }
                base_payload = {
                    "parameters": dict(self.base_parameters),
                    "texture": dict(self.base_texture),
                }
                for axis, value in zip(self.axes, values):
                    _set_dotted_value(base_payload, axis.target, value)
                for artifact_profile_id in self.artifact_profiles:
                    for replicate_index in range(max(1, self.repetitions)):
                        condition_id = stable_condition_id(
                            {
                                "sweep_id": self.sweep_id,
                                "structure_id": structure.structure_id,
                                "axis_values": axis_values,
                                "artifact_profile_id": artifact_profile_id,
                                "replicate_index": replicate_index,
                            }
                        )
                        conditions.append(
                            SimulationCondition(
                                condition_id=condition_id,
                                structure_id=structure.structure_id,
                                simulator=self.simulator,
                                peak_table_exporter=self.peak_table_exporter,
                                parameters=dict(base_payload["parameters"]),
                                texture=dict(base_payload["texture"]),
                                artifact_profile_id=artifact_profile_id,
                                replicate_index=replicate_index,
                                axis_values=dict(axis_values),
                                metadata={"sweep_id": self.sweep_id},
                            )
                        )
        return tuple(conditions)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "structures": self.structures,
            "ewald_hooks": {
                "simulator": self.simulator,
                "peak_table_exporter": self.peak_table_exporter,
            },
            "base_parameters": dict(self.base_parameters),
            "base_texture": dict(self.base_texture),
            "axes": [axis.to_mapping() for axis in self.axes],
            "artifact_profiles": list(self.artifact_profiles),
            "repetitions": self.repetitions,
            "split_policy": dict(self.split_policy),
            "metadata": dict(self.metadata),
        }


def read_simulation_sweep(path: str | Path) -> SimulationSweepSpec:
    """Parse a simulation sweep config from JSON/YAML."""

    return SimulationSweepSpec.from_file(path)


def stable_condition_id(payload: dict[str, Any]) -> str:
    """Return a compact deterministic id for a condition payload."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _set_dotted_value(
    target: dict[str, Any], dotted_path: str, value: Any
) -> None:
    keys = dotted_path.split(".")
    if not keys:
        raise ValueError("dotted path cannot be empty.")
    cursor = target
    for key in keys[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested value under {key!r}.")
        cursor = next_value
    cursor[keys[-1]] = value


def selected_artifact_profiles(
    conditions: Iterable[SimulationCondition],
) -> tuple[str, ...]:
    """Return unique artifact profile ids referenced by conditions."""

    return tuple(
        sorted({condition.artifact_profile_id for condition in conditions})
    )
