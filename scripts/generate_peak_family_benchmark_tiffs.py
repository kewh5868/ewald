"""Generate deterministic peak-family benchmark TIFF fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "tests" / "fixtures" / "known_structure_simulations" / "peak_family"
)
AXIS_RANGES = (-2.2, 2.2, 0.0, 2.4)
IMAGE_SHAPE = (256, 320)


def _peak(
    peak_id: str,
    label: str,
    qxy: float,
    qz: float,
    amplitude: float,
) -> dict[str, Any]:
    return {
        "peak_id": peak_id,
        "label": label,
        "qxy": float(qxy),
        "qz": float(qz),
        "amplitude": float(amplitude),
    }


BENCHMARK_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "cubic_missing_fundamental",
        "structure_name": "cubic perovskite-like missing fundamental",
        "description": (
            "A cubic-like qxy harmonic series where the qxy fundamental is "
            "absent; the grouping algorithm should infer the latent step."
        ),
        "peaks": [
            _peak("cubic_200", "Cubic (200)", 0.90, 0.42, 0.95),
            _peak("cubic_300", "Cubic (300)", 1.35, 1.16, 0.82),
            _peak("cubic_400", "Cubic (400)", 1.80, 1.72, 0.70),
            _peak("cubic_off_axis", "Cubic off-axis", -0.38, 2.02, 0.45),
        ],
        "expected_families": [
            {
                "family_id": "cubic_qxy_orders_2_3_4",
                "kind": "q_xy multiples",
                "peak_ids": ["cubic_200", "cubic_300", "cubic_400"],
                "orders": [2, 3, 4],
            }
        ],
    },
    {
        "case_id": "layered_qz_missing_fundamental",
        "structure_name": "layered halide-like missing qz fundamental",
        "description": (
            "A lamellar qz progression where the first order is outside the "
            "captured range or too weak to mark."
        ),
        "peaks": [
            _peak("layered_002", "Layered (002)", -1.35, 0.72, 0.92),
            _peak("layered_003", "Layered (003)", -0.40, 1.08, 0.78),
            _peak("layered_004", "Layered (004)", 0.75, 1.44, 0.68),
            _peak("layered_sideband", "Layered sideband", 1.55, 0.28, 0.38),
        ],
        "expected_families": [
            {
                "family_id": "layered_qz_orders_2_3_4",
                "kind": "q_z multiples",
                "peak_ids": ["layered_002", "layered_003", "layered_004"],
                "orders": [2, 3, 4],
            }
        ],
    },
    {
        "case_id": "orthorhombic_cross_grid",
        "structure_name": "orthorhombic fiber-texture cross grid",
        "description": (
            "A simple projected orthorhombic grid with independent qxy and "
            "qz harmonic families."
        ),
        "peaks": [
            _peak("ortho_h10_l1", "Ortho h1 l1", 0.62, 0.55, 0.88),
            _peak("ortho_h20_l2", "Ortho h2 l2", 1.24, 1.10, 0.76),
            _peak("ortho_h30_l3", "Ortho h3 l3", 1.86, 1.65, 0.60),
            _peak("ortho_tilted", "Ortho tilted", -1.02, 0.92, 0.46),
        ],
        "expected_families": [
            {
                "family_id": "ortho_qxy_orders_1_2_3",
                "kind": "q_xy multiples",
                "peak_ids": ["ortho_h10_l1", "ortho_h20_l2", "ortho_h30_l3"],
                "orders": [1, 2, 3],
            },
            {
                "family_id": "ortho_qz_orders_1_2_3",
                "kind": "q_z multiples",
                "peak_ids": ["ortho_h10_l1", "ortho_h20_l2", "ortho_h30_l3"],
                "orders": [1, 2, 3],
            },
        ],
    },
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": (
            "Deterministic known-structure q-space TIFFs for validating "
            "Structure Analysis peak-family grouping."
        ),
        "axis_ranges": list(AXIS_RANGES),
        "image_shape": list(IMAGE_SHAPE),
        "cases": [],
    }
    for case in BENCHMARK_CASES:
        image = _render_case(case)
        tiff_name = f"{case['case_id']}.tiff"
        tifffile.imwrite(OUTPUT_DIR / tiff_name, image.astype(np.float32))
        manifest["cases"].append(
            {
                "case_id": case["case_id"],
                "structure_name": case["structure_name"],
                "description": case["description"],
                "tiff": tiff_name,
                "peaks": case["peaks"],
                "expected_families": case["expected_families"],
            }
        )
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(BENCHMARK_CASES)} benchmark TIFFs to {OUTPUT_DIR}")
    return 0


def _render_case(case: dict[str, Any]) -> np.ndarray:
    qxy_min, qxy_max, qz_min, qz_max = AXIS_RANGES
    qxy = np.linspace(qxy_min, qxy_max, IMAGE_SHAPE[1], dtype=float)
    qz = np.linspace(qz_min, qz_max, IMAGE_SHAPE[0], dtype=float)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    image = 0.015 + 0.01 * (qz_grid / max(qz_max, 1.0e-9))
    image += 0.006 * np.cos(2.5 * qxy_grid) ** 2
    for peak in case["peaks"]:
        qxy_width = 0.045 + 0.01 * abs(float(peak["qxy"]))
        qz_width = 0.040 + 0.008 * abs(float(peak["qz"]))
        image += float(peak["amplitude"]) * np.exp(
            -0.5
            * (
                ((qxy_grid - float(peak["qxy"])) / qxy_width) ** 2
                + ((qz_grid - float(peak["qz"])) / qz_width) ** 2
            )
        )
    image -= np.nanmin(image)
    image /= max(float(np.nanmax(image)), 1.0e-9)
    return image


if __name__ == "__main__":
    raise SystemExit(main())
