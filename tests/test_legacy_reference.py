"""Tests for preserving the legacy reference tree."""

from pathlib import Path


def test_legacy_reference_tree_contains_previous_ui_and_analysis():
    legacy = Path(__file__).resolve().parents[1] / "src" / "ewald" / "legacy"

    assert (legacy / "README.md").exists()
    assert (legacy / "analysis" / "reciprocal_calculator.py").exists()
    assert (legacy / "ui" / "main_window.py").exists()
    assert (legacy / "yaml" / "ewald.yml").exists()
