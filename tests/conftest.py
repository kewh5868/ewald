"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_collection_modifyitems(config, items):
    """Skip source-checkout example tests when local example data is
    absent."""

    example_dir = Path(__file__).resolve().parents[1] / "example"
    if example_dir.exists():
        return
    skip_example = pytest.mark.skip(
        reason="local example data directory is not present"
    )
    for item in items:
        if {
            "repo_root",
            "example_manifest_path",
        }.intersection(item.fixturenames):
            item.add_marker(skip_example)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def example_manifest_path(repo_root: Path) -> Path:
    path = repo_root / "example" / "manifest.json"
    if not path.exists():
        pytest.skip("local example manifest is not present")
    return path
