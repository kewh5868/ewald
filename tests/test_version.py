"""Package version tests."""

import ewald


def test_package_version_is_defined():
    assert hasattr(ewald, "__version__")
    assert ewald.__version__ != "0.0.0"
