"""Definition of the installed package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ewald")
except PackageNotFoundError:
    __version__ = "unknown"
