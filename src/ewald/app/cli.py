"""Command line entry point for the EWALD desktop application."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ewald.app.metadata import configure_qapplication_metadata
from ewald.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ewald",
        description=(
            "Launch Experimental WAXS Analysis for Lattice Determination."
        ),
    )
    parser.add_argument(
        "project",
        nargs="?",
        type=Path,
        help="Optional .ewld project file to open at startup.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ewald {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from qtpy import QtWidgets

    from ewald.io.project import load_project
    from ewald.ui.main_window import MainWindow

    configure_qapplication_metadata()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_qapplication_metadata(app)
    project = load_project(args.project) if args.project else None
    window = MainWindow(project=project, project_path=args.project)
    window.show()
    return int(app.exec())
