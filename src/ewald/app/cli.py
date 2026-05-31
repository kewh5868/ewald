"""Command line entry point for the EWALD desktop application."""

from __future__ import annotations

import argparse
import sys
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.project and not args.project.exists():
        parser.error(f"project file does not exist: {args.project}")

    from qtpy import QtWidgets

    from ewald.io.project import load_project
    from ewald.ui.main_window import MainWindow
    from ewald.ui.theme import apply_application_theme

    configure_qapplication_metadata()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_qapplication_metadata(app)
    apply_application_theme(app)
    try:
        project = load_project(args.project) if args.project else None
    except Exception as exc:
        print(f"ewald: failed to open project: {exc}", file=sys.stderr)
        return 2
    window = MainWindow(project=project, project_path=args.project)
    window.show()
    return int(app.exec())
