"""Launch the standalone pole-figure generator tool."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the deployable pole-figure generator application."""

    from qtpy import QtWidgets

    from ewald.ui.pole_figure import PoleFigureGeneratorWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        argv or sys.argv
    )
    window = PoleFigureGeneratorWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
