"""Launch the standalone GIWAXS simulation tool."""

from __future__ import annotations

import sys

from ewald.app.metadata import configure_qapplication_metadata


def main(argv: list[str] | None = None) -> int:
    """Run the deployable GIWAXS simulation application."""

    from qtpy import QtWidgets

    from ewald.ui.giwaxs_simulation import GIWAXSSimulationWindow

    configure_qapplication_metadata()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        argv or sys.argv
    )
    configure_qapplication_metadata(app)
    window = GIWAXSSimulationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
