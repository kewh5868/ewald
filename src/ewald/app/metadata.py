"""Application identity shared by EWALD Qt entry points."""

from __future__ import annotations

APPLICATION_NAME = "EWALD"
APPLICATION_DISPLAY_NAME = "EWALD"
ORGANIZATION_NAME = "EWALD"


def configure_qapplication_metadata(app=None) -> None:
    """Apply EWALD identity to Qt application metadata."""

    from qtpy import QtCore, QtGui

    QtCore.QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
    QtCore.QCoreApplication.setApplicationName(APPLICATION_NAME)
    QtGui.QGuiApplication.setApplicationDisplayName(APPLICATION_DISPLAY_NAME)
    if app is not None:
        app.setOrganizationName(ORGANIZATION_NAME)
        app.setApplicationName(APPLICATION_NAME)
        app.setApplicationDisplayName(APPLICATION_DISPLAY_NAME)
