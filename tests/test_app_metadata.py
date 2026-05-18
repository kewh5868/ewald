"""Tests for EWALD Qt application metadata."""

from qtpy import QtCore, QtGui, QtWidgets

from ewald.app.metadata import (
    APPLICATION_DISPLAY_NAME,
    APPLICATION_NAME,
    ORGANIZATION_NAME,
    configure_qapplication_metadata,
)


def test_qapplication_metadata_uses_ewald():
    app = QtWidgets.QApplication.instance()

    configure_qapplication_metadata(app)

    assert QtCore.QCoreApplication.applicationName() == APPLICATION_NAME
    assert QtCore.QCoreApplication.organizationName() == ORGANIZATION_NAME
    assert (
        QtGui.QGuiApplication.applicationDisplayName()
        == APPLICATION_DISPLAY_NAME
    )
    if app is not None:
        assert app.applicationName() == "EWALD"
        assert app.organizationName() == "EWALD"
        assert app.applicationDisplayName() == "EWALD"
