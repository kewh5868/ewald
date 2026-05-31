"""Shared Qt styling for the EWALD desktop interface."""

from __future__ import annotations

from qtpy import QtGui, QtWidgets

THEME_PROPERTY = "_ewald_theme_applied"


def apply_application_theme(app: QtWidgets.QApplication | None) -> None:
    """Apply a consistent EWALD palette and widget polish once per app."""

    if app is None or app.property(THEME_PROPERTY):
        return
    app.setProperty(THEME_PROPERTY, True)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setPalette(_application_palette())
    app.setStyleSheet(_application_stylesheet())


def _application_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f7f8fb"))
    palette.setColor(
        QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#1f2937")
    )
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
    palette.setColor(
        QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#edf2f7")
    )
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#111827"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#f1f5f9"))
    palette.setColor(
        QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#111827")
    )
    palette.setColor(
        QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2563eb")
    )
    palette.setColor(
        QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff")
    )
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#2563eb"))
    return palette


def _application_stylesheet() -> str:
    return """
QMainWindow, QDialog {
    background: #f7f8fb;
}

QMenuBar, QMenu {
    background: #ffffff;
    color: #111827;
}

QMenuBar::item:selected, QMenu::item:selected {
    background: #e8f0ff;
    color: #0f172a;
}

QToolBar {
    background: #ffffff;
    border: 0;
    border-bottom: 1px solid #d6dee9;
    spacing: 6px;
    padding: 5px 8px;
}

QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background: #eef2f7;
    border-bottom: 1px solid #d6dee9;
    padding: 6px;
    font-weight: 600;
}

QTabWidget::pane {
    border: 1px solid #d6dee9;
    background: #ffffff;
}

QTabBar::tab {
    background: #edf2f7;
    border: 1px solid #d6dee9;
    border-bottom: 0;
    padding: 7px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
}

QTreeWidget, QTableWidget, QTextEdit, QPlainTextEdit {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #d6dee9;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}

QHeaderView::section {
    background: #eef2f7;
    border: 0;
    border-right: 1px solid #d6dee9;
    border-bottom: 1px solid #d6dee9;
    padding: 5px 7px;
    font-weight: 600;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    min-height: 24px;
    padding: 2px 6px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2563eb;
}

QPushButton, QToolButton {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    color: #111827;
    min-height: 24px;
    padding: 3px 8px;
}

QToolButton {
    padding: 2px;
}

QPushButton:hover, QToolButton:hover {
    background: #e8f0ff;
    border-color: #93b4f7;
}

QPushButton:pressed, QToolButton:pressed {
    background: #cfe0ff;
}

QPushButton:disabled, QToolButton:disabled {
    color: #94a3b8;
    background: #f1f5f9;
    border-color: #e2e8f0;
}

QToolButton:checked {
    background: #dbeafe;
    border-color: #2563eb;
}

QGroupBox {
    border: 1px solid #d6dee9;
    border-radius: 5px;
    margin-top: 12px;
    padding-top: 12px;
    background: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #334155;
}

QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d6dee9;
}
"""
