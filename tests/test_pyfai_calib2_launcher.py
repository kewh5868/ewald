"""Tests for managed pyFAI-calib2 process launching."""

from __future__ import annotations

import sys

from ewald.ui.pyfai_calib2 import PyFAICalib2Launcher, PyFAICalib2Status


def test_pyfai_calib2_launcher_tracks_single_process_lifecycle(qtbot):
    launcher = PyFAICalib2Launcher(
        program_resolver=lambda: sys.executable,
        arguments=["-c", "import time; time.sleep(0.5)"],
    )

    try:
        with qtbot.waitSignal(launcher.statusChanged, timeout=1000) as change:
            assert launcher.launch()

        assert change.args == [PyFAICalib2Status.LAUNCHING.value]
        assert not launcher.launch()
        qtbot.waitUntil(
            lambda: launcher.status == PyFAICalib2Status.RUNNING,
            timeout=3000,
        )
        assert not launcher.launch()
        qtbot.waitUntil(
            lambda: launcher.status == PyFAICalib2Status.CLOSED,
            timeout=5000,
        )
    finally:
        launcher.terminate(timeout_ms=200)


def test_pyfai_calib2_launcher_reports_failed_start(qtbot):
    launcher = PyFAICalib2Launcher(
        program_resolver=lambda: "__missing_ewald_pyfai_calib2__",
    )

    assert launcher.launch()
    qtbot.waitUntil(
        lambda: launcher.status == PyFAICalib2Status.FAILED,
        timeout=3000,
    )

    assert launcher.last_error
