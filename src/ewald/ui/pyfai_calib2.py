"""Managed launcher for the external pyFAI-calib2 Qt application."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from enum import Enum

from qtpy import QtCore

PYFAI_CALIB2_COMMAND = "pyfai-calib2"


class PyFAICalib2Status(Enum):
    """User-visible lifecycle states for the pyFAI-calib2 process."""

    NOT_LAUNCHED = "not launched"
    LAUNCHING = "launching"
    RUNNING = "running"
    CLOSED = "closed"
    FAILED = "failed"


class PyFAICalib2Launcher(QtCore.QObject):
    """Start and track a single pyFAI-calib2 process."""

    statusChanged = QtCore.Signal(str)
    launchSkipped = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        program_resolver: Callable[[], str] | None = None,
        arguments: Sequence[str] | None = None,
        process_factory: (
            Callable[[QtCore.QObject], QtCore.QProcess] | None
        ) = None,
    ) -> None:
        super().__init__(parent)
        self._program_resolver = (
            program_resolver or _default_pyfai_calib2_program
        )
        self._arguments = list(arguments or [])
        self._process_factory = process_factory or QtCore.QProcess
        self._process: QtCore.QProcess | None = None
        self._status = PyFAICalib2Status.NOT_LAUNCHED
        self._last_error = ""
        self._stopping = False
        self._process_failed = False

    @property
    def status(self) -> PyFAICalib2Status:
        """Current process lifecycle status."""

        return self._status

    @property
    def status_text(self) -> str:
        """Lowercase status text suitable for display in the UI."""

        return self._status.value

    @property
    def last_error(self) -> str:
        """Most recent process error string."""

        return self._last_error

    @property
    def is_active(self) -> bool:
        """Whether a launch is already in progress or running."""

        return self._status in {
            PyFAICalib2Status.LAUNCHING,
            PyFAICalib2Status.RUNNING,
        }

    def launch(self) -> bool:
        """Launch pyFAI-calib2 unless one is already active."""

        if self.is_active:
            self.launchSkipped.emit(
                f"{PYFAI_CALIB2_COMMAND} is already {self.status_text}."
            )
            return False

        self._release_inactive_process()
        self._last_error = ""
        self._stopping = False
        self._process_failed = False

        process = self._process_factory(self)
        process.setProgram(self._program_resolver())
        process.setArguments(self._arguments)
        process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
        )
        process.started.connect(self._handle_started)
        process.finished.connect(self._handle_finished)
        process.errorOccurred.connect(self._handle_error)
        process.readyRead.connect(self._drain_process_output)
        self._process = process

        self._set_status(PyFAICalib2Status.LAUNCHING)
        process.start()
        return True

    def terminate(self, timeout_ms: int = 1000) -> None:
        """Terminate the managed process if it is still active."""

        if self._process is None or not self.is_active:
            return
        self._stopping = True
        self._process.terminate()
        if not self._process.waitForFinished(timeout_ms):
            self._process.kill()
            self._process.waitForFinished(timeout_ms)
        if self._process.state() == QtCore.QProcess.ProcessState.NotRunning:
            self._set_status(PyFAICalib2Status.CLOSED)

    def _handle_started(self) -> None:
        self._set_status(PyFAICalib2Status.RUNNING)

    def _handle_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        if self._stopping:
            self._set_status(PyFAICalib2Status.CLOSED)
        elif (
            not self._process_failed
            and exit_status == QtCore.QProcess.ExitStatus.NormalExit
            and exit_code == 0
        ):
            self._set_status(PyFAICalib2Status.CLOSED)
        else:
            if not self._last_error:
                self._last_error = (
                    f"{PYFAI_CALIB2_COMMAND} exited with code {exit_code}."
                )
            self._set_status(PyFAICalib2Status.FAILED)
        self._stopping = False
        self._release_inactive_process()

    def _handle_error(self, error: QtCore.QProcess.ProcessError) -> None:
        self._process_failed = True
        if self._process is not None:
            self._last_error = self._process.errorString()
        if self._stopping:
            return
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self._set_status(PyFAICalib2Status.FAILED)

    def _drain_process_output(self) -> None:
        if self._process is not None:
            self._process.readAll()

    def _set_status(self, status: PyFAICalib2Status) -> None:
        if status == self._status:
            return
        self._status = status
        self.statusChanged.emit(status.value)

    def _release_inactive_process(self) -> None:
        if (
            self._process is not None
            and self._process.state()
            == QtCore.QProcess.ProcessState.NotRunning
        ):
            self._process.deleteLater()
            self._process = None


def _default_pyfai_calib2_program() -> str:
    return shutil.which(PYFAI_CALIB2_COMMAND) or PYFAI_CALIB2_COMMAND
