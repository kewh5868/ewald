"""Corrected q-space data viewer with ROI definition controls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ewald.data.models import (
    ImageCorrectionState,
    ProjectState,
    ROIRegion,
    mark_roi_pole_figure_stale,
    roi_hkl_metadata,
    roi_pole_figure_status,
)
from ewald.ui.notation import (
    QXY_HTML,
    QXY_MATPLOTLIB,
    QZ_HTML,
    QZ_MATPLOTLIB,
    RichTextComboBox,
    data_image_rect,
    enable_rich_text_items,
    set_data_aspect_locked,
    set_data_image_plot_range,
    set_qspace_axis_labels,
    set_rich_text_table_headers,
)
from ewald.ui.orientation import sample_orientation_for_image_transform

try:  # pragma: no cover - exercised through widget tests when installed.
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

ARCH_CHI_LIMITS_DEG = (-90.0, 90.0)
ARCH_CHI_LOCKED_METADATA_KEY = "chi_locked"
ARCH_HANDLE_SIZE = 11
COUPLED_ROI_GROUP_METADATA_KEY = "coupling_id"
COUPLED_ROI_ID_METADATA_KEY = "coupled_roi_id"
COUPLED_ROI_IDS_METADATA_KEY = "coupled_roi_ids"
COUPLED_ROI_ROLE_METADATA_KEY = "coupled_role"
COUPLED_ROI_SHARED_CENTER_METADATA_KEY = "shared_center"
CHANNEL_MIME_TYPE = "application/x-ewald-integration-channel"
CHANNEL_DETECT_MAX_PEAKS_PER_TRACE = 12
CHANNEL_DETECT_MIN_HEIGHT_FRACTION = 0.05
ROI_COLOR_ARCH = "#f2a65a"
ROI_COLOR_BOX_HORIZONTAL = "#3da5d9"
ROI_COLOR_BOX_VERTICAL = "#2a9d8f"
ROI_COL_ID = 0
ROI_COL_CHANNEL_1 = 1
ROI_COL_CHANNEL_2 = 2
ROI_COL_TYPE = 3
ROI_COL_DIRECTION = 4
ROI_COL_QXY_MIN = 5
ROI_COL_QXY_MAX = 6
ROI_COL_QZ_MIN = 7
ROI_COL_QZ_MAX = 8
ROI_COL_QR_MIN = 9
ROI_COL_QR_MAX = 10
ROI_COL_CHI_MIN = 11
ROI_COL_CHI_MAX = 12
ROI_COL_CHI_LOCK = 13
ROI_COL_INTEGRATE = 14
ROI_COL_RADIUS = 15
ROI_COL_QXY_CENTER = 16
ROI_COL_QZ_CENTER = 17
ROI_COL_H = 18
ROI_COL_K = 19
ROI_COL_L = 20
ROI_COL_HKL_LABEL = 21
ROI_COL_POLE_FIGURE = 22
ROI_COL_COUPLED = 23
ROI_HKL_EDIT_COLUMNS = {
    ROI_COL_H,
    ROI_COL_K,
    ROI_COL_L,
    ROI_COL_HKL_LABEL,
}
IMAGE_COLORMAPS = ("viridis", "magma", "turbo", "gray")
ROI_TABLE_HEADERS = [
    "ROI",
    "Ch 1",
    "Ch 2",
    "Type",
    "Direction",
    f"{QXY_HTML} min",
    f"{QXY_HTML} max",
    f"{QZ_HTML} min",
    f"{QZ_HTML} max",
    "qr min",
    "qr max",
    "chi min",
    "chi max",
    "Chi lock",
    "Integrate",
    "Radius",
    f"{QXY_HTML} center",
    f"{QZ_HTML} center",
    "h",
    "k",
    "l",
    "hkl label",
    "Pole figure",
    "Coupled",
]


@dataclass(slots=True, frozen=True)
class ImageDisplayStyle:
    """Color and level scaling applied to corrected detector images."""

    colormap: str = "viridis"
    use_quantile: bool = True
    quantile_low: float = 1.0
    quantile_high: float = 99.0
    level_min: float = 0.0
    level_max: float = 1.0


class ImagePlotToolbar(QtWidgets.QWidget):
    """Shared color and navigation toolbar for image-backed plots."""

    def __init__(
        self,
        *,
        colormap_combo: QtWidgets.QComboBox,
        level_min: QtWidgets.QDoubleSpinBox,
        level_max: QtWidgets.QDoubleSpinBox,
        quantile_check: QtWidgets.QCheckBox,
        quantile_low: QtWidgets.QDoubleSpinBox,
        quantile_high: QtWidgets.QDoubleSpinBox,
        auto_contrast_button: QtWidgets.QToolButton,
        zoom_in_button: QtWidgets.QToolButton | None = None,
        zoom_out_button: QtWidgets.QToolButton | None = None,
        autoscale_button: QtWidgets.QToolButton | None = None,
        pan_button: QtWidgets.QToolButton | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ImagePlotToolbar")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel("Color"))
        layout.addWidget(colormap_combo)
        layout.addSpacing(12)
        layout.addWidget(QtWidgets.QLabel("Min"))
        layout.addWidget(level_min)
        layout.addWidget(QtWidgets.QLabel("Max"))
        layout.addWidget(level_max)
        layout.addWidget(quantile_check)
        layout.addWidget(QtWidgets.QLabel("Low"))
        layout.addWidget(quantile_low)
        layout.addWidget(QtWidgets.QLabel("High"))
        layout.addWidget(quantile_high)
        layout.addWidget(auto_contrast_button)
        if all(
            button is not None
            for button in (
                zoom_in_button,
                zoom_out_button,
                autoscale_button,
                pan_button,
            )
        ):
            assert autoscale_button is not None
            autoscale_button.setText("Autoscale")
            layout.addSpacing(12)
            layout.addWidget(QtWidgets.QLabel("View"))
            layout.addWidget(zoom_in_button)
            layout.addWidget(zoom_out_button)
            layout.addWidget(autoscale_button)
            layout.addWidget(pan_button)
        layout.addStretch(1)


@dataclass(slots=True)
class _IntegrationTrace:
    roi_id: str
    label: str
    mode: str
    x_label: str
    x_values: np.ndarray
    y_values: np.ndarray
    color: str


@dataclass(slots=True, frozen=True)
class IntegrationPeakMarker:
    """One peak marked on an integrated 1D channel trace."""

    marker_id: str
    channel: int
    roi_id: str
    roi_name: str
    mode: str
    integration_x: float
    integrated_intensity: float
    qxy: float
    qz: float
    label: str


class _DrawingViewBox(pg.ViewBox if pg is not None else object):
    """ViewBox that reports drag bounds for user-drawn ROIs."""

    def __init__(
        self,
        on_bounds_drawn: Callable[[float, float, float, float], None],
    ) -> None:
        if pg is not None:
            super().__init__()
        self.on_bounds_drawn = on_bounds_drawn
        self.drawing_enabled = False

    def mouseDragEvent(self, event: Any, axis: Any = None) -> None:
        if (
            self.drawing_enabled
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            if event.isFinish():
                start = self.mapSceneToView(event.buttonDownScenePos())
                end = self.mapSceneToView(event.scenePos())
                self.on_bounds_drawn(start.x(), end.x(), start.y(), end.y())
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


class _ImageAspectPlotFrame(QtWidgets.QWidget):
    """Center a pyqtgraph image plot at the displayed data aspect."""

    def __init__(
        self,
        plot_widget: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plot_widget = plot_widget
        self._data_rect: QtCore.QRectF | None = None
        self._sync_pending = False
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(
            plot_widget,
            0,
            0,
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )

    def set_data_rect(self, rect: QtCore.QRectF) -> None:
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self._data_rect = QtCore.QRectF(rect)
        self._schedule_sync()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._schedule_sync()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._schedule_sync()

    def _schedule_sync(self) -> None:
        if self._data_rect is None or self._sync_pending:
            return
        self._sync_pending = True
        QtCore.QTimer.singleShot(0, self._sync_plot_size)

    def _sync_plot_size(self) -> None:
        self._sync_pending = False
        if self._data_rect is None:
            return
        available = self.contentsRect().size()
        if available.width() <= 0 or available.height() <= 0:
            return
        aspect = self._data_rect.width() / self._data_rect.height()
        extra_width, extra_height = self._plot_chrome_size()
        view_width = max(1.0, float(available.width() - extra_width))
        view_height = max(1.0, float(available.height() - extra_height))
        if view_width / view_height > aspect:
            target_view_height = view_height
            target_view_width = target_view_height * aspect
        else:
            target_view_width = view_width
            target_view_height = target_view_width / aspect
        target_width = min(
            available.width(),
            max(1, int(round(target_view_width + extra_width))),
        )
        target_height = min(
            available.height(),
            max(1, int(round(target_view_height + extra_height))),
        )
        target_size = QtCore.QSize(target_width, target_height)
        if self.plot_widget.size() != target_size:
            self.plot_widget.setFixedSize(target_size)
        self.plot_widget.setRange(
            xRange=(self._data_rect.left(), self._data_rect.right()),
            yRange=(self._data_rect.top(), self._data_rect.bottom()),
            padding=0.0,
        )

    def _plot_chrome_size(self) -> tuple[float, float]:
        if not hasattr(self.plot_widget, "getViewBox"):
            return 0.0, 0.0
        view_box = self.plot_widget.getViewBox()
        view_rect = view_box.sceneBoundingRect()
        if view_rect.width() <= 1.0 or view_rect.height() <= 1.0:
            return 90.0, 70.0
        return (
            max(0.0, float(self.plot_widget.width()) - view_rect.width()),
            max(0.0, float(self.plot_widget.height()) - view_rect.height()),
        )


if pg is not None:

    class _BoxROI(pg.ROI):
        """Movable rectangular ROI with resize handles on every edge."""

        CORNER_HANDLE = "box-corner"
        LEFT_HANDLE = "box-left"
        RIGHT_HANDLE = "box-right"
        BOTTOM_HANDLE = "box-bottom"
        TOP_HANDLE = "box-top"

        def __init__(
            self,
            pos: tuple[float, float],
            size: tuple[float, float],
            pen,
        ) -> None:
            super().__init__(
                pos,
                size,
                pen=pen,
                movable=True,
                rotatable=False,
                resizable=True,
            )
            self.handleSize = 7
            self.addScaleHandle(
                (1.0, 1.0),
                (0.0, 0.0),
                name=self.CORNER_HANDLE,
            )
            self.addScaleHandle(
                (0.0, 0.5),
                (1.0, 0.5),
                name=self.LEFT_HANDLE,
            )
            self.addScaleHandle(
                (1.0, 0.5),
                (0.0, 0.5),
                name=self.RIGHT_HANDLE,
            )
            self.addScaleHandle(
                (0.5, 0.0),
                (0.5, 1.0),
                name=self.BOTTOM_HANDLE,
            )
            self.addScaleHandle(
                (0.5, 1.0),
                (0.5, 0.0),
                name=self.TOP_HANDLE,
            )

    class _ArchROI(pg.ROI):
        """Movable annular-sector ROI with polar shape controls."""

        THICKNESS_HANDLE = "arch-thickness"
        RADIUS_HANDLE = "arch-radius"
        CHI_MIN_HANDLE = "arch-chi-min"
        CHI_MAX_HANDLE = "arch-chi-max"

        def __init__(
            self,
            roi: ROIRegion,
            pen,
        ):
            super().__init__(
                (0.0, 0.0),
                (1.0, 1.0),
                pen=pen,
                movable=True,
                resizable=False,
                rotatable=False,
            )
            self.brush = pg.mkBrush(242, 166, 90, 45)
            self.handleSize = ARCH_HANDLE_SIZE
            self._fixed_resize_center: tuple[float, float] | None = None
            self.addFreeHandle((0.5, 0.8), name=self.RADIUS_HANDLE)
            self.addFreeHandle((0.5, 1.0), name=self.THICKNESS_HANDLE)
            self.addFreeHandle((0.0, 0.5), name=self.CHI_MIN_HANDLE)
            self.addFreeHandle((1.0, 0.5), name=self.CHI_MAX_HANDLE)
            self.set_arch_region(roi)

        def set_arch_region(self, roi: ROIRegion) -> None:
            self.qr_min = float(roi.qr_min or 0.0)
            self.qr_max = float(roi.qr_max or 0.0)
            self.chi_min = float(roi.chi_min or 0.0)
            self.chi_max = float(roi.chi_max or 0.0)
            self.chi_locked = _arch_chi_locked(roi)
            self.center_qxy = float(roi.qxy_center)
            self.center_qz = float(roi.qz_center)
            self._fixed_resize_center = None
            self._apply_arch_bounds()

        def arch_parameters(self) -> tuple[float, float, float, float]:
            return self.qr_min, self.qr_max, self.chi_min, self.chi_max

        def arch_center(self) -> tuple[float, float]:
            if self._fixed_resize_center is not None:
                return self._fixed_resize_center
            center = self._center_from_graphic_position()
            self.center_qxy, self.center_qz = center
            return center

        def _center_from_graphic_position(self) -> tuple[float, float]:
            x_min, _x_max, y_min, _y_max = _arch_local_bounds(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            position = self.pos()
            return float(position.x() - x_min), float(position.y() - y_min)

        def _begin_fixed_center_resize(self) -> None:
            if self._fixed_resize_center is None:
                self._fixed_resize_center = (
                    self._center_from_graphic_position()
                )
            self.center_qxy, self.center_qz = self._fixed_resize_center

        def _apply_arch_bounds(
            self,
            *,
            block_signals: bool = True,
            finish: bool = True,
        ) -> None:
            x_min, x_max, y_min, y_max = _arch_local_bounds(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            previous_signal_state = self.blockSignals(block_signals)
            try:
                self.setPos(
                    (self.center_qxy + x_min, self.center_qz + y_min),
                    update=False,
                )
                self.setSize(
                    (x_max - x_min, y_max - y_min),
                    update=False,
                )
                self.stateChanged(finish=finish)
            finally:
                self.blockSignals(previous_signal_state)
            self._sync_handle_positions()
            self.update()

        def movePoint(
            self,
            handle,
            pos,
            modifiers=None,
            finish: bool = True,
            coords: str = "parent",
        ) -> None:
            name = self._handle_name(handle)
            if name not in {
                self.RADIUS_HANDLE,
                self.THICKNESS_HANDLE,
                self.CHI_MIN_HANDLE,
                self.CHI_MAX_HANDLE,
            }:
                super().movePoint(
                    handle,
                    pos,
                    modifiers=modifiers,
                    finish=finish,
                    coords=coords,
                )
                return

            if coords == "scene":
                parent_pos = self.mapSceneToParent(pos)
            elif coords == "parent":
                parent_pos = pos
            else:
                raise ValueError("Handle position must be parent or scene.")
            local_pos = self.mapFromParent(parent_pos)
            self._resize_from_handle(
                name,
                float(local_pos.x()),
                float(local_pos.y()),
                finish=finish,
            )

        def _resize_from_handle(
            self,
            name: str,
            local_x: float,
            local_y: float,
            *,
            finish: bool,
        ) -> None:
            x_min, _x_max, y_min, _y_max = _arch_local_bounds(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            self._begin_fixed_center_resize()
            arch_x = local_x + x_min
            arch_y = local_y + y_min
            if name == self.RADIUS_HANDLE:
                radius = float(np.hypot(arch_x, arch_y))
                thickness = max(self.qr_max - self.qr_min, 1.0e-9)
                radius_center = max(radius, thickness / 2.0)
                self.qr_min = max(0.0, radius_center - thickness / 2.0)
                self.qr_max = max(
                    self.qr_min + 1.0e-9,
                    radius_center + thickness / 2.0,
                )
            elif name == self.THICKNESS_HANDLE:
                radius = float(np.hypot(arch_x, arch_y))
                self.qr_max = max(self.qr_min + 1.0e-9, radius)
            else:
                chi_value = _chi_from_local_point(arch_x, arch_y)
                if self.chi_locked:
                    self.chi_min, self.chi_max = _symmetric_chi_range(
                        -abs(chi_value),
                        abs(chi_value),
                    )
                elif name == self.CHI_MIN_HANDLE:
                    self.chi_min = min(chi_value, self.chi_max - 1.0e-6)
                    self.chi_min = max(
                        self.chi_min,
                        ARCH_CHI_LIMITS_DEG[0],
                    )
                else:
                    self.chi_max = max(chi_value, self.chi_min + 1.0e-6)
                    self.chi_max = min(
                        self.chi_max,
                        ARCH_CHI_LIMITS_DEG[1],
                    )
            self._apply_arch_bounds(block_signals=False, finish=finish)
            if finish:
                self._fixed_resize_center = None

        def _handle_name(self, handle) -> str | None:
            for info in self.handles:
                if info.get("item") is handle:
                    name = info.get("name")
                    return str(name) if name is not None else None
            return None

        def _sync_handle_positions(self) -> None:
            x_min, x_max, y_min, y_max = _arch_local_bounds(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            width = max(x_max - x_min, 1.0e-9)
            height = max(y_max - y_min, 1.0e-9)
            radius_center = (self.qr_min + self.qr_max) / 2.0
            chi_center = (self.chi_min + self.chi_max) / 2.0
            positions = {
                self.RADIUS_HANDLE: _polar_qxy_qz(
                    radius_center,
                    chi_center,
                ),
                self.THICKNESS_HANDLE: _polar_qxy_qz(
                    self.qr_max,
                    chi_center,
                ),
                self.CHI_MIN_HANDLE: _polar_qxy_qz(
                    radius_center,
                    self.chi_min,
                ),
                self.CHI_MAX_HANDLE: _polar_qxy_qz(
                    radius_center,
                    self.chi_max,
                ),
            }
            size = self.state["size"]
            for info in self.handles:
                name = info.get("name")
                if name not in positions:
                    continue
                x_value, y_value = positions[name]
                normalized = pg.Point(
                    (x_value - x_min) / width,
                    (y_value - y_min) / height,
                )
                info["pos"] = normalized
                info["item"].setPos(normalized * size)

        def paint(self, painter, option, widget=None) -> None:
            painter.setRenderHint(
                QtGui.QPainter.RenderHint.Antialiasing,
                True,
            )
            x_min, _x_max, y_min, _y_max = _arch_local_bounds(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            path = _arch_path(
                self.qr_min,
                self.qr_max,
                self.chi_min,
                self.chi_max,
            )
            path.translate(-x_min, -y_min)
            painter.setPen(self.currentPen)
            painter.setBrush(self.brush)
            painter.drawPath(path)


class _DragHandleLabel(QtWidgets.QLabel):
    """Small label that can start a channel tear-off drag."""

    dragFinished = QtCore.Signal(int, bool)

    def __init__(self, channel: int, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.channel = channel
        self._press_position: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press_position = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._press_position is None:
            super().mouseMoveEvent(event)
            return
        if not event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            super().mouseMoveEvent(event)
            return
        distance = (event.pos() - self._press_position).manhattanLength()
        if distance < QtWidgets.QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return
        mime_data = QtCore.QMimeData()
        mime_data.setData(
            CHANNEL_MIME_TYPE,
            str(self.channel).encode("ascii"),
        )
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        result = drag.exec(QtCore.Qt.DropAction.MoveAction)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.dragFinished.emit(
            self.channel,
            result == QtCore.Qt.DropAction.MoveAction,
        )


class _MatplotlibIntegrationWidget(QtWidgets.QWidget):
    """Matplotlib-backed line plot for one integration channel."""

    peakMarked = QtCore.Signal(str, float, float)
    markerDragPreviewed = QtCore.Signal(str, float, float)
    markerMoved = QtCore.Signal(str, float, float)
    markerDeleted = QtCore.Signal(str)

    def __init__(
        self,
        *,
        with_toolbar: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.series: list[_IntegrationTrace] = []
        self.markers: list[IntegrationPeakMarker] = []
        self.mode: str | None = None
        self.autosnap_enabled = True
        self._drag_marker_id: str | None = None
        self._drag_marker_roi_id: str | None = None
        self._drag_marker_moved = False
        self._drag_marker_delete_pending = False
        self._poof_timers: list[QtCore.QTimer] = []
        self._poof_artists: list[Any] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.backends.backend_qtagg import (
                NavigationToolbar2QT as NavigationToolbar,
            )
            from matplotlib.figure import Figure
        except Exception:
            self.figure = None
            self.axes = None
            self.canvas = None
            self.toolbar = None
            fallback = QtWidgets.QLabel("Integration channel")
            fallback.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
            return

        self.figure = Figure(figsize=(3.4, 2.2), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = (
            NavigationToolbar(self.canvas, self) if with_toolbar else None
        )
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect(
            "button_press_event",
            self._handle_mouse_press,
        )
        self.canvas.mpl_connect(
            "motion_notify_event",
            self._handle_mouse_motion,
        )
        self.canvas.mpl_connect(
            "button_release_event",
            self._handle_mouse_release,
        )
        self.set_series([], None)

    def set_series(
        self,
        series: list[_IntegrationTrace],
        mode: str | None,
        markers: list[IntegrationPeakMarker] | None = None,
    ) -> None:
        self.series = list(series)
        self.markers = list(markers or [])
        self.mode = mode
        if self.axes is None or self.canvas is None:
            return
        self._clear_poof_animations()
        self.axes.clear()
        for trace in self.series:
            self.axes.plot(
                trace.x_values,
                trace.y_values,
                color=trace.color,
                linewidth=1.6,
                label=trace.label,
            )
            trace_markers = [
                marker
                for marker in self.markers
                if marker.roi_id == trace.roi_id
            ]
            if trace_markers:
                self.axes.scatter(
                    [marker.integration_x for marker in trace_markers],
                    [marker.integrated_intensity for marker in trace_markers],
                    s=38,
                    marker="v",
                    color=trace.color,
                    edgecolor="#111111",
                    linewidth=0.7,
                    zorder=5,
                )
        if self.series:
            self.axes.set_xlabel(self.series[0].x_label)
            self.axes.set_ylabel("Integrated intensity")
            if len(self.series) > 1:
                self.axes.legend(fontsize=7, loc="best")
        elif mode is not None:
            self.axes.set_xlabel(_mode_x_label(mode))
            self.axes.set_ylabel("Integrated intensity")
        self.axes.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def set_autosnap_enabled(self, enabled: bool) -> None:
        self.autosnap_enabled = bool(enabled)

    def _handle_mouse_press(self, event: Any) -> None:
        if self.axes is None or event.inaxes is not self.axes:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if self._toolbar_is_active():
            return
        marker = self._nearest_marker(event)
        if marker is not None:
            self._drag_marker_id = marker.marker_id
            self._drag_marker_roi_id = marker.roi_id
            self._drag_marker_moved = False
            self._drag_marker_delete_pending = False
            if self.canvas is not None:
                self.canvas.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.markerDragPreviewed.emit(
                marker.marker_id,
                marker.integration_x,
                marker.integrated_intensity,
            )
            return
        self._mark_nearest_trace_point(event)

    def _handle_mouse_motion(self, event: Any) -> None:
        if self._drag_marker_id is None or self._drag_marker_roi_id is None:
            return
        if (
            self.axes is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._drag_marker_delete_pending = True
            if self.canvas is not None:
                self.canvas.setCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
            return
        self._drag_marker_delete_pending = False
        if self.canvas is not None:
            self.canvas.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        nearest = self._nearest_trace_point(
            float(event.xdata),
            float(event.ydata),
            roi_id=self._drag_marker_roi_id,
        )
        if nearest is None:
            return
        _trace, x_value, y_value = nearest
        if self._preview_marker_drag(
            self._drag_marker_id,
            x_value,
            y_value,
        ):
            self._drag_marker_moved = True
        self.markerDragPreviewed.emit(
            self._drag_marker_id,
            x_value,
            y_value,
        )

    def _handle_mouse_release(self, event: Any) -> None:
        if self._drag_marker_id is None:
            return
        marker_id = self._drag_marker_id
        delete_marker = (
            self._drag_marker_delete_pending
            or self.axes is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
        )
        self._drag_marker_id = None
        self._drag_marker_roi_id = None
        moved = self._drag_marker_moved
        self._drag_marker_moved = False
        self._drag_marker_delete_pending = False
        if self.canvas is not None:
            self.canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        if delete_marker:
            marker = self._marker_by_id(marker_id)
            self.markerDeleted.emit(marker_id)
            if marker is not None:
                self._start_marker_poof(
                    marker.integration_x,
                    marker.integrated_intensity,
                )
            return
        if not moved:
            return
        marker = self._marker_by_id(marker_id)
        if marker is None:
            return
        self.markerMoved.emit(
            marker.marker_id,
            marker.integration_x,
            marker.integrated_intensity,
        )

    def _mark_nearest_trace_point(self, event: Any) -> None:
        nearest = self._nearest_trace_point(
            float(event.xdata),
            float(event.ydata),
        )
        if nearest is None:
            return
        trace, x_value, y_value = nearest
        self.peakMarked.emit(
            trace.roi_id,
            float(x_value),
            float(y_value),
        )

    def _toolbar_is_active(self) -> bool:
        return self.toolbar is not None and bool(
            getattr(self.toolbar, "mode", "")
        )

    def _nearest_marker(
        self,
        event: Any,
        *,
        threshold_px: float = 10.0,
    ) -> IntegrationPeakMarker | None:
        if self.axes is None or event.x is None or event.y is None:
            return None
        nearest_marker: IntegrationPeakMarker | None = None
        nearest_distance = float("inf")
        for marker in self.markers:
            marker_x, marker_y = self.axes.transData.transform(
                (marker.integration_x, marker.integrated_intensity)
            )
            distance = float(np.hypot(marker_x - event.x, marker_y - event.y))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_marker = marker
        if nearest_distance <= threshold_px:
            return nearest_marker
        return None

    def _marker_by_id(
        self,
        marker_id: str,
    ) -> IntegrationPeakMarker | None:
        for marker in self.markers:
            if marker.marker_id == marker_id:
                return marker
        return None

    def _preview_marker_drag(
        self,
        marker_id: str,
        integration_x: float,
        integrated_intensity: float,
    ) -> bool:
        changed = False
        updated: list[IntegrationPeakMarker] = []
        for marker in self.markers:
            if marker.marker_id != marker_id:
                updated.append(marker)
                continue
            if (
                marker.integration_x == integration_x
                and marker.integrated_intensity == integrated_intensity
            ):
                updated.append(marker)
                continue
            updated.append(
                replace(
                    marker,
                    integration_x=float(integration_x),
                    integrated_intensity=float(integrated_intensity),
                )
            )
            changed = True
        if not changed:
            return False
        self.set_series(self.series, self.mode, updated)
        return True

    def _start_marker_poof(
        self,
        integration_x: float,
        integrated_intensity: float,
    ) -> None:
        if self.axes is None or self.canvas is None:
            return
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        x_span = max(abs(float(x_max) - float(x_min)), 1.0e-12)
        y_span = max(abs(float(y_max) - float(y_min)), 1.0e-12)
        center = np.asarray(
            [float(integration_x), float(integrated_intensity)],
            dtype=float,
        )
        directions = np.asarray(
            [
                (1.0, 0.0),
                (0.45, 0.9),
                (-0.45, 0.9),
                (-1.0, 0.0),
                (-0.45, -0.9),
                (0.45, -0.9),
            ],
            dtype=float,
        )
        spread = np.asarray([x_span * 0.035, y_span * 0.055], dtype=float)
        ring = self.axes.scatter(
            [center[0]],
            [center[1]],
            s=[36.0],
            marker="o",
            facecolors="none",
            edgecolors="#e76f51",
            linewidths=1.4,
            alpha=0.9,
            zorder=8,
        )
        sparks = self.axes.scatter(
            np.full(directions.shape[0], center[0]),
            np.full(directions.shape[0], center[1]),
            s=np.full(directions.shape[0], 14.0),
            marker="*",
            color="#f4a261",
            alpha=0.9,
            zorder=9,
        )
        artists = [ring, sparks]
        self._poof_artists.extend(artists)
        timer = QtCore.QTimer(self)
        frame = {"index": 0}
        frame_count = 9

        def advance_poof() -> None:
            progress = frame["index"] / max(frame_count - 1, 1)
            alpha = max(0.0, 1.0 - progress)
            ring.set_sizes([36.0 + 140.0 * progress])
            ring.set_alpha(alpha * 0.9)
            spark_offsets = center + directions * spread * progress
            sparks.set_offsets(spark_offsets)
            sparks.set_sizes(
                np.full(directions.shape[0], 14.0 + 22.0 * progress)
            )
            sparks.set_alpha(alpha * 0.85)
            if self.canvas is not None:
                self.canvas.draw_idle()
            frame["index"] += 1
            if frame["index"] >= frame_count:
                timer.stop()
                self._remove_poof_artists(artists)
                if timer in self._poof_timers:
                    self._poof_timers.remove(timer)
                timer.deleteLater()

        self._poof_timers.append(timer)
        timer.timeout.connect(advance_poof)
        timer.start(35)
        advance_poof()

    def _clear_poof_animations(self) -> None:
        for timer in list(self._poof_timers):
            timer.stop()
            timer.deleteLater()
        self._poof_timers.clear()
        self._remove_poof_artists(list(self._poof_artists))

    def _remove_poof_artists(self, artists: list[Any]) -> None:
        for artist in artists:
            try:
                artist.remove()
            except ValueError:
                pass
            if artist in self._poof_artists:
                self._poof_artists.remove(artist)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _nearest_trace_point(
        self,
        x_value: float,
        y_value: float,
        *,
        roi_id: str | None = None,
    ) -> tuple[_IntegrationTrace, float, float] | None:
        if not self.autosnap_enabled:
            return self._nearest_trace_sample(
                x_value,
                y_value,
                roi_id=roi_id,
            )
        nearest_peak = self._nearest_trace_local_maximum(
            x_value,
            y_value,
            roi_id=roi_id,
        )
        if nearest_peak is not None:
            return nearest_peak
        return self._nearest_trace_sample(
            x_value,
            y_value,
            roi_id=roi_id,
        )

    def _nearest_trace_local_maximum(
        self,
        x_value: float,
        y_value: float,
        *,
        roi_id: str | None = None,
    ) -> tuple[_IntegrationTrace, float, float] | None:
        nearest: tuple[_IntegrationTrace, float, float] | None = None
        nearest_distance = float("inf")
        maxima_by_trace = [
            (trace, _trace_local_maxima(trace))
            for trace in self.series
            if roi_id is None or trace.roi_id == roi_id
        ]
        if not any(maxima for _trace, maxima in maxima_by_trace):
            return None
        x_span, y_span = _trace_collection_spans(self.series)
        for trace, maxima in maxima_by_trace:
            for candidate_x, candidate_y, _index in maxima:
                distance = ((candidate_x - x_value) / x_span) ** 2 + (
                    (candidate_y - y_value) / y_span
                ) ** 2
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = (trace, candidate_x, candidate_y)
        return nearest

    def _nearest_trace_sample(
        self,
        x_value: float,
        y_value: float,
        *,
        roi_id: str | None = None,
    ) -> tuple[_IntegrationTrace, float, float] | None:
        nearest: tuple[_IntegrationTrace, float, float] | None = None
        nearest_distance = float("inf")
        x_span, y_span = _trace_collection_spans(self.series)
        for trace in self.series:
            if roi_id is not None and trace.roi_id != roi_id:
                continue
            x_values = np.asarray(trace.x_values, dtype=float)
            y_values = np.asarray(trace.y_values, dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(valid):
                continue
            valid_x = x_values[valid]
            valid_y = y_values[valid]
            index = int(np.argmin(np.abs(valid_x - x_value)))
            candidate_x = float(valid_x[index])
            candidate_y = float(valid_y[index])
            distance = ((candidate_x - x_value) / x_span) ** 2 + (
                (candidate_y - y_value) / y_span
            ) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = (trace, candidate_x, candidate_y)
        return nearest


def _trace_collection_spans(
    series: list[_IntegrationTrace],
) -> tuple[float, float]:
    x_span = max(
        (
            float(np.nanmax(trace.x_values) - np.nanmin(trace.x_values))
            for trace in series
            if trace.x_values.size and np.isfinite(trace.x_values).any()
        ),
        default=1.0,
    )
    y_span = max(
        (
            float(np.nanmax(trace.y_values) - np.nanmin(trace.y_values))
            for trace in series
            if trace.y_values.size and np.isfinite(trace.y_values).any()
        ),
        default=1.0,
    )
    return max(abs(x_span), 1.0e-12), max(abs(y_span), 1.0e-12)


def _trace_local_maxima(
    trace: _IntegrationTrace,
    *,
    include_edges: bool = False,
) -> list[tuple[float, float, int]]:
    x_values = np.asarray(trace.x_values, dtype=float)
    y_values = np.asarray(trace.y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid):
        return []
    compact_x = x_values[valid]
    compact_y = y_values[valid]
    indices = _local_maximum_indices(compact_y, include_edges=include_edges)
    return [
        (float(compact_x[index]), float(compact_y[index]), int(index))
        for index in indices
    ]


def _local_maximum_indices(
    values: np.ndarray,
    *,
    include_edges: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=int)
    if array.size == 1:
        return np.asarray([0], dtype=int) if include_edges else np.asarray([])
    indices: list[int] = []
    if include_edges and array[0] > array[1]:
        indices.append(0)
    if array.size > 2:
        middle = array[1:-1]
        left = array[:-2]
        right = array[2:]
        mask = (
            (middle >= left)
            & (middle >= right)
            & ((middle > left) | (middle > right))
        )
        indices.extend((np.flatnonzero(mask) + 1).astype(int).tolist())
    if include_edges and array[-1] > array[-2]:
        indices.append(array.size - 1)
    return np.asarray(indices, dtype=int)


def _auto_detect_trace_peaks(
    trace: _IntegrationTrace,
    *,
    max_peaks: int = CHANNEL_DETECT_MAX_PEAKS_PER_TRACE,
) -> list[tuple[float, float]]:
    x_values = np.asarray(trace.x_values, dtype=float)
    y_values = np.asarray(trace.y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid):
        return []
    compact_x = x_values[valid]
    compact_y = y_values[valid]
    maxima = _local_maximum_indices(compact_y, include_edges=False)
    if maxima.size == 0:
        return []
    y_min = float(np.nanmin(compact_y))
    y_max = float(np.nanmax(compact_y))
    dynamic_range = max(y_max - y_min, 1.0e-12)
    baseline = float(np.nanmedian(compact_y))
    noise = _robust_trace_noise(compact_y - baseline)
    min_height = max(
        dynamic_range * CHANNEL_DETECT_MIN_HEIGHT_FRACTION,
        min(noise * 3.0, dynamic_range * 0.25),
    )
    scored: list[tuple[float, int]] = []
    for index in maxima:
        height = float(compact_y[index] - baseline)
        if height < min_height:
            continue
        scored.append((height, int(index)))
    scored.sort(reverse=True)
    selected: list[int] = []
    min_separation = max(1, int(round(compact_x.size * 0.03)))
    for _height, index in scored:
        if len(selected) >= max(1, int(max_peaks)):
            break
        if any(abs(index - kept) < min_separation for kept in selected):
            continue
        selected.append(index)
    selected.sort(key=lambda index: float(compact_x[index]))
    return [
        (float(compact_x[index]), float(compact_y[index]))
        for index in selected
    ]


def _robust_trace_noise(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= 1.0e-12:
        q25, q75 = np.nanpercentile(finite, [25.0, 75.0])
        noise = float((q75 - q25) / 1.349)
    if not np.isfinite(noise) or noise <= 1.0e-12:
        noise = float(np.nanstd(finite))
    return max(noise, 0.0)


def _trace_x_tolerance(trace: _IntegrationTrace) -> float:
    x_values = np.asarray(trace.x_values, dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    if x_values.size < 2:
        return 1.0e-9
    diffs = np.diff(np.unique(np.sort(x_values)))
    positive = diffs[diffs > 0.0]
    if positive.size:
        return max(float(np.nanmin(positive)) / 2.0, 1.0e-9)
    span = float(np.nanmax(x_values) - np.nanmin(x_values))
    return max(abs(span) * 1.0e-9, 1.0e-9)


class _IntegrationChannelPanel(QtWidgets.QFrame):
    """Reserved home slot for one ROI integration channel."""

    clearRequested = QtCore.Signal(int)
    detachRequested = QtCore.Signal(int)
    reattachDropRequested = QtCore.Signal(int)
    markerRequested = QtCore.Signal(int, str, float, float)
    markerDragPreviewed = QtCore.Signal(int, str, float, float)
    markerMoved = QtCore.Signal(int, str, float, float)
    markerDeleted = QtCore.Signal(int, str)
    clearMarkersRequested = QtCore.Signal(int)
    pushMarkersRequested = QtCore.Signal(int)
    detectPeaksRequested = QtCore.Signal(int)
    autoSnapToggled = QtCore.Signal(int, bool)

    def __init__(
        self,
        channel: int,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel = channel
        self.series: list[_IntegrationTrace] = []
        self.mode: str | None = None
        self.detached = False
        self.setAcceptDrops(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setMinimumHeight(180)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(6, 4, 6, 2)
        self.drag_label = _DragHandleLabel(
            channel,
            _channel_header_text(channel, None),
        )
        self.drag_label.dragFinished.connect(self._handle_home_drag_finished)
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.clicked.connect(
            lambda _checked=False: self.clearRequested.emit(self.channel)
        )
        self.marker_count_label = QtWidgets.QLabel("0 marks")
        self.marker_count_label.setMinimumWidth(54)
        self.coordinate_readout_label = QtWidgets.QLabel("")
        self.coordinate_readout_label.setMinimumWidth(150)
        self.coordinate_readout_label.setStyleSheet("color: #475569;")
        self.clear_marks_button = QtWidgets.QToolButton()
        self.clear_marks_button.setText("Clear Marks")
        self.clear_marks_button.clicked.connect(
            lambda _checked=False: self.clearMarkersRequested.emit(
                self.channel
            )
        )
        self.detect_peaks_button = QtWidgets.QToolButton()
        self.detect_peaks_button.setText("Detect Peaks")
        self.detect_peaks_button.setEnabled(False)
        self.detect_peaks_button.clicked.connect(
            lambda _checked=False: self.detectPeaksRequested.emit(self.channel)
        )
        self.autosnap_button = QtWidgets.QToolButton()
        self.autosnap_button.setText("Autosnap")
        self.autosnap_button.setCheckable(True)
        self.autosnap_button.setChecked(True)
        self.autosnap_button.setToolTip(
            "Snap clicked or dragged peaks to local maxima"
        )
        self.autosnap_button.toggled.connect(
            lambda checked: self.autoSnapToggled.emit(self.channel, checked)
        )
        self.push_markers_button = QtWidgets.QToolButton()
        self.push_markers_button.setText("Push Peaks")
        self.push_markers_button.clicked.connect(
            lambda _checked=False: self.pushMarkersRequested.emit(self.channel)
        )
        self.detach_button = QtWidgets.QToolButton()
        self.detach_button.setText("\u2197")
        self.detach_button.setToolTip("Detach integration channel")
        self.detach_button.clicked.connect(
            lambda _checked=False: self.detachRequested.emit(self.channel)
        )
        header_layout.addWidget(self.drag_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.marker_count_label)
        header_layout.addWidget(self.coordinate_readout_label)
        header_layout.addWidget(self.clear_marks_button)
        header_layout.addWidget(self.detect_peaks_button)
        header_layout.addWidget(self.autosnap_button)
        header_layout.addWidget(self.push_markers_button)
        header_layout.addWidget(self.clear_button)
        header_layout.addWidget(self.detach_button)

        self.plot_widget = _MatplotlibIntegrationWidget(with_toolbar=False)
        self.plot_widget.peakMarked.connect(
            lambda roi_id, x_value, y_value: self.markerRequested.emit(
                self.channel,
                roi_id,
                x_value,
                y_value,
            )
        )
        self.plot_widget.markerDragPreviewed.connect(
            lambda marker_id, x_value, y_value: (
                self.markerDragPreviewed.emit(
                    self.channel,
                    marker_id,
                    x_value,
                    y_value,
                )
            )
        )
        self.plot_widget.markerMoved.connect(
            lambda marker_id, x_value, y_value: self.markerMoved.emit(
                self.channel,
                marker_id,
                x_value,
                y_value,
            )
        )
        self.plot_widget.markerDeleted.connect(
            lambda marker_id: self.markerDeleted.emit(
                self.channel,
                marker_id,
            )
        )
        self.placeholder = QtWidgets.QLabel(f"Channel {channel} detached")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.placeholder, stretch=1)

    def set_series(
        self,
        series: list[_IntegrationTrace],
        mode: str | None,
        markers: list[IntegrationPeakMarker] | None = None,
    ) -> None:
        self.series = list(series)
        self.mode = mode
        self.drag_label.setText(_channel_header_text(self.channel, mode))
        self.plot_widget.set_series(self.series, self.mode, markers)
        self.set_marker_count(len(markers or []))
        self.detect_peaks_button.setEnabled(bool(self.series))

    def set_marker_count(self, count: int) -> None:
        suffix = "mark" if count == 1 else "marks"
        self.marker_count_label.setText(f"{count} {suffix}")
        self.clear_marks_button.setEnabled(count > 0)
        self.push_markers_button.setEnabled(count > 0)

    def set_peak_readout(self, text: str) -> None:
        self.coordinate_readout_label.setText(text)

    def set_autosnap_enabled(self, enabled: bool) -> None:
        self.plot_widget.set_autosnap_enabled(enabled)
        previous = self.autosnap_button.blockSignals(True)
        try:
            self.autosnap_button.setChecked(bool(enabled))
        finally:
            self.autosnap_button.blockSignals(previous)

    def set_detached(self, detached: bool) -> None:
        self.detached = detached
        self.plot_widget.setVisible(not detached)
        self.placeholder.setVisible(detached)
        self.detach_button.setEnabled(not detached)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._accepts_channel_drop(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if self._accepts_channel_drop(event.mimeData()):
            self.reattachDropRequested.emit(self.channel)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _accepts_channel_drop(self, mime_data: QtCore.QMimeData) -> bool:
        if not self.detached or not mime_data.hasFormat(CHANNEL_MIME_TYPE):
            return False
        try:
            channel = int(bytes(mime_data.data(CHANNEL_MIME_TYPE)).decode())
        except ValueError:
            return False
        return channel == self.channel

    def _handle_home_drag_finished(
        self, _channel: int, accepted: bool
    ) -> None:
        if not accepted:
            self.detachRequested.emit(self.channel)


class _DetachedIntegrationWindow(QtWidgets.QDialog):
    """Floating matplotlib window for one detached integration
    channel."""

    reattachRequested = QtCore.Signal(int)
    clearRequested = QtCore.Signal(int)
    markerRequested = QtCore.Signal(int, str, float, float)
    markerDragPreviewed = QtCore.Signal(int, str, float, float)
    markerMoved = QtCore.Signal(int, str, float, float)
    markerDeleted = QtCore.Signal(int, str)
    clearMarkersRequested = QtCore.Signal(int)
    pushMarkersRequested = QtCore.Signal(int)
    detectPeaksRequested = QtCore.Signal(int)
    autoSnapToggled = QtCore.Signal(int, bool)

    def __init__(
        self,
        channel: int,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel = channel
        self._closing_from_viewer = False
        self.setWindowTitle(f"EWALD Integration Channel {channel}")
        self.resize(640, 460)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(6, 4, 6, 2)
        self.drag_label = _DragHandleLabel(
            channel,
            _channel_header_text(channel, None),
        )
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.clicked.connect(
            lambda _checked=False: self.clearRequested.emit(self.channel)
        )
        self.marker_count_label = QtWidgets.QLabel("0 marks")
        self.coordinate_readout_label = QtWidgets.QLabel("")
        self.coordinate_readout_label.setMinimumWidth(150)
        self.coordinate_readout_label.setStyleSheet("color: #475569;")
        self.clear_marks_button = QtWidgets.QToolButton()
        self.clear_marks_button.setText("Clear Marks")
        self.clear_marks_button.clicked.connect(
            lambda _checked=False: self.clearMarkersRequested.emit(
                self.channel
            )
        )
        self.detect_peaks_button = QtWidgets.QToolButton()
        self.detect_peaks_button.setText("Detect Peaks")
        self.detect_peaks_button.setEnabled(False)
        self.detect_peaks_button.clicked.connect(
            lambda _checked=False: self.detectPeaksRequested.emit(self.channel)
        )
        self.autosnap_button = QtWidgets.QToolButton()
        self.autosnap_button.setText("Autosnap")
        self.autosnap_button.setCheckable(True)
        self.autosnap_button.setChecked(True)
        self.autosnap_button.setToolTip(
            "Snap clicked or dragged peaks to local maxima"
        )
        self.autosnap_button.toggled.connect(
            lambda checked: self.autoSnapToggled.emit(self.channel, checked)
        )
        self.push_markers_button = QtWidgets.QToolButton()
        self.push_markers_button.setText("Push Peaks")
        self.push_markers_button.clicked.connect(
            lambda _checked=False: self.pushMarkersRequested.emit(self.channel)
        )
        self.return_button = QtWidgets.QToolButton()
        self.return_button.setText("Return")
        self.return_button.clicked.connect(
            lambda _checked=False: self.reattachRequested.emit(self.channel)
        )
        header_layout.addWidget(self.drag_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.marker_count_label)
        header_layout.addWidget(self.coordinate_readout_label)
        header_layout.addWidget(self.clear_marks_button)
        header_layout.addWidget(self.detect_peaks_button)
        header_layout.addWidget(self.autosnap_button)
        header_layout.addWidget(self.push_markers_button)
        header_layout.addWidget(self.clear_button)
        header_layout.addWidget(self.return_button)

        self.plot_widget = _MatplotlibIntegrationWidget(with_toolbar=True)
        self.plot_widget.peakMarked.connect(
            lambda roi_id, x_value, y_value: self.markerRequested.emit(
                self.channel,
                roi_id,
                x_value,
                y_value,
            )
        )
        self.plot_widget.markerDragPreviewed.connect(
            lambda marker_id, x_value, y_value: (
                self.markerDragPreviewed.emit(
                    self.channel,
                    marker_id,
                    x_value,
                    y_value,
                )
            )
        )
        self.plot_widget.markerMoved.connect(
            lambda marker_id, x_value, y_value: self.markerMoved.emit(
                self.channel,
                marker_id,
                x_value,
                y_value,
            )
        )
        self.plot_widget.markerDeleted.connect(
            lambda marker_id: self.markerDeleted.emit(
                self.channel,
                marker_id,
            )
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.plot_widget, stretch=1)

    def set_series(
        self,
        series: list[_IntegrationTrace],
        mode: str | None,
        markers: list[IntegrationPeakMarker] | None = None,
    ) -> None:
        self.drag_label.setText(_channel_header_text(self.channel, mode))
        self.plot_widget.set_series(series, mode, markers)
        self.set_marker_count(len(markers or []))
        self.detect_peaks_button.setEnabled(bool(series))

    def set_marker_count(self, count: int) -> None:
        suffix = "mark" if count == 1 else "marks"
        self.marker_count_label.setText(f"{count} {suffix}")
        self.clear_marks_button.setEnabled(count > 0)
        self.push_markers_button.setEnabled(count > 0)

    def set_peak_readout(self, text: str) -> None:
        self.coordinate_readout_label.setText(text)

    def set_autosnap_enabled(self, enabled: bool) -> None:
        self.plot_widget.set_autosnap_enabled(enabled)
        previous = self.autosnap_button.blockSignals(True)
        try:
            self.autosnap_button.setChecked(bool(enabled))
        finally:
            self.autosnap_button.blockSignals(previous)

    def close_from_viewer(self) -> None:
        self._closing_from_viewer = True
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._closing_from_viewer:
            self._closing_from_viewer = True
            self.reattachRequested.emit(self.channel)
        super().closeEvent(event)


class DataViewerPane(QtWidgets.QWidget):
    """Display corrected detector data and user-defined integration
    ROIs."""

    roiRegionsChanged = QtCore.Signal(str)
    previewOrientationChanged = QtCore.Signal(str)
    imageStyleChanged = QtCore.Signal(object)
    integrationPeakMarkersPushed = QtCore.Signal(str, object)
    poleFigureRequested = QtCore.Signal(str, object, object, object)

    def __init__(
        self,
        project: ProjectState,
        data_id: str,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.data_id = data_id
        self.axis_ranges: tuple[float, float, float, float] | None = None
        self.coordinate_space = "pixel"
        self.image_data = self._load_display_image()
        self.orientation_controls_enabled = (
            not project.image_corrections_confirmed(data_id)
            and self.coordinate_space == "pixel"
        )
        self.roi_controls_enabled = (
            project.image_corrections_confirmed(data_id)
            and self.coordinate_space == "qspace"
        )
        self.roi_graphics: dict[str, Any] = {}
        self.low_q_graphics: list[Any] = []
        self.channel_assignments: dict[int, set[str]] = {1: set(), 2: set()}
        self.channel_modes: dict[int, str | None] = {1: None, 2: None}
        self.channel_autosnap_enabled: dict[int, bool] = {
            1: True,
            2: True,
        }
        self.channel_panels: dict[int, _IntegrationChannelPanel] = {}
        self.channel_windows: dict[int, _DetachedIntegrationWindow] = {}
        self.integration_peak_markers: dict[
            int,
            list[IntegrationPeakMarker],
        ] = {1: [], 2: []}
        self._syncing_roi_table = False
        self.plot_frame: _ImageAspectPlotFrame | None = None

        self._build_controls()
        self._build_plot()
        self._build_channel_panels()
        self._build_roi_table()
        self._build_layout()
        self._set_roi_controls_enabled(self.roi_controls_enabled)
        self._set_initial_image()
        self._sync_roi_table()

    def add_roi_from_bounds(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> ROIRegion | None:
        """Create a persisted ROI from viewer coordinates."""

        if not self.roi_controls_enabled:
            return None
        if x0 == x1 or y0 == y1:
            return None
        x_min, x_max = sorted((float(x0), float(x1)))
        y_min, y_max = sorted((float(y0), float(y1)))
        existing_count = len(self.project.rois_for_target(self.data_id))
        if self.arch_button.isChecked():
            qr_min, qr_max, chi_min, chi_max = _arch_polar_from_plot_bounds(
                x_min,
                x_max,
                y_min,
                y_max,
            )
            roi = ROIRegion(
                target_id=self.data_id,
                kind="arch",
                name=f"Arch {existing_count + 1}",
                qr_min=qr_min,
                qr_max=qr_max,
                chi_min=chi_min,
                chi_max=chi_max,
                integration_axis="chi",
                integration_direction="azimuthal",
                metadata={ARCH_CHI_LOCKED_METADATA_KEY: True},
            )
        else:
            axis = str(self.box_axis_combo.currentData())
            roi = ROIRegion(
                target_id=self.data_id,
                kind="box",
                name=f"Box {existing_count + 1}",
                qxy_min=x_min,
                qxy_max=x_max,
                qz_min=y_min,
                qz_max=y_max,
                integration_axis=axis,
                integration_direction=(
                    "vertical" if axis == "qz" else "horizontal"
                ),
            )
        stored = self.project.add_roi_region(roi)
        self._add_roi_graphic(stored)
        self._sync_roi_table()
        self._select_roi(stored.roi_id)
        self.roiRegionsChanged.emit(self.data_id)
        return stored

    def add_coupled_roi_pair_from_bounds(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> tuple[ROIRegion, ROIRegion] | None:
        """Create coupled box and arch ROIs with a shared center."""

        if not self.roi_controls_enabled:
            return None
        if x0 == x1 or y0 == y1:
            return None
        x_min, x_max = sorted((float(x0), float(x1)))
        y_min, y_max = sorted((float(y0), float(y1)))
        center_qxy = (x_min + x_max) / 2.0
        center_qz = (y_min + y_max) / 2.0
        existing_count = len(self.project.rois_for_target(self.data_id))
        axis = str(self.box_axis_combo.currentData())
        group_id = _unique_coupled_roi_group_id(
            self.data_id, existing_count + 1
        )
        box = ROIRegion(
            target_id=self.data_id,
            kind="box",
            name=f"Coupled Box {existing_count + 1}",
            qxy_min=x_min,
            qxy_max=x_max,
            qz_min=y_min,
            qz_max=y_max,
            integration_axis=axis,
            integration_direction="vertical" if axis == "qz" else "horizontal",
            metadata={
                COUPLED_ROI_GROUP_METADATA_KEY: group_id,
                COUPLED_ROI_ROLE_METADATA_KEY: "box",
                COUPLED_ROI_SHARED_CENTER_METADATA_KEY: True,
            },
        )
        radius_center = self.arch_radius.value()
        if radius_center <= 0.0:
            radius_center = max(
                min(x_max - x_min, y_max - y_min) / 2.0, 1.0e-9
            )
        thickness = max(
            self.arch_thickness.value(),
            radius_center * 0.2,
            1.0e-9,
        )
        radius_center = max(radius_center, thickness / 2.0)
        chi_min, chi_max = sorted(
            (self.arch_chi_min.value(), self.arch_chi_max.value())
        )
        if chi_min == chi_max:
            chi_min, chi_max = -45.0, 45.0
        arch = ROIRegion(
            target_id=self.data_id,
            kind="arch",
            name=f"Coupled Arch {existing_count + 2}",
            qxy_center=center_qxy,
            qz_center=center_qz,
            qr_min=max(0.0, radius_center - thickness / 2.0),
            qr_max=max(1.0e-9, radius_center + thickness / 2.0),
            chi_min=chi_min,
            chi_max=chi_max,
            integration_axis="chi",
            integration_direction="azimuthal",
            metadata={
                ARCH_CHI_LOCKED_METADATA_KEY: False,
                COUPLED_ROI_GROUP_METADATA_KEY: group_id,
                COUPLED_ROI_ROLE_METADATA_KEY: "arch",
                COUPLED_ROI_SHARED_CENTER_METADATA_KEY: True,
            },
        )
        stored_box = self.project.add_roi_region(box)
        stored_arch = self.project.add_roi_region(arch)
        _couple_roi_pair(stored_box, stored_arch, group_id=group_id)
        self._add_roi_graphic(stored_box)
        self._add_roi_graphic(stored_arch)
        self._sync_roi_table()
        self._select_roi(stored_box.roi_id)
        self._set_roi_status(
            f"Created coupled ROI pair: box for {QXY_HTML}/{QZ_HTML} traces, "
            "arch for azimuthal traces."
        )
        self.roiRegionsChanged.emit(self.data_id)
        return stored_box, stored_arch

    def _add_coupled_roi_pair_from_view_range(self) -> None:
        if not self.roi_controls_enabled:
            return
        if pg is None or self.view_box is None:
            self.add_coupled_roi_pair_from_bounds(0.0, 1.0, 0.0, 1.0)
            return
        (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
        x_pad = (x_max - x_min) * 0.35
        y_pad = (y_max - y_min) * 0.35
        self.add_coupled_roi_pair_from_bounds(
            x_min + x_pad,
            x_max - x_pad,
            y_min + y_pad,
            y_max - y_pad,
        )

    def _build_controls(self) -> None:
        self.colormap_combo = QtWidgets.QComboBox()
        for name in IMAGE_COLORMAPS:
            self.colormap_combo.addItem(name.title(), name)
        self.colormap_combo.currentIndexChanged.connect(
            self._apply_image_style
        )

        self.level_min = _level_spinbox()
        self.level_max = _level_spinbox()
        self.level_min.valueChanged.connect(self._apply_manual_levels)
        self.level_max.valueChanged.connect(self._apply_manual_levels)

        self.quantile_check = QtWidgets.QCheckBox("Quantile")
        self.quantile_check.setChecked(True)
        self.quantile_check.toggled.connect(self._apply_image_style)
        self.quantile_low = _quantile_spinbox(1.0)
        self.quantile_high = _quantile_spinbox(99.0)
        self.quantile_low.valueChanged.connect(self._apply_image_style)
        self.quantile_high.valueChanged.connect(self._apply_image_style)

        self.auto_contrast_button = QtWidgets.QToolButton()
        self.auto_contrast_button.setText("Auto")
        self.auto_contrast_button.clicked.connect(self._set_quantile_levels)

        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("Zoom In")
        self.zoom_in_button.clicked.connect(lambda: self._zoom_image(0.75))
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("Zoom Out")
        self.zoom_out_button.clicked.connect(lambda: self._zoom_image(1.35))
        self.zoom_fit_button = QtWidgets.QToolButton()
        self.zoom_fit_button.setText("Autoscale")
        self.zoom_fit_button.clicked.connect(self._reset_image_zoom)
        self.pan_button = QtWidgets.QToolButton()
        self.pan_button.setText("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.toggled.connect(self._set_pan_mode)

        self.rotate_left_button = _orientation_tool_button(
            "\u21b6",
            "Rotate raw preview 90 deg counter-clockwise",
        )
        self.rotate_left_button.clicked.connect(
            lambda _checked=False: self._rotate_raw_preview(-90)
        )
        self.rotate_right_button = _orientation_tool_button(
            "\u21b7",
            "Rotate raw preview 90 deg clockwise",
        )
        self.rotate_right_button.clicked.connect(
            lambda _checked=False: self._rotate_raw_preview(90)
        )
        self.mirror_y_button = _orientation_tool_button(
            "\u21c4",
            "Mirror raw preview over the active y-axis",
        )
        self.mirror_y_button.setCheckable(True)
        state = self.project.image_corrections.get(self.data_id)
        self.mirror_y_button.setChecked(
            state.image_mirrored_y if state else False
        )
        self.mirror_y_button.clicked.connect(self._toggle_raw_preview_mirror)

        self.box_button = QtWidgets.QToolButton()
        self.box_button.setText("Box")
        self.box_button.setCheckable(True)
        self.box_button.setChecked(True)
        self.arch_button = QtWidgets.QToolButton()
        self.arch_button.setText("Arch")
        self.arch_button.setCheckable(True)
        self.roi_mode_group = QtWidgets.QButtonGroup(self)
        self.roi_mode_group.setExclusive(True)
        self.roi_mode_group.addButton(self.box_button)
        self.roi_mode_group.addButton(self.arch_button)

        self.box_axis_combo = RichTextComboBox()
        self.box_axis_combo.addItem(QZ_HTML, "qz")
        self.box_axis_combo.addItem(QXY_HTML, "qxy")
        self.arch_radius = _roi_radius_spinbox()
        self.arch_radius.setValue(0.5)
        self.arch_thickness = _roi_radius_spinbox()
        self.arch_thickness.setValue(0.1)
        self.arch_chi_min = _roi_chi_spinbox(-45.0)
        self.arch_chi_max = _roi_chi_spinbox(45.0)
        self.apply_arch_button = QtWidgets.QToolButton()
        self.apply_arch_button.setText("Apply Arch")
        self.apply_arch_button.clicked.connect(self._apply_arch_adjustments)

        self.draw_toggle = QtWidgets.QToolButton()
        self.draw_toggle.setText("Draw")
        self.draw_toggle.setCheckable(True)
        self.draw_toggle.toggled.connect(self._set_drawing_enabled)

        self.add_roi_button = QtWidgets.QToolButton()
        self.add_roi_button.setText("Add ROI")
        self.add_roi_button.clicked.connect(self._add_roi_from_view_range)

        self.add_coupled_roi_button = QtWidgets.QToolButton()
        self.add_coupled_roi_button.setText("Add Coupled Pair")
        self.add_coupled_roi_button.setToolTip(
            "Create a box ROI and an arch ROI with the same center."
        )
        self.add_coupled_roi_button.clicked.connect(
            self._add_coupled_roi_pair_from_view_range
        )

        self.remove_roi_button = QtWidgets.QToolButton()
        self.remove_roi_button.setText("Remove")
        self.remove_roi_button.clicked.connect(self._remove_selected_roi)

        self.remove_coupled_pair_button = QtWidgets.QToolButton()
        self.remove_coupled_pair_button.setText("Remove Pair")
        self.remove_coupled_pair_button.clicked.connect(
            self._remove_selected_coupled_pair
        )

        self.decouple_roi_button = QtWidgets.QToolButton()
        self.decouple_roi_button.setText("Decouple")
        self.decouple_roi_button.clicked.connect(self._decouple_selected_roi)

        self.clear_roi_button = QtWidgets.QToolButton()
        self.clear_roi_button.setText("Clear")
        self.clear_roi_button.clicked.connect(self._clear_rois)
        self.open_pole_figure_button = QtWidgets.QToolButton()
        self.open_pole_figure_button.setText("Open Pole Figure Generator")
        self.open_pole_figure_button.setToolTip(
            "Open the selected ROI in the Pole Figure Generator."
        )
        self.open_pole_figure_button.clicked.connect(self._request_pole_figure)
        self.roi_status_label = QtWidgets.QLabel("")
        self.roi_status_label.setWordWrap(True)
        self.roi_status_label.setStyleSheet("color: #9a3412;")

    def _build_plot(self) -> None:
        if pg is None:
            self.view_box = None
            self.image_item = None
            self.plot_widget = QtWidgets.QLabel("Corrected data")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            return

        self.view_box = _DrawingViewBox(self.add_roi_from_bounds)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        if self.coordinate_space == "qspace":
            set_qspace_axis_labels(self.plot_widget)
        else:
            self.plot_widget.setLabel("bottom", "x", units="px")
            self.plot_widget.setLabel("left", "y", units="px")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        set_data_aspect_locked(self.plot_widget)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)

    def _build_channel_panels(self) -> None:
        self.channel_stack = QtWidgets.QWidget()
        self.channel_stack.setMinimumWidth(360)
        self.channel_stack.setMaximumWidth(560)
        layout = QtWidgets.QVBoxLayout(self.channel_stack)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(6)
        for channel in (1, 2):
            panel = _IntegrationChannelPanel(channel)
            panel.clearRequested.connect(self._clear_channel)
            panel.detachRequested.connect(self._detach_channel)
            panel.reattachDropRequested.connect(self._reattach_channel)
            panel.markerRequested.connect(self._add_channel_peak_marker)
            panel.markerDragPreviewed.connect(
                self._preview_channel_peak_marker_coordinate
            )
            panel.markerMoved.connect(self._move_channel_peak_marker)
            panel.markerDeleted.connect(self._delete_channel_peak_marker)
            panel.clearMarkersRequested.connect(self._clear_channel_markers)
            panel.pushMarkersRequested.connect(self._push_channel_markers)
            panel.detectPeaksRequested.connect(self._detect_channel_peaks)
            panel.autoSnapToggled.connect(self._set_channel_autosnap_enabled)
            panel.set_autosnap_enabled(self.channel_autosnap_enabled[channel])
            self.channel_panels[channel] = panel
            layout.addWidget(panel, stretch=1)

    def _build_roi_table(self) -> None:
        self.roi_table = QtWidgets.QTableWidget(0, len(ROI_TABLE_HEADERS))
        set_rich_text_table_headers(self.roi_table, ROI_TABLE_HEADERS)
        enable_rich_text_items(self.roi_table)
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.roi_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.roi_table.setMinimumHeight(170)
        self.roi_table.itemSelectionChanged.connect(
            self._handle_roi_selection_changed
        )
        self.roi_table.itemChanged.connect(self._handle_roi_table_item_changed)

    def _build_layout(self) -> None:
        self.orientation_bar = QtWidgets.QWidget()
        orientation_layout = QtWidgets.QHBoxLayout(self.orientation_bar)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.addWidget(self.rotate_left_button)
        orientation_layout.addWidget(self.rotate_right_button)
        orientation_layout.addWidget(self.mirror_y_button)
        orientation_layout.addStretch(1)

        self.plot_toolbar = ImagePlotToolbar(
            colormap_combo=self.colormap_combo,
            level_min=self.level_min,
            level_max=self.level_max,
            quantile_check=self.quantile_check,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            auto_contrast_button=self.auto_contrast_button,
            zoom_in_button=self.zoom_in_button,
            zoom_out_button=self.zoom_out_button,
            autoscale_button=self.zoom_fit_button,
            pan_button=self.pan_button,
        )

        roi_layout = QtWidgets.QHBoxLayout()
        roi_layout.addWidget(self.box_button)
        roi_layout.addWidget(self.arch_button)
        roi_layout.addWidget(QtWidgets.QLabel("Box integrate"))
        roi_layout.addWidget(self.box_axis_combo)
        roi_layout.addSpacing(10)
        roi_layout.addWidget(QtWidgets.QLabel("Arch radius"))
        roi_layout.addWidget(self.arch_radius)
        roi_layout.addWidget(QtWidgets.QLabel("Arch thickness"))
        roi_layout.addWidget(self.arch_thickness)
        roi_layout.addWidget(QtWidgets.QLabel("chi min"))
        roi_layout.addWidget(self.arch_chi_min)
        roi_layout.addWidget(QtWidgets.QLabel("chi max"))
        roi_layout.addWidget(self.arch_chi_max)
        roi_layout.addWidget(self.apply_arch_button)
        roi_layout.addWidget(self.draw_toggle)
        roi_layout.addWidget(self.add_roi_button)
        roi_layout.addWidget(self.add_coupled_roi_button)
        roi_layout.addWidget(self.remove_roi_button)
        roi_layout.addWidget(self.remove_coupled_pair_button)
        roi_layout.addWidget(self.decouple_roi_button)
        roi_layout.addWidget(self.clear_roi_button)
        roi_layout.addWidget(self.open_pole_figure_button)
        roi_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.orientation_bar)
        layout.addWidget(self.plot_toolbar)
        self.plot_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        plot_area = self.plot_widget
        if pg is not None and self.image_item is not None:
            self.plot_frame = _ImageAspectPlotFrame(self.plot_widget)
            plot_area = self.plot_frame
        self.plot_splitter.addWidget(plot_area)
        self.plot_splitter.addWidget(self.channel_stack)
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 0)
        self.plot_splitter.setSizes([920, 440])
        layout.addWidget(self.plot_splitter, stretch=1)
        layout.addLayout(roi_layout)
        layout.addWidget(self.roi_status_label)
        layout.addWidget(self.roi_table)
        self._set_orientation_controls_enabled(
            self.orientation_controls_enabled
        )

    def _set_initial_image(self) -> None:
        if self.image_data is None:
            return
        finite = self.image_data[np.isfinite(self.image_data)]
        if finite.size:
            self.level_min.setValue(float(np.nanmin(finite)))
            self.level_max.setValue(float(np.nanmax(finite)))
        if self.image_item is not None:
            self.image_item.setImage(self.image_data, autoLevels=False)
            image_rect = data_image_rect(
                self.image_data.shape,
                self.axis_ranges,
            )
            self.image_item.setRect(image_rect)
            if self.plot_frame is not None:
                self.plot_frame.set_data_rect(image_rect)
        if pg is not None and self.plot_widget is not None:
            set_data_image_plot_range(
                self.plot_widget,
                self.image_data.shape,
                self.axis_ranges,
            )
        self._set_quantile_levels()
        self._add_low_q_feature_graphics()
        for roi in self.project.rois_for_target(self.data_id):
            self._add_roi_graphic(roi)

    def _load_display_image(self) -> np.ndarray | None:
        data_file = self.project.data_file_by_id(self.data_id)
        if data_file is None:
            return None
        state = self.project.image_corrections.get(self.data_id)
        try:
            import tifffile

            image = tifffile.imread(data_file.usable_path)
        except Exception:
            return None
        array = np.asarray(image)
        if array.ndim > 2:
            array = array[0]
        corrected = self._try_qspace_map(array)
        if corrected is not None:
            return corrected
        return _apply_image_orientation(
            array,
            state.image_rotation_deg if state else 0,
            state.image_mirrored_y if state else False,
        ).astype(float, copy=False)

    def _try_qspace_map(self, image: np.ndarray) -> np.ndarray | None:
        state = self.project.image_corrections.get(self.data_id)
        if not state or not state.confirmed or not state.calibrant_asset_id:
            return None
        try:
            calibrant = self.project.get_correction_asset(
                "calibrant",
                state.calibrant_asset_id,
            )
        except KeyError:
            return None
        if calibrant.usable_path is None:
            return None
        mask = None
        if state.mask_asset_id:
            try:
                mask_asset = self.project.get_correction_asset(
                    "mask",
                    state.mask_asset_id,
                )
                if mask_asset.usable_path is not None:
                    mask = _load_mask(mask_asset.usable_path)
            except Exception:
                mask = None
        try:
            from ewald.processing.qspace import (
                GrazingIncidenceConfig,
                map_grazing_incidence_qspace,
            )

            data_file = self.project.data_file_by_id(self.data_id)
            incidence_angle = 0.0
            if data_file is not None:
                incidence_angle = float(
                    data_file.metadata.get("incidence_angle_deg", 0.0)
                )
            qmap = map_grazing_incidence_qspace(
                image,
                calibrant.usable_path,
                config=GrazingIncidenceConfig(
                    npt_ip=512,
                    npt_oop=512,
                    xray_energy_kev=state.xray_energy_kev,
                    incident_angle_deg=incidence_angle,
                    sample_orientation=state.pyfai_sample_orientation,
                    correct_solid_angle=state.correct_solid_angle,
                    polarization_factor=state.polarization_factor,
                    normalization_factor=state.normalization_factor,
                    dummy=state.dummy,
                    delta_dummy=state.delta_dummy,
                ),
                mask=mask,
            )
        except Exception:
            return None
        q_ip = np.asarray(qmap.coords["q_ip"].values, dtype=float)
        q_oop = np.asarray(qmap.coords["q_oop"].values, dtype=float)
        self.axis_ranges = (
            float(np.nanmin(q_ip)),
            float(np.nanmax(q_ip)),
            float(np.nanmin(q_oop)),
            float(np.nanmax(q_oop)),
        )
        self.coordinate_space = "qspace"
        return np.asarray(qmap.values, dtype=float)

    def _set_drawing_enabled(self, enabled: bool) -> None:
        if enabled and getattr(self, "pan_button", None) is not None:
            self.pan_button.setChecked(False)
        if self.view_box is not None:
            self.view_box.drawing_enabled = enabled
        self._sync_view_interaction_mode()

    def _zoom_image(self, factor: float) -> None:
        if pg is None or self.view_box is None:
            return
        self.view_box.scaleBy((factor, factor))

    def _reset_image_zoom(self) -> None:
        if pg is None or self.plot_widget is None or self.image_data is None:
            return
        if self.axis_ranges is not None:
            x_min, x_max, y_min, y_max = self.axis_ranges
            self.plot_widget.setRange(
                xRange=(x_min, x_max),
                yRange=(y_min, y_max),
                padding=0.0,
            )
            return
        height, width = self.image_data.shape[-2:]
        self.plot_widget.setRange(
            xRange=(0, width),
            yRange=(0, height),
            padding=0.0,
        )

    def _set_pan_mode(self, enabled: bool) -> None:
        if enabled and getattr(self, "draw_toggle", None) is not None:
            self.draw_toggle.setChecked(False)
        self._sync_view_interaction_mode()

    def _sync_view_interaction_mode(self) -> None:
        pan_enabled = bool(
            getattr(self, "pan_button", None) is not None
            and self.pan_button.isChecked()
        )
        draw_enabled = bool(
            getattr(self, "draw_toggle", None) is not None
            and self.draw_toggle.isChecked()
            and self.draw_toggle.isEnabled()
        )
        if self.view_box is not None:
            self.view_box.drawing_enabled = draw_enabled
        self._set_view_drag_enabled(
            pan_enabled or not self.roi_controls_enabled
        )
        if self.plot_widget is not None:
            if pan_enabled:
                cursor = QtCore.Qt.CursorShape.OpenHandCursor
            elif draw_enabled:
                cursor = QtCore.Qt.CursorShape.CrossCursor
            else:
                cursor = QtCore.Qt.CursorShape.ArrowCursor
            self.plot_widget.setCursor(cursor)

    def _set_orientation_controls_enabled(self, enabled: bool) -> None:
        self.orientation_bar.setVisible(enabled)
        for widget in (
            self.rotate_left_button,
            self.rotate_right_button,
            self.mirror_y_button,
        ):
            widget.setVisible(enabled)
            widget.setEnabled(enabled)

    def _set_roi_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.box_button,
            self.arch_button,
            self.box_axis_combo,
            self.arch_radius,
            self.arch_thickness,
            self.arch_chi_min,
            self.arch_chi_max,
            self.apply_arch_button,
            self.draw_toggle,
            self.add_roi_button,
            self.add_coupled_roi_button,
            self.remove_roi_button,
            self.remove_coupled_pair_button,
            self.decouple_roi_button,
            self.clear_roi_button,
            self.open_pole_figure_button,
            self.roi_table,
        ):
            widget.setEnabled(enabled)
        for panel in self.channel_panels.values():
            panel.setEnabled(enabled)
        if enabled:
            self._sync_arch_controls_from_selection()
            self._refresh_all_channels()
        self._sync_view_interaction_mode()
        self._update_pole_figure_button()

    def _add_roi_from_view_range(self) -> None:
        if not self.roi_controls_enabled:
            return
        if pg is None or self.view_box is None:
            self.add_roi_from_bounds(0.0, 1.0, 0.0, 1.0)
            return
        (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
        x_pad = (x_max - x_min) * 0.35
        y_pad = (y_max - y_min) * 0.35
        self.add_roi_from_bounds(
            x_min + x_pad,
            x_max - x_pad,
            y_min + y_pad,
            y_max - y_pad,
        )

    def _remove_selected_roi(self) -> None:
        row = self.roi_table.currentRow()
        if row < 0:
            return
        roi_id_item = self.roi_table.item(row, 0)
        if roi_id_item is None:
            return
        roi_id = roi_id_item.data(QtCore.Qt.ItemDataRole.UserRole)
        roi = self._roi_by_id(str(roi_id))
        for partner in self._coupled_rois_for(roi):
            _decouple_roi(partner, str(roi_id))
        regions = [
            roi
            for roi in self.project.rois_for_target(self.data_id)
            if roi.roi_id != roi_id
        ]
        self.project.set_roi_regions(self.data_id, regions)
        self._remove_roi_graphic(str(roi_id))
        for assignments in self.channel_assignments.values():
            assignments.discard(str(roi_id))
        for channel in self.integration_peak_markers:
            self._remove_channel_markers(channel, roi_id=str(roi_id))
        self._sync_roi_table()
        self._refresh_all_channels()
        self.roiRegionsChanged.emit(self.data_id)
        self._sync_arch_controls_from_selection()

    def _remove_selected_coupled_pair(self) -> None:
        roi = self._selected_roi_region()
        if roi is None:
            self._set_roi_status(
                "Select a coupled ROI pair before removing it."
            )
            return
        pair_ids = {roi.roi_id or "", *self._coupled_roi_ids(roi)}
        pair_ids.discard("")
        if len(pair_ids) <= 1:
            self._set_roi_status("The selected ROI is not coupled.")
            return
        regions = [
            candidate
            for candidate in self.project.rois_for_target(self.data_id)
            if candidate.roi_id not in pair_ids
        ]
        self.project.set_roi_regions(self.data_id, regions)
        for roi_id in pair_ids:
            self._remove_roi_graphic(roi_id)
            for assignments in self.channel_assignments.values():
                assignments.discard(roi_id)
            for channel in self.integration_peak_markers:
                self._remove_channel_markers(channel, roi_id=roi_id)
        self._sync_roi_table()
        self._refresh_all_channels()
        self._set_roi_status("Removed coupled ROI pair.")
        self.roiRegionsChanged.emit(self.data_id)
        self._sync_arch_controls_from_selection()

    def _decouple_selected_roi(self) -> None:
        roi = self._selected_roi_region()
        if roi is None:
            self._set_roi_status("Select a coupled ROI before decoupling.")
            return
        coupled = self._coupled_rois_for(roi)
        if not coupled:
            self._set_roi_status("The selected ROI is not coupled.")
            return
        for candidate in [roi, *coupled]:
            _decouple_roi(candidate)
        self._sync_roi_table()
        self._select_roi(roi.roi_id)
        self._set_roi_status(
            "Decoupled ROI pair; each ROI can now move independently."
        )
        self.roiRegionsChanged.emit(self.data_id)
        self._sync_arch_controls_from_selection()

    def _clear_rois(self) -> None:
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id:
                self._remove_roi_graphic(roi.roi_id)
        self.project.set_roi_regions(self.data_id, [])
        for channel in self.channel_assignments:
            self.channel_assignments[channel].clear()
            self.channel_modes[channel] = None
            self.integration_peak_markers[channel].clear()
        self._sync_roi_table()
        self._refresh_all_channels()
        self.roiRegionsChanged.emit(self.data_id)
        self._sync_arch_controls_from_selection()

    def _sync_roi_table(self) -> None:
        regions = self.project.rois_for_target(self.data_id)
        self._syncing_roi_table = True
        self.roi_table.setRowCount(len(regions))
        try:
            for row, roi in enumerate(regions):
                self._populate_roi_table_row(row, roi)
            self.roi_table.resizeColumnsToContents()
        finally:
            self._syncing_roi_table = False
        self._sync_arch_controls_from_selection()
        self._update_pole_figure_button()

    def _populate_roi_table_row(self, row: int, roi: ROIRegion) -> None:
        values = {
            ROI_COL_ID: roi.name,
            ROI_COL_TYPE: roi.kind.title(),
            ROI_COL_DIRECTION: _roi_direction_label(roi),
            ROI_COL_QXY_MIN: _format_float(roi.qxy_min),
            ROI_COL_QXY_MAX: _format_float(roi.qxy_max),
            ROI_COL_QZ_MIN: _format_float(roi.qz_min),
            ROI_COL_QZ_MAX: _format_float(roi.qz_max),
            ROI_COL_QR_MIN: _format_float(roi.qr_min),
            ROI_COL_QR_MAX: _format_float(roi.qr_max),
            ROI_COL_CHI_MIN: _format_float(roi.chi_min),
            ROI_COL_CHI_MAX: _format_float(roi.chi_max),
            ROI_COL_INTEGRATE: _integration_label(roi),
            ROI_COL_RADIUS: _format_float(_roi_radius_center(roi)),
            ROI_COL_QXY_CENTER: _format_float(_roi_qxy_center(roi)),
            ROI_COL_QZ_CENTER: _format_float(_roi_qz_center(roi)),
            ROI_COL_POLE_FIGURE: roi_pole_figure_status(roi),
            ROI_COL_COUPLED: _coupled_roi_label(roi),
        }
        hkl = roi_hkl_metadata(roi)
        values.update(
            {
                ROI_COL_H: _optional_text(hkl["h"]),
                ROI_COL_K: _optional_text(hkl["k"]),
                ROI_COL_L: _optional_text(hkl["l"]),
                ROI_COL_HKL_LABEL: str(hkl["label"] or ""),
            }
        )
        for channel, column in (
            (1, ROI_COL_CHANNEL_1),
            (2, ROI_COL_CHANNEL_2),
        ):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(
                roi.roi_id in self.channel_assignments[channel]
            )
            checkbox.setEnabled(self.roi_controls_enabled)
            checkbox.stateChanged.connect(
                lambda state, roi_id=roi.roi_id, channel=channel: (
                    self._toggle_roi_channel(
                        roi_id,
                        channel,
                        _is_checked_state(state),
                    )
                )
            )
            self.roi_table.setCellWidget(
                row,
                column,
                _centered_widget(checkbox),
            )
        if roi.kind == "arch":
            lock_button = QtWidgets.QToolButton()
            lock_button.setText("Lock")
            lock_button.setCheckable(True)
            lock_button.setChecked(_arch_chi_locked(roi))
            lock_button.setEnabled(self.roi_controls_enabled)
            lock_button.setToolTip("Keep chi min and max symmetric")
            lock_button.toggled.connect(
                lambda checked, roi_id=roi.roi_id: (
                    self._toggle_arch_chi_lock(roi_id, checked)
                )
            )
            self.roi_table.setCellWidget(
                row,
                ROI_COL_CHI_LOCK,
                _centered_widget(lock_button),
            )
        else:
            lock_placeholder = QtWidgets.QToolButton()
            lock_placeholder.setText("")
            lock_placeholder.setEnabled(False)
            self.roi_table.setCellWidget(
                row,
                ROI_COL_CHI_LOCK,
                _centered_widget(lock_placeholder),
            )
        color = QtGui.QBrush(QtGui.QColor(_roi_color(roi)))
        for column, value in values.items():
            item = QtWidgets.QTableWidgetItem(value)
            if column == ROI_COL_ID:
                item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    roi.roi_id,
                )
            if column in {
                ROI_COL_TYPE,
                ROI_COL_DIRECTION,
                ROI_COL_INTEGRATE,
            }:
                item.setForeground(color)
            if column not in _editable_roi_columns(roi):
                item.setFlags(
                    item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                )
            self.roi_table.setItem(row, column, item)

    def _handle_roi_table_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._syncing_roi_table:
            return
        column = item.column()
        row = item.row()
        roi_id_item = self.roi_table.item(row, ROI_COL_ID)
        if roi_id_item is None:
            return
        roi_id = roi_id_item.data(QtCore.Qt.ItemDataRole.UserRole)
        roi = self._roi_by_id(str(roi_id))
        if roi is None or column not in _editable_roi_columns(roi):
            return
        if column == ROI_COL_ID:
            roi.name = item.text().strip() or roi.name
            self._sync_roi_table()
            self._select_roi(roi.roi_id)
            self.roiRegionsChanged.emit(self.data_id)
            return
        if column in ROI_HKL_EDIT_COLUMNS:
            try:
                self.project.set_roi_hkl_tag(
                    self.data_id,
                    roi.roi_id or "",
                    h=(
                        self.roi_table.item(row, ROI_COL_H).text()
                        if self.roi_table.item(row, ROI_COL_H)
                        else ""
                    ),
                    k=(
                        self.roi_table.item(row, ROI_COL_K).text()
                        if self.roi_table.item(row, ROI_COL_K)
                        else ""
                    ),
                    l=(
                        self.roi_table.item(row, ROI_COL_L).text()
                        if self.roi_table.item(row, ROI_COL_L)
                        else ""
                    ),
                    label=(
                        self.roi_table.item(row, ROI_COL_HKL_LABEL).text()
                        if self.roi_table.item(row, ROI_COL_HKL_LABEL)
                        else ""
                    ),
                )
            except (KeyError, ValueError) as exc:
                self._reject_roi_table_edit(roi, str(exc))
                return
            self._sync_roi_table()
            self._select_roi(roi.roi_id)
            self.roiRegionsChanged.emit(self.data_id)
            return
        if column == ROI_COL_DIRECTION and roi.kind == "box":
            direction = item.text().strip().lower()
            if direction.startswith("h") or "qxy" in direction:
                roi.integration_axis = "qxy"
                roi.integration_direction = "horizontal"
            elif direction.startswith("v") or "qz" in direction:
                roi.integration_axis = "qz"
                roi.integration_direction = "vertical"
            else:
                self._reject_roi_table_edit(
                    roi,
                    "ROI direction must be vertical or horizontal.",
                )
                return
            self._sync_roi_table()
            self._select_roi(roi.roi_id)
            self._refresh_channels_for_roi(roi.roi_id)
            self.roiRegionsChanged.emit(self.data_id)
            return
        try:
            value = float(item.text())
        except ValueError:
            self._reject_roi_table_edit(roi, "Enter a numeric ROI value.")
            return
        if not np.isfinite(value):
            self._reject_roi_table_edit(roi, "Enter a finite ROI value.")
            return
        if not self._apply_roi_table_numeric_edit(roi, column, value):
            self._reject_roi_table_edit(
                roi,
                "That ROI value would make the geometry invalid.",
            )
            return
        self._commit_roi_geometry_changed(
            roi,
            message=f"Updated {roi.name or 'ROI'} from the table.",
        )

    def _apply_roi_table_numeric_edit(
        self,
        roi: ROIRegion,
        column: int,
        value: float,
    ) -> bool:
        if roi.kind == "arch":
            return self._apply_arch_table_numeric_edit(roi, column, value)
        return self._apply_box_table_numeric_edit(roi, column, value)

    def _apply_box_table_numeric_edit(
        self,
        roi: ROIRegion,
        column: int,
        value: float,
    ) -> bool:
        qxy_min, qxy_max, qz_min, qz_max = _box_roi_bounds(roi)
        if None in {qxy_min, qxy_max, qz_min, qz_max}:
            return False
        qxy_min = float(qxy_min)
        qxy_max = float(qxy_max)
        qz_min = float(qz_min)
        qz_max = float(qz_max)
        if column == ROI_COL_QXY_MIN:
            qxy_min = value
        elif column == ROI_COL_QXY_MAX:
            qxy_max = value
        elif column == ROI_COL_QZ_MIN:
            qz_min = value
        elif column == ROI_COL_QZ_MAX:
            qz_max = value
        elif column == ROI_COL_QXY_CENTER:
            width = max(qxy_max - qxy_min, 1.0e-12)
            qxy_min = value - width / 2.0
            qxy_max = value + width / 2.0
        elif column == ROI_COL_QZ_CENTER:
            height = max(qz_max - qz_min, 1.0e-12)
            qz_min = value - height / 2.0
            qz_max = value + height / 2.0
        else:
            return False
        if qxy_max <= qxy_min or qz_max <= qz_min:
            return False
        roi.qxy_min = qxy_min
        roi.qxy_max = qxy_max
        roi.qz_min = qz_min
        roi.qz_max = qz_max
        return True

    def _apply_arch_table_numeric_edit(
        self,
        roi: ROIRegion,
        column: int,
        value: float,
    ) -> bool:
        qr_min = float(roi.qr_min or 0.0)
        qr_max = float(roi.qr_max or 0.0)
        chi_min = float(roi.chi_min or 0.0)
        chi_max = float(roi.chi_max or 0.0)
        if column == ROI_COL_QR_MIN:
            qr_min = value
        elif column == ROI_COL_QR_MAX:
            qr_max = value
        elif column == ROI_COL_RADIUS:
            thickness = max(qr_max - qr_min, 1.0e-9)
            radius_center = max(value, thickness / 2.0)
            qr_min = max(0.0, radius_center - thickness / 2.0)
            qr_max = max(qr_min + 1.0e-9, radius_center + thickness / 2.0)
        elif column == ROI_COL_CHI_MIN:
            chi_min = value
        elif column == ROI_COL_CHI_MAX:
            chi_max = value
        elif column == ROI_COL_QXY_CENTER:
            roi.qxy_center = value
            return True
        elif column == ROI_COL_QZ_CENTER:
            roi.qz_center = value
            return True
        else:
            return False
        if qr_min < 0.0 or qr_max <= qr_min:
            return False
        chi_min, chi_max = sorted((chi_min, chi_max))
        if _arch_chi_locked(roi):
            chi_min, chi_max = _symmetric_chi_range(chi_min, chi_max)
        elif chi_max <= chi_min:
            return False
        roi.qr_min = qr_min
        roi.qr_max = qr_max
        roi.chi_min = chi_min
        roi.chi_max = chi_max
        return True

    def _reject_roi_table_edit(
        self,
        roi: ROIRegion,
        message: str,
    ) -> None:
        self._set_roi_status(message)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), message)
        self._sync_roi_table()
        self._select_roi(roi.roi_id)

    def _select_roi(self, roi_id: str | None) -> None:
        if roi_id is None:
            return
        for row in range(self.roi_table.rowCount()):
            item = self.roi_table.item(row, 0)
            if item is None:
                continue
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == roi_id:
                self.roi_table.selectRow(row)
                return

    def _selected_roi_region(self) -> ROIRegion | None:
        row = self.roi_table.currentRow()
        if row < 0:
            return None
        roi_id_item = self.roi_table.item(row, 0)
        if roi_id_item is None:
            return None
        roi_id = roi_id_item.data(QtCore.Qt.ItemDataRole.UserRole)
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id == roi_id:
                return roi
        return None

    def _handle_roi_selection_changed(self) -> None:
        self._sync_arch_controls_from_selection()
        self._update_pole_figure_button()

    def _update_pole_figure_button(self) -> None:
        if not hasattr(self, "open_pole_figure_button"):
            return
        has_roi = self._selected_roi_region() is not None
        enabled = (
            self.roi_controls_enabled
            and has_roi
            and self.image_data is not None
            and self.axis_ranges is not None
        )
        self.open_pole_figure_button.setEnabled(enabled)

    def _request_pole_figure(self) -> None:
        roi = self._selected_roi_region()
        if roi is None:
            self._set_roi_status(
                "Select an ROI before opening the Pole Figure Generator."
            )
            return
        if self.image_data is None or self.axis_ranges is None:
            self._set_roi_status(
                "Pole figure generation requires corrected q-space data."
            )
            return
        self.poleFigureRequested.emit(
            self.data_id,
            roi,
            self.image_data,
            self.axis_ranges,
        )

    def refresh_roi_table(self) -> None:
        """Refresh ROI table metadata/status after an external tool
        save."""

        selected = self._selected_roi_region()
        selected_roi_id = selected.roi_id if selected is not None else None
        self._sync_roi_table()
        self._select_roi(selected_roi_id)

    def _sync_arch_controls_from_selection(self) -> None:
        roi = self._selected_roi_region()
        is_coupled = bool(self._coupled_rois_for(roi))
        self.decouple_roi_button.setEnabled(
            self.roi_controls_enabled and is_coupled
        )
        self.remove_coupled_pair_button.setEnabled(
            self.roi_controls_enabled and is_coupled
        )
        enabled = (
            self.roi_controls_enabled
            and roi is not None
            and roi.kind == "arch"
        )
        for widget in (
            self.arch_radius,
            self.arch_thickness,
            self.arch_chi_min,
            self.arch_chi_max,
            self.apply_arch_button,
        ):
            widget.setEnabled(enabled)
        if not enabled or roi is None:
            return
        self.arch_radius.blockSignals(True)
        self.arch_thickness.blockSignals(True)
        self.arch_chi_min.blockSignals(True)
        self.arch_chi_max.blockSignals(True)
        try:
            self.arch_radius.setValue(_roi_radius_center(roi) or 0.0)
            self.arch_thickness.setValue(
                max(0.0, float((roi.qr_max or 0.0) - (roi.qr_min or 0.0)))
            )
            self.arch_chi_min.setValue(float(roi.chi_min or 0.0))
            self.arch_chi_max.setValue(float(roi.chi_max or 0.0))
        finally:
            self.arch_radius.blockSignals(False)
            self.arch_thickness.blockSignals(False)
            self.arch_chi_min.blockSignals(False)
            self.arch_chi_max.blockSignals(False)

    def _apply_arch_adjustments(self) -> None:
        roi = self._selected_roi_region()
        if roi is None or roi.kind != "arch":
            return
        thickness = max(self.arch_thickness.value(), 1.0e-9)
        radius_center = max(self.arch_radius.value(), thickness / 2.0)
        roi.qr_min = max(0.0, radius_center - thickness / 2.0)
        roi.qr_max = max(roi.qr_min + 1.0e-9, radius_center + thickness / 2.0)
        chi_min, chi_max = sorted(
            (self.arch_chi_min.value(), self.arch_chi_max.value())
        )
        if _arch_chi_locked(roi):
            chi_min, chi_max = _symmetric_chi_range(chi_min, chi_max)
        elif chi_min == chi_max:
            chi_max = min(ARCH_CHI_LIMITS_DEG[1], chi_min + 1.0)
        roi.chi_min = chi_min
        roi.chi_max = chi_max
        self._commit_roi_geometry_changed(
            roi,
            message=f"Updated {roi.name or 'arch ROI'} controls.",
        )

    def _toggle_arch_chi_lock(
        self,
        roi_id: str | None,
        checked: bool,
    ) -> None:
        if self._syncing_roi_table or roi_id is None:
            return
        roi = self._roi_by_id(roi_id)
        if roi is None or roi.kind != "arch":
            return
        _set_arch_chi_locked(roi, checked)
        if checked:
            _symmetrize_arch_roi_chi(roi)
        self._commit_roi_geometry_changed(
            roi,
            message="Arch chi lock updated.",
            clear_markers=False,
        )

    def _toggle_roi_channel(
        self,
        roi_id: str | None,
        channel: int,
        checked: bool,
    ) -> None:
        if self._syncing_roi_table or roi_id is None:
            return
        roi = self._roi_by_id(roi_id)
        if roi is None:
            return
        if checked:
            mode = _roi_integration_mode(roi)
            channel_mode = self.channel_modes[channel]
            if channel_mode is None:
                self.channel_modes[channel] = mode
            elif channel_mode != mode:
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    f"Channel {channel} uses {_mode_label(channel_mode)}.",
                )
                self._sync_roi_table()
                return
            self.channel_assignments[channel].add(roi_id)
        else:
            self.channel_assignments[channel].discard(roi_id)
            self._remove_channel_markers(channel, roi_id=roi_id)
        self._refresh_channel(channel)
        self._sync_roi_table()

    def _clear_channel(self, channel: int) -> None:
        self.channel_assignments[channel].clear()
        self.channel_modes[channel] = None
        self.integration_peak_markers[channel].clear()
        self._set_channel_peak_readout(channel, "")
        self._refresh_channel(channel)
        self._sync_roi_table()

    def _detach_channel(self, channel: int) -> None:
        if channel in self.channel_windows:
            self.channel_windows[channel].raise_()
            self.channel_windows[channel].activateWindow()
            return
        window = _DetachedIntegrationWindow(channel, parent=self)
        window.reattachRequested.connect(self._reattach_channel)
        window.clearRequested.connect(self._clear_channel)
        window.markerRequested.connect(self._add_channel_peak_marker)
        window.markerDragPreviewed.connect(
            self._preview_channel_peak_marker_coordinate
        )
        window.markerMoved.connect(self._move_channel_peak_marker)
        window.markerDeleted.connect(self._delete_channel_peak_marker)
        window.clearMarkersRequested.connect(self._clear_channel_markers)
        window.pushMarkersRequested.connect(self._push_channel_markers)
        window.detectPeaksRequested.connect(self._detect_channel_peaks)
        window.autoSnapToggled.connect(self._set_channel_autosnap_enabled)
        window.set_autosnap_enabled(self.channel_autosnap_enabled[channel])
        self.channel_windows[channel] = window
        self.channel_panels[channel].set_detached(True)
        self._refresh_channel(channel)
        window.show()

    def _reattach_channel(self, channel: int) -> None:
        window = self.channel_windows.pop(channel, None)
        self.channel_panels[channel].set_detached(False)
        if window is not None:
            window.close_from_viewer()
        self._refresh_channel(channel)

    def _refresh_channels_for_roi(self, roi_id: str | None) -> None:
        if roi_id is None:
            return
        for channel, assignments in self.channel_assignments.items():
            if roi_id in assignments:
                self._refresh_channel(channel)

    def _refresh_all_channels(self) -> None:
        for channel in self.channel_panels:
            self._refresh_channel(channel)

    def _refresh_channel(self, channel: int) -> None:
        traces: list[_IntegrationTrace] = []
        assignments = self.channel_assignments[channel]
        if not assignments:
            self.channel_modes[channel] = None
        mode = self.channel_modes[channel]
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id not in assignments:
                continue
            trace = _integration_trace(
                roi,
                self.image_data,
                self.axis_ranges,
            )
            if trace is not None and (mode is None or trace.mode == mode):
                traces.append(trace)
        markers = self._markers_for_channel(
            channel,
            roi_ids={trace.roi_id for trace in traces},
        )
        self.channel_panels[channel].set_series(traces, mode, markers)
        window = self.channel_windows.get(channel)
        if window is not None:
            window.set_series(traces, mode, markers)

    def _add_channel_peak_marker(
        self,
        channel: int,
        roi_id: str,
        integration_x: float,
        integrated_intensity: float,
    ) -> None:
        roi = self._roi_by_id(roi_id)
        if roi is None:
            return
        existing_ids = {
            marker.marker_id
            for markers in self.integration_peak_markers.values()
            for marker in markers
        }
        marker = _integration_peak_marker(
            roi,
            channel,
            integration_x,
            integrated_intensity,
            existing_ids=existing_ids,
        )
        if marker is None:
            return
        self.integration_peak_markers[channel].append(marker)
        self._set_channel_peak_readout(
            channel,
            _channel_peak_readout_text(channel, roi, marker),
        )
        self._refresh_channel(channel)

    def _preview_channel_peak_marker_coordinate(
        self,
        channel: int,
        marker_id: str,
        integration_x: float,
        integrated_intensity: float,
    ) -> None:
        marker = self._channel_marker_by_id(channel, marker_id)
        if marker is None:
            return
        roi = self._roi_by_id(marker.roi_id)
        if roi is None:
            return
        preview_marker = replace(
            marker,
            integration_x=float(integration_x),
            integrated_intensity=float(integrated_intensity),
        )
        text = _channel_peak_readout_text(channel, roi, preview_marker)
        self._set_channel_peak_readout(channel, text)
        self._set_roi_status(text)

    def _detect_channel_peaks(self, channel: int) -> None:
        traces = list(self.channel_panels[channel].series)
        if not traces:
            self._set_roi_status(f"Channel {channel} has no ROI trace.")
            return
        added = 0
        existing_ids = {
            marker.marker_id
            for markers in self.integration_peak_markers.values()
            for marker in markers
        }
        for trace in traces:
            roi = self._roi_by_id(trace.roi_id)
            if roi is None:
                continue
            tolerance = _trace_x_tolerance(trace)
            for integration_x, intensity in _auto_detect_trace_peaks(trace):
                if self._channel_has_marker_near(
                    channel,
                    trace.roi_id,
                    integration_x,
                    tolerance,
                ):
                    continue
                marker = _integration_peak_marker(
                    roi,
                    channel,
                    integration_x,
                    intensity,
                    existing_ids=existing_ids,
                )
                if marker is None:
                    continue
                self.integration_peak_markers[channel].append(marker)
                added += 1
        self._refresh_channel(channel)
        suffix = "peak" if added == 1 else "peaks"
        self._set_roi_status(f"Detected {added} channel {channel} {suffix}.")

    def _set_channel_autosnap_enabled(
        self,
        channel: int,
        enabled: bool,
    ) -> None:
        self.channel_autosnap_enabled[channel] = bool(enabled)
        panel = self.channel_panels.get(channel)
        if panel is not None:
            panel.set_autosnap_enabled(enabled)
        window = self.channel_windows.get(channel)
        if window is not None:
            window.set_autosnap_enabled(enabled)
        state = "enabled" if enabled else "disabled"
        self._set_roi_status(f"Channel {channel} autosnap {state}.")

    def _move_channel_peak_marker(
        self,
        channel: int,
        marker_id: str,
        integration_x: float,
        integrated_intensity: float,
    ) -> None:
        markers = self.integration_peak_markers.get(channel)
        if markers is None:
            return
        for index, marker in enumerate(markers):
            if marker.marker_id != marker_id:
                continue
            roi = self._roi_by_id(marker.roi_id)
            if roi is None:
                return
            coordinate = _integration_peak_qspace_coordinate(
                roi,
                integration_x,
            )
            if coordinate is None:
                return
            qxy, qz = coordinate
            roi_name = roi.name or roi.roi_id or marker.roi_name
            markers[index] = replace(
                marker,
                roi_name=roi_name,
                mode=_roi_integration_mode(roi),
                integration_x=float(integration_x),
                integrated_intensity=float(integrated_intensity),
                qxy=qxy,
                qz=qz,
                label=(
                    f"Ch {channel} {roi_name} "
                    f"@ {_format_float(float(integration_x))}"
                ),
            )
            self._refresh_channel(channel)
            self._set_channel_peak_readout(
                channel,
                _channel_peak_readout_text(channel, roi, markers[index]),
            )
            return

    def _delete_channel_peak_marker(
        self,
        channel: int,
        marker_id: str,
    ) -> None:
        markers = self.integration_peak_markers.get(channel)
        if markers is None:
            return
        before = len(markers)
        self.integration_peak_markers[channel] = [
            marker for marker in markers if marker.marker_id != marker_id
        ]
        if len(self.integration_peak_markers[channel]) != before:
            self._set_channel_peak_readout(channel, "")
            self._refresh_channel(channel)

    def _channel_marker_by_id(
        self,
        channel: int,
        marker_id: str,
    ) -> IntegrationPeakMarker | None:
        for marker in self.integration_peak_markers.get(channel, []):
            if marker.marker_id == marker_id:
                return marker
        return None

    def _channel_has_marker_near(
        self,
        channel: int,
        roi_id: str,
        integration_x: float,
        tolerance: float,
    ) -> bool:
        for marker in self.integration_peak_markers.get(channel, []):
            if marker.roi_id != roi_id:
                continue
            if abs(marker.integration_x - integration_x) <= tolerance:
                return True
        return False

    def _markers_for_channel(
        self,
        channel: int,
        *,
        roi_ids: set[str] | None = None,
    ) -> list[IntegrationPeakMarker]:
        markers = list(self.integration_peak_markers.get(channel, []))
        if roi_ids is None:
            return markers
        return [marker for marker in markers if marker.roi_id in roi_ids]

    def _clear_channel_markers(self, channel: int) -> None:
        self.integration_peak_markers[channel].clear()
        self._set_channel_peak_readout(channel, "")
        self._refresh_channel(channel)

    def _remove_channel_markers(
        self,
        channel: int,
        *,
        roi_id: str | None = None,
    ) -> None:
        if roi_id is None:
            self.integration_peak_markers[channel].clear()
            self._set_channel_peak_readout(channel, "")
            self._refresh_channel(channel)
            return
        before = len(self.integration_peak_markers[channel])
        self.integration_peak_markers[channel] = [
            marker
            for marker in self.integration_peak_markers[channel]
            if marker.roi_id != roi_id
        ]
        if len(self.integration_peak_markers[channel]) != before:
            self._set_channel_peak_readout(channel, "")
            self._refresh_channel(channel)

    def _set_channel_peak_readout(self, channel: int, text: str) -> None:
        panel = self.channel_panels.get(channel)
        if panel is not None:
            panel.set_peak_readout(text)
        window = self.channel_windows.get(channel)
        if window is not None:
            window.set_peak_readout(text)

    def _push_channel_markers(self, channel: int) -> None:
        markers = self._markers_for_channel(
            channel,
            roi_ids=set(self.channel_assignments[channel]),
        )
        if not markers:
            return
        self.integrationPeakMarkersPushed.emit(self.data_id, markers)

    def _roi_by_id(self, roi_id: str) -> ROIRegion | None:
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id == roi_id:
                return roi
        return None

    def _coupled_roi_ids(self, roi: ROIRegion | None) -> set[str]:
        if roi is None:
            return set()
        return _coupled_roi_ids(roi)

    def _coupled_rois_for(self, roi: ROIRegion | None) -> list[ROIRegion]:
        if roi is None:
            return []
        coupled_ids = self._coupled_roi_ids(roi)
        if not coupled_ids:
            return []
        return [
            candidate
            for candidate in self.project.rois_for_target(self.data_id)
            if candidate.roi_id in coupled_ids
        ]

    def _sync_coupled_center_from(self, roi: ROIRegion) -> set[str]:
        if not _roi_has_shared_center(roi):
            return set()
        center = _roi_display_center(roi)
        if center[0] is None or center[1] is None:
            return set()
        changed_ids: set[str] = set()
        for partner in self._coupled_rois_for(roi):
            if _set_roi_center_preserving_shape(
                partner,
                float(center[0]),
                float(center[1]),
            ):
                changed_ids.add(partner.roi_id or "")
        changed_ids.discard("")
        return changed_ids

    def _commit_roi_geometry_changed(
        self,
        roi: ROIRegion,
        *,
        message: str,
        clear_markers: bool = True,
    ) -> None:
        changed_ids = {roi.roi_id or ""}
        changed_ids.update(self._sync_coupled_center_from(roi))
        changed_ids.discard("")
        for changed_id in changed_ids:
            changed_roi = self._roi_by_id(changed_id)
            graphic = self.roi_graphics.get(changed_id)
            if changed_roi is not None and graphic is not None:
                _set_graphic_bounds(
                    graphic,
                    _roi_bounds(changed_roi),
                    roi=changed_roi,
                )
        if roi.roi_id is not None:
            mark_roi_pole_figure_stale(roi)
            self.project.mark_roi_pole_figures_stale(self.data_id, roi.roi_id)
        cleared = 0
        if clear_markers:
            for changed_id in changed_ids:
                cleared += self._clear_temporary_channel_markers_for_roi(
                    changed_id
                )
        self._sync_roi_table()
        self._select_roi(roi.roi_id)
        for changed_id in changed_ids:
            self._refresh_channels_for_roi(changed_id)
        status = message
        if cleared:
            status = (
                f"{message} Cleared {cleared} temporary channel "
                f"marker(s) for the moved ROI."
            )
        self._set_roi_status(status)
        self.roiRegionsChanged.emit(self.data_id)

    def _clear_temporary_channel_markers_for_roi(
        self,
        roi_id: str | None,
    ) -> int:
        if roi_id is None:
            return 0
        cleared = 0
        for channel, markers in self.integration_peak_markers.items():
            retained = [
                marker for marker in markers if marker.roi_id != roi_id
            ]
            cleared += len(markers) - len(retained)
            self.integration_peak_markers[channel] = retained
        return cleared

    def _set_roi_status(self, message: str) -> None:
        self.roi_status_label.setText(message)

    def _add_roi_graphic(self, roi: ROIRegion) -> None:
        if pg is None or self.plot_widget is None or roi.roi_id is None:
            return
        if roi.roi_id in self.roi_graphics:
            return
        bounds = _roi_bounds(roi)
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds
        pen = pg.mkPen(_roi_color(roi), width=2)
        if roi.kind == "arch":
            graphic = _ArchROI(roi, pen=pen)
        else:
            graphic = _BoxROI(
                (x_min, y_min),
                (x_max - x_min, y_max - y_min),
                pen=pen,
            )
        graphic.setZValue(10)
        graphic.sigRegionChangeStarted.connect(
            lambda *_args: self._set_view_drag_enabled(False)
        )
        roi_id = roi.roi_id

        def sync_bounds(*_args: Any) -> None:
            self._handle_roi_graphic_changed(str(roi_id), graphic)

        graphic.sigRegionChangeFinished.connect(sync_bounds)
        self.plot_widget.addItem(graphic)
        self.roi_graphics[roi.roi_id] = graphic

    def _handle_roi_graphic_changed(self, roi_id: str, graphic: Any) -> None:
        self._set_view_drag_enabled(False)
        bounds = _graphic_bounds(graphic)
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds
        changed_roi: ROIRegion | None = None
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id != roi_id:
                continue
            if roi.kind == "arch":
                if hasattr(graphic, "arch_center"):
                    roi.qxy_center, roi.qz_center = graphic.arch_center()
                    if hasattr(graphic, "arch_parameters"):
                        (
                            roi.qr_min,
                            roi.qr_max,
                            roi.chi_min,
                            roi.chi_max,
                        ) = graphic.arch_parameters()
                else:
                    qr_min, qr_max, chi_min, chi_max = (
                        _arch_polar_from_plot_bounds(
                            x_min,
                            x_max,
                            y_min,
                            y_max,
                        )
                    )
                    roi.qr_min = qr_min
                    roi.qr_max = qr_max
                    roi.chi_min = chi_min
                    roi.chi_max = chi_max
                if _arch_chi_locked(roi):
                    _symmetrize_arch_roi_chi(roi)
                _set_graphic_bounds(graphic, _arch_plot_bounds(roi), roi=roi)
            else:
                roi.qxy_min = x_min
                roi.qxy_max = x_max
                roi.qz_min = y_min
                roi.qz_max = y_max
            changed_roi = roi
            break
        if changed_roi is not None:
            self._sync_coupled_center_from(changed_roi)
            self._commit_roi_geometry_changed(
                changed_roi,
                message=f"Moved {changed_roi.name or 'ROI'}.",
            )

    def _remove_roi_graphic(self, roi_id: str) -> None:
        graphic = self.roi_graphics.pop(roi_id, None)
        if graphic is not None and self.plot_widget is not None:
            self.plot_widget.removeItem(graphic)

    def _add_low_q_feature_graphics(self) -> None:
        if (
            pg is None
            or self.plot_widget is None
            or self.coordinate_space != "qspace"
        ):
            return
        state = self.project.image_corrections.get(self.data_id)
        if state is None:
            return
        for feature in state.metadata.get("low_q_features", []):
            graphic = _low_q_feature_graphic(feature)
            if graphic is None:
                continue
            self.plot_widget.addItem(graphic)
            self.low_q_graphics.append(graphic)

    def _apply_manual_levels(self) -> None:
        if not self.quantile_check.isChecked():
            self._apply_image_style()

    def image_display_style(self) -> ImageDisplayStyle:
        return ImageDisplayStyle(
            colormap=str(self.colormap_combo.currentData()),
            use_quantile=self.quantile_check.isChecked(),
            quantile_low=self.quantile_low.value(),
            quantile_high=self.quantile_high.value(),
            level_min=self.level_min.value(),
            level_max=self.level_max.value(),
        )

    def _set_quantile_levels(self) -> None:
        if self.image_data is None:
            return
        low = min(self.quantile_low.value(), self.quantile_high.value())
        high = max(self.quantile_low.value(), self.quantile_high.value())
        finite = self.image_data[np.isfinite(self.image_data)]
        if not finite.size:
            return
        levels = np.nanquantile(finite, [low / 100.0, high / 100.0])
        self.level_min.blockSignals(True)
        self.level_max.blockSignals(True)
        self.level_min.setValue(float(levels[0]))
        self.level_max.setValue(float(levels[1]))
        self.level_min.blockSignals(False)
        self.level_max.blockSignals(False)
        self._apply_image_style()

    def _apply_image_style(self) -> None:
        if self.image_item is None or self.image_data is None:
            return
        style = self.image_display_style()
        apply_image_display_style(self.image_item, self.image_data, style)
        self.imageStyleChanged.emit(style)

    def _apply_colormap(self) -> None:
        _apply_colormap_to_image_item(
            self.image_item,
            str(self.colormap_combo.currentData()),
        )

    def _rotate_raw_preview(self, delta_degrees: int) -> None:
        if not self.orientation_controls_enabled:
            return
        state = self._orientation_state()
        state.image_rotation_deg = (
            state.image_rotation_deg + delta_degrees
        ) % 360
        self._sync_orientation_state(state)
        self._reload_preview_image()

    def _toggle_raw_preview_mirror(self, *_args: Any) -> None:
        if not self.orientation_controls_enabled:
            return
        state = self._orientation_state()
        state.image_mirrored_y = not state.image_mirrored_y
        self.mirror_y_button.setChecked(state.image_mirrored_y)
        self._sync_orientation_state(state)
        self._reload_preview_image()

    def _orientation_state(self) -> ImageCorrectionState:
        state = self.project.image_corrections.get(self.data_id)
        if state is None:
            state = ImageCorrectionState(target_id=self.data_id)
            self.project.image_corrections[self.data_id] = state
        return state

    def _sync_orientation_state(self, state: ImageCorrectionState) -> None:
        state.pyfai_sample_orientation = (
            sample_orientation_for_image_transform(
                state.image_rotation_deg,
                mirrored_y=state.image_mirrored_y,
            )
        )
        self.previewOrientationChanged.emit(self.data_id)

    def _reload_preview_image(self) -> None:
        self.axis_ranges = None
        self.coordinate_space = "pixel"
        self.image_data = self._load_display_image()
        self._set_initial_image()
        self._refresh_all_channels()

    def _set_view_drag_enabled(self, enabled: bool) -> None:
        if self.view_box is not None and hasattr(
            self.view_box, "setMouseEnabled"
        ):
            self.view_box.setMouseEnabled(x=enabled, y=enabled)


def _roi_bounds(
    roi: ROIRegion,
) -> tuple[float, float, float, float] | None:
    if roi.kind == "arch":
        return _arch_plot_bounds(roi)
    else:
        values = (roi.qxy_min, roi.qxy_max, roi.qz_min, roi.qz_max)
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _box_roi_bounds(
    roi: ROIRegion,
) -> tuple[float | None, float | None, float | None, float | None]:
    return roi.qxy_min, roi.qxy_max, roi.qz_min, roi.qz_max


def _roi_radius_center(roi: ROIRegion) -> float | None:
    if roi.qr_min is None or roi.qr_max is None:
        return None
    return (float(roi.qr_min) + float(roi.qr_max)) / 2.0


def _roi_qxy_center(roi: ROIRegion) -> float | None:
    if roi.kind == "arch":
        return float(roi.qxy_center)
    if roi.qxy_min is None or roi.qxy_max is None:
        return None
    return (float(roi.qxy_min) + float(roi.qxy_max)) / 2.0


def _roi_qz_center(roi: ROIRegion) -> float | None:
    if roi.kind == "arch":
        return float(roi.qz_center)
    if roi.qz_min is None or roi.qz_max is None:
        return None
    return (float(roi.qz_min) + float(roi.qz_max)) / 2.0


def _editable_roi_columns(roi: ROIRegion) -> set[int]:
    if roi.kind == "arch":
        return {
            ROI_COL_ID,
            ROI_COL_QR_MIN,
            ROI_COL_QR_MAX,
            ROI_COL_CHI_MIN,
            ROI_COL_CHI_MAX,
            ROI_COL_RADIUS,
            ROI_COL_QXY_CENTER,
            ROI_COL_QZ_CENTER,
            *ROI_HKL_EDIT_COLUMNS,
        }
    return {
        ROI_COL_ID,
        ROI_COL_DIRECTION,
        ROI_COL_QXY_MIN,
        ROI_COL_QXY_MAX,
        ROI_COL_QZ_MIN,
        ROI_COL_QZ_MAX,
        ROI_COL_QXY_CENTER,
        ROI_COL_QZ_CENTER,
        *ROI_HKL_EDIT_COLUMNS,
    }


def _optional_text(value: Any) -> str:
    return "" if value is None else str(value)


def _centered_widget(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    layout.setAlignment(widget, QtCore.Qt.AlignmentFlag.AlignCenter)
    return container


def _is_checked_state(state: Any) -> bool:
    checked_value = getattr(
        QtCore.Qt.CheckState.Checked,
        "value",
        QtCore.Qt.CheckState.Checked,
    )
    return state in {
        QtCore.Qt.CheckState.Checked,
        checked_value,
    }


def _metadata_id_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    try:
        return {str(item) for item in value if str(item)}
    except TypeError:
        text = str(value)
        return {text} if text else set()


def _coupled_roi_ids(roi: ROIRegion) -> set[str]:
    coupled = _metadata_id_set(roi.metadata.get(COUPLED_ROI_IDS_METADATA_KEY))
    single = str(roi.metadata.get(COUPLED_ROI_ID_METADATA_KEY) or "")
    if single:
        coupled.add(single)
    coupled.discard(str(roi.roi_id or ""))
    return coupled


def _roi_has_shared_center(roi: ROIRegion) -> bool:
    return bool(
        roi.metadata.get(COUPLED_ROI_SHARED_CENTER_METADATA_KEY)
        and _coupled_roi_ids(roi)
    )


def _couple_roi_pair(
    box: ROIRegion,
    arch: ROIRegion,
    *,
    group_id: str,
) -> None:
    box.metadata[COUPLED_ROI_GROUP_METADATA_KEY] = group_id
    arch.metadata[COUPLED_ROI_GROUP_METADATA_KEY] = group_id
    box.metadata[COUPLED_ROI_ROLE_METADATA_KEY] = "box"
    arch.metadata[COUPLED_ROI_ROLE_METADATA_KEY] = "arch"
    box.metadata[COUPLED_ROI_SHARED_CENTER_METADATA_KEY] = True
    arch.metadata[COUPLED_ROI_SHARED_CENTER_METADATA_KEY] = True
    if arch.roi_id is not None:
        box.metadata[COUPLED_ROI_ID_METADATA_KEY] = arch.roi_id
        box.metadata[COUPLED_ROI_IDS_METADATA_KEY] = [arch.roi_id]
    if box.roi_id is not None:
        arch.metadata[COUPLED_ROI_ID_METADATA_KEY] = box.roi_id
        arch.metadata[COUPLED_ROI_IDS_METADATA_KEY] = [box.roi_id]


def _decouple_roi(
    roi: ROIRegion, removed_partner_id: str | None = None
) -> None:
    if removed_partner_id is None:
        for key in (
            COUPLED_ROI_GROUP_METADATA_KEY,
            COUPLED_ROI_ID_METADATA_KEY,
            COUPLED_ROI_IDS_METADATA_KEY,
            COUPLED_ROI_ROLE_METADATA_KEY,
            COUPLED_ROI_SHARED_CENTER_METADATA_KEY,
        ):
            roi.metadata.pop(key, None)
        return
    coupled = _coupled_roi_ids(roi)
    coupled.discard(removed_partner_id)
    if coupled:
        roi.metadata[COUPLED_ROI_IDS_METADATA_KEY] = sorted(coupled)
        roi.metadata[COUPLED_ROI_ID_METADATA_KEY] = sorted(coupled)[0]
    else:
        _decouple_roi(roi)


def _coupled_roi_label(roi: ROIRegion) -> str:
    coupled = sorted(_coupled_roi_ids(roi))
    if not coupled:
        return ""
    role = str(roi.metadata.get(COUPLED_ROI_ROLE_METADATA_KEY) or roi.kind)
    group = str(roi.metadata.get(COUPLED_ROI_GROUP_METADATA_KEY) or "pair")
    return f"{group} ({role})"


def _unique_coupled_roi_group_id(data_id: str, index: int) -> str:
    return f"{data_id}_coupled_roi_{index}"


def _roi_display_center(roi: ROIRegion) -> tuple[float | None, float | None]:
    return _roi_qxy_center(roi), _roi_qz_center(roi)


def _set_roi_center_preserving_shape(
    roi: ROIRegion,
    qxy_center: float,
    qz_center: float,
) -> bool:
    old_center = _roi_display_center(roi)
    if old_center[0] == qxy_center and old_center[1] == qz_center:
        return False
    if roi.kind == "arch":
        roi.qxy_center = float(qxy_center)
        roi.qz_center = float(qz_center)
        return True
    if (
        roi.qxy_min is None
        or roi.qxy_max is None
        or roi.qz_min is None
        or roi.qz_max is None
    ):
        return False
    width = float(roi.qxy_max) - float(roi.qxy_min)
    height = float(roi.qz_max) - float(roi.qz_min)
    if width <= 0.0 or height <= 0.0:
        return False
    roi.qxy_min = float(qxy_center) - width / 2.0
    roi.qxy_max = float(qxy_center) + width / 2.0
    roi.qz_min = float(qz_center) - height / 2.0
    roi.qz_max = float(qz_center) + height / 2.0
    return True


def _arch_chi_locked(roi: ROIRegion) -> bool:
    return bool(roi.metadata.get(ARCH_CHI_LOCKED_METADATA_KEY, False))


def _set_arch_chi_locked(roi: ROIRegion, locked: bool) -> None:
    roi.metadata[ARCH_CHI_LOCKED_METADATA_KEY] = bool(locked)


def _symmetrize_arch_roi_chi(roi: ROIRegion) -> None:
    roi.chi_min, roi.chi_max = _symmetric_chi_range(
        float(roi.chi_min or 0.0),
        float(roi.chi_max or 0.0),
    )


def _symmetric_chi_range(
    chi_min: float,
    chi_max: float,
) -> tuple[float, float]:
    extent = max(abs(float(chi_min)), abs(float(chi_max)), 1.0)
    limit = max(abs(limit) for limit in ARCH_CHI_LIMITS_DEG)
    extent = min(extent, limit)
    return -extent, extent


def _roi_color(roi: ROIRegion) -> str:
    if roi.kind == "arch":
        return ROI_COLOR_ARCH
    if roi.integration_axis == "qxy":
        return ROI_COLOR_BOX_HORIZONTAL
    return ROI_COLOR_BOX_VERTICAL


def _roi_direction_label(roi: ROIRegion) -> str:
    if roi.kind == "arch":
        return "azimuthal"
    if roi.integration_axis == "qxy":
        return "horizontal"
    return "vertical"


def _roi_integration_mode(roi: ROIRegion) -> str:
    if roi.kind == "arch":
        return "arch"
    if roi.integration_axis == "qxy":
        return "box:qxy"
    return "box:qz"


def _mode_label(mode: str) -> str:
    if mode == "arch":
        return "arch integrations"
    if mode == "box:qxy":
        return "horizontal box integrations"
    return "vertical box integrations"


def _mode_header_label(mode: str) -> str:
    if mode == "arch":
        return "Arch"
    if mode == "box:qxy":
        return "Horizontal Box"
    return "Vertical Box"


def _channel_header_text(channel: int, mode: str | None) -> str:
    title = f"Channel {channel}"
    if mode is None:
        return title
    return f"{title} ({_mode_header_label(mode)})"


def _mode_x_label(mode: str) -> str:
    if mode == "arch":
        return "chi (deg)"
    if mode == "box:qxy":
        return QXY_MATPLOTLIB
    return QZ_MATPLOTLIB


def _integration_trace(
    roi: ROIRegion,
    image_data: np.ndarray | None,
    axis_ranges: tuple[float, float, float, float] | None,
) -> _IntegrationTrace | None:
    if image_data is None or axis_ranges is None:
        return None
    image = np.asarray(image_data, dtype=float)
    if image.ndim != 2:
        return None
    x_axis, y_axis = _image_axes(image.shape, axis_ranges)
    if roi.kind == "arch":
        return _arch_integration_trace(roi, image, x_axis, y_axis)
    return _box_integration_trace(roi, image, x_axis, y_axis)


def _image_axes(
    shape: tuple[int, int],
    axis_ranges: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x_min, x_max, y_min, y_max = axis_ranges
    return (
        np.linspace(x_min, x_max, max(width, 1)),
        np.linspace(y_min, y_max, max(height, 1)),
    )


def _box_integration_trace(
    roi: ROIRegion,
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> _IntegrationTrace | None:
    values = (roi.qxy_min, roi.qxy_max, roi.qz_min, roi.qz_max)
    if any(value is None for value in values) or roi.roi_id is None:
        return None
    qxy_min, qxy_max = sorted((float(roi.qxy_min), float(roi.qxy_max)))
    qz_min, qz_max = sorted((float(roi.qz_min), float(roi.qz_max)))
    x_mask = (x_axis >= qxy_min) & (x_axis <= qxy_max)
    y_mask = (y_axis >= qz_min) & (y_axis <= qz_max)
    if not np.any(x_mask) or not np.any(y_mask):
        return None
    sub_image = image[np.ix_(y_mask, x_mask)]
    mode = _roi_integration_mode(roi)
    if mode == "box:qxy":
        x_values = x_axis[x_mask]
        y_values = np.nansum(sub_image, axis=0)
    else:
        x_values = y_axis[y_mask]
        y_values = np.nansum(sub_image, axis=1)
    return _IntegrationTrace(
        roi_id=roi.roi_id,
        label=roi.name or roi.roi_id,
        mode=mode,
        x_label=_mode_x_label(mode),
        x_values=np.asarray(x_values, dtype=float),
        y_values=np.asarray(y_values, dtype=float),
        color=_roi_color(roi),
    )


def _arch_integration_trace(
    roi: ROIRegion,
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> _IntegrationTrace | None:
    values = (roi.qr_min, roi.qr_max, roi.chi_min, roi.chi_max)
    if any(value is None for value in values) or roi.roi_id is None:
        return None
    qr_min, qr_max = sorted((float(roi.qr_min), float(roi.qr_max)))
    chi_min, chi_max = sorted((float(roi.chi_min), float(roi.chi_max)))
    if qr_max <= qr_min or chi_max <= chi_min:
        return None
    qxy_grid, qz_grid = np.meshgrid(x_axis, y_axis)
    qxy_relative = qxy_grid - roi.qxy_center
    qz_relative = qz_grid - roi.qz_center
    radius = np.hypot(qxy_relative, qz_relative)
    chi = np.degrees(np.arctan2(qxy_relative, qz_relative))
    mask = (
        (radius >= qr_min)
        & (radius <= qr_max)
        & (chi >= chi_min)
        & (chi <= chi_max)
    )
    if not np.any(mask):
        return None
    bin_count = max(12, min(180, int(np.ceil(abs(chi_max - chi_min)))))
    edges = np.linspace(chi_min, chi_max, bin_count + 1)
    bin_index = np.digitize(chi[mask], edges) - 1
    valid = (bin_index >= 0) & (bin_index < bin_count)
    if not np.any(valid):
        return None
    weights = np.nan_to_num(image[mask][valid], nan=0.0)
    integrated = np.bincount(
        bin_index[valid],
        weights=weights,
        minlength=bin_count,
    )
    counts = np.bincount(bin_index[valid], minlength=bin_count)
    integrated = np.where(counts > 0, integrated, np.nan)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mode = _roi_integration_mode(roi)
    return _IntegrationTrace(
        roi_id=roi.roi_id,
        label=roi.name or roi.roi_id,
        mode=mode,
        x_label=_mode_x_label(mode),
        x_values=centers,
        y_values=integrated,
        color=_roi_color(roi),
    )


def _integration_peak_marker(
    roi: ROIRegion,
    channel: int,
    integration_x: float,
    integrated_intensity: float,
    *,
    existing_ids: set[str],
) -> IntegrationPeakMarker | None:
    if roi.roi_id is None:
        return None
    coordinate = _integration_peak_qspace_coordinate(roi, integration_x)
    if coordinate is None:
        return None
    qxy, qz = coordinate
    marker_id = _unique_integration_marker_id(roi.target_id, existing_ids)
    roi_name = roi.name or roi.roi_id
    return IntegrationPeakMarker(
        marker_id=marker_id,
        channel=channel,
        roi_id=roi.roi_id,
        roi_name=roi_name,
        mode=_roi_integration_mode(roi),
        integration_x=float(integration_x),
        integrated_intensity=float(integrated_intensity),
        qxy=qxy,
        qz=qz,
        label=(
            f"Ch {channel} {roi_name} "
            f"@ {_format_float(float(integration_x))}"
        ),
    )


def _channel_peak_readout_text(
    channel: int,
    roi: ROIRegion,
    marker: IntegrationPeakMarker,
) -> str:
    axis_label = _roi_integration_axis_label(roi)
    coordinate = _integration_peak_qspace_coordinate(
        roi,
        marker.integration_x,
    )
    base = (
        f"Ch {channel} active peak: "
        f"trace {axis_label}={_format_float(marker.integration_x)}, "
        f"I={_format_float(marker.integrated_intensity)}"
    )
    if coordinate is None:
        return base
    qxy, qz = coordinate
    return f"{base}, " f"qxy={_format_float(qxy)}, " f"qz={_format_float(qz)}"


def _roi_integration_axis_label(roi: ROIRegion) -> str:
    mode = _roi_integration_mode(roi)
    if mode == "arch":
        return "chi"
    if mode == "box:qxy":
        return "qxy"
    return "qz"


def _integration_peak_qspace_coordinate(
    roi: ROIRegion,
    integration_x: float,
) -> tuple[float, float] | None:
    mode = _roi_integration_mode(roi)
    if mode == "box:qxy":
        if roi.qz_min is None or roi.qz_max is None:
            return None
        qz = (float(roi.qz_min) + float(roi.qz_max)) / 2.0
        return float(integration_x), qz
    if mode == "box:qz":
        if roi.qxy_min is None or roi.qxy_max is None:
            return None
        qxy = (float(roi.qxy_min) + float(roi.qxy_max)) / 2.0
        return qxy, float(integration_x)
    values = (roi.qr_min, roi.qr_max, roi.chi_min, roi.chi_max)
    if any(value is None for value in values):
        return None
    radius = (float(roi.qr_min) + float(roi.qr_max)) / 2.0
    delta_qxy, delta_qz = _polar_qxy_qz(radius, float(integration_x))
    return roi.qxy_center + delta_qxy, roi.qz_center + delta_qz


def _unique_integration_marker_id(
    data_id: str,
    existing_ids: set[str],
) -> str:
    base = f"{data_id}_integration_peak"
    index = 1
    candidate = f"{base}_{index}"
    while candidate in existing_ids:
        index += 1
        candidate = f"{base}_{index}"
    existing_ids.add(candidate)
    return candidate


def _low_q_feature_graphic(feature: dict[str, Any]) -> Any | None:
    if pg is None:
        return None
    display = feature.get("display", "point")
    qxy = feature.get("qxy")
    qz = feature.get("qz")
    if display == "horizontal_line" and qz is not None:
        line = pg.InfiniteLine(
            pos=float(qz),
            angle=0,
            pen=pg.mkPen(
                "#e76f51", width=1.5, style=QtCore.Qt.PenStyle.DashLine
            ),
        )
        line.setZValue(9)
        return line
    if qxy is None or qz is None:
        return None
    scatter = pg.ScatterPlotItem(
        x=[float(qxy)],
        y=[float(qz)],
        size=10,
        brush=pg.mkBrush("#e76f51"),
        pen=pg.mkPen("#ffffff", width=1),
    )
    scatter.setZValue(11)
    return scatter


def _graphic_bounds(graphic: Any) -> tuple[float, float, float, float] | None:
    try:
        position = graphic.pos()
        size = graphic.size()
        x0, y0 = _point_xy(position)
        width, height = _point_xy(size)
    except Exception:
        return None
    x_min, x_max = sorted((x0, x0 + width))
    y_min, y_max = sorted((y0, y0 + height))
    return x_min, x_max, y_min, y_max


def _point_xy(point: Any) -> tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x()), float(point.y())
    return float(point[0]), float(point[1])


def _arch_plot_bounds(
    roi: ROIRegion,
) -> tuple[float, float, float, float] | None:
    values = (roi.qr_min, roi.qr_max, roi.chi_min, roi.chi_max)
    if any(value is None for value in values):
        return None
    x_min, x_max, y_min, y_max = _arch_local_bounds(
        float(roi.qr_min),
        float(roi.qr_max),
        float(roi.chi_min),
        float(roi.chi_max),
    )
    return (
        roi.qxy_center + x_min,
        roi.qxy_center + x_max,
        roi.qz_center + y_min,
        roi.qz_center + y_max,
    )


def _arch_local_bounds(
    qr_min: float,
    qr_max: float,
    chi_min: float,
    chi_max: float,
) -> tuple[float, float, float, float]:
    points = _arch_points(qr_min, qr_max, chi_min, chi_max)
    if not points:
        return 0.0, 1.0e-9, 0.0, 1.0e-9
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _arch_polar_from_plot_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, float, float]:
    max_abs_qxy = max(abs(x_min), abs(x_max))
    qr_min = max(0.0, min(abs(y_min), abs(y_max)))
    qr_max = max(qr_min + 1.0e-9, max(abs(y_min), abs(y_max)))
    qr_center = max((qr_min + qr_max) / 2.0, 1.0e-9)
    chi_extent = np.degrees(np.arctan2(max_abs_qxy, qr_center))
    chi_limit = max(abs(limit) for limit in ARCH_CHI_LIMITS_DEG)
    chi_extent = float(np.clip(chi_extent, 1.0, chi_limit))
    return qr_min, qr_max, -chi_extent, chi_extent


def _arch_path(
    qr_min: float,
    qr_max: float,
    chi_min: float,
    chi_max: float,
) -> QtGui.QPainterPath:
    points = _arch_points(qr_min, qr_max, chi_min, chi_max)
    path = QtGui.QPainterPath()
    if not points:
        return path
    path.moveTo(points[0][0], points[0][1])
    for x_value, y_value in points[1:]:
        path.lineTo(x_value, y_value)
    path.closeSubpath()
    return path


def _arch_points(
    qr_min: float,
    qr_max: float,
    chi_min: float,
    chi_max: float,
    *,
    samples: int = 96,
) -> list[tuple[float, float]]:
    radius_min, radius_max = sorted((max(0.0, qr_min), max(0.0, qr_max)))
    angle_min, angle_max = sorted(
        (
            float(np.clip(chi_min, *ARCH_CHI_LIMITS_DEG)),
            float(np.clip(chi_max, *ARCH_CHI_LIMITS_DEG)),
        )
    )
    if radius_max <= 0.0 or angle_min == angle_max:
        return []
    angles = np.linspace(angle_min, angle_max, samples)
    outer = [_polar_qxy_qz(radius_max, angle) for angle in angles]
    inner = [_polar_qxy_qz(radius_min, angle) for angle in angles[::-1]]
    return outer + inner


def _polar_qxy_qz(radius: float, chi_deg: float) -> tuple[float, float]:
    angle = np.radians(chi_deg)
    return float(radius * np.sin(angle)), float(radius * np.cos(angle))


def _chi_from_local_point(qxy: float, qz: float) -> float:
    chi = float(np.degrees(np.arctan2(qxy, qz)))
    return float(np.clip(chi, *ARCH_CHI_LIMITS_DEG))


def _set_graphic_bounds(
    graphic: Any,
    bounds: tuple[float, float, float, float] | None,
    *,
    roi: ROIRegion | None = None,
) -> None:
    if bounds is None:
        return
    if roi is not None and hasattr(graphic, "set_arch_region"):
        graphic.set_arch_region(roi)
        return
    x_min, x_max, y_min, y_max = bounds
    if hasattr(graphic, "blockSignals"):
        graphic.blockSignals(True)
    try:
        graphic.setPos((x_min, y_min))
        graphic.setSize((x_max - x_min, y_max - y_min))
    finally:
        if hasattr(graphic, "blockSignals"):
            graphic.blockSignals(False)
    graphic.update()


def image_display_levels(
    image_data: np.ndarray | None,
    style: ImageDisplayStyle,
) -> tuple[float, float] | None:
    if image_data is None:
        return None
    if style.use_quantile:
        low = min(style.quantile_low, style.quantile_high)
        high = max(style.quantile_low, style.quantile_high)
        finite = image_data[np.isfinite(image_data)]
        if finite.size:
            levels = np.nanquantile(finite, [low / 100.0, high / 100.0])
        else:
            levels = (style.level_min, style.level_max)
    else:
        levels = (style.level_min, style.level_max)
    low_level = float(levels[0])
    high_level = float(levels[1])
    if not np.isfinite(low_level) or not np.isfinite(high_level):
        return None
    if low_level == high_level:
        high_level = low_level + 1.0
    return low_level, high_level


def apply_image_display_style(
    image_item: Any,
    image_data: np.ndarray | None,
    style: ImageDisplayStyle,
) -> None:
    if image_item is None or image_data is None:
        return
    levels = image_display_levels(image_data, style)
    if levels is not None:
        image_item.setLevels(levels)
    _apply_colormap_to_image_item(image_item, style.colormap)


def _apply_colormap_to_image_item(image_item: Any, name: str) -> None:
    if pg is None or image_item is None:
        return
    try:
        cmap = pg.colormap.get(name)
        image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    except Exception:
        if name == "gray":
            image_item.setLookupTable(None)


def _level_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(-1.0e12, 1.0e12)
    spinbox.setDecimals(4)
    spinbox.setKeyboardTracking(False)
    spinbox.setMaximumWidth(110)
    return spinbox


def _roi_radius_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 1.0e6)
    spinbox.setDecimals(5)
    spinbox.setSingleStep(0.01)
    spinbox.setMaximumWidth(100)
    return spinbox


def _roi_chi_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(*ARCH_CHI_LIMITS_DEG)
    spinbox.setDecimals(2)
    spinbox.setSingleStep(1.0)
    spinbox.setSuffix(" deg")
    spinbox.setValue(value)
    spinbox.setMaximumWidth(100)
    return spinbox


def _quantile_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 100.0)
    spinbox.setDecimals(2)
    spinbox.setSingleStep(0.5)
    spinbox.setSuffix("%")
    spinbox.setValue(value)
    spinbox.setMaximumWidth(90)
    return spinbox


def _orientation_tool_button(
    text: str,
    tooltip: str,
) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setText(text)
    button.setToolTip(tooltip)
    button.setFixedSize(32, 32)
    return button


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.5g}"


def _integration_label(roi: ROIRegion) -> str:
    if roi.kind == "arch":
        return "chi azimuthal"
    if roi.integration_axis == "qxy":
        return f"{QXY_HTML} horizontal"
    return f"{QZ_HTML} vertical"


def _load_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    if mask_path.suffix.lower() == ".edf":
        import fabio

        return np.asarray(fabio.open(str(mask_path)).data).astype(bool)
    if mask_path.suffix.lower() in {".npy", ".npz"}:
        loaded = np.load(mask_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = loaded.files[0]
            return np.asarray(loaded[key]).astype(bool)
        return np.asarray(loaded).astype(bool)
    import tifffile

    return np.asarray(tifffile.imread(mask_path)).astype(bool)


def _apply_image_rotation(array: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation = rotation_deg % 360
    if rotation == 0:
        return array
    if rotation not in {90, 180, 270}:
        return array
    return np.rot90(array, k=-(rotation // 90))


def _apply_image_orientation(
    array: np.ndarray,
    rotation_deg: int,
    mirrored_y: bool,
) -> np.ndarray:
    oriented = _apply_image_rotation(array, rotation_deg)
    if mirrored_y:
        return np.fliplr(oriented)
    return oriented
