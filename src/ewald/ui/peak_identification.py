"""Peak identification tab for corrected q-space detector images."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ewald.crystallography import (
    CRYSTAL_SYSTEMS,
    CrystalOverlayCalculator,
    CrystalOverlayParameters,
)
from ewald.crystallography.overlay import (
    apply_crystal_system_constraints,
    compose_quaternions,
    euler_angles_from_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_euler_angles,
)
from ewald.data.models import (
    PEAK_POINT_KIND_COMMITTED,
    PEAK_POINT_KIND_GAP_ESTIMATED,
    ProjectState,
)
from ewald.processing.peak_detection import (
    LocalMaxPeakFinderConfig,
    find_local_maxima_peaks,
)
from ewald.processing.peak_fitting import (
    PEAK_FIT_INTEGRATIONS,
    compute_peak_fit_integrations,
    evaluate_peak_fit_2d,
    fit_peak_integration,
    fit_peak_integrations,
    fit_peak_roi_2d,
    slice_peak_roi,
)
from ewald.ui.data_viewer import (
    IMAGE_COLORMAPS,
    ImageDisplayStyle,
    _apply_image_orientation,
    _ImageAspectPlotFrame,
    _level_spinbox,
    _load_mask,
    _quantile_spinbox,
    apply_image_display_style,
)
from ewald.ui.notation import (
    QSPACE_UNITS_HTML,
    QXY_HTML,
    QXY_MATPLOTLIB_SYMBOL,
    QZ_HTML,
    QZ_MATPLOTLIB_SYMBOL,
    RichTextCheckBox,
    RichTextComboBox,
    data_image_rect,
    enable_rich_text_items,
    qt_tooltip,
    rich_label,
    set_data_aspect_locked,
    set_data_image_plot_range,
    set_qspace_axis_labels,
    set_rich_text_table_headers,
)
from ewald.ui.structure_analysis import sync_structure_peak_from_fit

try:  # pragma: no cover - exercised by Qt tests when installed.
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

PEAK_TABLE_HEADERS = [
    "Peak",
    "Source",
    QXY_HTML,
    QZ_HTML,
    "Intensity",
    f"ROI {QXY_HTML} min",
    f"ROI {QXY_HTML} max",
    f"ROI {QZ_HTML} min",
    f"ROI {QZ_HTML} max",
    "Integrations",
    "1D Fits",
    "2D Fit",
    f"Fit {QXY_HTML}",
    f"Fit {QZ_HTML}",
]
CRYSTAL_PEAK_TABLE_HEADERS = ["h", "k", "l", QXY_HTML, QZ_HTML]
FIT_DETAIL_HEADERS = ["Quantity", "Value", "Status"]
FIT_INTEGRATION_LABELS = {
    "qxy": QXY_HTML,
    "qz": QZ_HTML,
    "azimuthal": "Azimuthal",
}
FIT_INTEGRATION_MATPLOTLIB_LABELS = {
    "qxy": QXY_MATPLOTLIB_SYMBOL,
    "qz": QZ_MATPLOTLIB_SYMBOL,
    "azimuthal": "Azimuthal",
}
CRYSTAL_HKL_LABEL_LIMIT = 500
HKL_LABEL_MODE_ALL = "all"
HKL_LABEL_MODE_PARTIAL = "partial"
HKL_LABEL_MODE_NONE = "none"
HKL_LABEL_MODE_CHOICES = [
    ("Partial hkl", HKL_LABEL_MODE_PARTIAL),
    ("All hkl", HKL_LABEL_MODE_ALL),
    ("No hkl", HKL_LABEL_MODE_NONE),
]
DEFAULT_PHASE_TAG = "Phase 1 / main phase"
DIRECT_BEAM_VECTOR = np.array((0.0, 1.0, 0.0), dtype=float)
FALLBACK_SCATTER_VECTOR = np.array((1.0, 0.0, 1.0), dtype=float)
ORIENTATION_ANGLE_LIMITS_DEG = (-180.0, 180.0)
ORIENTATION_SLIDER_SCALE = 10.0
PEAK_UNDO_LIMIT = 50
PEAK_ACTION_ICON_SIZE = QtCore.QSize(18, 18)
MIRROR_SOURCE_SELECTED = "selected"
MIRROR_SOURCE_POSITIVE_QXY = "positive-qxy"
MIRROR_SOURCE_NEGATIVE_QXY = "negative-qxy"
ROI_RESIZE_SYMMETRIC = "symmetric"
ROI_RESIZE_QZ = "qz"
ROI_RESIZE_QXY = "qxy"
ROI_RESIZE_HOTKEYS = {
    QtCore.Qt.Key.Key_S: ROI_RESIZE_SYMMETRIC,
    QtCore.Qt.Key.Key_V: ROI_RESIZE_QZ,
    QtCore.Qt.Key.Key_H: ROI_RESIZE_QXY,
}
PEAK_FINDER_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "global": {
        "threshold_percentile": 99.5,
        "adaptive_threshold": False,
        "adaptive_floor_percentile": 94.0,
        "min_snr": 4.5,
        "background_radius_px": 18,
        "max_peaks": 500,
        "min_distance_px": 8,
        "neighborhood_radius_px": 2,
    },
    "adaptive": {
        "threshold_percentile": 99.5,
        "adaptive_threshold": True,
        "adaptive_floor_percentile": 94.0,
        "min_snr": 4.5,
        "background_radius_px": 18,
        "max_peaks": 600,
        "min_distance_px": 8,
        "neighborhood_radius_px": 2,
    },
    "sensitive": {
        "threshold_percentile": 99.5,
        "adaptive_threshold": True,
        "adaptive_floor_percentile": 94.0,
        "min_snr": 3.5,
        "background_radius_px": 18,
        "max_peaks": 900,
        "min_distance_px": 8,
        "neighborhood_radius_px": 2,
    },
}
PEAK_FINDER_PRESET_TOOLTIPS = {
    "global": (
        "Use one image-wide intensity cutoff. Best when the background is "
        "fairly uniform and only the strongest, clearest peaks are needed."
    ),
    "adaptive": (
        "Use local background and noise estimates so peaks can be accepted "
        "even when the detector background changes across the image."
    ),
    "sensitive": (
        "Lower the adaptive SNR requirement and allow more candidates. Useful "
        "for weak peaks, but expect more false positives to review."
    ),
}
PEAK_FINDER_SETTING_TOOLTIPS = {
    "threshold": (
        "Global percentile cutoff for candidate peak intensity. Higher values "
        "keep only brighter pixels; lower values admit more candidates."
    ),
    "adaptive": (
        "Use local background and noise estimates so weaker real peaks can "
        "pass even when global intensity changes."
    ),
    "adaptive_floor": (
        "Lowest percentile allowed as the adaptive local floor. Raising this "
        "makes the adaptive detector more selective in noisy backgrounds."
    ),
    "min_snr": (
        "Minimum signal-to-noise ratio above the local background for adaptive "
        "peak acceptance."
    ),
    "background_px": (
        "Pixel radius used to estimate the local background and noise around "
        "each candidate."
    ),
    "max_peaks": (
        "Maximum number of peaks to add or consolidate from one Find Peaks run."
    ),
    "distance_px": (
        "Minimum pixel spacing between accepted candidates. Increase this to "
        "avoid multiple points on the same broad peak."
    ),
    "window_px": (
        "Neighborhood radius used to decide whether a pixel is a local maximum."
    ),
    "min_qz": (
        f"Reject candidates below this {QZ_HTML} value. Useful for excluding "
        "beamstop and low-q artifacts."
    ),
    "ignore_nonpositive": (
        "Ignore zero and negative intensity pixels during peak detection."
    ),
    "consolidate": (
        "Move compatible manual or channel-derived peaks onto detected local "
        "maxima instead of creating duplicates nearby."
    ),
    "find_peaks": (
        "Run peak detection using the current settings and update the peak "
        "table."
    ),
}


class _PeakViewBox(pg.ViewBox if pg is not None else object):
    """ViewBox that forwards plot clicks to the peak pane."""

    def __init__(self, on_click) -> None:
        if pg is not None:
            super().__init__()
        self.on_click = on_click
        self.pan_enabled = False

    def mouseClickEvent(self, event: Any) -> None:
        if self.pan_enabled:
            super().mouseClickEvent(event)
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self.mapSceneToView(event.scenePos())
            self.on_click(float(point.x()), float(point.y()))
            event.accept()
            return
        super().mouseClickEvent(event)


if pg is not None:

    class _DraggablePeakScatter(pg.ScatterPlotItem):
        """Scatter points that report drag gestures in data
        coordinates."""

        peakClicked = QtCore.Signal(str)
        peakDragStarted = QtCore.Signal(str)
        peakDragMoved = QtCore.Signal(str, float, float, bool)
        peakDragFinished = QtCore.Signal(str, float, float, bool)

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._drag_peak_id: str | None = None

        def mouseClickEvent(self, event: Any) -> None:
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                super().mouseClickEvent(event)
                return
            points = self.pointsAt(event.pos())
            if not points:
                event.ignore()
                return
            self.peakClicked.emit(str(points[0].data()))
            event.accept()

        def mouseDragEvent(self, event: Any) -> None:
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                super().mouseDragEvent(event)
                return
            if event.isStart():
                points = self.pointsAt(event.buttonDownPos())
                if not points:
                    event.ignore()
                    return
                self._drag_peak_id = str(points[0].data())
                self.peakDragStarted.emit(self._drag_peak_id)
            if self._drag_peak_id is None:
                event.ignore()
                return
            view_pos = self.mapToParent(event.pos())
            off_plot = not self._scene_pos_in_plot(event.scenePos())
            if event.isFinish():
                self.peakDragFinished.emit(
                    self._drag_peak_id,
                    float(view_pos.x()),
                    float(view_pos.y()),
                    off_plot,
                )
                self._drag_peak_id = None
            else:
                self.peakDragMoved.emit(
                    self._drag_peak_id,
                    float(view_pos.x()),
                    float(view_pos.y()),
                    off_plot,
                )
            event.accept()

        def _scene_pos_in_plot(self, scene_pos: Any) -> bool:
            view = self.getViewBox()
            if view is None:
                return True
            try:
                return bool(view.sceneBoundingRect().contains(scene_pos))
            except Exception:
                return True


class _PeakFitIntegrationStack(QtWidgets.QWidget):
    """Vertical matplotlib stack for qxy, qz, and azimuthal profiles."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.integrations: dict[str, dict[str, Any]] = {}
        self.fits: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, dict[str, Any]] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
        except Exception:
            self.figure = None
            self.axes = []
            self.canvas = None
            fallback = QtWidgets.QLabel("Peak integrations")
            fallback.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
            return

        self.figure = Figure(figsize=(4.8, 5.4), constrained_layout=True)
        self.axes = list(self.figure.subplots(3, 1))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.set_profiles({}, {})

    def set_profiles(
        self,
        integrations: dict[str, dict[str, Any]],
        fits: dict[str, dict[str, Any]],
        failures: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.integrations = dict(integrations)
        self.fits = dict(fits)
        self.failures = dict(failures or {})
        if self.canvas is None:
            return
        for axis, name in zip(self.axes, PEAK_FIT_INTEGRATIONS, strict=True):
            axis.clear()
            profile = self.integrations.get(name)
            axis.set_title(
                f"{FIT_INTEGRATION_MATPLOTLIB_LABELS[name]} integration"
            )
            axis.set_ylabel("Intensity")
            if profile is None:
                axis.text(
                    0.5,
                    0.5,
                    "Not run",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#6b7280",
                )
                axis.grid(True, alpha=0.2)
                continue
            x_values = np.asarray(profile.get("x_values", []), dtype=float)
            y_values = np.asarray(profile.get("y_values", []), dtype=float)
            axis.plot(x_values, y_values, color="#2563eb", linewidth=1.4)
            fit = self.fits.get(name)
            if fit is not None:
                axis.plot(
                    x_values,
                    _gaussian_1d_values(x_values, fit),
                    color="#dc2626",
                    linewidth=1.2,
                    linestyle="--",
                )
                annotation = _fit_metrics_text(name, fit)
            else:
                failure = self.failures.get(name)
                annotation = (
                    f"Fit failed: {failure.get('message', '')}"
                    if failure
                    else "Fit: not run"
                )
            axis.text(
                0.98,
                0.95,
                annotation,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#111827" if fit is not None else "#6b7280",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "alpha": 0.82,
                },
            )
            axis.set_xlabel(
                FIT_INTEGRATION_MATPLOTLIB_LABELS.get(
                    str(profile.get("name", "")),
                    str(profile.get("x_label", "")),
                )
            )
            axis.grid(True, alpha=0.2)
        self.canvas.draw_idle()


class _PeakFitMeshWidget(QtWidgets.QWidget):
    """3D ROI topology with the 2D Gaussian fit overlaid as a mesh."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        empty_text: str = "Run a 2D fit",
    ) -> None:
        super().__init__(parent)
        self.empty_text = empty_text
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
        except Exception:
            self.figure = None
            self.axes = None
            self.canvas = None
            fallback = QtWidgets.QLabel("2D fit mesh")
            fallback.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
            return

        self.figure = Figure(figsize=(4.8, 3.8), constrained_layout=True)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.set_mesh(None)

    def set_mesh(
        self,
        mesh: (
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
            ]
            | None
        ),
        fit: dict[str, Any] | None = None,
    ) -> None:
        if self.canvas is None or self.axes is None:
            return
        if not hasattr(self, "metrics_label"):
            self.metrics_label = QtWidgets.QLabel()
            self.metrics_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            self.layout().insertWidget(0, self.metrics_label)
        self.metrics_label.setText(_fit_2d_metrics_html(fit))
        self.axes.clear()
        if mesh is None:
            self.axes.text2D(
                0.5,
                0.5,
                self.empty_text,
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )
            self.axes.set_axis_off()
            self.canvas.draw_idle()
            return
        qxy_grid, qz_grid, intensity, model = mesh
        self.axes.set_axis_on()
        self.axes.plot_surface(
            qxy_grid,
            qz_grid,
            intensity,
            cmap="viridis",
            alpha=0.72,
            linewidth=0,
            antialiased=True,
        )
        if model is not None:
            self.axes.plot_wireframe(
                qxy_grid,
                qz_grid,
                model,
                color="#f97316",
                linewidth=0.8,
                rstride=1,
                cstride=1,
            )
        self.axes.view_init(elev=32.0, azim=-45.0)
        self.axes.set_xlabel(QXY_MATPLOTLIB_SYMBOL)
        self.axes.set_ylabel(QZ_MATPLOTLIB_SYMBOL)
        self.axes.set_zlabel("Intensity")
        self.canvas.draw_idle()


class _CrystalOrientationViewer(QtWidgets.QWidget):
    """Antialiased direct-cell viewer with drag and wheel
    interaction."""

    orientationDeltaRequested = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._corners = np.empty((0, 3), dtype=float)
        self._edges: list[tuple[int, int]] = []
        self._q_vectors = np.empty((0, 3), dtype=float)
        self._zoom = 1.0
        self._last_position: QtCore.QPointF | None = None
        self.setMinimumHeight(230)
        self.setMouseTracking(True)
        self.setToolTip("Drag to rotate. Scroll to zoom.")
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def set_result(self, result: Any) -> None:
        self._corners = np.asarray(result.cell_corners, dtype=float)
        self._edges = list(result.cell_edges)
        self._q_vectors = np.asarray(result.q_vectors, dtype=float)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_position = _event_position(event)
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._last_position is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            position = _event_position(event)
            delta = position - self._last_position
            self._last_position = position
            self.orientationDeltaRequested.emit(
                float(delta.x()),
                float(delta.y()),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_position = None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._last_position = None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        steps = float(event.angleDelta().y()) / 120.0
        self._zoom = float(
            np.clip(self._zoom * (1.0 + steps * 0.08), 0.45, 3.0)
        )
        self.update()
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QtGui.QColor("#101827"))
        gradient.setColorAt(1.0, QtGui.QColor("#182033"))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtGui.QPen(QtGui.QColor("#2a354a"), 1))
        painter.drawRoundedRect(rect, 8, 8)
        self._paint_grid(painter, rect)
        if self._corners.size == 0:
            painter.setPen(QtGui.QColor("#9fb0c8"))
            painter.drawText(
                rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No crystal"
            )
            painter.end()
            return
        projected = self._project_points(self._corners, rect)
        self._paint_reciprocal_points(painter, rect)
        self._paint_edges(painter, projected)
        self._paint_axes(painter, projected)
        self._paint_beams(painter, projected, rect)
        painter.end()

    def _paint_grid(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
    ) -> None:
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 18), 1)
        painter.setPen(pen)
        spacing = max(24, rect.height() // 7)
        for x in range(rect.left() + spacing, rect.right(), spacing):
            painter.drawLine(x, rect.top() + 8, x, rect.bottom() - 8)
        for y in range(rect.top() + spacing, rect.bottom(), spacing):
            painter.drawLine(rect.left() + 8, y, rect.right() - 8, y)

    def _paint_edges(
        self,
        painter: QtGui.QPainter,
        projected: list[QtCore.QPointF],
    ) -> None:
        if not self._edges:
            return
        depths = self._corners[:, 1]
        depth_min = float(np.nanmin(depths))
        depth_span = max(float(np.nanmax(depths) - depth_min), 1.0e-12)
        edge_depths = [
            (
                float((depths[start] + depths[end]) / 2.0),
                start,
                end,
            )
            for start, end in self._edges
        ]
        for depth, start, end in sorted(edge_depths):
            alpha = int(125 + 95 * ((depth - depth_min) / depth_span))
            pen = QtGui.QPen(QtGui.QColor(118, 211, 194, alpha), 2)
            painter.setPen(pen)
            painter.drawLine(projected[start], projected[end])
        for point in projected:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#f7d06f"))
            painter.drawEllipse(point, 3.4, 3.4)

    def _paint_axes(
        self,
        painter: QtGui.QPainter,
        projected: list[QtCore.QPointF],
    ) -> None:
        if len(projected) < 4:
            return
        axes = [
            (1, "a", "#ff8f70"),
            (2, "b", "#75d4c3"),
            (3, "c", "#8ab4ff"),
        ]
        origin = projected[0]
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        for index, label, color in axes:
            pen = QtGui.QPen(QtGui.QColor(color), 2.4)
            painter.setPen(pen)
            painter.drawLine(origin, projected[index])
            painter.setPen(QtGui.QColor(color))
            painter.drawText(projected[index] + QtCore.QPointF(5, -5), label)

    def _paint_beams(
        self,
        painter: QtGui.QPainter,
        projected: list[QtCore.QPointF],
        rect: QtCore.QRect,
    ) -> None:
        if not projected:
            return
        origin = projected[0]
        distances = [
            float(
                np.hypot(
                    point.x() - origin.x(),
                    point.y() - origin.y(),
                )
            )
            for point in projected
        ]
        span = max(
            max(distances, default=0.0),
            min(rect.width(), rect.height()) * 0.24,
        )

        direct_direction = _screen_unit_vector(
            _preview_unit_vector(DIRECT_BEAM_VECTOR)
        )
        direct_tail = _offset_point(origin, direct_direction, -span * 0.72)
        direct_tip = _offset_point(origin, direct_direction, span * 0.82)
        self._paint_beam_arrow(
            painter,
            direct_tail,
            direct_tip,
            color="#ff5b5b",
            label="Direct beam",
        )

        scattered_direction = _screen_unit_vector(
            _scattered_beam_preview_direction(self._q_vectors)
        )
        scattered_tip = _offset_point(origin, scattered_direction, span * 0.95)
        self._paint_beam_arrow(
            painter,
            origin,
            scattered_tip,
            color="#ffbd59",
            label="Scattered beam",
        )

    def _paint_beam_arrow(
        self,
        painter: QtGui.QPainter,
        tail: QtCore.QPointF,
        tip: QtCore.QPointF,
        *,
        color: str,
        label: str,
    ) -> None:
        painter.save()
        beam_color = QtGui.QColor(color)
        beam_color.setAlpha(230)
        pen = QtGui.QPen(beam_color, 2.7)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(tail, tip)

        head = _beam_arrowhead_points(_point_array(tail), _point_array(tip))
        path = QtGui.QPainterPath()
        path.moveTo(QtCore.QPointF(float(head[0, 0]), float(head[0, 1])))
        path.lineTo(QtCore.QPointF(float(head[1, 0]), float(head[1, 1])))
        path.lineTo(QtCore.QPointF(float(head[2, 0]), float(head[2, 1])))
        painter.drawPath(path)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize() - 1, 8))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(color))
        label_direction = _unit_vector(_point_array(tip) - _point_array(tail))
        label_position = _offset_point(tip, label_direction, 10.0)
        painter.drawText(label_position + QtCore.QPointF(5.0, -4.0), label)
        painter.restore()

    def _paint_reciprocal_points(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
    ) -> None:
        if self._q_vectors.size == 0:
            return
        q_vectors = np.asarray(self._q_vectors, dtype=float)
        q_norm = np.linalg.norm(q_vectors, axis=1)
        max_q = max(float(np.nanmax(q_norm)), 1.0e-12)
        cell_span = max(
            float(np.nanmax(np.linalg.norm(self._corners, axis=1))),
            1.0,
        )
        scaled = q_vectors / max_q * cell_span * 0.45
        points = self._project_points(scaled, rect)
        depths = scaled[:, 1]
        depth_min = float(np.nanmin(depths))
        depth_span = max(float(np.nanmax(depths) - depth_min), 1.0e-12)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for point, depth in sorted(
            zip(points, depths, strict=False), key=lambda item: item[1]
        ):
            alpha = int(90 + 105 * ((float(depth) - depth_min) / depth_span))
            painter.setBrush(QtGui.QColor(255, 189, 89, alpha))
            painter.drawEllipse(point, 2.1, 2.1)

    def _project_points(
        self,
        points: np.ndarray,
        rect: QtCore.QRect,
    ) -> list[QtCore.QPointF]:
        array = np.asarray(points, dtype=float)
        if array.size == 0:
            return []
        projected = _project_cell_preview(array)
        projected_x = projected[:, 0]
        projected_y = projected[:, 1]
        span = max(
            float(np.nanmax(projected_x) - np.nanmin(projected_x)),
            float(np.nanmax(projected_y) - np.nanmin(projected_y)),
            1.0,
        )
        scale = min(rect.width(), rect.height()) / span * 0.58 * self._zoom
        center = rect.center()
        return [
            QtCore.QPointF(
                center.x()
                + (float(x) - float(np.nanmean(projected_x))) * scale,
                center.y()
                - (float(y) - float(np.nanmean(projected_y))) * scale,
            )
            for x, y in zip(projected_x, projected_y, strict=False)
        ]


class PeakIdentificationPane(QtWidgets.QWidget):
    """Identify peak centers and rectangular fit ROIs on corrected
    data."""

    peakSetChanged = QtCore.Signal(str)

    def __init__(
        self,
        project: ProjectState,
        data_id: str,
        *,
        image_style: ImageDisplayStyle | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.data_id = data_id
        self.axis_ranges: tuple[float, float, float, float] | None = None
        self.coordinate_space = "pixel"
        self.image_data = self._load_display_image()
        self.image_style = image_style or ImageDisplayStyle()
        self.active_peak_id: str | None = None
        self.roi_graphics: list[Any] = []
        self.selected_roi_resize_mode = ROI_RESIZE_SYMMETRIC
        self._held_roi_resize_mode: str | None = None
        self._roi_drag_originals: dict[str, dict[str, Any]] = {}
        self._pending_peak_drag_snapshot: list[dict[str, Any]] | None = None
        self._undo_stack: list[list[dict[str, Any]]] = []
        self._redo_stack: list[list[dict[str, Any]]] = []
        self._restoring_history = False
        self.crystal_quaternion = (0.0, 0.0, 0.0, 1.0)
        self.crystal_orientation_angles = (0.0, 0.0, 0.0)
        self.crystal_calculator = CrystalOverlayCalculator()
        self.plot_frame: _ImageAspectPlotFrame | None = None
        self.crystal_overlay_graphics: list[Any] = []
        self.crystal_edge_graphics: list[Any] = []
        self.crystal_update_timer = QtCore.QTimer(self)
        self.crystal_update_timer.setSingleShot(True)
        self.crystal_update_timer.setInterval(50)
        self.crystal_update_timer.timeout.connect(self._update_crystal_overlay)

        self._build_controls()
        self._build_plot()
        self._build_table()
        self._build_layout()
        restored_crystal_overlay = self._restore_crystal_overlay_state()
        self._set_initial_image()
        self._sync_table()
        self._sync_fit_peak_combo()
        self._refresh_fit_view()
        self._refresh_peak_graphics()
        if restored_crystal_overlay:
            self._schedule_crystal_overlay_update()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._sync_history_buttons()

    def run_peak_finder(self) -> None:
        """Run the automatic local-maximum peak finder."""

        if self.image_data is None:
            self._set_peak_finder_status("No image is loaded.")
            return
        self._set_peak_finder_status("Finding peaks...")
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            self._run_peak_finder()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _run_peak_finder(self) -> None:
        x_axis, y_axis = self._image_axes()
        valid_mask = self._valid_peak_mask(y_axis)
        candidates = find_local_maxima_peaks(
            self.image_data,
            x_axis=x_axis,
            y_axis=y_axis,
            valid_mask=valid_mask,
            config=LocalMaxPeakFinderConfig(
                threshold_percentile=self.threshold_percentile.value(),
                adaptive_threshold=self.adaptive_peak_threshold_check.isChecked(),
                adaptive_floor_percentile=self.adaptive_floor_percentile.value(),
                min_snr=self.min_snr.value(),
                background_radius_px=self.background_radius_px.value(),
                max_peaks=self.max_peaks.value(),
                min_distance_px=self.min_distance_px.value(),
                neighborhood_radius_px=self.neighborhood_radius_px.value(),
                ignore_nonpositive=self.ignore_nonpositive_check.isChecked(),
            ),
        )
        records = copy.deepcopy(self.peaks())
        used = {_peak_id(record) for record in records}
        changed_peak_ids: list[str] = []
        consolidated_ids: set[str] = set()
        added_count = 0
        consolidated_count = 0
        skipped_nearby_count = 0
        for candidate in candidates:
            nearby = self._nearby_peak_records_for_candidate(
                candidate,
                records,
            )
            consolidate_target = None
            if self.consolidate_peaks_check.isChecked():
                for record, _distance in nearby:
                    peak_id = _peak_id(record)
                    if peak_id in consolidated_ids:
                        continue
                    if self._can_consolidate_peak_record(record):
                        consolidate_target = record
                        break
            if consolidate_target is not None:
                self._consolidate_peak_record(consolidate_target, candidate)
                peak_id = _peak_id(consolidate_target)
                consolidated_ids.add(peak_id)
                changed_peak_ids.append(peak_id)
                consolidated_count += 1
                continue
            if nearby:
                skipped_nearby_count += 1
                continue
            record = self._peak_record(
                candidate.x,
                candidate.y,
                candidate.intensity,
                source="auto-local-maximum",
                used_ids=used,
            )
            self._store_peak_finder_metadata(record, candidate)
            records.append(record)
            changed_peak_ids.append(_peak_id(record))
            added_count += 1
        status = self._peak_finder_status_text(
            len(candidates),
            added_count,
            consolidated_count,
            skipped_nearby_count,
        )
        if not changed_peak_ids:
            self._set_peak_finder_status(status)
            return
        self._push_undo_state()
        self._set_peaks(records)
        self.active_peak_id = changed_peak_ids[0]
        self._sync_after_peak_change()
        self._set_peak_finder_status(status)

    def _apply_peak_finder_preset(self, name: str) -> None:
        preset = PEAK_FINDER_PRESETS.get(name)
        if preset is None:
            return
        self.threshold_percentile.setValue(
            float(preset["threshold_percentile"])
        )
        self.adaptive_peak_threshold_check.setChecked(
            bool(preset["adaptive_threshold"])
        )
        self.adaptive_floor_percentile.setValue(
            float(preset["adaptive_floor_percentile"])
        )
        self.min_snr.setValue(float(preset["min_snr"]))
        self.background_radius_px.setValue(int(preset["background_radius_px"]))
        self.max_peaks.setValue(int(preset["max_peaks"]))
        self.min_distance_px.setValue(int(preset["min_distance_px"]))
        self.neighborhood_radius_px.setValue(
            int(preset["neighborhood_radius_px"])
        )
        self._set_peak_finder_status(f"Preset: {name.title()}")

    def _peak_finder_status_text(
        self,
        candidate_count: int,
        added_count: int,
        consolidated_count: int,
        skipped_nearby_count: int,
    ) -> str:
        changed = added_count + consolidated_count
        if candidate_count == 0:
            return "No peak candidates matched the current thresholds."
        parts = [
            f"{candidate_count} candidate(s)",
            f"{added_count} added",
            f"{consolidated_count} consolidated",
        ]
        if skipped_nearby_count:
            parts.append(f"{skipped_nearby_count} already present")
        if changed == 0:
            parts.append("no changes")
        return "; ".join(parts) + "."

    def _set_peak_finder_status(self, message: str) -> None:
        self.peak_finder_status_label.setText(message)

    def _nearby_peak_records_for_candidate(
        self,
        candidate: Any,
        records: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        x_radius, y_radius = self._finder_merge_radii()
        nearby: list[tuple[dict[str, Any], float]] = []
        for record in records:
            distance = (
                (_peak_qxy(record) - float(candidate.x)) / x_radius
            ) ** 2 + ((_peak_qz(record) - float(candidate.y)) / y_radius) ** 2
            if distance <= 1.0:
                nearby.append((record, distance))
        nearby.sort(key=lambda item: item[1])
        return nearby

    def _finder_merge_radii(self) -> tuple[float, float]:
        radius_px = max(float(self.min_distance_px.value()), 1.0)
        x_step, y_step = self._axis_pixel_steps()
        return x_step * radius_px, y_step * radius_px

    def _axis_pixel_steps(self) -> tuple[float, float]:
        if self.image_data is None or self.axis_ranges is None:
            return 1.0, 1.0
        height, width = self.image_data.shape
        x_min, x_max, y_min, y_max = self.axis_ranges
        x_step = abs(float(x_max) - float(x_min)) / max(width - 1, 1)
        y_step = abs(float(y_max) - float(y_min)) / max(height - 1, 1)
        return max(x_step, 1.0e-12), max(y_step, 1.0e-12)

    def _can_consolidate_peak_record(
        self,
        record: dict[str, Any],
    ) -> bool:
        if _is_gap_estimated_peak(record):
            return False
        source = str(record.get("source", "")).lower()
        return (
            "manual" in source
            or "integration" in source
            or "channel" in source
        )

    def _consolidate_peak_record(
        self,
        record: dict[str, Any],
        candidate: Any,
    ) -> None:
        source = str(record.get("source", ""))
        record["qxy"] = float(candidate.x)
        record["qz"] = float(candidate.y)
        record["intensity"] = float(candidate.intensity)
        if "integration" in source.lower() or "channel" in source.lower():
            record["source"] = "integration-channel-local-maximum"
        else:
            record["source"] = "manual-local-maximum"
        metadata = dict(record.get("metadata", {}))
        metadata["consolidated_by"] = "find-peaks"
        if source:
            metadata.setdefault("consolidated_from_source", source)
        record["metadata"] = metadata
        self._store_peak_finder_metadata(record, candidate)
        if record.get("roi"):
            self._apply_roi_to_record(record)

    def _store_peak_finder_metadata(
        self,
        record: dict[str, Any],
        candidate: Any,
    ) -> None:
        metadata = dict(record.get("metadata", {}))
        finder = {
            "adaptive_threshold": self.adaptive_peak_threshold_check.isChecked(),
            "threshold_percentile": float(self.threshold_percentile.value()),
            "adaptive_floor_percentile": float(
                self.adaptive_floor_percentile.value()
            ),
            "min_snr": float(self.min_snr.value()),
            "background_radius_px": int(self.background_radius_px.value()),
            "min_distance_px": int(self.min_distance_px.value()),
            "neighborhood_radius_px": int(self.neighborhood_radius_px.value()),
        }
        for key in ("background", "noise", "snr", "prominence", "score"):
            value = getattr(candidate, key, None)
            if value is None:
                continue
            try:
                finder[key] = float(value)
            except (TypeError, ValueError):
                continue
        metadata["peak_finder"] = finder
        record["metadata"] = metadata

    def add_peak_at(
        self,
        qxy: float,
        qz: float,
        *,
        source: str = "manual",
        record_history: bool = True,
    ) -> dict[str, Any]:
        """Add one manual peak at the supplied plot coordinate."""

        if record_history:
            self._push_undo_state()
        records = self.peaks()
        used = {_peak_id(record) for record in records}
        target = (
            self._snap_target_near(qxy, qz)
            if source == "manual" and self._coordinate_is_masked(qxy, qz)
            else None
        )
        if target is not None:
            qxy = float(target["qxy"])
            qz = float(target["qz"])
            intensity = float(target["intensity"])
            source = str(target["source"])
        else:
            intensity = self._intensity_at(qxy, qz)
        record = self._peak_record(
            qxy,
            qz,
            intensity,
            source=source,
            used_ids=used,
        )
        if target is not None:
            self._apply_peak_target_metadata(record, target)
        records.append(record)
        self._set_peaks(records)
        self.active_peak_id = _peak_id(record)
        self._sync_after_peak_change()
        if target is not None and target.get("kind") == "masked-gap":
            self.snap_feedback_label.setText(
                "Placed masked-gap peak estimate from side maxima."
            )
        return record

    def add_integration_markers(
        self,
        markers: list[Any],
    ) -> list[dict[str, Any]]:
        """Add q-space peaks generated from integration-channel
        markers."""

        records = self.peaks()
        used = {_peak_id(record) for record in records}
        existing_marker_ids = {
            str(record.get("metadata", {}).get("integration_marker_id"))
            for record in records
            if record.get("metadata", {}).get("integration_marker_id")
        }
        added: list[dict[str, Any]] = []
        pushed_history = False
        for marker in markers:
            marker_id = str(_marker_field(marker, "marker_id", ""))
            if marker_id and marker_id in existing_marker_ids:
                continue
            qxy = float(_marker_field(marker, "qxy", 0.0))
            qz = float(_marker_field(marker, "qz", 0.0))
            intensity = float(
                _marker_field(marker, "integrated_intensity", 0.0)
            )
            record = self._peak_record(
                qxy,
                qz,
                intensity,
                source="integration-channel",
                used_ids=used,
            )
            marker_label = _marker_field(marker, "label", "")
            if marker_label:
                record["label"] = str(marker_label)
            record["metadata"] = {
                "integration_marker_id": marker_id,
                "integration_channel": _marker_field(marker, "channel"),
                "integration_roi_id": _marker_field(marker, "roi_id"),
                "integration_roi_name": _marker_field(marker, "roi_name"),
                "integration_mode": _marker_field(marker, "mode"),
                "integration_x": _marker_field(marker, "integration_x"),
            }
            records.append(record)
            added.append(record)
            if marker_id:
                existing_marker_ids.add(marker_id)
        if not added:
            return []
        if not pushed_history:
            self._push_undo_state()
        self._set_peaks(records)
        self.active_peak_id = _peak_id(added[0])
        self._sync_after_peak_change()
        return added

    def remove_peak(
        self,
        peak_id: str | None,
        *,
        record_history: bool = True,
    ) -> None:
        """Remove one peak record by id."""

        if peak_id is None:
            return
        if record_history:
            self._push_undo_state()
        records = [
            record for record in self.peaks() if _peak_id(record) != peak_id
        ]
        self._set_peaks(records)
        self.active_peak_id = _peak_id(records[0]) if records else None
        self._sync_after_peak_change()

    def delete_selected_peak(self) -> None:
        self.remove_peak(self.active_peak_id)

    def clear_peaks(self) -> None:
        if not self.peaks():
            return
        self._push_undo_state()
        self._set_peaks([])
        self.active_peak_id = None
        self._sync_after_peak_change()

    def snap_selected_peak_to_local_maximum(self) -> None:
        """Move the active peak to the brightest local pixel nearby."""

        record = self._active_record()
        if record is None:
            return
        target = self._snap_target_near(
            _peak_qxy(record),
            _peak_qz(record),
        )
        if target is None:
            return
        self._push_undo_state()
        self._apply_peak_target_to_record(record, target)
        if record.get("roi"):
            self._apply_roi_to_record(record)
        self._sync_after_peak_change()

    def snap_all_peaks_to_local_maxima(self) -> None:
        """Move every peak to the brightest nearby local pixel."""

        records = self.peaks()
        if not records:
            return
        updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for record in records:
            target = self._snap_target_near(
                _peak_qxy(record),
                _peak_qz(record),
            )
            if target is not None:
                updates.append((record, target))
        if not updates:
            self.snap_feedback_label.setText("No finite local maxima found.")
            return
        self._push_undo_state()
        for record, target in updates:
            self._apply_peak_target_to_record(record, target)
            if record.get("roi"):
                self._apply_roi_to_record(record)
        self._set_peaks(records)
        self.snap_feedback_label.setText(
            f"Snapped {len(updates)} peak(s) to nearby maxima."
        )
        self._sync_after_peak_change()

    def create_gap_estimated_peak(self) -> None:
        """Create a gap-estimated peak at the current plot center."""

        if self.image_data is None:
            return
        if pg is not None and hasattr(self, "plot_widget"):
            view_range = self.plot_widget.viewRange()
            qxy = float(sum(view_range[0]) / 2.0)
            qz = float(sum(view_range[1]) / 2.0)
        elif self.axis_ranges is not None:
            x_min, x_max, y_min, y_max = self.axis_ranges
            qxy = float((x_min + x_max) / 2.0)
            qz = float((y_min + y_max) / 2.0)
        else:
            height, width = self.image_data.shape
            qxy = float(width / 2.0)
            qz = float(height / 2.0)
        record = self.add_peak_at(qxy, qz, source="gap estimate")
        record.setdefault("metadata", {})["gap_estimate"] = True
        record.setdefault("metadata", {})[
            "estimate_method"
        ] = "manual plot center"
        record["point_kind"] = PEAK_POINT_KIND_GAP_ESTIMATED
        record["gap_estimated"] = True
        record["phase_tag"] = "gap-estimated"
        self._sync_after_peak_change()

    def toggle_active_gap_estimate(self) -> None:
        """Toggle whether the active peak is marked as gap-estimated."""

        record = self._active_record()
        if record is None:
            return
        self._push_undo_state()
        metadata = record.setdefault("metadata", {})
        enabled = not bool(metadata.get("gap_estimate"))
        metadata["gap_estimate"] = enabled
        metadata["estimate_method"] = "manual toggle" if enabled else ""
        record["source"] = "gap estimate" if enabled else "manual"
        record["point_kind"] = (
            PEAK_POINT_KIND_GAP_ESTIMATED
            if enabled
            else PEAK_POINT_KIND_COMMITTED
        )
        record["gap_estimated"] = enabled
        record["phase_tag"] = "gap-estimated" if enabled else DEFAULT_PHASE_TAG
        self._sync_after_peak_change()

    def check_peak_symmetry(self) -> None:
        """Summarize mirrored peak matches about the qz axis."""

        records = self.peaks()
        if not records:
            self.symmetry_summary_label.setText("No peaks to compare.")
            return
        qxy_tolerance = self.symmetry_qxy_tolerance.value()
        qz_tolerance = self.symmetry_qz_tolerance.value()
        unmatched = 0
        for record in records:
            qxy = _peak_qxy(record)
            qz = _peak_qz(record)
            if abs(qxy) <= qxy_tolerance:
                continue
            mirrored = any(
                abs(_peak_qxy(other) + qxy) <= qxy_tolerance
                and abs(_peak_qz(other) - qz) <= qz_tolerance
                for other in records
                if other is not record
            )
            if not mirrored:
                unmatched += 1
        matched = len(records) - unmatched
        self.symmetry_summary_label.setText(
            f"{matched}/{len(records)} peak(s) have a mirrored partner."
        )

    def mirror_missing_peaks(self) -> None:
        """Add gap-estimated mirror partners across the qz axis."""

        records = self.peaks()
        if not records:
            self.symmetry_summary_label.setText("No peaks to mirror.")
            return
        qxy_tolerance = self.symmetry_qxy_tolerance.value()
        qz_tolerance = self.symmetry_qz_tolerance.value()
        source_records = self._mirror_source_peak_records(
            records,
            qxy_tolerance=qxy_tolerance,
        )
        if not source_records:
            self.symmetry_summary_label.setText(
                "Select peaks or choose a populated qxy side to mirror."
            )
            return

        used = {_peak_id(record) for record in records}
        updated_records = copy.deepcopy(records)
        added: list[dict[str, Any]] = []
        skipped_existing = 0
        skipped_axis = 0
        for source in source_records:
            source_qxy = _peak_qxy(source)
            if abs(source_qxy) <= qxy_tolerance:
                skipped_axis += 1
                continue
            target_qxy = -source_qxy
            target_qz = _peak_qz(source)
            if self._has_peak_near(
                updated_records,
                target_qxy,
                target_qz,
                qxy_tolerance=qxy_tolerance,
                qz_tolerance=qz_tolerance,
            ):
                skipped_existing += 1
                continue
            mirrored = self._mirrored_gap_peak_record(
                source,
                target_qxy,
                target_qz,
                used_ids=used,
            )
            updated_records.append(mirrored)
            added.append(mirrored)

        if not added:
            self.symmetry_summary_label.setText(
                "No missing mirrored partners found "
                f"({skipped_existing} already matched, "
                f"{skipped_axis} near the qz axis)."
            )
            return

        self._push_undo_state()
        self._set_peaks(updated_records)
        self.active_peak_id = _peak_id(added[0])
        self._sync_after_peak_change()
        self.symmetry_summary_label.setText(
            f"Added {len(added)} mirrored gap estimate(s); "
            f"skipped {skipped_existing} existing partner(s)."
        )

    def _mirror_source_peak_records(
        self,
        records: list[dict[str, Any]],
        *,
        qxy_tolerance: float,
    ) -> list[dict[str, Any]]:
        mode = str(
            self.mirror_source_combo.currentData() or MIRROR_SOURCE_SELECTED
        )
        if mode == MIRROR_SOURCE_POSITIVE_QXY:
            return [
                record
                for record in records
                if _peak_qxy(record) > qxy_tolerance
            ]
        if mode == MIRROR_SOURCE_NEGATIVE_QXY:
            return [
                record
                for record in records
                if _peak_qxy(record) < -qxy_tolerance
            ]
        selected = self._selected_peak_records(records)
        if selected:
            return selected
        active = self._active_record()
        return [active] if active is not None else []

    def _selected_peak_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selection_model = self.peak_table.selectionModel()
        if selection_model is None:
            return []
        rows = sorted(
            {index.row() for index in selection_model.selectedRows()}
        )
        return [records[row] for row in rows if 0 <= row < len(records)]

    def _has_peak_near(
        self,
        records: list[dict[str, Any]],
        qxy: float,
        qz: float,
        *,
        qxy_tolerance: float,
        qz_tolerance: float,
    ) -> bool:
        return any(
            abs(_peak_qxy(record) - qxy) <= qxy_tolerance
            and abs(_peak_qz(record) - qz) <= qz_tolerance
            for record in records
        )

    def _mirrored_gap_peak_record(
        self,
        source: dict[str, Any],
        qxy: float,
        qz: float,
        *,
        used_ids: set[str],
    ) -> dict[str, Any]:
        record = self._peak_record(
            qxy,
            qz,
            self._intensity_at(qxy, qz),
            source="gap estimate",
            used_ids=used_ids,
        )
        record["point_kind"] = PEAK_POINT_KIND_GAP_ESTIMATED
        record["gap_estimated"] = True
        record["phase_tag"] = "gap-estimated"
        record["metadata"] = {
            "gap_estimate": True,
            "estimate_method": "mirror across qz axis",
            "mirror_axis": "qz",
            "mirror_source_peak_id": _peak_id(source),
            "mirror_source_qxy": _peak_qxy(source),
            "mirror_source_qz": _peak_qz(source),
        }
        return record

    def apply_roi_to_selected_peak(self) -> None:
        record = self._active_record()
        if record is None:
            return
        self._push_undo_state()
        self._apply_roi_to_record(record)
        self._sync_after_peak_change()

    def apply_roi_to_all_peaks(self) -> None:
        self._push_undo_state()
        for record in self.peaks():
            self._apply_roi_to_record(record)
        self._sync_after_peak_change()

    def run_integrations_for_selected_roi(self) -> None:
        record = self._active_record()
        if record is None:
            self._set_fit_status("Select a peak before integrating.")
            return
        if not record.get("roi"):
            self._set_fit_status("The selected peak does not have an ROI.")
            return
        if self._compute_integrations_for_record(record):
            self._set_fit_status(
                f"Computed {QXY_HTML}, {QZ_HTML}, and azimuthal profiles."
            )
            self._sync_after_fit_change(record)

    def run_integrations_for_all_rois(self) -> None:
        count = 0
        for record in self.peaks():
            if record.get("roi") and self._compute_integrations_for_record(
                record
            ):
                count += 1
        self._set_fit_status(f"Computed integrations for {count} ROI(s).")
        self._sync_after_fit_change(self._active_record())

    def run_fit_for_selected_integration(self) -> None:
        name = str(self.fit_integration_combo.currentData() or "qxy")
        self._run_fit_for_selected_integration_name(name)

    def run_fit_for_selected_qxy_trace(self) -> None:
        self._run_fit_for_selected_integration_name("qxy")

    def run_fit_for_selected_qz_trace(self) -> None:
        self._run_fit_for_selected_integration_name("qz")

    def run_fit_for_selected_azimuthal_trace(self) -> None:
        self._run_fit_for_selected_integration_name("azimuthal")

    def _run_fit_for_selected_integration_name(self, name: str) -> None:
        record = self._active_record()
        if record is None or not record.get("roi"):
            self._set_fit_status("Select a peak with an ROI before fitting.")
            return
        store = self._fit_record_for_peak(record, create=True)
        if not store.get("integrations"):
            self._set_fit_status(
                "Integrate the selected ROI before fitting a 1D trace."
            )
            return
        integration = store.get("integrations", {}).get(name)
        if integration is None:
            self._set_fit_status(f"No {FIT_INTEGRATION_LABELS[name]} profile.")
            return
        fit = fit_peak_integration(integration)
        if fit is None:
            store.setdefault("integration_fits", {}).pop(name, None)
            store.setdefault("fit_failures", {})[name] = {
                "status": "failed",
                "message": f"Could not fit {FIT_INTEGRATION_LABELS[name]} profile.",
            }
            store["integrations"][name].pop("fit", None)
            self._set_fit_status(
                f"Could not fit {FIT_INTEGRATION_LABELS[name]} profile."
            )
            self._sync_after_fit_change(record)
            return
        store.setdefault("integration_fits", {})[name] = fit
        store.setdefault("fit_failures", {}).pop(name, None)
        store["integrations"][name]["fit"] = fit
        self._set_fit_status(f"Fit {FIT_INTEGRATION_LABELS[name]} profile.")
        self._sync_after_fit_change(record)

    def run_all_integration_fits_for_selected_roi(self) -> None:
        record = self._active_record()
        if record is None or not record.get("roi"):
            self._set_fit_status("Select a peak with an ROI before fitting.")
            return
        store = self._fit_record_for_peak(record, create=True)
        if not store.get("integrations"):
            self._set_fit_status(
                "Integrate the selected ROI before fitting integrated traces."
            )
            return
        fits: dict[str, dict[str, Any]] = {}
        failures: dict[str, dict[str, Any]] = {}
        for name, integration in store.get("integrations", {}).items():
            fit = fit_peak_integration(integration)
            if fit is None:
                failures[name] = {
                    "status": "failed",
                    "message": (
                        f"Could not fit {FIT_INTEGRATION_LABELS.get(name, name)} "
                        "profile."
                    ),
                }
            else:
                fits[name] = fit
        store["integration_fits"] = fits
        store["fit_failures"] = failures
        for name, fit in fits.items():
            store["integrations"][name]["fit"] = fit
        for name in failures:
            store["integrations"][name].pop("fit", None)
        self._set_fit_status(
            f"Fit {len(fits)} currently integrated trace(s) for selected ROI."
        )
        self._sync_after_fit_change(record)

    def run_2d_fit_for_selected_roi(self) -> None:
        record = self._active_record()
        if record is None:
            self._set_fit_status("Select a peak before fitting.")
            return
        if self._fit_2d_for_record(record, prepare_missing=True):
            self._set_fit_status("Fit the selected ROI with a 2D Gaussian.")
            self._sync_after_fit_change(record)

    def batch_process_all_peak_fits(self) -> None:
        count = 0
        for record in self.peaks():
            if self._fit_2d_for_record(record, prepare_missing=True):
                count += 1
        self._set_fit_status(
            f"Completed full fit workflow for {count} ROI(s)."
        )
        self._sync_after_fit_change(self._active_record())

    def peaks(self) -> list[dict[str, Any]]:
        return list(self.project.peak_sets.get(self.data_id, []))

    def _set_peaks(self, records: list[dict[str, Any]]) -> None:
        self.project.peak_sets[self.data_id] = records

    def _snapshot_peaks(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.peaks())

    def _push_undo_state(self) -> None:
        if self._restoring_history:
            return
        self._undo_stack.append(self._snapshot_peaks())
        if len(self._undo_stack) > PEAK_UNDO_LIMIT:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._sync_history_buttons()

    def undo_peak_action(self) -> None:
        if not self._undo_stack:
            return
        current = self._snapshot_peaks()
        records = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_peak_snapshot(records)

    def redo_peak_action(self) -> None:
        if not self._redo_stack:
            return
        current = self._snapshot_peaks()
        records = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._restore_peak_snapshot(records)

    def _restore_peak_snapshot(self, records: list[dict[str, Any]]) -> None:
        self._restoring_history = True
        try:
            self._set_peaks(copy.deepcopy(records))
            ids = {_peak_id(record) for record in records}
            if self.active_peak_id not in ids:
                self.active_peak_id = _peak_id(records[0]) if records else None
            self._sync_after_peak_change()
        finally:
            self._restoring_history = False
            self._sync_history_buttons()

    def _sync_history_buttons(self) -> None:
        for button, stack in (
            (getattr(self, "undo_button", None), self._undo_stack),
            (getattr(self, "redo_button", None), self._redo_stack),
        ):
            if button is not None:
                button.setEnabled(bool(stack))

    def _set_roi_resize_mode(self, mode: str) -> None:
        self.selected_roi_resize_mode = mode
        labels = {
            ROI_RESIZE_SYMMETRIC: "Symmetric",
            ROI_RESIZE_QZ: f"{QZ_HTML} only",
            ROI_RESIZE_QXY: f"{QXY_HTML} only",
        }
        self.resize_mode_label.setText(
            f"Mode: {labels.get(mode, labels[ROI_RESIZE_SYMMETRIC])}"
        )

    def _build_controls(self) -> None:
        self.colormap_combo = QtWidgets.QComboBox()
        for name in IMAGE_COLORMAPS:
            self.colormap_combo.addItem(name.title(), name)
        self.level_min = _level_spinbox()
        self.level_max = _level_spinbox()
        self.quantile_check = QtWidgets.QCheckBox("Quantile")
        self.quantile_low = _quantile_spinbox(self.image_style.quantile_low)
        self.quantile_high = _quantile_spinbox(self.image_style.quantile_high)
        self.auto_contrast_button = QtWidgets.QToolButton()
        self.auto_contrast_button.setText("Auto")
        self._set_image_style_controls(self.image_style)
        self.colormap_combo.currentIndexChanged.connect(
            self._apply_image_style_from_controls
        )
        self.level_min.valueChanged.connect(self._apply_manual_levels)
        self.level_max.valueChanged.connect(self._apply_manual_levels)
        self.quantile_check.toggled.connect(
            self._apply_image_style_from_controls
        )
        self.quantile_low.valueChanged.connect(
            self._apply_image_style_from_controls
        )
        self.quantile_high.valueChanged.connect(
            self._apply_image_style_from_controls
        )
        self.auto_contrast_button.clicked.connect(self._set_quantile_levels)

        self.threshold_percentile = QtWidgets.QDoubleSpinBox()
        self.threshold_percentile.setRange(0.0, 100.0)
        self.threshold_percentile.setDecimals(2)
        self.threshold_percentile.setSingleStep(0.1)
        self.threshold_percentile.setSuffix("%")
        self.threshold_percentile.setValue(99.5)
        self.threshold_percentile.setMaximumWidth(95)

        self.adaptive_peak_threshold_check = QtWidgets.QCheckBox("Adaptive")
        self.adaptive_peak_threshold_check.setChecked(False)

        self.adaptive_floor_percentile = QtWidgets.QDoubleSpinBox()
        self.adaptive_floor_percentile.setRange(0.0, 100.0)
        self.adaptive_floor_percentile.setDecimals(2)
        self.adaptive_floor_percentile.setSingleStep(0.5)
        self.adaptive_floor_percentile.setSuffix("%")
        self.adaptive_floor_percentile.setValue(94.0)
        self.adaptive_floor_percentile.setMaximumWidth(95)

        self.min_snr = QtWidgets.QDoubleSpinBox()
        self.min_snr.setRange(0.0, 1000.0)
        self.min_snr.setDecimals(2)
        self.min_snr.setSingleStep(0.25)
        self.min_snr.setValue(4.5)
        self.min_snr.setMaximumWidth(80)

        self.background_radius_px = QtWidgets.QSpinBox()
        self.background_radius_px.setRange(3, 1000)
        self.background_radius_px.setValue(18)
        self.background_radius_px.setMaximumWidth(72)

        self.max_peaks = QtWidgets.QSpinBox()
        self.max_peaks.setRange(1, 10000)
        self.max_peaks.setValue(
            int(PEAK_FINDER_PRESETS["global"]["max_peaks"])
        )
        self.max_peaks.setMaximumWidth(80)

        self.min_distance_px = QtWidgets.QSpinBox()
        self.min_distance_px.setRange(0, 1000)
        self.min_distance_px.setValue(8)
        self.min_distance_px.setMaximumWidth(72)

        self.neighborhood_radius_px = QtWidgets.QSpinBox()
        self.neighborhood_radius_px.setRange(1, 100)
        self.neighborhood_radius_px.setValue(2)
        self.neighborhood_radius_px.setMaximumWidth(72)

        self.min_qz = QtWidgets.QDoubleSpinBox()
        self.min_qz.setRange(-1.0e6, 1.0e6)
        self.min_qz.setDecimals(4)
        self.min_qz.setSingleStep(0.01)
        self.min_qz.setMaximumWidth(100)
        if self.axis_ranges is not None:
            self.min_qz.setValue(self.axis_ranges[2])

        self.ignore_nonpositive_check = QtWidgets.QCheckBox("Ignore <= 0")
        self.ignore_nonpositive_check.setChecked(True)
        self.consolidate_peaks_check = QtWidgets.QCheckBox(
            "Consolidate manual/channel"
        )
        self.consolidate_peaks_check.setChecked(True)

        self.find_peaks_button = QtWidgets.QToolButton()
        self.find_peaks_button.setText("Find Peaks")
        self.find_peaks_button.clicked.connect(self.run_peak_finder)
        self.global_peak_preset_button = QtWidgets.QToolButton()
        self.global_peak_preset_button.setText("Global")
        self.global_peak_preset_button.clicked.connect(
            lambda: self._apply_peak_finder_preset("global")
        )
        self.adaptive_peak_preset_button = QtWidgets.QToolButton()
        self.adaptive_peak_preset_button.setText("Adaptive")
        self.adaptive_peak_preset_button.clicked.connect(
            lambda: self._apply_peak_finder_preset("adaptive")
        )
        self.sensitive_peak_preset_button = QtWidgets.QToolButton()
        self.sensitive_peak_preset_button.setText("Sensitive")
        self.sensitive_peak_preset_button.clicked.connect(
            lambda: self._apply_peak_finder_preset("sensitive")
        )
        self._set_peak_finder_tooltips()
        self.peak_finder_status_label = QtWidgets.QLabel("Ready.")
        self.peak_finder_status_label.setWordWrap(True)
        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("Zoom In")
        self.zoom_in_button.clicked.connect(lambda: self._zoom_image(0.75))
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("Zoom Out")
        self.zoom_out_button.clicked.connect(lambda: self._zoom_image(1.35))
        self.zoom_fit_button = QtWidgets.QToolButton()
        self.zoom_fit_button.setText("Fit")
        self.zoom_fit_button.clicked.connect(self._reset_image_zoom)
        self.pan_button = QtWidgets.QToolButton()
        self.pan_button.setText("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.toggled.connect(self._set_pan_mode)

        self.add_point_button = QtWidgets.QToolButton()
        self.add_point_button.setText("Add Point")
        self.add_point_button.setCheckable(True)
        self.add_point_button.setChecked(True)
        self.remove_point_button = QtWidgets.QToolButton()
        self.remove_point_button.setText("Remove Point")
        self.remove_point_button.setCheckable(True)
        self.click_mode_group = QtWidgets.QButtonGroup(self)
        self.click_mode_group.setExclusive(True)
        self.click_mode_group.addButton(self.add_point_button)
        self.click_mode_group.addButton(self.remove_point_button)

        self.snap_peak_button = QtWidgets.QToolButton()
        self.snap_peak_button.setText("Snap to Max")
        self.snap_peak_button.clicked.connect(
            self.snap_selected_peak_to_local_maximum
        )
        self.delete_peak_button = QtWidgets.QToolButton()
        self.delete_peak_button.setText("Delete")
        self.delete_peak_button.clicked.connect(self.delete_selected_peak)
        style = QtWidgets.QApplication.style()
        self.clear_peaks_button = QtWidgets.QToolButton()
        self._configure_peak_action_button(
            self.clear_peaks_button,
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon),
            "Clear all peaks",
        )
        self.clear_peaks_button.clicked.connect(self.clear_peaks)
        self.undo_button = QtWidgets.QToolButton()
        self._configure_peak_action_button(
            self.undo_button,
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack),
            "Undo peak edit",
        )
        self.undo_button.clicked.connect(self.undo_peak_action)
        self.undo_button.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_button = QtWidgets.QToolButton()
        self._configure_peak_action_button(
            self.redo_button,
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_ArrowForward
            ),
            "Redo peak edit",
        )
        self.redo_button.clicked.connect(self.redo_peak_action)
        self.redo_button.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.snap_all_button = QtWidgets.QToolButton()
        self.snap_all_button.setText("Snap All to Max")
        self.snap_all_button.clicked.connect(
            self.snap_all_peaks_to_local_maxima
        )
        self.snap_window_px = QtWidgets.QSpinBox()
        self.snap_window_px.setRange(1, 200)
        self.snap_window_px.setValue(8)
        self.snap_window_px.setMaximumWidth(72)
        self.snap_feedback_label = QtWidgets.QLabel("")
        self.snap_feedback_label.setWordWrap(True)
        self.cursor_coordinate_label = QtWidgets.QLabel("")
        self.cursor_coordinate_label.setWordWrap(True)

        self.roi_width = _roi_size_spinbox()
        self.roi_height = _roi_size_spinbox()
        self._set_default_roi_dimensions()
        self.roi_shape_combo = QtWidgets.QComboBox()
        self.roi_shape_combo.addItem("Box", "box")
        self.snap_drag_check = QtWidgets.QCheckBox("Snap drag")
        self.snap_drag_check.setChecked(True)
        self.resize_symmetric_button = QtWidgets.QToolButton()
        self.resize_symmetric_button.setText("Symmetric")
        self.resize_symmetric_button.setCheckable(True)
        self.resize_qz_button = QtWidgets.QToolButton()
        self.resize_qz_button.setText("Vertical only")
        self.resize_qz_button.setToolTip(
            f"<qt>Resize only along {QZ_HTML}.</qt>"
        )
        self.resize_qz_button.setCheckable(True)
        self.resize_qxy_button = QtWidgets.QToolButton()
        self.resize_qxy_button.setText("Horizontal only")
        self.resize_qxy_button.setToolTip(
            f"<qt>Resize only along {QXY_HTML}.</qt>"
        )
        self.resize_qxy_button.setCheckable(True)
        self.resize_mode_group = QtWidgets.QButtonGroup(self)
        self.resize_mode_group.setExclusive(True)
        self.resize_mode_group.addButton(self.resize_symmetric_button)
        self.resize_mode_group.addButton(self.resize_qz_button)
        self.resize_mode_group.addButton(self.resize_qxy_button)
        self.resize_symmetric_button.setChecked(True)
        self.resize_symmetric_button.clicked.connect(
            lambda: self._set_roi_resize_mode(ROI_RESIZE_SYMMETRIC)
        )
        self.resize_qz_button.clicked.connect(
            lambda: self._set_roi_resize_mode(ROI_RESIZE_QZ)
        )
        self.resize_qxy_button.clicked.connect(
            lambda: self._set_roi_resize_mode(ROI_RESIZE_QXY)
        )
        self.resize_mode_label = QtWidgets.QLabel("Mode: Symmetric")
        self.resize_mode_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.resize_hotkey_note = rich_label(
            f"Hold S for symmetric, V for {QZ_HTML}-only, "
            f"H for {QXY_HTML}-only while resizing."
        )
        self.active_roi_mesh_widget = _PeakFitMeshWidget(
            empty_text="Select a peak ROI"
        )
        self.active_roi_mesh_widget.setMinimumHeight(230)
        self.apply_roi_button = QtWidgets.QToolButton()
        self.apply_roi_button.setText("Set ROI")
        self.apply_roi_button.clicked.connect(self.apply_roi_to_selected_peak)
        self.apply_all_rois_button = QtWidgets.QToolButton()
        self.apply_all_rois_button.setText("Set All ROIs")
        self.apply_all_rois_button.clicked.connect(self.apply_roi_to_all_peaks)
        self.gap_estimate_button = QtWidgets.QToolButton()
        self.gap_estimate_button.setText("Add Gap Estimate")
        self.gap_estimate_button.clicked.connect(
            self.create_gap_estimated_peak
        )
        self.tag_gap_button = QtWidgets.QToolButton()
        self.tag_gap_button.setText("Tag Gap")
        self.tag_gap_button.clicked.connect(self.toggle_active_gap_estimate)
        self.symmetry_qxy_tolerance = _roi_size_spinbox()
        self.symmetry_qxy_tolerance.setValue(0.03)
        self.symmetry_qz_tolerance = _roi_size_spinbox()
        self.symmetry_qz_tolerance.setValue(0.03)
        self.mirror_source_combo = RichTextComboBox()
        self.mirror_source_combo.addItem(
            "Selected peaks",
            MIRROR_SOURCE_SELECTED,
        )
        self.mirror_source_combo.addItem(
            f"{QXY_HTML} > 0",
            MIRROR_SOURCE_POSITIVE_QXY,
        )
        self.mirror_source_combo.addItem(
            f"{QXY_HTML} < 0",
            MIRROR_SOURCE_NEGATIVE_QXY,
        )
        self.mirror_missing_button = QtWidgets.QToolButton()
        self.mirror_missing_button.setText("Mirror Missing")
        self.mirror_missing_button.setToolTip(
            "<qt>Add gap-estimated mirror partners across the "
            f"{QZ_HTML} axis when no matching peak exists.</qt>"
        )
        self.mirror_missing_button.clicked.connect(self.mirror_missing_peaks)
        self.symmetry_check_button = QtWidgets.QToolButton()
        self.symmetry_check_button.setText("Check Symmetry")
        self.symmetry_check_button.clicked.connect(self.check_peak_symmetry)
        self.symmetry_summary_label = QtWidgets.QLabel("")
        self.symmetry_summary_label.setWordWrap(True)
        self._build_peak_fit_controls()
        self._build_crystal_overlay_controls()

    def _set_peak_finder_tooltips(self) -> None:
        controls = {
            self.threshold_percentile: "threshold",
            self.adaptive_peak_threshold_check: "adaptive",
            self.adaptive_floor_percentile: "adaptive_floor",
            self.min_snr: "min_snr",
            self.background_radius_px: "background_px",
            self.max_peaks: "max_peaks",
            self.min_distance_px: "distance_px",
            self.neighborhood_radius_px: "window_px",
            self.min_qz: "min_qz",
            self.ignore_nonpositive_check: "ignore_nonpositive",
            self.consolidate_peaks_check: "consolidate",
            self.find_peaks_button: "find_peaks",
        }
        for widget, key in controls.items():
            widget.setToolTip(qt_tooltip(PEAK_FINDER_SETTING_TOOLTIPS[key]))

        presets = {
            self.global_peak_preset_button: "global",
            self.adaptive_peak_preset_button: "adaptive",
            self.sensitive_peak_preset_button: "sensitive",
        }
        for widget, key in presets.items():
            widget.setToolTip(qt_tooltip(PEAK_FINDER_PRESET_TOOLTIPS[key]))

    @staticmethod
    def _peak_finder_label(text: str, tooltip_key: str) -> QtWidgets.QLabel:
        label = rich_label(text)
        label.setToolTip(qt_tooltip(PEAK_FINDER_SETTING_TOOLTIPS[tooltip_key]))
        return label

    @staticmethod
    def _configure_peak_action_button(
        button: QtWidgets.QToolButton,
        icon: QtGui.QIcon,
        tooltip: str,
    ) -> None:
        button.setIcon(icon)
        button.setIconSize(PEAK_ACTION_ICON_SIZE)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setAutoRaise(True)

    def _build_peak_fit_controls(self) -> None:
        self.fit_peak_combo = RichTextComboBox()
        self.fit_peak_combo.currentIndexChanged.connect(
            self._handle_fit_peak_combo_changed
        )

        self.fit_integration_combo = RichTextComboBox()
        for name in PEAK_FIT_INTEGRATIONS:
            self.fit_integration_combo.addItem(
                FIT_INTEGRATION_LABELS[name], name
            )

        self.run_selected_integrations_button = QtWidgets.QToolButton()
        self.run_selected_integrations_button.setText("Integrate Selected ROI")
        self.run_selected_integrations_button.clicked.connect(
            self.run_integrations_for_selected_roi
        )
        self.run_all_integrations_button = QtWidgets.QToolButton()
        self.run_all_integrations_button.setText("Integrate All ROIs")
        self.run_all_integrations_button.clicked.connect(
            self.run_integrations_for_all_rois
        )
        self.fit_selected_integration_button = QtWidgets.QToolButton()
        self.fit_selected_integration_button.setText(
            "Fit Trace Selected Above"
        )
        self.fit_selected_integration_button.clicked.connect(
            self.run_fit_for_selected_integration
        )
        self.fit_qxy_trace_button = QtWidgets.QToolButton()
        self.fit_qxy_trace_button.setText("Fit Horizontal Trace")
        self.fit_qxy_trace_button.setToolTip(
            f"<qt>Fit the selected {QXY_HTML} 1D trace.</qt>"
        )
        self.fit_qxy_trace_button.clicked.connect(
            self.run_fit_for_selected_qxy_trace
        )
        self.fit_qz_trace_button = QtWidgets.QToolButton()
        self.fit_qz_trace_button.setText("Fit Vertical Trace")
        self.fit_qz_trace_button.setToolTip(
            f"<qt>Fit the selected {QZ_HTML} 1D trace.</qt>"
        )
        self.fit_qz_trace_button.clicked.connect(
            self.run_fit_for_selected_qz_trace
        )
        self.fit_azimuthal_trace_button = QtWidgets.QToolButton()
        self.fit_azimuthal_trace_button.setText("Fit Azimuthal Trace")
        self.fit_azimuthal_trace_button.clicked.connect(
            self.run_fit_for_selected_azimuthal_trace
        )
        self.fit_all_integrations_button = QtWidgets.QToolButton()
        self.fit_all_integrations_button.setText(
            "Fit All Integrated Traces for Selected ROI"
        )
        self.fit_all_integrations_button.clicked.connect(
            self.run_all_integration_fits_for_selected_roi
        )
        self.run_2d_fit_button = QtWidgets.QToolButton()
        self.run_2d_fit_button.setText("Fit Selected ROI in 2D")
        self.run_2d_fit_button.clicked.connect(
            self.run_2d_fit_for_selected_roi
        )
        self.batch_fit_button = QtWidgets.QToolButton()
        self.batch_fit_button.setText("Fit All ROIs Full Workflow")
        self.batch_fit_button.clicked.connect(self.batch_process_all_peak_fits)

        self.fit_status_label = QtWidgets.QLabel("Select a peak with an ROI.")
        self.fit_status_label.setWordWrap(True)
        self.fit_status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.fit_integration_stack = _PeakFitIntegrationStack()
        self.fit_integration_stack.setMinimumHeight(420)
        self.fit_mesh_widget = _PeakFitMeshWidget()
        self.fit_mesh_widget.setMinimumHeight(320)
        self.fit_detail_tree = QtWidgets.QTreeWidget()
        self.fit_detail_tree.setColumnCount(len(FIT_DETAIL_HEADERS))
        self.fit_detail_tree.setHeaderLabels(FIT_DETAIL_HEADERS)
        enable_rich_text_items(self.fit_detail_tree)
        self.fit_detail_tree.setMinimumHeight(180)

    def _build_crystal_overlay_controls(self) -> None:
        self.crystal_system_combo = QtWidgets.QComboBox()
        self.crystal_system_combo.addItems(CRYSTAL_SYSTEMS.keys())

        self.lattice_a = _lattice_spinbox(6.3)
        self.lattice_b = _lattice_spinbox(6.3)
        self.lattice_c = _lattice_spinbox(6.3)
        self.lattice_alpha = _angle_spinbox(90.0)
        self.lattice_beta = _angle_spinbox(90.0)
        self.lattice_gamma = _angle_spinbox(90.0)

        self.h_max = _hkl_spinbox(3)
        self.k_max = _hkl_spinbox(3)
        self.l_max = _hkl_spinbox(3)

        self.positive_qz_check = RichTextCheckBox(f"Positive {QZ_HTML}")
        self.positive_qz_check.setChecked(True)
        self.show_crystal_overlay_check = QtWidgets.QCheckBox("Show overlay")
        self.show_crystal_overlay_check.setChecked(True)
        self.show_crystal_hkl_labels_check = QtWidgets.QCheckBox(
            "Show (hkl) labels"
        )
        self.show_crystal_hkl_labels_check.setChecked(True)
        self.hkl_label_mode_combo = QtWidgets.QComboBox()
        for label, mode in HKL_LABEL_MODE_CHOICES:
            self.hkl_label_mode_combo.addItem(label, mode)
        self.hkl_label_mode_combo.setCurrentIndex(0)

        self.orientation_x_slider = _orientation_slider()
        self.orientation_y_slider = _orientation_slider()
        self.orientation_z_slider = _orientation_slider()
        self.orientation_x_spin = _orientation_spinbox()
        self.orientation_y_spin = _orientation_spinbox()
        self.orientation_z_spin = _orientation_spinbox()

        self.rotation_step = QtWidgets.QDoubleSpinBox()
        self.rotation_step.setRange(0.1, 45.0)
        self.rotation_step.setDecimals(1)
        self.rotation_step.setSingleStep(0.5)
        self.rotation_step.setValue(5.0)
        self.rotation_step.setSuffix(" deg")
        self.rotation_step.setMaximumWidth(92)

        self.rotate_x_neg_button = QtWidgets.QToolButton()
        self.rotate_x_neg_button.setText("X-")
        self.rotate_x_pos_button = QtWidgets.QToolButton()
        self.rotate_x_pos_button.setText("X+")
        self.rotate_y_neg_button = QtWidgets.QToolButton()
        self.rotate_y_neg_button.setText("Y-")
        self.rotate_y_pos_button = QtWidgets.QToolButton()
        self.rotate_y_pos_button.setText("Y+")
        self.rotate_z_neg_button = QtWidgets.QToolButton()
        self.rotate_z_neg_button.setText("Z-")
        self.rotate_z_pos_button = QtWidgets.QToolButton()
        self.rotate_z_pos_button.setText("Z+")
        self.reset_orientation_button = QtWidgets.QToolButton()
        self.reset_orientation_button.setText("Reset")

        self.update_crystal_overlay_button = QtWidgets.QToolButton()
        self.update_crystal_overlay_button.setText("Update Overlay")
        self.update_crystal_overlay_button.clicked.connect(
            self._update_crystal_overlay_now
        )
        self.auto_update_crystal_overlay_button = QtWidgets.QToolButton()
        self.auto_update_crystal_overlay_button.setText("Auto Update")
        self.auto_update_crystal_overlay_button.setCheckable(True)
        self.auto_update_crystal_overlay_button.setChecked(True)
        self.auto_update_crystal_overlay_button.toggled.connect(
            self._handle_crystal_auto_update_toggled
        )

        self.orientation_label = QtWidgets.QLabel("q = [0, 0, 0, 1]")
        self.orientation_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.orientation_label.setWordWrap(True)

        self.crystal_peak_table = QtWidgets.QTableWidget(
            0,
            len(CRYSTAL_PEAK_TABLE_HEADERS),
        )
        set_rich_text_table_headers(
            self.crystal_peak_table,
            CRYSTAL_PEAK_TABLE_HEADERS,
        )
        enable_rich_text_items(self.crystal_peak_table)
        self.crystal_peak_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.crystal_peak_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.crystal_peak_table.setMaximumHeight(170)

        self.crystal_preview_widget = _CrystalOrientationViewer()
        self.crystal_preview_widget.orientationDeltaRequested.connect(
            self._handle_orientation_view_drag
        )

        self.crystal_system_combo.currentTextChanged.connect(
            self._handle_crystal_system_changed
        )
        for spinbox in self._lattice_spinboxes().values():
            spinbox.valueChanged.connect(self._handle_crystal_lattice_changed)
        for spinbox in (self.h_max, self.k_max, self.l_max):
            spinbox.valueChanged.connect(self._schedule_crystal_overlay_update)
        self.positive_qz_check.toggled.connect(
            self._schedule_crystal_overlay_update
        )
        self.show_crystal_overlay_check.toggled.connect(
            self._schedule_crystal_overlay_update
        )
        self.show_crystal_hkl_labels_check.toggled.connect(
            self._handle_hkl_label_check_toggled
        )
        self.hkl_label_mode_combo.currentIndexChanged.connect(
            self._handle_hkl_label_mode_changed
        )
        self._connect_orientation_pair(
            self.orientation_x_slider,
            self.orientation_x_spin,
        )
        self._connect_orientation_pair(
            self.orientation_y_slider,
            self.orientation_y_spin,
        )
        self._connect_orientation_pair(
            self.orientation_z_slider,
            self.orientation_z_spin,
        )
        self.rotate_x_neg_button.clicked.connect(
            lambda: self._rotate_crystal(
                (1.0, 0.0, 0.0), -self.rotation_step.value()
            )
        )
        self.rotate_x_pos_button.clicked.connect(
            lambda: self._rotate_crystal(
                (1.0, 0.0, 0.0), self.rotation_step.value()
            )
        )
        self.rotate_y_neg_button.clicked.connect(
            lambda: self._rotate_crystal(
                (0.0, 1.0, 0.0), -self.rotation_step.value()
            )
        )
        self.rotate_y_pos_button.clicked.connect(
            lambda: self._rotate_crystal(
                (0.0, 1.0, 0.0), self.rotation_step.value()
            )
        )
        self.rotate_z_neg_button.clicked.connect(
            lambda: self._rotate_crystal(
                (0.0, 0.0, 1.0), -self.rotation_step.value()
            )
        )
        self.rotate_z_pos_button.clicked.connect(
            lambda: self._rotate_crystal(
                (0.0, 0.0, 1.0), self.rotation_step.value()
            )
        )
        self.reset_orientation_button.clicked.connect(
            self._reset_crystal_orientation
        )

    def _build_plot(self) -> None:
        if pg is None:
            self.view_box = None
            self.image_item = None
            self.plot_widget = QtWidgets.QLabel("Corrected data")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.peak_scatter = None
            self.active_peak_scatter = None
            self.gap_peak_scatter = None
            self.active_gap_peak_scatter = None
            self.crystal_overlay_scatter = None
            return
        self.view_box = _PeakViewBox(self._handle_plot_click)
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
        self.peak_scatter = _DraggablePeakScatter(
            size=8,
            brush=pg.mkBrush("#d62828"),
            pen=pg.mkPen("#ffffff", width=0.5),
        )
        self.active_peak_scatter = _DraggablePeakScatter(
            size=13,
            brush=pg.mkBrush("#2f80ed"),
            pen=pg.mkPen("#ffffff", width=1.0),
        )
        self.gap_peak_scatter = _DraggablePeakScatter(
            size=9,
            symbol="d",
            brush=pg.mkBrush("#f59e0b"),
            pen=pg.mkPen("#3b2700", width=0.8),
        )
        self.active_gap_peak_scatter = _DraggablePeakScatter(
            size=14,
            symbol="d",
            brush=pg.mkBrush("#facc15"),
            pen=pg.mkPen("#111827", width=1.1),
        )
        self.crystal_overlay_scatter = pg.ScatterPlotItem(
            size=7,
            brush=pg.mkBrush(138, 92, 246, 130),
            pen=pg.mkPen("#2f1847", width=0.7),
        )
        self.peak_scatter.setZValue(12)
        self.active_peak_scatter.setZValue(13)
        self.gap_peak_scatter.setZValue(12.5)
        self.active_gap_peak_scatter.setZValue(13.5)
        self.crystal_overlay_scatter.setZValue(10)
        for scatter in (
            self.peak_scatter,
            self.active_peak_scatter,
            self.gap_peak_scatter,
            self.active_gap_peak_scatter,
        ):
            scatter.peakClicked.connect(self._handle_peak_clicked)
            scatter.peakDragStarted.connect(self._handle_peak_drag_started)
            scatter.peakDragMoved.connect(self._handle_peak_drag_moved)
            scatter.peakDragFinished.connect(self._handle_peak_drag_finished)
            self.plot_widget.addItem(scatter)
        self.plot_widget.addItem(self.crystal_overlay_scatter)

    def _build_table(self) -> None:
        self.peak_table = QtWidgets.QTableWidget(0, len(PEAK_TABLE_HEADERS))
        set_rich_text_table_headers(self.peak_table, PEAK_TABLE_HEADERS)
        enable_rich_text_items(self.peak_table)
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.peak_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.peak_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.peak_table.setMinimumHeight(170)
        self.peak_table.itemSelectionChanged.connect(
            self._handle_table_selection
        )

    def _build_layout(self) -> None:
        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.setMinimumWidth(420)
        self.side_tabs.setMaximumWidth(640)
        self.side_tabs.addTab(self._peak_finder_tab(), "Peak Finder")
        self.side_tabs.addTab(self._roi_selection_tab(), "ROI Selection")
        self.side_tabs.addTab(self._peak_fit_tab(), "Peak Fit")

        contrast_layout = QtWidgets.QHBoxLayout()
        contrast_layout.addWidget(QtWidgets.QLabel("Color"))
        contrast_layout.addWidget(self.colormap_combo)
        contrast_layout.addSpacing(12)
        contrast_layout.addWidget(QtWidgets.QLabel("Min"))
        contrast_layout.addWidget(self.level_min)
        contrast_layout.addWidget(QtWidgets.QLabel("Max"))
        contrast_layout.addWidget(self.level_max)
        contrast_layout.addWidget(self.quantile_check)
        contrast_layout.addWidget(QtWidgets.QLabel("Low"))
        contrast_layout.addWidget(self.quantile_low)
        contrast_layout.addWidget(QtWidgets.QLabel("High"))
        contrast_layout.addWidget(self.quantile_high)
        contrast_layout.addWidget(self.auto_contrast_button)
        contrast_layout.addStretch(1)

        plot_area = self.plot_widget
        if pg is not None and self.image_item is not None:
            self.plot_frame = _ImageAspectPlotFrame(self.plot_widget)
            plot_area = self.plot_frame

        image_layout = QtWidgets.QVBoxLayout()
        image_layout.addLayout(contrast_layout)
        image_layout.addWidget(plot_area, stretch=1)

        plot_layout = QtWidgets.QHBoxLayout()
        plot_layout.addLayout(image_layout, stretch=1)
        plot_layout.addWidget(self.side_tabs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(plot_layout, stretch=1)
        layout.addWidget(self.peak_table)

    def _peak_finder_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.peak_action_bar = QtWidgets.QWidget()
        peak_action_layout = QtWidgets.QHBoxLayout(self.peak_action_bar)
        peak_action_layout.setContentsMargins(0, 0, 0, 4)
        peak_action_layout.setSpacing(4)
        peak_action_layout.addStretch(1)
        peak_action_layout.addWidget(self.undo_button)
        peak_action_layout.addWidget(self.redo_button)
        peak_action_layout.addWidget(self.clear_peaks_button)
        layout.addWidget(self.peak_action_bar)

        self.peak_finder_subtabs = QtWidgets.QTabWidget()
        self.peak_finder_subtabs.setObjectName("PeakFinderSubTabs")
        self.peak_finder_subtabs.addTab(
            self._peak_detection_tab(), "Peak Detection"
        )
        self.peak_finder_subtabs.addTab(
            self._crystal_overlay_tab(), "Crystal Overlay"
        )
        layout.addWidget(self.peak_finder_subtabs, stretch=1)
        return tab

    def _peak_detection_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)

        finder_group = QtWidgets.QGroupBox("Find Peaks")
        finder_form = QtWidgets.QFormLayout()
        finder_form.addRow(
            self._peak_finder_label("Threshold", "threshold"),
            self.threshold_percentile,
        )
        finder_form.addRow(self.adaptive_peak_threshold_check)
        finder_form.addRow(
            self._peak_finder_label("Adaptive floor", "adaptive_floor"),
            self.adaptive_floor_percentile,
        )
        finder_form.addRow(
            self._peak_finder_label("Min SNR", "min_snr"),
            self.min_snr,
        )
        finder_form.addRow(
            self._peak_finder_label("Background px", "background_px"),
            self.background_radius_px,
        )
        finder_form.addRow(
            self._peak_finder_label("Max peaks", "max_peaks"),
            self.max_peaks,
        )
        finder_form.addRow(
            self._peak_finder_label("Distance px", "distance_px"),
            self.min_distance_px,
        )
        finder_form.addRow(
            self._peak_finder_label("Window px", "window_px"),
            self.neighborhood_radius_px,
        )
        finder_form.addRow(
            self._peak_finder_label(f"Min {QZ_HTML}", "min_qz"),
            self.min_qz,
        )
        finder_form.addRow(self.ignore_nonpositive_check)
        finder_form.addRow(self.consolidate_peaks_check)
        finder_layout = QtWidgets.QVBoxLayout(finder_group)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(self.global_peak_preset_button)
        preset_row.addWidget(self.adaptive_peak_preset_button)
        preset_row.addWidget(self.sensitive_peak_preset_button)
        finder_layout.addLayout(preset_row)
        finder_layout.addLayout(finder_form)
        finder_layout.addWidget(self.find_peaks_button)
        finder_layout.addWidget(self.peak_finder_status_label)
        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.addWidget(self.zoom_in_button)
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.zoom_fit_button)
        zoom_row.addWidget(self.pan_button)
        finder_layout.addLayout(zoom_row)
        content_layout.addWidget(finder_group)

        edit_group = QtWidgets.QGroupBox("Manual Peaks")
        edit_layout = QtWidgets.QGridLayout(edit_group)
        edit_layout.addWidget(self.add_point_button, 0, 0)
        edit_layout.addWidget(self.remove_point_button, 0, 1)
        edit_layout.addWidget(self.snap_peak_button, 1, 0)
        edit_layout.addWidget(self.delete_peak_button, 1, 1)
        edit_layout.addWidget(self.snap_all_button, 2, 0, 1, 2)
        edit_layout.addWidget(QtWidgets.QLabel("Snap window px"), 3, 0)
        edit_layout.addWidget(self.snap_window_px, 3, 1)
        edit_layout.addWidget(self.cursor_coordinate_label, 4, 0, 1, 2)
        edit_layout.addWidget(self.snap_feedback_label, 5, 0, 1, 2)
        content_layout.addWidget(edit_group)

        analysis_group = QtWidgets.QGroupBox("Symmetry & Gap Estimates")
        analysis_layout = QtWidgets.QGridLayout(analysis_group)
        analysis_layout.addWidget(self.gap_estimate_button, 0, 0)
        analysis_layout.addWidget(self.tag_gap_button, 0, 1)
        analysis_layout.addWidget(rich_label(f"{QXY_HTML} tol"), 1, 0)
        analysis_layout.addWidget(self.symmetry_qxy_tolerance, 1, 1)
        analysis_layout.addWidget(rich_label(f"{QZ_HTML} tol"), 2, 0)
        analysis_layout.addWidget(self.symmetry_qz_tolerance, 2, 1)
        analysis_layout.addWidget(QtWidgets.QLabel("Mirror source"), 3, 0)
        analysis_layout.addWidget(self.mirror_source_combo, 3, 1)
        analysis_layout.addWidget(self.mirror_missing_button, 4, 0, 1, 2)
        analysis_layout.addWidget(self.symmetry_check_button, 5, 0, 1, 2)
        analysis_layout.addWidget(self.symmetry_summary_label, 6, 0, 1, 2)
        content_layout.addWidget(analysis_group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)
        return tab

    def _roi_selection_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)

        self.roi_tools_group = QtWidgets.QGroupBox("ROI Tools")
        roi_group = self.roi_tools_group
        roi_group.setStyleSheet(
            "QGroupBox { font-weight: 600; border: 1px solid #9ca3af; "
            "border-radius: 4px; margin-top: 8px; padding-top: 8px; }"
        )
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        roi_form = QtWidgets.QFormLayout()
        roi_form.addRow("Shape", self.roi_shape_combo)
        roi_form.addRow(rich_label(f"{QXY_HTML} size"), self.roi_width)
        roi_form.addRow(rich_label(f"{QZ_HTML} size"), self.roi_height)
        roi_layout.addLayout(roi_form)
        roi_buttons = QtWidgets.QHBoxLayout()
        roi_buttons.addWidget(self.apply_roi_button)
        roi_buttons.addWidget(self.apply_all_rois_button)
        roi_layout.addLayout(roi_buttons)
        resize_row = QtWidgets.QHBoxLayout()
        resize_row.addWidget(self.resize_symmetric_button)
        resize_row.addWidget(self.resize_qz_button)
        resize_row.addWidget(self.resize_qxy_button)
        roi_layout.addLayout(resize_row)
        roi_layout.addWidget(self.resize_mode_label)
        roi_layout.addWidget(self.resize_hotkey_note)
        roi_layout.addWidget(self.snap_drag_check)
        roi_layout.addWidget(self.active_roi_mesh_widget)
        content_layout.addWidget(roi_group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)
        return tab

    def _peak_fit_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)

        controls = QtWidgets.QGroupBox("Fit Controls")
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.addWidget(QtWidgets.QLabel("Peak ROI"), 0, 0)
        controls_layout.addWidget(self.fit_peak_combo, 0, 1, 1, 3)
        controls_layout.addWidget(
            self.run_selected_integrations_button,
            1,
            0,
            1,
            2,
        )
        controls_layout.addWidget(self.run_all_integrations_button, 1, 2, 1, 2)
        controls_layout.addWidget(rich_label(f"{QXY_HTML} 1D trace"), 2, 0)
        controls_layout.addWidget(self.fit_qxy_trace_button, 2, 1)
        controls_layout.addWidget(rich_label(f"{QZ_HTML} 1D trace"), 3, 0)
        controls_layout.addWidget(self.fit_qz_trace_button, 3, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Azimuthal trace"), 4, 0)
        controls_layout.addWidget(self.fit_azimuthal_trace_button, 4, 1)
        controls_layout.addWidget(self.fit_all_integrations_button, 5, 0, 1, 4)
        controls_layout.addWidget(self.run_2d_fit_button, 6, 0, 1, 2)
        controls_layout.addWidget(self.batch_fit_button, 6, 2, 1, 2)
        controls_layout.addWidget(self.fit_status_label, 7, 0, 1, 4)
        content_layout.addWidget(controls)

        content_layout.addWidget(self.fit_integration_stack)
        content_layout.addWidget(self.fit_mesh_widget)
        content_layout.addWidget(self.fit_detail_tree)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        self.peak_fit_scroll_area = scroll
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        scroll_row = QtWidgets.QHBoxLayout()
        scroll_row.setContentsMargins(0, 0, 0, 0)
        scroll_row.setSpacing(2)
        scroll_row.addWidget(scroll, stretch=1)
        scroll_row.addWidget(self._peak_fit_scroll_controls())
        layout.addLayout(scroll_row, stretch=1)
        self._sync_peak_fit_scroll_controls()
        return tab

    def _peak_fit_scroll_controls(self) -> QtWidgets.QWidget:
        controls = QtWidgets.QWidget()
        width = max(
            22,
            self.peak_fit_scroll_area.verticalScrollBar().sizeHint().width(),
        )
        controls.setFixedWidth(width)
        layout = QtWidgets.QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.peak_fit_scroll_up_button = QtWidgets.QToolButton()
        self.peak_fit_scroll_up_button.setArrowType(
            QtCore.Qt.ArrowType.UpArrow
        )
        self.peak_fit_scroll_up_button.setAutoRepeat(True)
        self.peak_fit_scroll_up_button.setToolTip(
            "Scroll Peak Fit controls up"
        )
        self.peak_fit_scroll_bar = QtWidgets.QScrollBar(
            QtCore.Qt.Orientation.Vertical
        )
        self.peak_fit_scroll_down_button = QtWidgets.QToolButton()
        self.peak_fit_scroll_down_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
        )
        self.peak_fit_scroll_down_button.setAutoRepeat(True)
        self.peak_fit_scroll_down_button.setToolTip(
            "Scroll Peak Fit controls down"
        )

        for button in (
            self.peak_fit_scroll_up_button,
            self.peak_fit_scroll_down_button,
        ):
            button.setFixedSize(width, width)

        source = self.peak_fit_scroll_area.verticalScrollBar()
        source.rangeChanged.connect(self._sync_peak_fit_scroll_controls)
        source.valueChanged.connect(self._sync_peak_fit_scroll_controls)
        self.peak_fit_scroll_bar.valueChanged.connect(source.setValue)
        self.peak_fit_scroll_up_button.clicked.connect(
            lambda: self._step_peak_fit_scroll(-1)
        )
        self.peak_fit_scroll_down_button.clicked.connect(
            lambda: self._step_peak_fit_scroll(1)
        )

        layout.addWidget(self.peak_fit_scroll_up_button)
        layout.addWidget(self.peak_fit_scroll_bar, stretch=1)
        layout.addWidget(self.peak_fit_scroll_down_button)
        return controls

    def _sync_peak_fit_scroll_controls(self, *_args: Any) -> None:
        scroll = getattr(self, "peak_fit_scroll_area", None)
        scrollbar = getattr(self, "peak_fit_scroll_bar", None)
        if scroll is None or scrollbar is None:
            return
        source = scroll.verticalScrollBar()
        scrollbar.blockSignals(True)
        try:
            scrollbar.setRange(source.minimum(), source.maximum())
            scrollbar.setPageStep(source.pageStep())
            scrollbar.setSingleStep(max(source.singleStep(), 24))
            scrollbar.setValue(source.value())
        finally:
            scrollbar.blockSignals(False)
        enabled = source.maximum() > source.minimum()
        scrollbar.setEnabled(enabled)
        self.peak_fit_scroll_up_button.setEnabled(enabled)
        self.peak_fit_scroll_down_button.setEnabled(enabled)

    def _step_peak_fit_scroll(self, direction: int) -> None:
        scrollbar = getattr(self, "peak_fit_scroll_bar", None)
        if scrollbar is None:
            return
        step = max(scrollbar.singleStep(), 24)
        scrollbar.setValue(scrollbar.value() + int(direction) * step)

    def _crystal_overlay_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(content)

        lattice_group = QtWidgets.QGroupBox("Lattice & Overlay Peaks")
        lattice_form = QtWidgets.QFormLayout(lattice_group)
        lattice_form.addRow("Crystal system", self.crystal_system_combo)
        lattice_form.addRow("a", self.lattice_a)
        lattice_form.addRow("b", self.lattice_b)
        lattice_form.addRow("c", self.lattice_c)
        lattice_form.addRow("alpha", self.lattice_alpha)
        lattice_form.addRow("beta", self.lattice_beta)
        lattice_form.addRow("gamma", self.lattice_gamma)
        lattice_form.addRow("h max", self.h_max)
        lattice_form.addRow("k max", self.k_max)
        lattice_form.addRow("l max", self.l_max)
        lattice_form.addRow(self.positive_qz_check)
        lattice_form.addRow(self.show_crystal_overlay_check)
        lattice_form.addRow("(hkl) labels", self.hkl_label_mode_combo)
        overlay_button_row = QtWidgets.QWidget()
        overlay_button_layout = QtWidgets.QHBoxLayout(overlay_button_row)
        overlay_button_layout.setContentsMargins(0, 0, 0, 0)
        overlay_button_layout.addWidget(
            self.auto_update_crystal_overlay_button
        )
        overlay_button_layout.addWidget(self.update_crystal_overlay_button)
        lattice_form.addRow(overlay_button_row)
        form_layout.addWidget(lattice_group)

        orientation_group = QtWidgets.QGroupBox("Crystal Orientation")
        orientation_layout = QtWidgets.QHBoxLayout(orientation_group)
        orientation_controls = QtWidgets.QWidget()
        orientation_controls.setObjectName("CrystalOrientationControls")
        orientation_controls_layout = QtWidgets.QVBoxLayout(
            orientation_controls
        )
        orientation_controls_layout.setContentsMargins(0, 0, 0, 0)
        slider_form = QtWidgets.QFormLayout()
        slider_form.addRow(
            "X",
            _orientation_control_row(
                self.orientation_x_slider,
                self.orientation_x_spin,
            ),
        )
        slider_form.addRow(
            "Y",
            _orientation_control_row(
                self.orientation_y_slider,
                self.orientation_y_spin,
            ),
        )
        slider_form.addRow(
            "Z",
            _orientation_control_row(
                self.orientation_z_slider,
                self.orientation_z_spin,
            ),
        )
        orientation_controls_layout.addLayout(slider_form)
        step_layout = QtWidgets.QHBoxLayout()
        step_layout.addWidget(QtWidgets.QLabel("Step"))
        step_layout.addWidget(self.rotation_step)
        step_layout.addStretch(1)
        orientation_controls_layout.addLayout(step_layout)
        rotate_grid = QtWidgets.QGridLayout()
        rotate_grid.addWidget(self.rotate_x_neg_button, 0, 0)
        rotate_grid.addWidget(self.rotate_x_pos_button, 0, 1)
        rotate_grid.addWidget(self.rotate_y_neg_button, 1, 0)
        rotate_grid.addWidget(self.rotate_y_pos_button, 1, 1)
        rotate_grid.addWidget(self.rotate_z_neg_button, 2, 0)
        rotate_grid.addWidget(self.rotate_z_pos_button, 2, 1)
        rotate_grid.addWidget(self.reset_orientation_button, 3, 0, 1, 2)
        orientation_controls_layout.addLayout(rotate_grid)
        orientation_controls_layout.addWidget(self.orientation_label)
        orientation_controls_layout.addStretch(1)
        orientation_layout.addWidget(orientation_controls)
        orientation_layout.addWidget(self.crystal_preview_widget, stretch=1)
        form_layout.addWidget(orientation_group)

        form_layout.addWidget(self.crystal_peak_table)
        form_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return tab

    def _set_initial_image(self) -> None:
        if self.image_data is None:
            return
        if self.image_item is not None:
            self.image_item.setImage(self.image_data, autoLevels=False)
            image_rect = data_image_rect(
                self.image_data.shape,
                self.axis_ranges,
            )
            self.image_item.setRect(image_rect)
            if self.plot_frame is not None:
                self.plot_frame.set_data_rect(image_rect)
            self.apply_image_style(self.image_style)
        if pg is not None and self.plot_widget is not None:
            set_data_image_plot_range(
                self.plot_widget,
                self.image_data.shape,
                self.axis_ranges,
            )

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
            )
            return
        height, width = self.image_data.shape[-2:]
        self.plot_widget.setRange(xRange=(0, width), yRange=(0, height))

    def _set_pan_mode(self, enabled: bool) -> None:
        if self.view_box is not None:
            self.view_box.pan_enabled = enabled
        self.plot_widget.setCursor(
            QtCore.Qt.CursorShape.OpenHandCursor
            if enabled
            else QtCore.Qt.CursorShape.CrossCursor
        )

    def _apply_manual_levels(self) -> None:
        if not self.quantile_check.isChecked():
            self._apply_image_style_from_controls()

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
        self._apply_image_style_from_controls()

    def _apply_image_style_from_controls(self) -> None:
        self.apply_image_style(
            self.image_display_style(),
            sync_controls=False,
        )

    def apply_image_style(
        self,
        style: ImageDisplayStyle,
        *,
        sync_controls: bool = True,
    ) -> None:
        self.image_style = style
        if sync_controls:
            self._set_image_style_controls(style)
        apply_image_display_style(
            self.image_item,
            self.image_data,
            self.image_style,
        )

    def _set_image_style_controls(self, style: ImageDisplayStyle) -> None:
        controls = (
            self.colormap_combo,
            self.level_min,
            self.level_max,
            self.quantile_check,
            self.quantile_low,
            self.quantile_high,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            index = self.colormap_combo.findData(style.colormap)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)
            self.level_min.setValue(style.level_min)
            self.level_max.setValue(style.level_max)
            self.quantile_check.setChecked(style.use_quantile)
            self.quantile_low.setValue(style.quantile_low)
            self.quantile_high.setValue(style.quantile_high)
        finally:
            for control in controls:
                control.blockSignals(False)

    def _restore_crystal_overlay_state(self) -> bool:
        state = self.project.analysis_results.get("crystal_overlays", {}).get(
            self.data_id, {}
        )
        params = CrystalOverlayParameters.from_dict(
            state.get("parameters", state)
        ).constrained()
        controls = [
            self.crystal_system_combo,
            self.lattice_a,
            self.lattice_b,
            self.lattice_c,
            self.lattice_alpha,
            self.lattice_beta,
            self.lattice_gamma,
            self.h_max,
            self.k_max,
            self.l_max,
            self.positive_qz_check,
            self.show_crystal_overlay_check,
            self.show_crystal_hkl_labels_check,
            self.hkl_label_mode_combo,
            self.auto_update_crystal_overlay_button,
        ]
        for control in controls:
            control.blockSignals(True)
        try:
            self.crystal_system_combo.setCurrentText(params.crystal_system)
            self.lattice_a.setValue(params.a)
            self.lattice_b.setValue(params.b)
            self.lattice_c.setValue(params.c)
            self.lattice_alpha.setValue(params.alpha)
            self.lattice_beta.setValue(params.beta)
            self.lattice_gamma.setValue(params.gamma)
            self.h_max.setValue(params.h_max)
            self.k_max.setValue(params.k_max)
            self.l_max.setValue(params.l_max)
            self.positive_qz_check.setChecked(params.positive_qz_only)
            self.show_crystal_overlay_check.setChecked(
                bool(state.get("show_overlay", True))
            )
            mode = state.get("hkl_label_mode")
            if mode is None:
                mode = (
                    HKL_LABEL_MODE_PARTIAL
                    if bool(state.get("show_hkl_labels", True))
                    else HKL_LABEL_MODE_NONE
                )
            self._set_hkl_label_mode(str(mode))
            self.auto_update_crystal_overlay_button.setChecked(
                bool(state.get("auto_update_overlay", True))
            )
        finally:
            for control in controls:
                control.blockSignals(False)
        angles = _orientation_angles_from_state(
            state.get("orientation_angles_deg"),
            params.orientation_quaternion,
        )
        self._set_crystal_orientation_angles(*angles, schedule_update=False)
        self._apply_crystal_constraints_to_controls()
        self._update_orientation_label()
        return bool(state)

    def restore_crystal_overlay_from_project(self) -> None:
        """Reload Crystal Overlay controls from project state and
        redraw."""

        self._restore_crystal_overlay_state()
        self._update_crystal_overlay_now()

    def _lattice_spinboxes(
        self,
    ) -> dict[str, QtWidgets.QDoubleSpinBox]:
        return {
            "a": self.lattice_a,
            "b": self.lattice_b,
            "c": self.lattice_c,
            "alpha": self.lattice_alpha,
            "beta": self.lattice_beta,
            "gamma": self.lattice_gamma,
        }

    def _crystal_parameters(self) -> CrystalOverlayParameters:
        return CrystalOverlayParameters(
            crystal_system=self.crystal_system_combo.currentText(),
            a=self.lattice_a.value(),
            b=self.lattice_b.value(),
            c=self.lattice_c.value(),
            alpha=self.lattice_alpha.value(),
            beta=self.lattice_beta.value(),
            gamma=self.lattice_gamma.value(),
            h_max=self.h_max.value(),
            k_max=self.k_max.value(),
            l_max=self.l_max.value(),
            orientation_quaternion=self.crystal_quaternion,
            positive_qz_only=self.positive_qz_check.isChecked(),
        ).constrained()

    def _handle_crystal_system_changed(self, *_args: Any) -> None:
        self._apply_crystal_constraints_to_controls()
        self._update_crystal_overlay_now()

    def _handle_crystal_lattice_changed(self, *_args: Any) -> None:
        self._apply_crystal_constraints_to_controls()
        self._update_crystal_overlay_now()

    def _handle_hkl_label_check_toggled(self, checked: bool) -> None:
        if checked and self._hkl_label_mode() == HKL_LABEL_MODE_NONE:
            self._set_hkl_label_mode(HKL_LABEL_MODE_PARTIAL)
        elif not checked:
            self._set_hkl_label_mode(HKL_LABEL_MODE_NONE)
        self._schedule_crystal_overlay_update()

    def _handle_hkl_label_mode_changed(self, *_args: Any) -> None:
        self.show_crystal_hkl_labels_check.blockSignals(True)
        try:
            self.show_crystal_hkl_labels_check.setChecked(
                self._hkl_label_mode() != HKL_LABEL_MODE_NONE
            )
        finally:
            self.show_crystal_hkl_labels_check.blockSignals(False)
        self._schedule_crystal_overlay_update()

    def _hkl_label_mode(self) -> str:
        mode = self.hkl_label_mode_combo.currentData()
        if mode in {
            HKL_LABEL_MODE_ALL,
            HKL_LABEL_MODE_PARTIAL,
            HKL_LABEL_MODE_NONE,
        }:
            return str(mode)
        return HKL_LABEL_MODE_PARTIAL

    def _set_hkl_label_mode(self, mode: str) -> None:
        normalized = (
            mode
            if mode
            in {
                HKL_LABEL_MODE_ALL,
                HKL_LABEL_MODE_PARTIAL,
                HKL_LABEL_MODE_NONE,
            }
            else HKL_LABEL_MODE_PARTIAL
        )
        combo_index = self.hkl_label_mode_combo.findData(normalized)
        if combo_index >= 0:
            self.hkl_label_mode_combo.setCurrentIndex(combo_index)
        self.show_crystal_hkl_labels_check.setChecked(
            normalized != HKL_LABEL_MODE_NONE
        )

    def _apply_crystal_constraints_to_controls(self) -> None:
        values = {
            "crystal_system": self.crystal_system_combo.currentText(),
            "a": self.lattice_a.value(),
            "b": self.lattice_b.value(),
            "c": self.lattice_c.value(),
            "alpha": self.lattice_alpha.value(),
            "beta": self.lattice_beta.value(),
            "gamma": self.lattice_gamma.value(),
        }
        apply_crystal_system_constraints(values)
        disabled = set(
            CRYSTAL_SYSTEMS.get(
                self.crystal_system_combo.currentText(),
                CRYSTAL_SYSTEMS["Triclinic"],
            ).get("disabled", [])
        )
        for name, spinbox in self._lattice_spinboxes().items():
            spinbox.blockSignals(True)
            try:
                spinbox.setEnabled(name not in disabled)
                spinbox.setValue(float(values[name]))
            finally:
                spinbox.blockSignals(False)

    def _connect_orientation_pair(
        self,
        slider: QtWidgets.QSlider,
        spinbox: QtWidgets.QDoubleSpinBox,
    ) -> None:
        slider.valueChanged.connect(
            lambda value, control=spinbox: _set_spinbox_blocked(
                control,
                _slider_value_to_degrees(value),
            )
        )
        slider.valueChanged.connect(self._handle_orientation_slider_changed)
        spinbox.valueChanged.connect(
            lambda value, control=slider: _set_slider_blocked(control, value)
        )
        spinbox.valueChanged.connect(self._handle_orientation_slider_changed)

    def _orientation_controls(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self.orientation_x_slider,
            self.orientation_y_slider,
            self.orientation_z_slider,
            self.orientation_x_spin,
            self.orientation_y_spin,
            self.orientation_z_spin,
        )

    def _set_orientation_control_values(
        self,
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
    ) -> None:
        controls = self._orientation_controls()
        for control in controls:
            control.blockSignals(True)
        try:
            for slider, spinbox, value in (
                (
                    self.orientation_x_slider,
                    self.orientation_x_spin,
                    x_degrees,
                ),
                (
                    self.orientation_y_slider,
                    self.orientation_y_spin,
                    y_degrees,
                ),
                (
                    self.orientation_z_slider,
                    self.orientation_z_spin,
                    z_degrees,
                ),
            ):
                slider.setValue(_degrees_to_slider_value(value))
                spinbox.setValue(_clamp_orientation_angle(value))
        finally:
            for control in controls:
                control.blockSignals(False)

    def _handle_orientation_slider_changed(self, *_args: Any) -> None:
        self._set_crystal_orientation_angles(
            _slider_value_to_degrees(self.orientation_x_slider.value()),
            _slider_value_to_degrees(self.orientation_y_slider.value()),
            _slider_value_to_degrees(self.orientation_z_slider.value()),
        )

    def _set_crystal_orientation_angles(
        self,
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
        *,
        schedule_update: bool = True,
    ) -> None:
        angles = tuple(
            _clamp_orientation_angle(value)
            for value in (x_degrees, y_degrees, z_degrees)
        )
        self.crystal_orientation_angles = angles
        self.crystal_quaternion = quaternion_from_euler_angles(*angles)
        self._set_orientation_control_values(*angles)
        self._update_orientation_label()
        if schedule_update:
            self._schedule_crystal_overlay_update()

    def _handle_orientation_view_drag(
        self,
        delta_x: float,
        delta_y: float,
    ) -> None:
        x_degrees, y_degrees, z_degrees = self.crystal_orientation_angles
        self._set_crystal_orientation_angles(
            x_degrees - delta_y * 0.35,
            y_degrees + delta_x * 0.35,
            z_degrees,
        )

    def _rotate_crystal(
        self,
        axis: tuple[float, float, float],
        angle_degrees: float,
    ) -> None:
        x_degrees, y_degrees, z_degrees = self.crystal_orientation_angles
        if np.allclose(axis, (1.0, 0.0, 0.0)):
            x_degrees += angle_degrees
        elif np.allclose(axis, (0.0, 1.0, 0.0)):
            y_degrees += angle_degrees
        elif np.allclose(axis, (0.0, 0.0, 1.0)):
            z_degrees += angle_degrees
        else:
            delta = quaternion_from_axis_angle(axis, angle_degrees)
            quaternion = compose_quaternions(self.crystal_quaternion, delta)
            x_degrees, y_degrees, z_degrees = euler_angles_from_quaternion(
                quaternion
            )
        self._set_crystal_orientation_angles(
            x_degrees,
            y_degrees,
            z_degrees,
        )

    def _reset_crystal_orientation(self) -> None:
        self._set_crystal_orientation_angles(0.0, 0.0, 0.0)

    def _handle_crystal_auto_update_toggled(self, checked: bool) -> None:
        if checked:
            self._schedule_crystal_overlay_update()
            return
        self.crystal_update_timer.stop()

    def _schedule_crystal_overlay_update(self, *_args: Any) -> None:
        if not self.auto_update_crystal_overlay_button.isChecked():
            return
        self.crystal_update_timer.start()

    def _update_crystal_overlay_now(self) -> None:
        self.crystal_update_timer.stop()
        self._update_crystal_overlay()

    def _update_orientation_label(self) -> None:
        x, y, z, w = self.crystal_quaternion
        x_deg, y_deg, z_deg = self.crystal_orientation_angles
        self.orientation_label.setText(
            f"X {x_deg:.1f} deg, Y {y_deg:.1f} deg, Z {z_deg:.1f} deg\n"
            f"q = [{x:.4f}, {y:.4f}, {z:.4f}, {w:.4f}]"
        )

    def _update_crystal_overlay(self) -> None:
        params = self._crystal_parameters()
        result = self.crystal_calculator.project(params)
        self._clear_crystal_overlay_graphics()
        if pg is not None and self.crystal_overlay_scatter is not None:
            if self.show_crystal_overlay_check.isChecked():
                self.crystal_overlay_scatter.setData(
                    spots=_crystal_overlay_spots(result),
                    hoverable=True,
                    tip=_crystal_hkl_tip,
                )
                mode = self._hkl_label_mode()
                if mode != HKL_LABEL_MODE_NONE:
                    self._add_crystal_hkl_labels(result, mode=mode)
            else:
                self.crystal_overlay_scatter.setData(spots=[])
        self._update_crystal_preview(result)
        self._sync_crystal_peak_table(result)
        self._store_crystal_overlay_state(params, result)

    def _clear_crystal_overlay_graphics(self) -> None:
        if not hasattr(self.plot_widget, "removeItem"):
            self.crystal_overlay_graphics.clear()
            return
        for graphic in self.crystal_overlay_graphics:
            self.plot_widget.removeItem(graphic)
        self.crystal_overlay_graphics.clear()

    def _add_crystal_hkl_labels(self, result: Any, *, mode: str) -> None:
        if pg is None or not hasattr(self.plot_widget, "addItem"):
            return
        indices = _crystal_hkl_label_indices(result, mode)
        if not indices:
            return
        border = pg.mkPen(47, 24, 71, 120, width=0.4)
        fill = pg.mkBrush(255, 255, 255, 180)
        color = QtGui.QColor("#2f1847")
        for index in indices:
            qxy = result.qxy[index]
            qz = result.qz[index]
            hkl = result.hkl[index]
            label = pg.TextItem(
                _format_hkl_label(hkl),
                color=color,
                anchor=(0.0, 1.0),
                border=border,
                fill=fill,
            )
            label.setPos(float(qxy), float(qz))
            label.setZValue(10.5)
            self.plot_widget.addItem(label)
            self.crystal_overlay_graphics.append(label)

    def _update_crystal_preview(self, result: Any) -> None:
        if hasattr(self.crystal_preview_widget, "set_result"):
            self.crystal_preview_widget.set_result(result)
            return
        for item in self.crystal_edge_graphics:
            self.crystal_preview_widget.removeItem(item)
        self.crystal_edge_graphics.clear()
        corners = np.asarray(result.cell_corners, dtype=float)
        if corners.size == 0:
            return
        screen = _project_cell_preview(corners)
        for start, end in result.cell_edges:
            xs = [screen[start, 0], screen[end, 0]]
            zs = [screen[start, 1], screen[end, 1]]
            depth = float((corners[start, 1] + corners[end, 1]) / 2.0)
            alpha = int(np.clip(160 + 55 * np.tanh(depth / 8.0), 95, 235))
            item = pg.PlotDataItem(
                xs,
                zs,
                pen=pg.mkPen(QtGui.QColor(31, 122, 140, alpha), width=1.6),
                symbol="o",
                symbolSize=4,
                symbolBrush="#bfdbf7",
            )
            self.crystal_preview_widget.addItem(item)
            self.crystal_edge_graphics.append(item)
        origin = _project_cell_preview(np.array([[0.0, 0.0, 0.0]]))[0]
        self._add_crystal_beam_graphics(result, origin, screen)
        self.crystal_preview_widget.enableAutoRange()

    def _add_crystal_beam_graphics(
        self,
        result: Any,
        origin: np.ndarray,
        screen: np.ndarray,
    ) -> None:
        span = max(
            float(np.ptp(screen[:, 0])),
            float(np.ptp(screen[:, 1])),
            1.0,
        )
        direct_direction = _preview_unit_vector(DIRECT_BEAM_VECTOR)
        direct_tail = origin - direct_direction * span * 0.55
        direct_tip = origin + direct_direction * span * 0.65
        self._add_labeled_beam(
            direct_tail,
            direct_tip,
            color="#d62828",
            label="Direct beam",
            label_offset=direct_direction * span * 0.05,
        )

        scattered_direction = _scattered_beam_preview_direction(
            result.q_vectors
        )
        scattered_tip = origin + scattered_direction * span * 0.72
        self._add_labeled_beam(
            origin,
            scattered_tip,
            color="#f77f00",
            label="Scattered beam",
            label_offset=scattered_direction * span * 0.05,
        )

    def _add_labeled_beam(
        self,
        tail: np.ndarray,
        tip: np.ndarray,
        *,
        color: str,
        label: str,
        label_offset: np.ndarray,
    ) -> None:
        pen = pg.mkPen(color, width=2.2)
        line = pg.PlotDataItem(
            [float(tail[0]), float(tip[0])],
            [float(tail[1]), float(tip[1])],
            pen=pen,
        )
        line.setZValue(6)
        self.crystal_preview_widget.addItem(line)
        self.crystal_edge_graphics.append(line)

        head_points = _beam_arrowhead_points(tail, tip)
        head = pg.PlotDataItem(
            [float(point[0]) for point in head_points],
            [float(point[1]) for point in head_points],
            pen=pen,
        )
        head.setZValue(7)
        self.crystal_preview_widget.addItem(head)
        self.crystal_edge_graphics.append(head)

        label_item = pg.TextItem(
            label,
            color=QtGui.QColor(color),
            anchor=(0.0, 0.5),
        )
        label_item.setPos(
            float(tip[0] + label_offset[0]),
            float(tip[1] + label_offset[1]),
        )
        label_item.setZValue(8)
        self.crystal_preview_widget.addItem(label_item)
        self.crystal_edge_graphics.append(label_item)

    def _sync_crystal_peak_table(self, result: Any) -> None:
        rows = result.peak_rows(limit=250)
        self.crystal_peak_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["h"],
                row["k"],
                row["l"],
                _format_float(row["qxy"]),
                _format_float(row["qz"]),
            ]
            for column, value in enumerate(values):
                self.crystal_peak_table.setItem(
                    row_index,
                    column,
                    QtWidgets.QTableWidgetItem(str(value)),
                )
        self.crystal_peak_table.resizeColumnsToContents()

    def _store_crystal_overlay_state(
        self,
        params: CrystalOverlayParameters,
        result: Any,
    ) -> None:
        overlays = self.project.analysis_results.setdefault(
            "crystal_overlays",
            {},
        )
        overlays[self.data_id] = {
            "parameters": params.as_dict(),
            "show_overlay": self.show_crystal_overlay_check.isChecked(),
            "show_hkl_labels": (self._hkl_label_mode() != HKL_LABEL_MODE_NONE),
            "hkl_label_mode": self._hkl_label_mode(),
            "auto_update_overlay": (
                self.auto_update_crystal_overlay_button.isChecked()
            ),
            "orientation_angles_deg": [
                float(value) for value in self.crystal_orientation_angles
            ],
            "peak_count": int(len(result.hkl)),
            "preview_peaks": result.peak_rows(limit=500),
        }

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

    def _sync_after_peak_change(self) -> None:
        self._sync_table()
        self._sync_fit_peak_combo()
        self._refresh_fit_view()
        self._refresh_peak_graphics()
        self._refresh_active_roi_preview()
        self._sync_history_buttons()
        self.peakSetChanged.emit(self.data_id)

    def _sync_after_fit_change(
        self,
        record: dict[str, Any] | None,
    ) -> None:
        if record is not None:
            self._update_record_fit_summary(record)
            store = self._fit_record_for_peak(record, create=False)
            sync_structure_peak_from_fit(
                self.project,
                self.data_id,
                record,
                store,
            )
        self._sync_table()
        self._sync_fit_peak_combo()
        self._refresh_fit_view()
        self._refresh_active_roi_preview()
        self.peakSetChanged.emit(self.data_id)

    def _sync_table(self) -> None:
        records = self.peaks()
        self.peak_table.setRowCount(len(records))
        selected_row = 0
        for row, record in enumerate(records):
            peak_id = _peak_id(record)
            if peak_id == self.active_peak_id:
                selected_row = row
            summary = self._fit_summary(record)
            values = [
                record.get("label", peak_id),
                record.get("source", ""),
                _format_float(_peak_qxy(record)),
                _format_float(_peak_qz(record)),
                _format_float(record.get("intensity")),
                _format_float(record.get("roi", {}).get("qxy_min")),
                _format_float(record.get("roi", {}).get("qxy_max")),
                _format_float(record.get("roi", {}).get("qz_min")),
                _format_float(record.get("roi", {}).get("qz_max")),
                summary["integrations"],
                summary["one_d_fits"],
                summary["two_d_fit"],
                _format_float(summary["center_qxy"]),
                _format_float(summary["center_qz"]),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, peak_id)
                self.peak_table.setItem(row, column, item)
        self.peak_table.resizeColumnsToContents()
        if records and self.active_peak_id is not None:
            self.peak_table.selectRow(selected_row)

    def _sync_fit_peak_combo(self) -> None:
        current_peak_id = self.active_peak_id
        self.fit_peak_combo.blockSignals(True)
        try:
            self.fit_peak_combo.clear()
            current_index = -1
            for index, record in enumerate(self.peaks()):
                peak_id = _peak_id(record)
                suffix = "ROI" if record.get("roi") else "no ROI"
                label = self._fit_peak_combo_label(record, suffix)
                self.fit_peak_combo.addItem(label, peak_id)
                if peak_id == current_peak_id:
                    current_index = index
            if current_index >= 0:
                self.fit_peak_combo.setCurrentIndex(current_index)
        finally:
            self.fit_peak_combo.blockSignals(False)

    def _handle_fit_peak_combo_changed(self) -> None:
        peak_id = self.fit_peak_combo.currentData()
        if peak_id is None:
            return
        self.active_peak_id = str(peak_id)
        self._sync_table()
        self._refresh_peak_graphics()
        self._refresh_fit_view()

    def _fit_peak_combo_label(
        self,
        record: dict[str, Any],
        suffix: str,
    ) -> str:
        peak_id = _peak_id(record)
        summary = self._fit_summary(record)
        qxy = summary.get("center_qxy")
        qz = summary.get("center_qz")
        if qxy is None:
            qxy = _peak_qxy(record)
        if qz is None:
            qz = _peak_qz(record)
        parts = [
            f"{record.get('label', peak_id)} ({suffix})",
            f"{QXY_HTML} = {_format_float(qxy)} {QSPACE_UNITS_HTML}",
            f"{QZ_HTML} = {_format_float(qz)} {QSPACE_UNITS_HTML}",
        ]
        source = str(record.get("source", ""))
        status = str(record.get("status", ""))
        if "gap" in source.lower() or "gap" in status.lower():
            parts.append("gap-estimated")
        hkl = record.get("hkl_label") or record.get("hkl")
        if hkl:
            parts.append(_format_hkl_label_from_record(hkl))
        phase = str(record.get("phase_tag", ""))
        if phase and phase != DEFAULT_PHASE_TAG:
            parts.append(phase)
        return " | ".join(parts)

    def _refresh_fit_view(self) -> None:
        record = self._active_record()
        store = self._fit_record_for_peak(record, create=False)
        integrations = dict(store.get("integrations", {})) if store else {}
        fits = dict(store.get("integration_fits", {})) if store else {}
        failures = dict(store.get("fit_failures", {})) if store else {}
        self.fit_integration_stack.set_profiles(integrations, fits, failures)
        mesh = None
        fit_2d = None
        if record is not None and record.get("roi") and store:
            fit_2d = store.get("fit_2d")
            if fit_2d:
                mesh = evaluate_peak_fit_2d(
                    self.image_data,
                    self.axis_ranges,
                    record["roi"],
                    fit_2d,
                )
        self.fit_mesh_widget.set_mesh(
            mesh, fit_2d if mesh is not None else None
        )
        self._sync_fit_detail_tree(record, store)
        self._set_fit_buttons_enabled(
            record is not None and bool(record.get("roi"))
        )

    def _refresh_active_roi_preview(self) -> None:
        if not hasattr(self, "active_roi_mesh_widget"):
            return
        record = self._active_record()
        if record is None or not record.get("roi"):
            self.active_roi_mesh_widget.set_mesh(None)
            return
        store = self._fit_record_for_peak(record, create=False)
        fit_2d = store.get("fit_2d") if store else None
        if fit_2d:
            mesh = evaluate_peak_fit_2d(
                self.image_data,
                self.axis_ranges,
                record["roi"],
                fit_2d,
            )
            self.active_roi_mesh_widget.set_mesh(mesh, fit_2d)
            return
        sliced = slice_peak_roi(
            self.image_data, self.axis_ranges, record["roi"]
        )
        if sliced is None:
            self.active_roi_mesh_widget.set_mesh(None)
            return
        qxy_grid, qz_grid, intensity = sliced.as_mesh()
        self.active_roi_mesh_widget.set_mesh(
            (qxy_grid, qz_grid, intensity, None),
            None,
        )

    def _refresh_peak_graphics(self) -> None:
        if pg is None or self.plot_widget is None:
            return
        for graphic in self.roi_graphics:
            self.plot_widget.removeItem(graphic)
        self.roi_graphics.clear()
        measured_spots: list[dict[str, Any]] = []
        active_measured_spots: list[dict[str, Any]] = []
        gap_spots: list[dict[str, Any]] = []
        active_gap_spots: list[dict[str, Any]] = []
        for record in self.peaks():
            peak_id = _peak_id(record)
            qxy = _peak_qxy(record)
            qz = _peak_qz(record)
            spot = {
                "pos": (qxy, qz),
                "data": peak_id,
            }
            is_gap = _is_gap_estimated_peak(record)
            if peak_id == self.active_peak_id:
                if is_gap:
                    active_gap_spots.append(spot)
                else:
                    active_measured_spots.append(spot)
            else:
                if is_gap:
                    gap_spots.append(spot)
                else:
                    measured_spots.append(spot)
            roi = record.get("roi")
            if roi:
                self._add_roi_graphic(
                    roi,
                    active=peak_id == self.active_peak_id,
                    peak_id=peak_id,
                )
        self.peak_scatter.setData(spots=measured_spots)
        self.active_peak_scatter.setData(spots=active_measured_spots)
        self.gap_peak_scatter.setData(spots=gap_spots)
        self.active_gap_peak_scatter.setData(spots=active_gap_spots)

    def _handle_peak_drag_started(self, peak_id: str) -> None:
        self.active_peak_id = peak_id
        self._push_undo_state()
        self._sync_table()
        self._sync_fit_peak_combo()

    def _handle_peak_drag_moved(
        self,
        peak_id: str,
        qxy: float,
        qz: float,
        off_plot: bool = False,
    ) -> None:
        if off_plot:
            return
        record = self._record_by_id(peak_id)
        if record is None:
            return
        record["qxy"] = float(qxy)
        record["qz"] = float(qz)
        record["intensity"] = self._intensity_at(float(qxy), float(qz))
        record["source"] = "manual-drag"
        if record.get("roi"):
            self._apply_roi_to_record(record)
        self._sync_table()
        self._refresh_fit_view()

    def _handle_peak_drag_finished(
        self,
        peak_id: str,
        qxy: float,
        qz: float,
        off_plot: bool = False,
    ) -> None:
        if off_plot:
            self._sync_after_peak_change()
            return
        record = self._record_by_id(peak_id)
        if record is None:
            return
        if self.snap_drag_check.isChecked():
            target = self._snap_target_near(float(qxy), float(qz))
            if target is not None:
                self._apply_peak_target_to_record(record, target)
        self._sync_after_peak_change()

    def _handle_peak_clicked(self, peak_id: str) -> None:
        if self.remove_point_button.isChecked():
            self.remove_peak(peak_id)
            return
        self.active_peak_id = peak_id
        self._sync_table()
        self._sync_fit_peak_combo()
        self._refresh_peak_graphics()
        self._refresh_fit_view()
        self._refresh_active_roi_preview()

    def _record_by_id(self, peak_id: str) -> dict[str, Any] | None:
        for record in self.peaks():
            if _peak_id(record) == peak_id:
                return record
        return None

    def _add_roi_graphic(
        self,
        roi: dict[str, Any],
        *,
        active: bool,
        peak_id: str,
    ) -> None:
        values = (
            roi.get("qxy_min"),
            roi.get("qxy_max"),
            roi.get("qz_min"),
            roi.get("qz_max"),
        )
        if any(value is None for value in values):
            return
        qxy_min, qxy_max, qz_min, qz_max = (float(value) for value in values)
        pen = pg.mkPen("#2f80ed" if active else "#d62828", width=1.4)
        graphic = pg.RectROI(
            (qxy_min, qz_min),
            (qxy_max - qxy_min, qz_max - qz_min),
            pen=pen,
            movable=active,
            resizable=active,
            rotatable=False,
            sideScalers=active,
        )
        graphic.setZValue(11)
        if active:
            graphic.sigRegionChangeStarted.connect(
                lambda *_args, peak_id=peak_id: (
                    self._begin_peak_roi_change(peak_id)
                )
            )
            graphic.sigRegionChangeFinished.connect(
                lambda *_args, peak_id=peak_id, graphic=graphic: (
                    self._handle_peak_roi_graphic_changed(peak_id, graphic)
                )
            )
        self.plot_widget.addItem(graphic)
        self.roi_graphics.append(graphic)

    def _begin_peak_roi_change(self, peak_id: str) -> None:
        self._push_undo_state()
        record = self._record_by_id(peak_id)
        if record is not None and isinstance(record.get("roi"), dict):
            self._roi_drag_originals[peak_id] = dict(record["roi"])

    def _handle_peak_roi_graphic_changed(
        self,
        peak_id: str,
        graphic: Any,
    ) -> None:
        record = self._record_by_id(peak_id)
        if record is None or not isinstance(record.get("roi"), dict):
            return
        position = graphic.pos()
        size = graphic.size()
        x0 = float(position.x())
        y0 = float(position.y())
        width = float(size.x())
        height = float(size.y())
        qxy_min, qxy_max = sorted((x0, x0 + width))
        qz_min, qz_max = sorted((y0, y0 + height))
        original = self._roi_drag_originals.pop(peak_id, {})
        mode = self._held_roi_resize_mode or self.selected_roi_resize_mode
        if mode == ROI_RESIZE_QZ and original:
            qxy_min = float(original.get("qxy_min", qxy_min))
            qxy_max = float(original.get("qxy_max", qxy_max))
        elif mode == ROI_RESIZE_QXY and original:
            qz_min = float(original.get("qz_min", qz_min))
            qz_max = float(original.get("qz_max", qz_max))
        roi = record["roi"]
        roi.update(
            {
                "qxy_min": qxy_min,
                "qxy_max": qxy_max,
                "qz_min": qz_min,
                "qz_max": qz_max,
                "center_qxy": (qxy_min + qxy_max) / 2.0,
                "center_qz": (qz_min + qz_max) / 2.0,
                "width": max(qxy_max - qxy_min, 1.0e-12),
                "height": max(qz_max - qz_min, 1.0e-12),
            }
        )
        self._sync_after_peak_change()

    def _handle_plot_click(self, qxy: float, qz: float) -> None:
        if self.remove_point_button.isChecked():
            self.remove_peak(self._nearest_peak_id(qxy, qz))
            return
        self.add_peak_at(qxy, qz, source="manual")

    def _handle_table_selection(self) -> None:
        row = self.peak_table.currentRow()
        if row < 0:
            return
        item = self.peak_table.item(row, 0)
        if item is None:
            return
        self.active_peak_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._refresh_peak_graphics()
        self._sync_fit_peak_combo()
        self._refresh_fit_view()
        self._refresh_active_roi_preview()

    def _compute_integrations_for_record(
        self,
        record: dict[str, Any],
    ) -> bool:
        roi = record.get("roi")
        if not roi:
            return False
        integrations = compute_peak_fit_integrations(
            self.image_data,
            self.axis_ranges,
            roi,
            azimuthal_roi=record.get("azimuthal_roi")
            or roi.get("azimuthal_roi"),
        )
        if not integrations:
            self._set_fit_status("No finite ROI pixels were available.")
            return False
        store = self._fit_record_for_peak(record, create=True)
        store["roi"] = dict(roi)
        if record.get("azimuthal_roi") or roi.get("azimuthal_roi"):
            store["azimuthal_roi"] = dict(
                record.get("azimuthal_roi") or roi.get("azimuthal_roi")
            )
        store["integrations"] = integrations
        store["integration_fits"] = {}
        store["fit_failures"] = {}
        store.pop("fit_2d", None)
        self._update_record_fit_summary(record)
        return True

    def _fit_2d_for_record(
        self,
        record: dict[str, Any],
        *,
        prepare_missing: bool,
    ) -> bool:
        if not record.get("roi"):
            return False
        store = self._fit_record_for_peak(record, create=True)
        if prepare_missing and not store.get("integrations"):
            if not self._compute_integrations_for_record(record):
                return False
            store = self._fit_record_for_peak(record, create=True)
        if prepare_missing and len(store.get("integration_fits", {})) < 3:
            fits = fit_peak_integrations(store.get("integrations", {}))
            store["integration_fits"] = fits
            for name, fit in fits.items():
                store["integrations"][name]["fit"] = fit
        fit = fit_peak_roi_2d(
            self.image_data,
            self.axis_ranges,
            record["roi"],
            store.get("integration_fits", {}),
        )
        if fit is None:
            store.pop("fit_2d", None)
            store["fit_2d_failure"] = {
                "status": "failed",
                "message": "The 2D fit could not be computed.",
            }
            self._set_fit_status("The 2D fit could not be computed.")
            return False
        store["fit_2d"] = fit
        store.pop("fit_2d_failure", None)
        self._update_record_fit_summary(record)
        return True

    def _fit_store(self, *, create: bool) -> dict[str, Any]:
        container = self.project.fits.get(self.data_id)
        if container is None:
            if not create:
                return {}
            container = {}
            self.project.fits[self.data_id] = container
        if not isinstance(container, dict):
            if not create:
                return {}
            container = {"legacy": container}
            self.project.fits[self.data_id] = container
        peak_fit = container.get("peak_fit")
        if peak_fit is None:
            if not create:
                return {}
            peak_fit = {}
            container["peak_fit"] = peak_fit
        if not isinstance(peak_fit, dict):
            if not create:
                return {}
            peak_fit = {}
            container["peak_fit"] = peak_fit
        return peak_fit

    def _fit_record_for_peak(
        self,
        record: dict[str, Any] | None,
        *,
        create: bool,
    ) -> dict[str, Any]:
        if record is None:
            return {}
        peak_id = _peak_id(record)
        store = self._fit_store(create=create)
        if not create:
            value = store.get(peak_id, {})
            return value if isinstance(value, dict) else {}
        value = store.setdefault(
            peak_id,
            {
                "peak_id": peak_id,
                "label": record.get("label", peak_id),
            },
        )
        if isinstance(value, dict):
            value["label"] = record.get("label", peak_id)
            return value
        store[peak_id] = {
            "peak_id": peak_id,
            "label": record.get("label", peak_id),
        }
        return store[peak_id]

    def _fit_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        store = self._fit_record_for_peak(record, create=False)
        if not store:
            return {
                "integrations": "0/3",
                "one_d_fits": "0/3",
                "two_d_fit": "No",
                "center_qxy": None,
                "center_qz": None,
            }
        integrations = store.get("integrations", {})
        one_d_fits = store.get("integration_fits", {})
        fit_2d = store.get("fit_2d")
        return {
            "integrations": f"{len(integrations)}/3",
            "one_d_fits": f"{len(one_d_fits)}/3",
            "two_d_fit": "Yes" if fit_2d else "No",
            "center_qxy": fit_2d.get("center_qxy") if fit_2d else None,
            "center_qz": fit_2d.get("center_qz") if fit_2d else None,
        }

    def _update_record_fit_summary(self, record: dict[str, Any]) -> None:
        record["fit_summary"] = self._fit_summary(record)

    def _sync_fit_detail_tree(
        self,
        record: dict[str, Any] | None,
        store: dict[str, Any],
    ) -> None:
        self.fit_detail_tree.clear()
        if record is None:
            self.fit_detail_tree.addTopLevelItem(
                QtWidgets.QTreeWidgetItem(["Peak", "None selected", ""])
            )
            return
        label = str(record.get("label", _peak_id(record)))
        roi_status = "Ready" if record.get("roi") else "Missing ROI"
        root = QtWidgets.QTreeWidgetItem(["Peak", label, roi_status])
        self.fit_detail_tree.addTopLevelItem(root)

        integrations_group = QtWidgets.QTreeWidgetItem(
            ["Integrations", f"{len(store.get('integrations', {}))}/3", ""]
        )
        root.addChild(integrations_group)
        for name in PEAK_FIT_INTEGRATIONS:
            profile = store.get("integrations", {}).get(name)
            status = "Ready" if profile else "Not run"
            item = QtWidgets.QTreeWidgetItem(
                [FIT_INTEGRATION_LABELS[name], "", status]
            )
            integrations_group.addChild(item)
            if profile:
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [
                            "Samples",
                            str(len(profile.get("x_values", []))),
                            "",
                        ]
                    )
                )

        fits_group = QtWidgets.QTreeWidgetItem(
            [
                "1D Gaussian Fits",
                f"{len(store.get('integration_fits', {}))}/3",
                "",
            ]
        )
        root.addChild(fits_group)
        for name in PEAK_FIT_INTEGRATIONS:
            fit = store.get("integration_fits", {}).get(name)
            failure = store.get("fit_failures", {}).get(name)
            status = (
                str(fit.get("status", ""))
                if fit
                else (
                    str(failure.get("status", "failed"))
                    if failure
                    else "Not run"
                )
            )
            item = QtWidgets.QTreeWidgetItem(
                [FIT_INTEGRATION_LABELS[name], "", status]
            )
            fits_group.addChild(item)
            if fit:
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        ["Center", _format_float(fit.get("center")), ""]
                    )
                )
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [
                            "Width FWHM",
                            _format_float(fit.get("width_fwhm")),
                            "",
                        ]
                    )
                )
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [
                            "R_w",
                            _format_float(
                                fit.get("statistics", {}).get("r_w")
                            ),
                            "",
                        ]
                    )
                )
            elif failure:
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        ["Message", str(failure.get("message", "")), ""]
                    )
                )

        fit_2d = store.get("fit_2d")
        fit_2d_failure = store.get("fit_2d_failure")
        fit_2d_group = QtWidgets.QTreeWidgetItem(
            [
                "2D Gaussian Fit",
                str(fit_2d.get("model_name", "")) if fit_2d else "",
                (
                    str(fit_2d.get("status", "Not run"))
                    if fit_2d
                    else (
                        str(fit_2d_failure.get("status", "failed"))
                        if fit_2d_failure
                        else "Not run"
                    )
                ),
            ]
        )
        root.addChild(fit_2d_group)
        if fit_2d:
            for label_text, key in (
                (f"Center {QXY_HTML}", "center_qxy"),
                (f"Center {QZ_HTML}", "center_qz"),
                (f"Width {QXY_HTML} FWHM", "width_qxy_fwhm"),
                (f"Width {QZ_HTML} FWHM", "width_qz_fwhm"),
                ("R_w", "r_w"),
                ("RMSE", "rmse"),
                ("R squared", "r_squared"),
            ):
                value = (
                    fit_2d.get("statistics", {}).get(key)
                    if key in {"r_w", "rmse", "r_squared"}
                    else fit_2d.get(key)
                )
                fit_2d_group.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [label_text, _format_float(value), ""]
                    )
                )
            fit_2d_group.addChild(
                QtWidgets.QTreeWidgetItem(
                    ["Expression", str(fit_2d.get("expression", "")), ""]
                )
            )
        elif fit_2d_failure:
            fit_2d_group.addChild(
                QtWidgets.QTreeWidgetItem(
                    ["Message", str(fit_2d_failure.get("message", "")), ""]
                )
            )
        self.fit_detail_tree.expandItem(root)
        self.fit_detail_tree.expandItem(integrations_group)
        self.fit_detail_tree.expandItem(fits_group)
        self.fit_detail_tree.expandItem(fit_2d_group)
        self.fit_detail_tree.resizeColumnToContents(0)

    def _set_fit_buttons_enabled(self, enabled: bool) -> None:
        has_any_roi = any(bool(record.get("roi")) for record in self.peaks())
        record = self._active_record()
        has_active_roi = bool(record and record.get("roi"))
        store = self._fit_record_for_peak(record, create=False)
        integrations = store.get("integrations", {}) if store else {}
        self.run_selected_integrations_button.setEnabled(
            enabled and has_active_roi
        )
        self.run_2d_fit_button.setEnabled(enabled and has_active_roi)
        self.fit_selected_integration_button.setEnabled(
            enabled and bool(integrations)
        )
        self.fit_qxy_trace_button.setEnabled(enabled and "qxy" in integrations)
        self.fit_qz_trace_button.setEnabled(enabled and "qz" in integrations)
        self.fit_azimuthal_trace_button.setEnabled(
            enabled and "azimuthal" in integrations
        )
        self.fit_all_integrations_button.setEnabled(
            enabled and bool(integrations)
        )
        self.fit_integration_combo.setEnabled(enabled and bool(integrations))
        self.run_all_integrations_button.setEnabled(enabled and has_any_roi)
        self.batch_fit_button.setEnabled(enabled and has_any_roi)

    def _set_fit_status(self, message: str) -> None:
        self.fit_status_label.setText(message)

    def _active_record(self) -> dict[str, Any] | None:
        for record in self.peaks():
            if _peak_id(record) == self.active_peak_id:
                return record
        return None

    def _nearest_peak_id(self, qxy: float, qz: float) -> str | None:
        records = self.peaks()
        if not records:
            return None
        x_scale, y_scale = self._axis_scales()
        distances = [
            ((_peak_qxy(record) - qxy) / x_scale) ** 2
            + ((_peak_qz(record) - qz) / y_scale) ** 2
            for record in records
        ]
        return _peak_id(records[int(np.argmin(distances))])

    def _peak_record(
        self,
        qxy: float,
        qz: float,
        intensity: float,
        *,
        source: str,
        used_ids: set[str],
    ) -> dict[str, Any]:
        peak_id = _unique_peak_id(self.data_id, used_ids)
        used_ids.add(peak_id)
        return {
            "peak_id": peak_id,
            "label": f"P{len(used_ids)}",
            "qxy": float(qxy),
            "qz": float(qz),
            "intensity": float(intensity),
            "source": source,
            "point_kind": PEAK_POINT_KIND_COMMITTED,
            "phase_tag": DEFAULT_PHASE_TAG,
        }

    def _apply_roi_to_record(self, record: dict[str, Any]) -> None:
        width = max(self.roi_width.value(), 1.0e-12)
        height = max(self.roi_height.value(), 1.0e-12)
        qxy = _peak_qxy(record)
        qz = _peak_qz(record)
        record["roi"] = {
            "kind": "box",
            "center_qxy": qxy,
            "center_qz": qz,
            "width": width,
            "height": height,
            "qxy_min": qxy - width / 2.0,
            "qxy_max": qxy + width / 2.0,
            "qz_min": qz - height / 2.0,
            "qz_max": qz + height / 2.0,
        }
        store = self._fit_record_for_peak(record, create=False)
        if store:
            store["roi"] = dict(record["roi"])
            store.pop("integrations", None)
            store.pop("integration_fits", None)
            store.pop("fit_2d", None)
        record.pop("fit_summary", None)

    def _snap_target_near(
        self,
        qxy: float,
        qz: float,
    ) -> dict[str, Any] | None:
        gap_estimate = self._masked_gap_peak_estimate(qxy, qz)
        if gap_estimate is not None:
            return gap_estimate
        maximum = self._local_maximum_near(qxy, qz)
        if maximum is None:
            return None
        peak_qxy, peak_qz, intensity = maximum
        return {
            "kind": "local-maximum",
            "qxy": peak_qxy,
            "qz": peak_qz,
            "intensity": intensity,
            "source": "manual-local-maximum",
        }

    def _apply_peak_target_to_record(
        self,
        record: dict[str, Any],
        target: dict[str, Any],
    ) -> None:
        record["qxy"] = float(target["qxy"])
        record["qz"] = float(target["qz"])
        record["intensity"] = float(target["intensity"])
        record["source"] = str(target["source"])
        self._apply_peak_target_metadata(record, target)

    def _apply_peak_target_metadata(
        self,
        record: dict[str, Any],
        target: dict[str, Any],
    ) -> None:
        if target.get("kind") != "masked-gap":
            return
        record["point_kind"] = PEAK_POINT_KIND_GAP_ESTIMATED
        record["gap_estimated"] = True
        record["phase_tag"] = "gap-estimated"
        record["metadata"] = dict(target.get("metadata", {}))

    def _coordinate_is_masked(self, qxy: float, qz: float) -> bool:
        if self.image_data is None:
            return False
        x_axis, y_axis = self._image_axes()
        if not x_axis.size or not y_axis.size:
            return False
        x_index = int(np.argmin(np.abs(x_axis - qxy)))
        y_index = int(np.argmin(np.abs(y_axis - qz)))
        image = np.asarray(self.image_data, dtype=float)
        qxy_gap = _masked_gap_line_mask(image[y_index, :])
        qz_gap = _masked_gap_line_mask(image[:, x_index])
        return bool(qxy_gap[x_index] or qz_gap[y_index])

    def _masked_gap_peak_estimate(
        self,
        qxy: float,
        qz: float,
    ) -> dict[str, Any] | None:
        if self.image_data is None:
            return None
        x_axis, y_axis = self._image_axes()
        if not x_axis.size or not y_axis.size:
            return None
        x_index = int(np.argmin(np.abs(x_axis - qxy)))
        y_index = int(np.argmin(np.abs(y_axis - qz)))

        radius = max(
            self.snap_window_px.value(),
            self.min_distance_px.value(),
            self.neighborhood_radius_px.value() * 4,
            3,
        )
        candidates = [
            candidate
            for candidate in (
                self._masked_gap_axis_estimate(
                    x_index,
                    y_index,
                    axis="qxy",
                    radius=radius,
                    x_axis=x_axis,
                    y_axis=y_axis,
                ),
                self._masked_gap_axis_estimate(
                    x_index,
                    y_index,
                    axis="qz",
                    radius=radius,
                    x_axis=x_axis,
                    y_axis=y_axis,
                ),
            )
            if candidate is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: float(candidate["score"]))

    def _masked_gap_axis_estimate(
        self,
        x_index: int,
        y_index: int,
        *,
        axis: str,
        radius: int,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> dict[str, Any] | None:
        if self.image_data is None:
            return None
        image = np.asarray(self.image_data, dtype=float)
        if axis == "qxy":
            primary_values = x_axis
            primary_index = x_index
            masked_line = _masked_gap_line_mask(image[y_index, :])
        else:
            primary_values = y_axis
            primary_index = y_index
            masked_line = _masked_gap_line_mask(image[:, x_index])
        if not masked_line[primary_index]:
            return None
        gap_start, gap_end = _contiguous_true_span(masked_line, primary_index)
        minus_anchor = plus_anchor = None
        side_radius = int(radius)
        for scale in (1, 2, 4):
            side_radius = int(radius * scale)
            perp_radius = max(2, side_radius // 2)
            minus_anchor, plus_anchor = self._masked_gap_side_anchors(
                x_index,
                y_index,
                axis=axis,
                gap_start=gap_start,
                gap_end=gap_end,
                side_radius=side_radius,
                perp_radius=perp_radius,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            if minus_anchor is not None and plus_anchor is not None:
                break
        if minus_anchor is None or plus_anchor is None:
            return None

        profile_values, profile = self._masked_gap_profile(
            x_index,
            y_index,
            axis=axis,
            gap_start=gap_start,
            gap_end=gap_end,
            side_radius=side_radius,
            perp_radius=max(2, side_radius // 2),
            x_axis=x_axis,
            y_axis=y_axis,
        )
        if profile_values.size == 0:
            return None
        gap_min = float(
            min(primary_values[gap_start], primary_values[gap_end])
        )
        gap_max = float(
            max(primary_values[gap_start], primary_values[gap_end])
        )
        fit = self._fit_masked_gap_gaussian(
            profile_values,
            profile,
            gap_min=gap_min,
            gap_max=gap_max,
            minus_anchor=minus_anchor,
            plus_anchor=plus_anchor,
            axis=axis,
        )
        if fit is None:
            return None
        center = float(fit["center"])
        weights = [
            max(float(minus_anchor["intensity"]), 1.0e-12),
            max(float(plus_anchor["intensity"]), 1.0e-12),
        ]
        if axis == "qxy":
            qxy = center
            qz = float(
                np.average(
                    [minus_anchor["qz"], plus_anchor["qz"]],
                    weights=weights,
                )
            )
        else:
            qxy = float(
                np.average(
                    [minus_anchor["qxy"], plus_anchor["qxy"]],
                    weights=weights,
                )
            )
            qz = center
        metadata = {
            "gap_estimate": True,
            "estimate_method": "masked gap gaussian",
            "masked_gap": True,
            "gap_axis": axis,
            f"gap_{axis}_min": gap_min,
            f"gap_{axis}_max": gap_max,
            "minus_anchor_qxy": float(minus_anchor["qxy"]),
            "minus_anchor_qz": float(minus_anchor["qz"]),
            "minus_anchor_intensity": float(minus_anchor["intensity"]),
            "plus_anchor_qxy": float(plus_anchor["qxy"]),
            "plus_anchor_qz": float(plus_anchor["qz"]),
            "plus_anchor_intensity": float(plus_anchor["intensity"]),
            "gaussian_fit": fit["fit_kind"],
        }
        return {
            "kind": "masked-gap",
            "qxy": qxy,
            "qz": qz,
            "intensity": float(fit["intensity"]),
            "source": "gap estimate",
            "score": float(fit["score"]),
            "metadata": metadata,
        }

    def _masked_gap_side_anchors(
        self,
        x_index: int,
        y_index: int,
        *,
        axis: str,
        gap_start: int,
        gap_end: int,
        side_radius: int,
        perp_radius: int,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.image_data is None:
            return None, None
        height, width = self.image_data.shape
        if axis == "qxy":
            y0 = max(0, y_index - perp_radius)
            y1 = min(height, y_index + perp_radius + 1)
            minus = self._finite_window_max(
                max(0, gap_start - side_radius),
                gap_start,
                y0,
                y1,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            plus = self._finite_window_max(
                gap_end + 1,
                min(width, gap_end + side_radius + 1),
                y0,
                y1,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            return minus, plus

        x0 = max(0, x_index - perp_radius)
        x1 = min(width, x_index + perp_radius + 1)
        minus = self._finite_window_max(
            x0,
            x1,
            max(0, gap_start - side_radius),
            gap_start,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        plus = self._finite_window_max(
            x0,
            x1,
            gap_end + 1,
            min(height, gap_end + side_radius + 1),
            x_axis=x_axis,
            y_axis=y_axis,
        )
        return minus, plus

    def _finite_window_max(
        self,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        *,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> dict[str, Any] | None:
        if self.image_data is None or x1 <= x0 or y1 <= y0:
            return None
        window = np.asarray(self.image_data[y0:y1, x0:x1], dtype=float)
        finite = np.isfinite(window)
        if not finite.any():
            return None
        filled = np.where(finite, window, -np.inf)
        local = int(np.argmax(filled))
        local_y, local_x = np.unravel_index(local, window.shape)
        peak_x = x0 + int(local_x)
        peak_y = y0 + int(local_y)
        return {
            "x_index": peak_x,
            "y_index": peak_y,
            "qxy": float(x_axis[peak_x]),
            "qz": float(y_axis[peak_y]),
            "intensity": float(self.image_data[peak_y, peak_x]),
        }

    def _masked_gap_profile(
        self,
        x_index: int,
        y_index: int,
        *,
        axis: str,
        gap_start: int,
        gap_end: int,
        side_radius: int,
        perp_radius: int,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.image_data is None:
            return np.array([]), np.array([])
        height, width = self.image_data.shape
        if axis == "qxy":
            x0 = max(0, gap_start - side_radius)
            x1 = min(width, gap_end + side_radius + 1)
            y0 = max(0, y_index - perp_radius)
            y1 = min(height, y_index + perp_radius + 1)
            values = np.asarray(self.image_data[y0:y1, x0:x1], dtype=float)
            return x_axis[x0:x1], _nanmax_axis(values, axis=0)
        x0 = max(0, x_index - perp_radius)
        x1 = min(width, x_index + perp_radius + 1)
        y0 = max(0, gap_start - side_radius)
        y1 = min(height, gap_end + side_radius + 1)
        values = np.asarray(self.image_data[y0:y1, x0:x1], dtype=float)
        return y_axis[y0:y1], _nanmax_axis(values, axis=1)

    def _fit_masked_gap_gaussian(
        self,
        primary_values: np.ndarray,
        profile: np.ndarray,
        *,
        gap_min: float,
        gap_max: float,
        minus_anchor: dict[str, Any],
        plus_anchor: dict[str, Any],
        axis: str,
    ) -> dict[str, Any] | None:
        finite = np.isfinite(profile)
        if not finite.any():
            return None
        finite_profile = profile[finite]
        baseline = float(np.nanpercentile(finite_profile, 10.0))
        signal = profile - baseline
        finite_signal = signal[np.isfinite(signal)]
        if not finite_signal.size:
            return None
        max_signal = float(np.nanmax(finite_signal))
        threshold = max(max_signal * 0.03, 1.0e-12)
        fit_mask = finite & (signal > threshold)
        center = None
        intensity = None
        fit_kind = "anchor-weighted"
        if np.count_nonzero(fit_mask) >= 3:
            try:
                coefficients = np.polyfit(
                    primary_values[fit_mask],
                    np.log(signal[fit_mask]),
                    2,
                )
                curvature, slope, intercept = coefficients
                if curvature < 0.0:
                    fitted_center = -slope / (2.0 * curvature)
                    if np.isfinite(fitted_center):
                        center = float(
                            np.clip(fitted_center, gap_min, gap_max)
                        )
                        log_signal = float(np.polyval(coefficients, center))
                        intensity = baseline + float(np.exp(log_signal))
                        fit_kind = "log-quadratic-gaussian"
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                center = None
        if center is None:
            minus_value = float(minus_anchor[axis])
            plus_value = float(plus_anchor[axis])
            weights = [
                max(float(minus_anchor["intensity"]) - baseline, 1.0e-12),
                max(float(plus_anchor["intensity"]) - baseline, 1.0e-12),
            ]
            center = float(
                np.clip(
                    np.average([minus_value, plus_value], weights=weights),
                    gap_min,
                    gap_max,
                )
            )
            intensity = max(
                float(minus_anchor["intensity"]),
                float(plus_anchor["intensity"]),
            )
        score = min(
            max(float(minus_anchor["intensity"]) - baseline, 0.0),
            max(float(plus_anchor["intensity"]) - baseline, 0.0),
        )
        return {
            "center": center,
            "intensity": float(intensity),
            "score": score,
            "fit_kind": fit_kind,
        }

    def _local_maximum_near(
        self,
        qxy: float,
        qz: float,
    ) -> tuple[float, float, float] | None:
        if self.image_data is None:
            return None
        x_axis, y_axis = self._image_axes()
        x_index = int(np.argmin(np.abs(x_axis - qxy)))
        y_index = int(np.argmin(np.abs(y_axis - qz)))
        radius = max(
            self.neighborhood_radius_px.value() * 2,
            self.min_distance_px.value(),
            1,
        )
        x0 = max(0, x_index - radius)
        x1 = min(self.image_data.shape[1], x_index + radius + 1)
        y0 = max(0, y_index - radius)
        y1 = min(self.image_data.shape[0], y_index + radius + 1)
        window = np.asarray(self.image_data[y0:y1, x0:x1], dtype=float)
        if not window.size or not np.isfinite(window).any():
            return None
        local = np.nanargmax(window)
        local_y, local_x = np.unravel_index(local, window.shape)
        peak_y = y0 + int(local_y)
        peak_x = x0 + int(local_x)
        return (
            float(x_axis[peak_x]),
            float(y_axis[peak_y]),
            float(self.image_data[peak_y, peak_x]),
        )

    def _intensity_at(self, qxy: float, qz: float) -> float:
        if self.image_data is None:
            return float("nan")
        x_axis, y_axis = self._image_axes()
        x_index = int(np.argmin(np.abs(x_axis - qxy)))
        y_index = int(np.argmin(np.abs(y_axis - qz)))
        return float(self.image_data[y_index, x_index])

    def _image_axes(self) -> tuple[np.ndarray, np.ndarray]:
        if self.image_data is None:
            return np.array([]), np.array([])
        height, width = self.image_data.shape
        if self.axis_ranges is None:
            return np.arange(width, dtype=float), np.arange(
                height, dtype=float
            )
        x_min, x_max, y_min, y_max = self.axis_ranges
        return (
            np.linspace(x_min, x_max, width),
            np.linspace(y_min, y_max, height),
        )

    def _valid_peak_mask(self, y_axis: np.ndarray) -> np.ndarray:
        if self.image_data is None:
            return np.zeros((0, 0), dtype=bool)
        valid = np.isfinite(self.image_data)
        if y_axis.size == self.image_data.shape[0]:
            valid &= y_axis[:, np.newaxis] >= self.min_qz.value()
        return valid

    def _axis_scales(self) -> tuple[float, float]:
        if self.axis_ranges is None:
            return 1.0, 1.0
        x_min, x_max, y_min, y_max = self.axis_ranges
        return max(abs(x_max - x_min), 1.0e-12), max(
            abs(y_max - y_min),
            1.0e-12,
        )

    def _set_default_roi_dimensions(self) -> None:
        if self.axis_ranges is None:
            self.roi_width.setValue(20.0)
            self.roi_height.setValue(20.0)
            return
        x_min, x_max, y_min, y_max = self.axis_ranges
        self.roi_width.setValue(max(abs(x_max - x_min) / 30.0, 0.01))
        self.roi_height.setValue(max(abs(y_max - y_min) / 30.0, 0.01))


def _roi_size_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(1.0e-9, 1.0e6)
    spinbox.setDecimals(5)
    spinbox.setSingleStep(0.01)
    spinbox.setMaximumWidth(95)
    return spinbox


def _orientation_control_row(
    slider: QtWidgets.QSlider,
    spinbox: QtWidgets.QDoubleSpinBox,
) -> QtWidgets.QWidget:
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(slider, stretch=1)
    layout.addWidget(spinbox)
    return row


def _orientation_slider() -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    low, high = ORIENTATION_ANGLE_LIMITS_DEG
    slider.setRange(
        int(low * ORIENTATION_SLIDER_SCALE),
        int(high * ORIENTATION_SLIDER_SCALE),
    )
    slider.setSingleStep(5)
    slider.setPageStep(50)
    slider.setTickInterval(300)
    slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
    slider.setMaximumWidth(170)
    return slider


def _orientation_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(*ORIENTATION_ANGLE_LIMITS_DEG)
    spinbox.setDecimals(1)
    spinbox.setSingleStep(1.0)
    spinbox.setSuffix(" deg")
    spinbox.setMaximumWidth(95)
    return spinbox


def _clamp_orientation_angle(value: float) -> float:
    low, high = ORIENTATION_ANGLE_LIMITS_DEG
    return float(np.clip(value, low, high))


def _degrees_to_slider_value(value: float) -> int:
    return int(
        round(_clamp_orientation_angle(value) * ORIENTATION_SLIDER_SCALE)
    )


def _slider_value_to_degrees(value: int) -> float:
    return _clamp_orientation_angle(float(value) / ORIENTATION_SLIDER_SCALE)


def _set_slider_blocked(
    slider: QtWidgets.QSlider,
    value: float,
) -> None:
    slider.blockSignals(True)
    try:
        slider.setValue(_degrees_to_slider_value(value))
    finally:
        slider.blockSignals(False)


def _set_spinbox_blocked(
    spinbox: QtWidgets.QDoubleSpinBox,
    value: float,
) -> None:
    spinbox.blockSignals(True)
    try:
        spinbox.setValue(_clamp_orientation_angle(value))
    finally:
        spinbox.blockSignals(False)


def _orientation_angles_from_state(
    payload: Any,
    fallback_quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    if isinstance(payload, (list, tuple)) and len(payload) == 3:
        try:
            return tuple(_clamp_orientation_angle(value) for value in payload)
        except (TypeError, ValueError):
            pass
    return euler_angles_from_quaternion(fallback_quaternion)


def _event_position(
    event: QtGui.QMouseEvent | QtGui.QWheelEvent,
) -> QtCore.QPointF:
    if hasattr(event, "position"):
        return event.position()
    return QtCore.QPointF(event.pos())


def _lattice_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0001, 1000.0)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(0.1)
    spinbox.setValue(value)
    spinbox.setMaximumWidth(110)
    return spinbox


def _angle_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.01, 179.99)
    spinbox.setDecimals(3)
    spinbox.setSingleStep(1.0)
    spinbox.setValue(value)
    spinbox.setMaximumWidth(110)
    return spinbox


def _hkl_spinbox(value: int) -> QtWidgets.QSpinBox:
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(0, 12)
    spinbox.setValue(value)
    spinbox.setMaximumWidth(82)
    return spinbox


def _project_cell_preview(points: np.ndarray) -> np.ndarray:
    """Project 3D cell coordinates into a lightweight oblique
    preview."""

    array = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            array[:, 0] + 0.35 * array[:, 1],
            array[:, 2] + 0.20 * array[:, 1],
        )
    )


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.zeros_like(array, dtype=float)
    return array / norm


def _preview_unit_vector(vector: np.ndarray) -> np.ndarray:
    direction = _project_cell_preview(np.asarray([vector], dtype=float))[0]
    normalized = _unit_vector(direction)
    if not np.any(normalized):
        return np.array((1.0, 0.0), dtype=float)
    return normalized


def _screen_unit_vector(preview_vector: np.ndarray) -> np.ndarray:
    direction = np.array(
        (float(preview_vector[0]), -float(preview_vector[1])),
        dtype=float,
    )
    normalized = _unit_vector(direction)
    if not np.any(normalized):
        return np.array((1.0, 0.0), dtype=float)
    return normalized


def _point_array(point: QtCore.QPointF) -> np.ndarray:
    return np.array((float(point.x()), float(point.y())), dtype=float)


def _offset_point(
    point: QtCore.QPointF,
    direction: np.ndarray,
    distance: float,
) -> QtCore.QPointF:
    vector = _unit_vector(direction)
    return QtCore.QPointF(
        float(point.x() + vector[0] * distance),
        float(point.y() + vector[1] * distance),
    )


def _representative_q_direction(q_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(q_vectors, dtype=float).reshape((-1, 3))
    if vectors.size == 0:
        return _unit_vector(FALLBACK_SCATTER_VECTOR)
    norms = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(vectors).all(axis=1) & (norms > 1.0e-12)
    if not np.any(finite):
        return _unit_vector(FALLBACK_SCATTER_VECTOR)

    candidates = vectors[finite] / norms[finite, np.newaxis]
    direct_preview = _preview_unit_vector(DIRECT_BEAM_VECTOR)
    candidate_previews = np.vstack(
        [_preview_unit_vector(candidate) for candidate in candidates]
    )
    separation = np.abs(
        direct_preview[0] * candidate_previews[:, 1]
        - direct_preview[1] * candidate_previews[:, 0]
    )
    depth_bias = 0.05 * np.abs(candidates[:, 2])
    return candidates[int(np.argmax(separation + depth_bias))]


def _scattered_beam_preview_direction(q_vectors: np.ndarray) -> np.ndarray:
    q_direction = _representative_q_direction(q_vectors)
    outgoing = DIRECT_BEAM_VECTOR + 0.85 * q_direction
    if not np.any(_unit_vector(outgoing)):
        outgoing = DIRECT_BEAM_VECTOR + 0.85 * FALLBACK_SCATTER_VECTOR

    direction = _preview_unit_vector(outgoing)
    direct_direction = _preview_unit_vector(DIRECT_BEAM_VECTOR)
    separation = abs(
        direct_direction[0] * direction[1] - direct_direction[1] * direction[0]
    )
    if separation < 0.08:
        direction = _preview_unit_vector(
            DIRECT_BEAM_VECTOR + 0.85 * FALLBACK_SCATTER_VECTOR
        )
    return direction


def _beam_arrowhead_points(tail: np.ndarray, tip: np.ndarray) -> np.ndarray:
    tail_point = np.asarray(tail, dtype=float)
    tip_point = np.asarray(tip, dtype=float)
    ray = tip_point - tail_point
    length = float(np.linalg.norm(ray))
    if length <= 1.0e-12 or not np.isfinite(length):
        return np.vstack((tip_point, tip_point, tip_point))

    direction = ray / length
    normal = np.array((-direction[1], direction[0]), dtype=float)
    head_length = max(length * 0.16, 0.08)
    base = tip_point - direction * head_length
    return np.vstack(
        (
            base + normal * head_length * 0.45,
            tip_point,
            base - normal * head_length * 0.45,
        )
    )


def _crystal_overlay_spots(result: Any) -> list[dict[str, Any]]:
    if pg is None:
        return []
    spots: list[dict[str, Any]] = []
    for qxy, qz, hkl in zip(result.qxy, result.qz, result.hkl, strict=False):
        spots.append(
            {
                "pos": (float(qxy), float(qz)),
                "data": _format_hkl_label(hkl),
                "symbol": "o",
                "size": 9,
                "brush": pg.mkBrush(255, 190, 77, 155),
                "pen": pg.mkPen("#332208", width=1.0),
            }
        )
    return spots


def _format_hkl_label(hkl: Any) -> str:
    h, k, ell = (int(value) for value in hkl)
    return f"({h} {k} {ell})"


def _format_hkl_label_from_record(hkl: Any) -> str:
    if isinstance(hkl, str):
        stripped = hkl.strip()
        if not stripped:
            return ""
        if stripped.startswith("(") and stripped.endswith(")"):
            return stripped
        parts = stripped.replace(",", " ").split()
    else:
        parts = list(hkl)
    if len(parts) != 3:
        return str(hkl)
    try:
        return _format_hkl_label(parts)
    except (TypeError, ValueError):
        return str(hkl)


def _crystal_hkl_label_indices(result: Any, mode: str) -> list[int]:
    count = len(result.hkl)
    if count == 0 or mode == HKL_LABEL_MODE_NONE:
        return []
    if mode == HKL_LABEL_MODE_ALL:
        return list(range(count))

    hkl = np.asarray(result.hkl, dtype=int)
    qxy = np.asarray(result.qxy, dtype=float)
    qz = np.asarray(result.qz, dtype=float)
    q_magnitude = np.hypot(qxy, qz)
    order = np.sum(np.abs(hkl), axis=1)
    ranked = sorted(
        range(count),
        key=lambda index: (
            int(order[index]),
            float(q_magnitude[index]),
            tuple(int(value) for value in np.abs(hkl[index])),
        ),
    )
    selected: list[int] = []
    seen_families: set[tuple[int, int, int]] = set()
    for index in ranked:
        family = tuple(
            sorted((int(abs(value)) for value in hkl[index]), reverse=True)
        )
        important = int(order[index]) <= 2 or family not in seen_families
        if not important:
            continue
        selected.append(index)
        seen_families.add(family)
        if len(selected) >= CRYSTAL_HKL_LABEL_LIMIT:
            break
    return sorted(selected)


def _crystal_hkl_tip(*, x: float, y: float, data: Any) -> str:
    return str(data)


def _unique_peak_id(data_id: str, used_ids: set[str]) -> str:
    base = f"{data_id}_peak"
    index = 1
    candidate = f"{base}_{index}"
    while candidate in used_ids:
        index += 1
        candidate = f"{base}_{index}"
    return candidate


def _peak_id(record: dict[str, Any]) -> str:
    return str(record.get("peak_id") or record.get("id") or "peak")


def _peak_qxy(record: dict[str, Any]) -> float:
    return float(record.get("qxy", record.get("qx", record.get("x", 0.0))))


def _peak_qz(record: dict[str, Any]) -> float:
    return float(record.get("qz", record.get("y", 0.0)))


def _contiguous_true_span(values: np.ndarray, index: int) -> tuple[int, int]:
    start = int(index)
    end = int(index)
    while start > 0 and bool(values[start - 1]):
        start -= 1
    while end + 1 < values.size and bool(values[end + 1]):
        end += 1
    return start, end


def _nanmax_axis(values: np.ndarray, *, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    if values.size == 0:
        return np.array([])
    has_finite = np.any(finite, axis=axis)
    collapsed = np.max(np.where(finite, values, -np.inf), axis=axis)
    return np.where(has_finite, collapsed, np.nan)


def _masked_gap_line_mask(values: np.ndarray) -> np.ndarray:
    line = np.asarray(values, dtype=float)
    mask = ~np.isfinite(line)
    finite = line[np.isfinite(line)]
    if not finite.size:
        return mask
    floor = float(np.nanmin(finite))
    ceiling = float(np.nanpercentile(finite, 99.0))
    span = ceiling - floor
    if span <= 0.0:
        return mask
    floor_tolerance = max(span * 1.0e-6, 1.0e-12)
    return mask | (line <= floor + floor_tolerance)


def _is_gap_estimated_peak(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return (
        bool(metadata.get("gap_estimate"))
        or bool(record.get("gap_estimated"))
        or record.get("point_kind") == PEAK_POINT_KIND_GAP_ESTIMATED
        or str(record.get("source", "")).lower() == "gap estimate"
    )


def _peak_tip(record: dict[str, Any]) -> str:
    label = str(record.get("label", _peak_id(record)))
    return (
        f"<qt>{label}<br>"
        f"{QXY_HTML}: {_format_float(_peak_qxy(record))} "
        f"{QSPACE_UNITS_HTML}<br>"
        f"{QZ_HTML}: {_format_float(_peak_qz(record))} "
        f"{QSPACE_UNITS_HTML}</qt>"
    )


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.5g}"


def _format_metric(value: Any) -> str:
    text = _format_float(value)
    return text if text else "--"


def _fit_metrics_text(name: str, fit: dict[str, Any]) -> str:
    center = _format_metric(fit.get("center"))
    width = _format_metric(fit.get("width_fwhm"))
    r_w = _format_metric(fit.get("statistics", {}).get("r_w"))
    if name == "qxy":
        unit = r"Å$^{-1}$"
        label = QXY_MATPLOTLIB_SYMBOL
    elif name == "qz":
        unit = r"Å$^{-1}$"
        label = QZ_MATPLOTLIB_SYMBOL
    else:
        unit = "deg"
        label = "chi"
    return (
        f"{label} center {center} {unit}\n"
        f"FWHM {width} {unit}\n"
        f"$R_w$ {r_w}"
    )


def _fit_2d_metrics_html(fit: dict[str, Any] | None) -> str:
    if not fit:
        return "2D ROI fit metrics: not fitted"
    status = str(fit.get("status", ""))
    center_qxy = _format_metric(fit.get("center_qxy"))
    center_qz = _format_metric(fit.get("center_qz"))
    width_qxy = _format_metric(fit.get("width_qxy_fwhm"))
    width_qz = _format_metric(fit.get("width_qz_fwhm"))
    r_w = _format_metric(fit.get("statistics", {}).get("r_w"))
    return (
        f"2D ROI fit ({status}): "
        f"{QXY_HTML} center {center_qxy} {QSPACE_UNITS_HTML}; "
        f"{QZ_HTML} center {center_qz} {QSPACE_UNITS_HTML}; "
        f"FWHM {QXY_HTML} {width_qxy} {QSPACE_UNITS_HTML}; "
        f"FWHM {QZ_HTML} {width_qz} {QSPACE_UNITS_HTML}; "
        f"R<sub>w</sub> {r_w}"
    )


def _gaussian_1d_values(
    x_values: np.ndarray,
    fit: dict[str, Any],
) -> np.ndarray:
    sigma = max(abs(float(fit.get("sigma", 1.0))), 1.0e-12)
    center = float(fit.get("center", 0.0))
    amplitude = float(fit.get("amplitude", 0.0))
    offset = float(fit.get("offset", 0.0))
    return offset + amplitude * np.exp(
        -0.5 * ((np.asarray(x_values, dtype=float) - center) / sigma) ** 2
    )


def _marker_field(marker: Any, name: str, default: Any = None) -> Any:
    if isinstance(marker, dict):
        return marker.get(name, default)
    return getattr(marker, name, default)
