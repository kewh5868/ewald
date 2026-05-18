"""Shared display notation for reciprocal-space UI labels."""

from __future__ import annotations

from html import escape

from qtpy import QtCore, QtGui, QtWidgets

QXY_TEXT = "q_{xy}"
QZ_TEXT = "q_{z}"
QXY_HTML = "q<sub>xy</sub>"
QZ_HTML = "q<sub>z</sub>"
QSPACE_UNITS_TEXT = "Å^{-1}"
QSPACE_UNITS_HTML = "Å<sup>-1</sup>"
QXY_MATPLOTLIB = r"$q_{xy}$ (Å$^{-1}$)"
QZ_MATPLOTLIB = r"$q_{z}$ (Å$^{-1}$)"
QXY_MATPLOTLIB_SYMBOL = r"$q_{xy}$"
QZ_MATPLOTLIB_SYMBOL = r"$q_z$"


def rich_label(text: str) -> QtWidgets.QLabel:
    """Return a QLabel configured for Qt rich text."""

    label = QtWidgets.QLabel(text)
    label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    return label


class RichTextComboBox(QtWidgets.QComboBox):
    """Combo box that renders rich text in the popup and current label."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setItemDelegate(_RichTextItemDelegate(self))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        text = self.currentText()
        if not _contains_rich_text(text):
            super().paintEvent(event)
            return
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = ""
        painter.drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
        )
        rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
            QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        _draw_rich_text(
            painter,
            rect,
            text,
            self.font(),
            self.palette(),
            option.state,
        )


class RichTextCheckBox(QtWidgets.QCheckBox):
    """Check box that renders rich text in its label."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        text = self.text()
        if not _contains_rich_text(text):
            super().paintEvent(event)
            return
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        painter.drawControl(
            QtWidgets.QStyle.ControlElement.CE_CheckBox,
            option,
        )
        rect = self.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_CheckBoxContents,
            option,
            self,
        )
        _draw_rich_text(
            painter,
            rect,
            text,
            self.font(),
            self.palette(),
            option.state,
        )

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        text = self.text()
        if not _contains_rich_text(text):
            return size
        document = _rich_text_document(text, self.font(), self.palette())
        width = int(document.idealWidth()) + 26
        height = max(size.height(), int(document.size().height()) + 4)
        return QtCore.QSize(width, height)


def enable_rich_text_items(view: QtWidgets.QAbstractItemView) -> None:
    """Render HTML snippets in item-view cells."""

    view.setItemDelegate(_RichTextItemDelegate(view))


def set_rich_text_table_headers(
    table: QtWidgets.QTableWidget,
    labels: list[str],
) -> None:
    """Apply table headers that can render Qt rich text."""

    table.setHorizontalHeader(_RichTextHeaderView(table))
    table.setHorizontalHeaderLabels(labels)


def set_qspace_axis_labels(plot_widget) -> None:
    """Apply shared q-space labels to a pyqtgraph plot widget."""

    plot_widget.setLabel("bottom", QXY_HTML, units=QSPACE_UNITS_HTML)
    plot_widget.setLabel("left", QZ_HTML, units=QSPACE_UNITS_HTML)


def set_data_aspect_locked(plot_widget) -> None:
    """Keep one plotted x unit visually equal to one plotted y unit."""

    if hasattr(plot_widget, "setAspectLocked"):
        plot_widget.setAspectLocked(True, ratio=1)
        return
    if not hasattr(plot_widget, "getViewBox"):
        return
    view_box = plot_widget.getViewBox()
    if hasattr(view_box, "setAspectLocked"):
        view_box.setAspectLocked(True, ratio=1)


def data_image_rect(
    shape: tuple[int, ...],
    axis_ranges: tuple[float, float, float, float] | None,
) -> QtCore.QRectF:
    """Return the plot rect used by pyqtgraph image items."""

    height, width = shape[-2:]
    if axis_ranges is None:
        return QtCore.QRectF(0.0, 0.0, float(width), float(height))
    x_min, x_max, y_min, y_max = axis_ranges
    return QtCore.QRectF(
        float(x_min),
        float(y_min),
        float(x_max - x_min),
        float(y_max - y_min),
    )


def set_data_image_plot_range(
    plot_widget,
    shape: tuple[int, ...],
    axis_ranges: tuple[float, float, float, float] | None,
) -> None:
    """Set image plot bounds without adding padding around the data."""

    rect = data_image_rect(shape, axis_ranges)
    plot_widget.setRange(
        xRange=(rect.left(), rect.right()),
        yRange=(rect.top(), rect.bottom()),
        padding=0.0,
    )


def qt_tooltip(text: str) -> str:
    """Wrap rich tooltip text so Qt treats it as HTML."""

    return f"<qt>{text}</qt>"


def _contains_rich_text(text: str) -> bool:
    return any(tag in text for tag in ("<sub>", "<sup>", "<br", "<qt>"))


class _RichTextItemDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate that paints item text as rich text when needed."""

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        item_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        text = item_option.text
        if not _contains_rich_text(text):
            super().paint(painter, option, index)
            return
        item_option.text = ""
        style = _style(item_option.widget)
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
            item_option,
            painter,
            item_option.widget,
        )
        rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText,
            item_option,
            item_option.widget,
        )
        _draw_rich_text(
            painter,
            rect,
            text,
            item_option.font,
            item_option.palette,
            item_option.state,
        )

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        item_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        text = item_option.text
        if not _contains_rich_text(text):
            return super().sizeHint(option, index)
        document = _rich_text_document(
            text,
            item_option.font,
            item_option.palette,
            item_option.state,
        )
        size = document.size().toSize()
        size.setWidth(int(document.idealWidth()) + 4)
        size.setHeight(size.height() + 4)
        return size


class _RichTextHeaderView(QtWidgets.QHeaderView):
    """Horizontal header that paints section text as rich text."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)

    def paintSection(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        logical_index: int,
    ) -> None:
        if not rect.isValid():
            return
        text = str(
            self.model().headerData(
                logical_index,
                self.orientation(),
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            or ""
        )
        if not _contains_rich_text(text):
            super().paintSection(painter, rect, logical_index)
            return
        painter.save()
        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        option.text = ""
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_Header,
            option,
            painter,
            self,
        )
        text_rect = self.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_HeaderLabel,
            option,
            self,
        )
        _draw_rich_text(
            painter,
            text_rect,
            text,
            self.font(),
            self.palette(),
            option.state,
        )
        painter.restore()

    def sectionSizeFromContents(self, logical_index: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logical_index)
        text = str(
            self.model().headerData(
                logical_index,
                self.orientation(),
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            or ""
        )
        if not _contains_rich_text(text):
            return size
        document = _rich_text_document(text, self.font(), self.palette())
        return QtCore.QSize(
            max(size.width(), int(document.idealWidth()) + 18),
            max(size.height(), int(document.size().height()) + 10),
        )


def _draw_rich_text(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    text: str,
    font: QtGui.QFont,
    palette: QtGui.QPalette,
    state: QtWidgets.QStyle.StateFlag | QtWidgets.QStyle.State | None = None,
) -> None:
    document = _rich_text_document(text, font, palette, state)
    painter.save()
    painter.translate(rect.topLeft())
    clip = QtCore.QRectF(0, 0, rect.width(), rect.height())
    painter.setClipRect(clip)
    y_offset = max(0.0, (rect.height() - document.size().height()) / 2.0)
    painter.translate(0, y_offset)
    document.drawContents(painter, clip)
    painter.restore()


def _rich_text_document(
    text: str,
    font: QtGui.QFont,
    palette: QtGui.QPalette,
    state: QtWidgets.QStyle.StateFlag | QtWidgets.QStyle.State | None = None,
) -> QtGui.QTextDocument:
    document = QtGui.QTextDocument()
    document.setDefaultFont(font)
    selected_state = getattr(
        getattr(QtWidgets.QStyle, "StateFlag", QtWidgets.QStyle),
        "State_Selected",
    )
    selected = bool(state is not None and state & selected_state)
    role = (
        QtGui.QPalette.ColorRole.HighlightedText
        if selected
        else QtGui.QPalette.ColorRole.Text
    )
    color = palette.color(role).name()
    document.setHtml(f'<span style="color:{escape(color)}">{text}</span>')
    document.setDocumentMargin(0)
    return document


def _style(
    widget: QtWidgets.QWidget | None,
) -> QtWidgets.QStyle:
    if widget is not None:
        return widget.style()
    return QtWidgets.QApplication.style()
