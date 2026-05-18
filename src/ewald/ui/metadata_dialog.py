"""Manual metadata review dialog for unresolved filename tokens."""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

from ewald.data.models import DataGroupRef

_TIME_REVIEW_KEYS = {
    "duration_candidates_s",
    "frame_timestamp_s",
    "exposure_time_s",
}


class ManualMetadataDialog(QtWidgets.QDialog):
    """Collect user-provided metadata names for unresolved tokens."""

    def __init__(
        self,
        group: DataGroupRef,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.group = group
        self.setWindowTitle("Review Metadata")
        self.resize(760, 420)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["File", "Token", "Attribute Name", "Value"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.time_table = QtWidgets.QTableWidget(0, 3)
        self.time_table.setHorizontalHeaderLabels(
            ["File", "Frame timestamp", "Exposure time"]
        )
        self.time_table.horizontalHeader().setStretchLastSection(True)
        self.time_table.setVisible(False)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.time_table)
        layout.addWidget(self.table)
        layout.addWidget(buttons)
        self._populate()

    def has_review_rows(self) -> bool:
        return self.table.rowCount() > 0 or self.time_table.rowCount() > 0

    def accept(self) -> None:
        self.apply_metadata()
        super().accept()

    def apply_metadata(self) -> None:
        self._apply_time_candidate_selections()
        for row in range(self.table.rowCount()):
            data_id = self.table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            token = self.table.item(row, 1).text()
            attribute = self.table.item(row, 2).text().strip()
            value = self.table.item(row, 3).text().strip()
            original_key = self.table.item(row, 2).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            if not attribute:
                continue
            for data_file in self.group.data_files:
                if data_file.data_id != data_id:
                    continue
                if original_key and original_key != attribute:
                    data_file.metadata.pop(original_key, None)
                data_file.metadata[attribute] = value or token
                self._rename_metadata_field(
                    data_file.metadata, original_key, attribute
                )
                unresolved = list(
                    data_file.metadata.get("_unresolved_tokens", [])
                )
                if token in unresolved:
                    unresolved.remove(token)
                if unresolved:
                    data_file.metadata["_unresolved_tokens"] = unresolved
                else:
                    data_file.metadata.pop("_unresolved_tokens", None)

    def _populate(self) -> None:
        for data_file in self.group.data_files:
            self._add_time_candidate_row(data_file)
            for field in data_file.metadata.get("_metadata_fields", []):
                if field.get("key") in _TIME_REVIEW_KEYS:
                    continue
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._set_file_cell(row, data_file.data_id)
                self.table.setItem(
                    row,
                    1,
                    QtWidgets.QTableWidgetItem(
                        str(field.get("raw_token", ""))
                    ),
                )
                attribute_item = QtWidgets.QTableWidgetItem(
                    str(field.get("key", ""))
                )
                attribute_item.setData(
                    QtCore.Qt.ItemDataRole.UserRole, field.get("key")
                )
                self.table.setItem(row, 2, attribute_item)
                self.table.setItem(
                    row,
                    3,
                    QtWidgets.QTableWidgetItem(str(field.get("value", ""))),
                )
            unresolved = data_file.metadata.get("_unresolved_tokens", [])
            for token in unresolved:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._set_file_cell(row, data_file.data_id)
                self.table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(str(token))
                )
                self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
                self.table.setItem(
                    row, 3, QtWidgets.QTableWidgetItem(str(token))
                )

    def _add_time_candidate_row(self, data_file) -> None:
        candidates = _time_candidates(data_file.metadata)
        if not candidates:
            return
        self.time_table.setVisible(True)
        row = self.time_table.rowCount()
        self.time_table.insertRow(row)
        file_item = QtWidgets.QTableWidgetItem(data_file.data_id or "")
        file_item.setData(QtCore.Qt.ItemDataRole.UserRole, data_file.data_id)
        self.time_table.setItem(row, 0, file_item)
        self.time_table.setCellWidget(
            row,
            1,
            _time_candidate_combo(
                candidates,
                data_file.metadata.get("frame_timestamp_s"),
            ),
        )
        self.time_table.setCellWidget(
            row,
            2,
            _time_candidate_combo(
                candidates,
                data_file.metadata.get("exposure_time_s"),
            ),
        )

    def _apply_time_candidate_selections(self) -> None:
        for row in range(self.time_table.rowCount()):
            file_item = self.time_table.item(row, 0)
            if file_item is None:
                continue
            data_id = file_item.data(QtCore.Qt.ItemDataRole.UserRole)
            data_file = next(
                (
                    item
                    for item in self.group.data_files
                    if item.data_id == data_id
                ),
                None,
            )
            if data_file is None:
                continue
            for column, key in (
                (1, "frame_timestamp_s"),
                (2, "exposure_time_s"),
            ):
                combo = self.time_table.cellWidget(row, column)
                if not isinstance(combo, QtWidgets.QComboBox):
                    continue
                payload = combo.currentData()
                if not isinstance(payload, dict):
                    continue
                value = payload.get("value")
                if value is None:
                    continue
                raw_token = str(payload.get("token") or value)
                data_file.metadata[key] = value
                self._upsert_metadata_field(
                    data_file.metadata,
                    key,
                    value,
                    raw_token,
                    unit="s",
                    confidence=0.95,
                )

    def _set_file_cell(self, row: int, data_id: str | None) -> None:
        file_item = QtWidgets.QTableWidgetItem(data_id or "")
        file_item.setData(QtCore.Qt.ItemDataRole.UserRole, data_id)
        self.table.setItem(row, 0, file_item)

    def _rename_metadata_field(
        self,
        metadata: dict,
        original_key: str | None,
        attribute: str,
    ) -> None:
        if not original_key:
            return
        for field in metadata.get("_metadata_fields", []):
            if field.get("key") == original_key:
                field["key"] = attribute

    def _upsert_metadata_field(
        self,
        metadata: dict,
        key: str,
        value: Any,
        raw_token: str,
        *,
        unit: str | None = None,
        confidence: float = 0.9,
    ) -> None:
        fields = metadata.setdefault("_metadata_fields", [])
        for field in fields:
            if field.get("key") == key:
                field.update(
                    {
                        "value": value,
                        "raw_token": raw_token,
                        "unit": unit,
                        "confidence": confidence,
                    }
                )
                return
        fields.append(
            {
                "key": key,
                "value": value,
                "raw_token": raw_token,
                "unit": unit,
                "confidence": confidence,
            }
        )


def _time_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    values = metadata.get("duration_candidates_s") or []
    if not isinstance(values, list):
        return []
    raw_tokens = _duration_candidate_tokens(metadata)
    candidates: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            continue
        token = raw_tokens[index] if index < len(raw_tokens) else ""
        candidates.append({"value": numeric_value, "token": token})
    return candidates


def _duration_candidate_tokens(metadata: dict[str, Any]) -> list[str]:
    for field in metadata.get("_metadata_fields", []):
        if field.get("key") != "duration_candidates_s":
            continue
        raw_token = str(field.get("raw_token") or "")
        return [token for token in raw_token.split(",") if token]
    return []


def _time_candidate_combo(
    candidates: list[dict[str, Any]],
    selected_value: Any,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    for candidate in candidates:
        value = candidate["value"]
        token = candidate.get("token") or f"{value:g}s"
        combo.addItem(f"{token} ({value:g} s)", candidate)
    selected = _coerce_float(selected_value)
    if selected is not None:
        for index in range(combo.count()):
            payload = combo.itemData(index)
            if isinstance(payload, dict) and payload.get("value") == selected:
                combo.setCurrentIndex(index)
                break
    return combo


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
