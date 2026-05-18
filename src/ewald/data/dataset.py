"""xarray/dask detector image containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class DetectorImageSet:
    """A lazily loadable detector image collection backed by xarray."""

    dataset: Any
    source_paths: list[Path] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        if "image" not in self.dataset.dims:
            return 1
        return int(self.dataset.sizes["image"])

    def summary(self) -> dict[str, Any]:
        return {
            "image_count": self.image_count,
            "dims": dict(self.dataset.sizes),
            "source_paths": [str(path) for path in self.source_paths],
            "data_vars": list(self.dataset.data_vars),
        }

    @classmethod
    def from_data_array(
        cls,
        data_array: Any,
        *,
        variable_name: str = "intensity",
        source_paths: Iterable[str | Path] = (),
    ) -> "DetectorImageSet":
        import xarray as xr

        if not hasattr(data_array, "to_dataset"):
            data_array = xr.DataArray(data_array)
        dataset = data_array.to_dataset(name=variable_name)
        return cls(
            dataset=dataset,
            source_paths=[Path(path) for path in source_paths],
        )


def open_detector_images(
    paths: Iterable[str | Path],
    *,
    chunks: str | tuple[int, ...] = "auto",
) -> DetectorImageSet:
    """Open detector images as a lazy xarray dataset.

    The first frame is read immediately to infer shape and dtype. Remaining
    images are wrapped as dask delayed tasks so large collections can be
    processed without loading every frame into memory at once.
    """

    import dask.array as da
    import tifffile
    import xarray as xr
    from dask import delayed

    source_paths = [Path(path) for path in paths]
    if not source_paths:
        raise ValueError("At least one detector image path is required.")

    sample = tifffile.imread(source_paths[0])

    @delayed
    def _read(path: Path) -> Any:
        return tifffile.imread(path)

    arrays = [
        da.from_delayed(
            _read(path),
            shape=sample.shape,
            dtype=sample.dtype,
        )
        for path in source_paths
    ]
    stack = da.stack(arrays, axis=0).rechunk(chunks)
    data = xr.DataArray(
        stack,
        dims=("image", "y", "x"),
        coords={"path": ("image", [str(path) for path in source_paths])},
        name="intensity",
    )
    return DetectorImageSet.from_data_array(
        data,
        variable_name="intensity",
        source_paths=source_paths,
    )
