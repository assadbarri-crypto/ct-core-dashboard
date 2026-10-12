"""DICOM folder discovery + volume loading."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pydicom


@dataclass
class ScanMeta:
    folder: Path
    files: list[Path]
    n_slices: int
    rows: int
    cols: int
    pixel_spacing_mm: tuple[float, float] | None
    slice_thickness_mm: float | None
    rescale_slope: float
    rescale_intercept: float


def find_dicom_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.name.startswith("CT.") or f.suffix.lower() == ".dcm":
            files.append(f)
    files.sort()
    return files


def read_dicom_meta(file_path: str | Path) -> dict:
    ds = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
    return {
        "rows": int(getattr(ds, "Rows", 0)),
        "cols": int(getattr(ds, "Columns", 0)),
        "pixel_spacing": tuple(float(x) for x in (getattr(ds, "PixelSpacing", None) or [])),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0)) or None,
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
    }


def scan_folder(folder: str | Path) -> ScanMeta:
    files = find_dicom_files(folder)
    if not files:
        raise RuntimeError(f"No DICOM files found in {folder}")
    m = read_dicom_meta(files[0])
    return ScanMeta(
        folder=Path(folder),
        files=files,
        n_slices=len(files),
        rows=m["rows"],
        cols=m["cols"],
        pixel_spacing_mm=m["pixel_spacing"] or None,
        slice_thickness_mm=m["slice_thickness"],
        rescale_slope=m["rescale_slope"],
        rescale_intercept=m["rescale_intercept"],
    )


def load_slice(file_path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(file_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return arr