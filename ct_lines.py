"""Line geometry + sampling for the CT dashboard.

Line families:
    H_*  = horizontal, offset vertically
    V_*  = vertical,   offset horizontally
    D_*  = main diagonal  "\\"  (top-left to bottom-right)
    E_*  = anti diagonal  "/"   (bottom-left to top-right)

Label convention:
    X_A{i}  : above the center
    X_C0    : through the center
    X_B{i}  : below the center
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from skimage.draw import disk, line as sk_line


# ------------------------------------------------------------------
# ROI mask
# ------------------------------------------------------------------
def circle_mask(shape: tuple[int, int], center: tuple[int, int],
                diameter_px: int) -> np.ndarray:
    """Boolean mask of a filled circle."""
    mask = np.zeros(shape, dtype=bool)
    r = int(diameter_px // 2)
    rr, cc = disk((int(center[0]), int(center[1])), r, shape=shape)
    mask[rr, cc] = True
    return mask


# ------------------------------------------------------------------
# Data holder for one line
# ------------------------------------------------------------------
@dataclass
class LineData:
    label: str
    family: str
    position: tuple[int, int]
    length: int
    rr: np.ndarray
    cc: np.ndarray
    values: list[float]
    selected_indices: list[int]
    average_ct: float
    valid: bool = True


# ------------------------------------------------------------------
# Chord through a circle at perpendicular distance d
# ------------------------------------------------------------------
def chord_length(radius: float, d: float) -> float:
    if d >= radius:
        return 0.0
    return 2.0 * np.sqrt(radius**2 - d**2)


# ------------------------------------------------------------------
# Sample N selected points around the center of a value list
# ------------------------------------------------------------------
def sample_selected(values: list[float], n_points: int) -> list[int]:
    n = len(values)
    if n == 0:
        return []
    half = (n_points - 1) // 2
    center = n // 2
    if center >= half and n - center - 1 >= half:
        return list(range(center - half, center + half + 1))
    left = min(half, center)
    right = min(half, n - center - 1)
    idx = list(range(center - left, center + right + 1))
    while len(idx) < n_points:
        idx.append(-1)
    return idx


# ------------------------------------------------------------------
# Build one line of a family
# ------------------------------------------------------------------
def _make_line(slice_data: np.ndarray, family: str, offset_idx: int,
               center: tuple[int, int], diameter_px: int,
               line_length_ratio: float,
               num_selected_points: int,
               n_lines_total: int) -> LineData:
    rows, cols = slice_data.shape
    cr, cc = int(center[0]), int(center[1])
    radius = diameter_px / 2.0
    spacing = diameter_px / (n_lines_total + 1)

    # label
    if offset_idx < 0:
        label = f"{family}_A{abs(offset_idx)}"
    elif offset_idx == 0:
        label = f"{family}_C0"
    else:
        label = f"{family}_B{offset_idx}"

    # ---------- anchor + perpendicular distance ----------
    if family == 'H':
        line_row = int(round(cr + offset_idx * spacing))
        line_col = cc
        d_perp = abs(line_row - cr)
        anchor = (line_row, line_col)

    elif family == 'V':
        line_row = cr
        line_col = int(round(cc + offset_idx * spacing))
        d_perp = abs(line_col - cc)
        anchor = (line_row, line_col)

    elif family in ('D', 'E'):
        if family == 'D':
            dr, dc = 1.0, 1.0
        else:
            dr, dc = 1.0, -1.0
        n = np.hypot(dr, dc); dr, dc = dr / n, dc / n
        pr, pc = -dc, dr
        if pr > 0:
            pr, pc = -pr, -pc
        line_row = int(round(cr + pr * offset_idx * spacing))
        line_col = int(round(cc + pc * offset_idx * spacing))
        d_perp = abs((line_row - cr) * pr + (line_col - cc) * pc)
        anchor = (line_row, line_col)

    else:
        raise ValueError(f"Unknown family: {family}")

    # ---------- length from chord ----------
    L_chord = chord_length(radius, d_perp)
    if L_chord <= 0:
        return LineData(label=label, family=family, position=anchor,
                        length=0, rr=np.array([], dtype=int),
                        cc=np.array([], dtype=int), values=[],
                        selected_indices=[], average_ct=float('nan'),
                        valid=False)

    L = int(L_chord * line_length_ratio)
    half = L // 2

    # ---------- endpoints + pixel trace ----------
    if family == 'H':
        c0 = max(0, min(cols - 1, line_col - half))
        c1 = max(0, min(cols - 1, line_col + half))
        rr, cc_arr = sk_line(line_row, c0, line_row, c1)

    elif family == 'V':
        r0 = max(0, min(rows - 1, line_row - half))
        r1 = max(0, min(rows - 1, line_row + half))
        rr, cc_arr = sk_line(r0, line_col, r1, line_col)

    else:  # D or E
        r0 = int(round(line_row - dr * half)); c0 = int(round(line_col - dc * half))
        r1 = int(round(line_row + dr * half)); c1 = int(round(line_col + dc * half))
        r0 = max(0, min(rows - 1, r0)); r1 = max(0, min(rows - 1, r1))
        c0 = max(0, min(cols - 1, c0)); c1 = max(0, min(cols - 1, c1))
        rr, cc_arr = sk_line(r0, c0, r1, c1)

    values = [float(slice_data[r, c]) for r, c in zip(rr, cc_arr)]
    avg = float(np.mean(values)) if values else float('nan')
    sel_idx = sample_selected(values, num_selected_points)

    return LineData(label=label, family=family, position=anchor,
                    length=len(values), rr=rr, cc=cc_arr,
                    values=values, selected_indices=sel_idx,
                    average_ct=avg, valid=True)


# ------------------------------------------------------------------
# Build all lines of a family
# ------------------------------------------------------------------
def build_family(slice_data: np.ndarray, family: str,
                 n_lines_above: int, n_lines_below: int,
                 center: tuple[int, int], diameter_px: int,
                 line_length_ratio: float, num_selected_points: int
                 ) -> dict[str, LineData]:
    n_total = 1 + n_lines_above + n_lines_below
    out: dict[str, LineData] = {}

    for i in range(n_lines_above, 0, -1):
        ld = _make_line(slice_data, family, -i, center, diameter_px,
                        line_length_ratio, num_selected_points, n_total)
        out[ld.label] = ld
    ld = _make_line(slice_data, family, 0, center, diameter_px,
                    line_length_ratio, num_selected_points, n_total)
    out[ld.label] = ld
    for i in range(1, n_lines_below + 1):
        ld = _make_line(slice_data, family, i, center, diameter_px,
                        line_length_ratio, num_selected_points, n_total)
        out[ld.label] = ld

    return out


def labels_for_family(family: str, n_lines_above: int, n_lines_below: int) -> list[str]:
    return ([f"{family}_A{i}" for i in range(n_lines_above, 0, -1)] +
            [f"{family}_C0"] +
            [f"{family}_B{i}" for i in range(1, n_lines_below + 1)])


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def stats_dict(values) -> dict:
    """Return avg/mean/median/sd/min/max for a 1-D array-like of numbers."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float('nan') for k in
                ('average', 'mean', 'median', 'sd', 'min', 'max')}
    return {
        'average': float(arr.mean()),
        'mean':    float(arr.mean()),
        'median':  float(np.median(arr)),
        'sd':      float(arr.std(ddof=0)),
        'min':     float(arr.min()),
        'max':     float(arr.max()),
    }