"""Dry vs Saturated comparison helpers for the CT dashboard.

Everything here is condition-agnostic: it takes a "pair" of folders
(reference + target) and computes comparisons. It reuses the same
line geometry and filters used in the single-condition dashboard.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from ct_io import scan_folder, load_slice
from ct_lines import build_family, circle_mask, stats_dict


# ------------------------------------------------------------------
# Find partner condition folder
# ------------------------------------------------------------------
def find_partner_folder(condition_folder: Path) -> Path | None:
    """
    Given .../SW1_dry, find the sibling .../SW1_saturated (or vice versa).
    Returns None if no partner exists.
    """
    parent = condition_folder.parent
    if not parent.exists():
        return None

    name = condition_folder.name.lower()
    candidates = [p for p in parent.iterdir() if p.is_dir()]

    def has_dcm(p: Path) -> bool:
        try:
            return any(
                f.is_file() and (f.name.startswith("CT.") or f.suffix.lower() == ".dcm")
                for f in p.iterdir()
            )
        except Exception:
            return False

    candidates = [p for p in candidates if p != condition_folder and has_dcm(p)]
    if not candidates:
        return None

    # If we're in a "dry" folder, look for "sat"; if we're in "sat", look for "dry"
    want = "sat" if "dry" in name else ("dry" if "sat" in name else None)
    if want is None:
        return None

    for p in candidates:
        if want in p.name.lower():
            return p
    return None


# ------------------------------------------------------------------
# Load the whole ROI volume once for a folder
# ------------------------------------------------------------------
def load_roi_volume(folder: str | Path, center: tuple[int, int],
                    diameter: int, n_slices: int,
                    downsample: int = 1) -> np.ndarray:
    """Return a (n_slices, H, W) array cropped? No — full slices stacked.
    For comparison we want full slices so we can apply ROI or line later.
    """
    m = scan_folder(folder)
    imgs = []
    for f in m.files[:n_slices]:
        imgs.append(load_slice(str(f)))
    vol = np.stack(imgs, axis=0)          # (Z, Y, X)
    if downsample > 1:
        vol = vol[::downsample]
    return vol


# ------------------------------------------------------------------
# Per-slice ROI statistics (min / mean / max / std)
# ------------------------------------------------------------------
def per_slice_roi_stats(vol: np.ndarray,
                        center: tuple[int, int],
                        diameter: int,
                        lo: float, hi: float,
                        slice_numbers: list[int] | None = None) -> dict:
    """
    vol: (Z, Y, X). Returns dict with slices, mean, min, max, std arrays.
    """
    Z, Y, X = vol.shape
    mk = circle_mask((Y, X), center, diameter)

    slices, means, mins, maxs, stds = [], [], [], [], []
    for i in range(Z):
        v = vol[i][mk]
        v = v[np.isfinite(v)]
        v = v[(v >= lo) & (v <= hi)]
        slices.append(slice_numbers[i] if slice_numbers else i + 1)
        if v.size:
            means.append(float(v.mean()))
            mins.append(float(v.min()))
            maxs.append(float(v.max()))
            stds.append(float(v.std(ddof=0)))
        else:
            means.append(np.nan); mins.append(np.nan)
            maxs.append(np.nan); stds.append(np.nan)
    return {'slices': slices, 'mean': means, 'min': mins,
            'max': maxs, 'std': stds}


# ------------------------------------------------------------------
# Per-slice, per-line selected-point matrices for one folder
# ------------------------------------------------------------------
def per_slice_line_matrices(vol: np.ndarray,
                            center: tuple[int, int],
                            diameter: int,
                            line_length_ratio: float,
                            num_selected_points: int,
                            num_lines_above: int, num_lines_below: int,
                            num_lines_left: int,  num_lines_right: int,
                            num_diag_above: int,  num_diag_below: int,
                            h_label: str, v_label: str,
                            d_label: str, e_label: str,
                            lo: float, hi: float,
                            slice_numbers: list[int] | None = None) -> dict:
    """
    Returns {family: (Z x P) matrix of CT values at 21 selected points}.
    """
    Z = vol.shape[0]
    P = int(num_selected_points)
    fams = ['H', 'V', 'D', 'E']
    labels = {'H': h_label, 'V': v_label, 'D': d_label, 'E': e_label}
    M = {fam: np.full((Z, P), np.nan) for fam in fams}

    for i in range(Z):
        img = vol[i]
        h_fam = build_family(img, 'H', num_lines_above, num_lines_below,
                             center, diameter, line_length_ratio, P)
        v_fam = build_family(img, 'V', num_lines_left, num_lines_right,
                             center, diameter, line_length_ratio, P)
        d_fam = build_family(img, 'D', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, P)
        e_fam = build_family(img, 'E', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, P)

        for fam, fam_data, lbl in (('H', h_fam, h_label),
                                   ('V', v_fam, v_label),
                                   ('D', d_fam, d_label),
                                   ('E', e_fam, e_label)):
            if lbl in fam_data and fam_data[lbl].valid:
                vals = np.asarray(fam_data[lbl].values, dtype=float)
                sel_idx = fam_data[lbl].selected_indices
                for j, idx in enumerate(sel_idx):
                    if 0 <= idx < len(vals) and j < P:
                        v = vals[idx]
                        if np.isfinite(v) and lo <= v <= hi:
                            M[fam][i, j] = v
    return {'matrices': M, 'labels': labels, 'num_points': P,
            'slices': slice_numbers if slice_numbers else list(range(1, Z + 1))}