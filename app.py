"""CT Core Sample Dashboard — Phase 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8
Run:  streamlit run app.py
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import ndimage as ndi

from ct_io import scan_folder, load_slice
from ct_metadata import parse_general_data, find_sample
from ct_lines import (
    build_family, labels_for_family, circle_mask, stats_dict,
)
from ct_compare import (
    find_partner_folder, load_roi_volume,
    per_slice_roi_stats, per_slice_line_matrices,
)

import plotly.io as pio
pio.templates.default = "plotly_white"


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
with open(HERE / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ACTIVE_SAMPLE = CFG["active_sample"]
DEFAULTS = CFG["defaults"]
TRIMS = CFG.get("trims", {})
DEFAULT_SET1 = CFG.get("set1", "")
DEFAULT_SET2 = CFG.get("set2", "")

# Auto-detect root based on this file's location, so the project works
# from any folder as long as the layout is preserved:
#   <root>/<sample>/CT_Dashboard/app.py
_AUTO_ROOT = HERE.parent.parent
_AUTO_GEN = _AUTO_ROOT / "General_data.txt"

ROOT = Path(CFG["root"]) if CFG.get("root") else _AUTO_ROOT
GEN_DATA = Path(CFG["general_data_file"]) if CFG.get("general_data_file") else _AUTO_GEN

# ---- Cloud-layout shim --------------------------------------------
# If CFG["set1"] points to a real folder containing DICOMs, treat
# set1 / set2 as the condition folders directly and bypass the
# <root>/<sample>/<condition>/ lookup used locally.
def _looks_like_condition_folder(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        return any(
            f.is_file() and (f.name.startswith("CT.") or f.suffix.lower() == ".dcm")
            for f in p.iterdir()
        )
    except Exception:
        return False

_DIRECT_SET1 = Path(CFG.get("set1", "") or "")
_DIRECT_SET2 = Path(CFG.get("set2", "") or "")
_USE_DIRECT_LAYOUT = _looks_like_condition_folder(_DIRECT_SET1)

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
st.set_page_config(page_title="CT Core Dashboard", layout="wide")
st.title("CT Core Sample Dashboard")
st.caption(f"Root: {ROOT}   |   Active sample: {ACTIVE_SAMPLE}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def apply_filter(arr, lo: float, hi: float):
    a = np.asarray(arr, dtype=float).ravel()
    a = np.isfinite(a)
    return np.asarray(arr, dtype=float).ravel()[a][
        (np.asarray(arr, dtype=float).ravel()[a] >= lo) &
        (np.asarray(arr, dtype=float).ravel()[a] <= hi)]


def list_sample_folders(root: Path):
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_condition_folders(sample_folder: Path):
    if not sample_folder.exists():
        return []
    keep = []
    for p in sample_folder.iterdir():
        if not p.is_dir() or p.name == "CT_Dashboard":
            continue
        try:
            has_dcm = any(
                f.is_file() and (f.name.startswith("CT.") or f.suffix.lower() == ".dcm")
                for f in p.iterdir()
            )
            if has_dcm:
                keep.append(p)
        except Exception:
            pass
    return sorted(keep)


def detect_center_and_diameter(slice_img, threshold=1300.0):
    mask = slice_img > threshold
    if mask.sum() == 0:
        h, w = slice_img.shape
        return h // 2, w // 2, 0
    labels, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = (labels == largest)
    mask = ndi.binary_fill_holes(mask)
    rows, cols = np.where(mask)
    cr = int(round(rows.mean()))
    cc = int(round(cols.mean()))
    diameter = int(round(2 * np.sqrt(mask.sum() / np.pi)))
    return cr, cc, diameter


def trim_slice_indices(n_slices: int, skip_first: int, skip_last: int) -> list[int]:
    """Return 0-based indices of the slices to keep."""
    lo = max(0, int(skip_first))
    hi = max(0, int(skip_last))
    start = lo
    stop = n_slices - hi
    if stop <= start:
        return []
    return list(range(start, stop))

def build_family_labels(family: str, n_per_side: int) -> list[str]:
    """Return the full ordered list of labels for a family.

    Order matches the numbering convention:
        A{n} … A1, C0, B1 … B{n}
    """
    labels = []
    for i in range(n_per_side, 0, -1):
        labels.append(f"{family}_A{i}")
    labels.append(f"{family}_C0")
    for i in range(1, n_per_side + 1):
        labels.append(f"{family}_B{i}")
    return labels

def pool_line_values(slice_img, family: str, labels: list[str],
                     center, diameter, line_length_ratio, num_selected_points,
                     n_per_side):
    """
    Build all requested lines of a family and return their values pooled.

    Returns:
        values  : flat list of all CT values across all selected lines
        all_rr  : numpy array of row indices (concatenated)
        all_cc  : numpy array of col indices (concatenated)
        per_line: dict {label: LineData} for the individual lines
        points_matrix: (n_points, n_lines) array of the sampled points;
                       axis 0 = point index, axis 1 = line
    """
    fam_data = build_family(slice_img, family, n_per_side, n_per_side,
                            center, diameter, line_length_ratio,
                            num_selected_points)
    values = []
    rrs, ccs = [], []
    per_line = {}
    points_cols = []

    for lb in labels:
        if lb not in fam_data or not fam_data[lb].valid:
            continue
        ld = fam_data[lb]
        per_line[lb] = ld
        values.extend(ld.values)
        rrs.append(ld.rr)
        ccs.append(ld.cc)
        # pad selected_values to num_selected_points
        sv = list(ld.values[ld.selected_indices[0]:ld.selected_indices[-1] + 1]) \
             if ld.selected_indices else []
        # ld.selected_indices indexes into ld.values; the sampled 21 points
        # are already contiguous, so we can take them directly
        sel = [ld.values[i] for i in ld.selected_indices if 0 <= i < len(ld.values)]
        while len(sel) < num_selected_points:
            sel.append(np.nan)
        points_cols.append(sel[:num_selected_points])

    if rrs:
        all_rr = np.concatenate(rrs)
        all_cc = np.concatenate(ccs)
    else:
        all_rr = np.array([], dtype=int)
        all_cc = np.array([], dtype=int)

    if points_cols:
        points_matrix = np.array(points_cols).T   # shape (num_points, n_lines)
    else:
        points_matrix = np.full((num_selected_points, 0), np.nan)

    return values, all_rr, all_cc, per_line, points_matrix

# ------------------------------------------------------------------
# Sample info
# ------------------------------------------------------------------
sample_map = {}
try:
    sample_map = parse_general_data(GEN_DATA)
except Exception as e:
    st.warning(f"Could not read General_data.txt: {e}")
info = find_sample(sample_map, ACTIVE_SAMPLE) if sample_map else None


# ------------------------------------------------------------------
# Sidebar: sample + two-set selection
# ------------------------------------------------------------------
st.sidebar.header("Sample selection")


def _idx_or_zero(name, names):
    return names.index(name) if name in names else 0


# ---- Determine sample_folder and condition_names -----------------
if _USE_DIRECT_LAYOUT:
    st.sidebar.caption("Layout: direct (cloud)")

    _direct_folders = []
    for p in [_DIRECT_SET1, _DIRECT_SET2]:
        if p and str(p) and p.exists() and p.is_dir():
            _direct_folders.append(p)

    condition_names = [p.name for p in _direct_folders]
    sample_folder = _direct_folders[0].parent if _direct_folders else Path(".")

    if not condition_names:
        st.error("Direct layout selected but no condition folders found.")
        st.stop()

    st.sidebar.caption(f"Found: {', '.join(condition_names)}")

else:
    sample_dirs = list_sample_folders(ROOT)
    sample_names = [p.name for p in sample_dirs] or [ACTIVE_SAMPLE]
    default_idx = sample_names.index(ACTIVE_SAMPLE) if ACTIVE_SAMPLE in sample_names else 0
    sample_choice = st.sidebar.selectbox("Sample folder", sample_names, index=default_idx)
    sample_folder = ROOT / sample_choice

    condition_dirs = list_condition_folders(sample_folder)
    if not condition_dirs:
        st.error(f"No DICOM sub-folders found under {sample_folder}")
        st.stop()

    condition_names = [p.name for p in condition_dirs]


# ---- Comparison sets (Set 1 required, Set 2 optional) ------------
st.sidebar.markdown("**Comparison sets**")

set1_choice = st.sidebar.selectbox(
    "Set 1 (primary)", condition_names,
    index=_idx_or_zero(DEFAULT_SET1, condition_names),
    key="set1_choice")

_available_for_set2 = [n for n in condition_names if n != set1_choice]
_default_enable = bool(DEFAULT_SET2) and (DEFAULT_SET2 in _available_for_set2)

enable_set2 = st.sidebar.checkbox(
    "Compare with a second set",
    value=_default_enable,
    help="Uncheck if this sample has only one condition.",
    key="enable_set2")

if enable_set2 and _available_for_set2:
    _default_set2_idx = (
        _available_for_set2.index(DEFAULT_SET2)
        if DEFAULT_SET2 in _available_for_set2 else 0
    )
    set2_choice = st.sidebar.selectbox(
        "Set 2 (comparison)", _available_for_set2,
        index=_default_set2_idx,
        key="set2_choice")
    set1_folder = sample_folder / set1_choice
    set2_folder = sample_folder / set2_choice
    st.sidebar.caption(f"Set 1: **{set1_choice}**   |   Set 2: **{set2_choice}**")
else:
    if enable_set2 and not _available_for_set2:
        st.sidebar.warning("Only one condition available — Set 2 disabled.")
    set2_choice = None
    set1_folder = sample_folder / set1_choice
    set2_folder = None
    st.sidebar.caption(f"Set 1: **{set1_choice}**   |   Set 2: *disabled*")

# Downstream aliases
dicom_folder = set1_folder
partner_folder = set2_folder

if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()


# ------------------------------------------------------------------
# DICOM metadata
# ------------------------------------------------------------------
try:
    meta = scan_folder(dicom_folder)
except Exception as e:
    st.error(f"Could not read DICOM folder {dicom_folder}: {e}")
    st.stop()


# ------------------------------------------------------------------
# Sidebar: slice trimming
# ------------------------------------------------------------------
st.sidebar.header("Slice trimming")

# figure out which condition key to use for the defaults
_default_skip_first = int(TRIMS.get("skip_first", 0))
_default_skip_last  = int(TRIMS.get("skip_last", 0))

apply_trim = st.sidebar.checkbox(
    "Apply slice trimming",
    value=(_default_skip_first + _default_skip_last) > 0,
    help="Skip the first/last slices of the core, where air corrupts the ROI."
)

skip_first = st.sidebar.number_input(
    "Skip first N slices", min_value=0, max_value=max(0, meta.n_slices - 2),
    value=_default_skip_first, step=1,
    disabled=not apply_trim,
    help="Enabled only when 'Apply slice trimming' is ticked.")
skip_last = st.sidebar.number_input(
    "Skip last N slices", min_value=0, max_value=max(0, meta.n_slices - 2),
    value=_default_skip_last, step=1,
    disabled=not apply_trim,
    help="Enabled only when 'Apply slice trimming' is ticked.")

if not apply_trim:
    skip_first = 0
    skip_last = 0

kept_indices = trim_slice_indices(meta.n_slices, skip_first, skip_last)
if len(kept_indices) < 3:
    st.sidebar.error("Trimming removed too many slices. Reduce skip_first/skip_last.")
    st.stop()

st.sidebar.caption(
    f"Using slices **{kept_indices[0] + 1} – {kept_indices[-1] + 1}** "
    f"({len(kept_indices)} of {meta.n_slices})"
)


# ------------------------------------------------------------------
# Cached loaders (all take an explicit list of slice indices)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_one(path: str) -> np.ndarray:
    return load_slice(path)


@st.cache_data(show_spinner="Computing statistics over the kept slices …")
def _all_slices_stats(folder: str, indices: tuple[int, ...],
                      center: tuple[int, int], diameter: int,
                      lo: float, hi: float) -> tuple[dict, int, int]:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    rows, cols = m.rows, m.cols
    mk = circle_mask((rows, cols), center, diameter)
    chunks, kept_total, seen_total = [], 0, 0
    for i in indices:
        img = _load(str(m.files[i]))
        vals = img[mk]
        seen_total += vals.size
        v = vals[np.isfinite(vals)]
        v = v[(v >= lo) & (v <= hi)]
        kept_total += v.size
        if v.size:
            chunks.append(v)
    if not chunks:
        return stats_dict([]), 0, seen_total
    return stats_dict(np.concatenate(chunks)), kept_total, seen_total


@st.cache_data(show_spinner="Computing per-slice statistics …")
def _per_slice_stats(folder: str, indices: tuple[int, ...],
                     center: tuple[int, int], diameter: int,
                     lo: float, hi: float) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    rows, cols = m.rows, m.cols
    mk = circle_mask((rows, cols), center, diameter)
    slices, mins, means, maxs, stds = [], [], [], [], []
    for i in indices:
        img = _load(str(m.files[i]))
        v = img[mk]
        v = v[np.isfinite(v)]
        v = v[(v >= lo) & (v <= hi)]
        slices.append(i + 1)
        if v.size:
            mins.append(float(v.min())); means.append(float(v.mean()))
            maxs.append(float(v.max())); stds.append(float(v.std(ddof=0)))
        else:
            mins.append(np.nan); means.append(np.nan)
            maxs.append(np.nan); stds.append(np.nan)
    return {'slices': slices, 'min': mins, 'mean': means, 'max': maxs, 'std': stds}


@st.cache_data(show_spinner="Computing per-slice stats for the selected lines …")
def _per_slice_line_stats(folder: str, indices: tuple[int, ...],
                          center: tuple[int, int], diameter: int,
                          lo: float, hi: float,
                          line_length_ratio: float, num_selected_points: int,
                          num_lines_above: int, num_lines_below: int,
                          num_lines_left: int,  num_lines_right: int,
                          num_diag_above: int,  num_diag_below: int,
                          h_label: str, v_label: str,
                          d_label: str, e_label: str) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    out = {
        'H': {'label': h_label, 'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []},
        'V': {'label': v_label, 'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []},
        'D': {'label': d_label, 'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []},
        'E': {'label': e_label, 'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []},
    }
    for i in indices:
        img = _load(str(m.files[i]))
        h_fam = build_family(img, 'H', num_lines_above, num_lines_below,
                             center, diameter, line_length_ratio, num_selected_points)
        v_fam = build_family(img, 'V', num_lines_left, num_lines_right,
                             center, diameter, line_length_ratio, num_selected_points)
        d_fam = build_family(img, 'D', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, num_selected_points)
        e_fam = build_family(img, 'E', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, num_selected_points)
        for fam, fam_data, label in (('H', h_fam, h_label),
                                     ('V', v_fam, v_label),
                                     ('D', d_fam, d_label),
                                     ('E', e_fam, e_label)):
            entry = out[fam]
            entry['slices'].append(i + 1)
            if label in fam_data and fam_data[label].valid:
                vals = fam_data[label].values
                v = np.asarray(vals, dtype=float)
                v = v[np.isfinite(v)]
                v = v[(v >= lo) & (v <= hi)]
                if v.size:
                    entry['min'].append(float(v.min()))
                    entry['mean'].append(float(v.mean()))
                    entry['max'].append(float(v.max()))
                    entry['std'].append(float(v.std(ddof=0)))
                    continue
            entry['min'].append(np.nan); entry['mean'].append(np.nan)
            entry['max'].append(np.nan); entry['std'].append(np.nan)
    return out

@st.cache_data(show_spinner="Computing per-slice stats for the selected lines …")
def _per_slice_line_stats_multi(folder: str, indices: tuple[int, ...],
                                center: tuple[int, int], diameter: int,
                                lo: float, hi: float,
                                line_length_ratio: float,
                                num_selected_points: int,
                                n_per_side: int,
                                labels: tuple[str, ...]) -> dict:
    """
    Per-slice min/mean/max/std pooled across the selected lines of one
    family. All pixels from all selected lines are combined per slice.
    """
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)

    family = labels[0].split('_')[0]   # 'H', 'V', 'D', 'E'

    out = {'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []}

    for i in indices:
        img = _load(str(m.files[i]))
        fam_data = build_family(img, family, n_per_side, n_per_side,
                                center, diameter, line_length_ratio,
                                num_selected_points)
        pooled = []
        for lb in labels:
            if lb in fam_data and fam_data[lb].valid:
                pooled.extend(fam_data[lb].values)

        v = np.asarray(pooled, dtype=float)
        v = v[np.isfinite(v)]
        v = v[(v >= lo) & (v <= hi)]

        out['slices'].append(i + 1)
        if v.size:
            out['min'].append(float(v.min()))
            out['mean'].append(float(v.mean()))
            out['max'].append(float(v.max()))
            out['std'].append(float(v.std(ddof=0)))
        else:
            out['min'].append(np.nan)
            out['mean'].append(np.nan)
            out['max'].append(np.nan)
            out['std'].append(np.nan)

    return out

@st.cache_data(show_spinner="Gathering 3D data for the selected lines …")
def _per_slice_line_points(folder: str, indices: tuple[int, ...],
                           center: tuple[int, int], diameter: int,
                           lo: float, hi: float,
                           line_length_ratio: float, num_selected_points: int,
                           num_lines_above: int, num_lines_below: int,
                           num_lines_left: int,  num_lines_right: int,
                           num_diag_above: int,  num_diag_below: int,
                           h_label: str, v_label: str,
                           d_label: str, e_label: str) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    P = int(num_selected_points)
    fams = ['H', 'V', 'D', 'E']
    labels = {'H': h_label, 'V': v_label, 'D': d_label, 'E': e_label}
    matrices = {fam: np.full((len(indices), P), np.nan) for fam in fams}
    slice_numbers = [i + 1 for i in indices]
    for row, i in enumerate(indices):
        img = _load(str(m.files[i]))
        h_fam = build_family(img, 'H', num_lines_above, num_lines_below,
                             center, diameter, line_length_ratio, num_selected_points)
        v_fam = build_family(img, 'V', num_lines_left, num_lines_right,
                             center, diameter, line_length_ratio, num_selected_points)
        d_fam = build_family(img, 'D', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, num_selected_points)
        e_fam = build_family(img, 'E', num_diag_above, num_diag_below,
                             center, diameter, line_length_ratio, num_selected_points)
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
                            matrices[fam][row, j] = v
    return {'slices': slice_numbers, 'labels': labels,
            'matrices': matrices, 'num_points': P}


@st.cache_data(show_spinner="Computing partner-condition per-slice ROI stats …")
def _partner_roi_stats(folder: str, indices: tuple[int, ...],
                       center: tuple[int, int], diameter: int,
                       lo: float, hi: float) -> dict:
    vol = load_roi_volume(folder, center, diameter, max(indices) + 1)
    vol = vol[list(indices)]
    return per_slice_roi_stats(vol, center, diameter, lo, hi,
                               slice_numbers=[i + 1 for i in indices])


@st.cache_data(show_spinner="Computing partner-condition per-line matrices …")
def _partner_line_matrices(folder: str, indices: tuple[int, ...],
                           center: tuple[int, int], diameter: int,
                           lo: float, hi: float,
                           line_length_ratio: float, num_selected_points: int,
                           num_lines_above: int, num_lines_below: int,
                           num_lines_left: int,  num_lines_right: int,
                           num_diag_above: int,  num_diag_below: int,
                           h_label: str, v_label: str,
                           d_label: str, e_label: str) -> dict:
    vol = load_roi_volume(folder, center, diameter, max(indices) + 1)
    vol = vol[list(indices)]
    return per_slice_line_matrices(vol, center, diameter,
                                   line_length_ratio, num_selected_points,
                                   num_lines_above, num_lines_below,
                                   num_lines_left, num_lines_right,
                                   num_diag_above, num_diag_below,
                                   h_label, v_label, d_label, e_label,
                                   lo, hi,
                                   slice_numbers=[i + 1 for i in indices])


# ------------------------------------------------------------------
# Sidebar: ROI
# ------------------------------------------------------------------
st.sidebar.header("Region of interest")

if "center_row" not in st.session_state:
    st.session_state.center_row = int(DEFAULTS["center_px"][0])
if "center_col" not in st.session_state:
    st.session_state.center_col = int(DEFAULTS["center_px"][1])
if "region_diameter" not in st.session_state:
    st.session_state.region_diameter = int(DEFAULTS["region_diameter_px"])

if st.sidebar.button("Auto-detect center & diameter"):
    try:
        mid = kept_indices[len(kept_indices) // 2]
        mid_img = _load_one(str(meta.files[mid]))
        cr, cc, dia = detect_center_and_diameter(
            mid_img, threshold=float(DEFAULTS.get("auto_center_threshold", 1300)))
        if dia > 0:
            st.session_state.center_row = cr
            st.session_state.center_col = cc
            st.session_state.region_diameter = dia
            st.sidebar.success(f"Detected: row={cr}, col={cc}, diameter={dia} px")
        else:
            st.sidebar.warning("No sample body found above the threshold.")
    except Exception as e:
        st.sidebar.error(f"Auto-detect failed: {e}")

region_diameter = st.sidebar.slider(
    "Region diameter (px)",
    min_value=50, max_value=min(meta.rows, meta.cols),
    value=int(st.session_state.region_diameter), step=5)
st.session_state.region_diameter = region_diameter

center_row = st.sidebar.number_input(
    "Center row (px)", 0, meta.rows - 1,
    value=int(st.session_state.center_row), step=1)
center_col = st.sidebar.number_input(
    "Center col (px)", 0, meta.cols - 1,
    value=int(st.session_state.center_col), step=1)
st.session_state.center_row = int(center_row)
st.session_state.center_col = int(center_col)
center = (int(center_row), int(center_col))


# ------------------------------------------------------------------
# Sidebar: line selection (new multi-select UI)
# ------------------------------------------------------------------
st.sidebar.header("Line selection")

# --- one global control: how many lines per side per family ---
n_per_side = st.sidebar.number_input(
    "Lines per side (all families)",
    min_value=1, max_value=10,
    value=int(DEFAULTS.get("num_lines_above", 2)),
    step=1,
    help="Applies equally to H, V, D, E. 2 gives 5 lines per family, "
         "3 gives 7, and so on."
)

# --- build the label lists for each family ---
h_all = build_family_labels('H', n_per_side)
v_all = build_family_labels('V', n_per_side)
d_all = build_family_labels('D', n_per_side)
e_all = build_family_labels('E', n_per_side)

st.sidebar.markdown("---")
st.sidebar.markdown("**Line families**")

# ---- H family ----
h_enabled = st.sidebar.checkbox("Enable H (horizontal)", value=True,
                                key="fam_enable_H")
if h_enabled:
    h_selected = st.sidebar.multiselect(
        "H lines", options=h_all,
        default=[lb for lb in h_all if lb == 'H_C0'],
        key="fam_lines_H")
else:
    h_selected = []
    st.sidebar.caption("H: disabled")

# ---- V family ----
v_enabled = st.sidebar.checkbox("Enable V (vertical)", value=True,
                                key="fam_enable_V")
if v_enabled:
    v_selected = st.sidebar.multiselect(
        "V lines", options=v_all,
        default=[lb for lb in v_all if lb == 'V_C0'],
        key="fam_lines_V")
else:
    v_selected = []
    st.sidebar.caption("V: disabled")

# ---- D family ----
d_enabled = st.sidebar.checkbox("Enable D (diagonal \\)", value=True,
                                key="fam_enable_D")
if d_enabled:
    d_selected = st.sidebar.multiselect(
        "D lines", options=d_all,
        default=[lb for lb in d_all if lb == 'D_C0'],
        key="fam_lines_D")
else:
    d_selected = []
    st.sidebar.caption("D: disabled")

# ---- E family ----
e_enabled = st.sidebar.checkbox("Enable E (diagonal /)", value=True,
                                key="fam_enable_E")
if e_enabled:
    e_selected = st.sidebar.multiselect(
        "E lines", options=e_all,
        default=[lb for lb in e_all if lb == 'E_C0'],
        key="fam_lines_E")
else:
    e_selected = []
    st.sidebar.caption("E: disabled")

# --- summary ---
_chosen_count = len(h_selected) + len(v_selected) + len(d_selected) + len(e_selected)
st.sidebar.caption(
    f"Selected: H={len(h_selected)} | V={len(v_selected)} | "
    f"D={len(d_selected)} | E={len(e_selected)}  ({_chosen_count} total)")

# --- the four single-line dropdowns are still needed for the current
#     downstream code. They default to the center line of each family,
#     and fall back to the first selected line if the center isn't picked.
h_choices = h_all
v_choices = v_all
d_choices = d_all
e_choices = e_all

def _default_index(labels: list[str], fallback: str = 'H_C0'):
    # try the family center first, otherwise the middle of the list
    for i, lb in enumerate(labels):
        if lb == fallback:
            return i
    return len(labels) // 2

h_sel = st.sidebar.selectbox(
    "H line (single — legacy)", h_choices,
    index=_default_index(h_all, 'H_C0'), key="legacy_h_sel")
v_sel = st.sidebar.selectbox(
    "V line (single — legacy)", v_choices,
    index=_default_index(v_all, 'V_C0'), key="legacy_v_sel")
d_sel = st.sidebar.selectbox(
    "D line (single — legacy)", d_choices,
    index=_default_index(d_all, 'D_C0'), key="legacy_d_sel")
e_sel = st.sidebar.selectbox(
    "E line (single — legacy)", e_choices,
    index=_default_index(e_all, 'E_C0'), key="legacy_e_sel")

# --- points per line (unchanged) ---
num_pts = st.sidebar.number_input(
    "Points per line",
    min_value=5, max_value=201,
    value=int(DEFAULTS["num_selected_points"]),
    step=2,
    help="Odd number. Number of evenly-spaced points sampled along each line."
)
if num_pts % 2 == 0:
    num_pts += 1
DEFAULTS["num_selected_points"] = num_pts

# --- propagate n_per_side into DEFAULTS so downstream calls pick it up ---
DEFAULTS["num_lines_above"] = n_per_side
DEFAULTS["num_lines_below"] = n_per_side
DEFAULTS["num_lines_left"] = n_per_side
DEFAULTS["num_lines_right"] = n_per_side
DEFAULTS["num_diag_above"] = n_per_side
DEFAULTS["num_diag_below"] = n_per_side

# ------------------------------------------------------------------
# Sidebar: statistics filter
# ------------------------------------------------------------------
st.sidebar.header("Statistics filter")

use_two_filters = st.sidebar.checkbox(
    "Use separate filter per condition (comparison)", value=False)

if not use_two_filters:
    filter_min = st.sidebar.number_input("Filter min (inclusive)",
        min_value=-10000, max_value=10000, value=1000, step=10)
    filter_max = st.sidebar.number_input("Filter max (inclusive)",
        min_value=-10000, max_value=10000, value=1771, step=10)
    filter_ref_min, filter_ref_max = filter_min, filter_max
    filter_tgt_min, filter_tgt_max = filter_min, filter_max
else:
    st.sidebar.markdown("**Reference filter (dry)**")
    filter_ref_min = st.sidebar.number_input("Ref min", -10000, 10000, 1000, 10, key="ref_min")
    filter_ref_max = st.sidebar.number_input("Ref max", -10000, 10000, 1771, 10, key="ref_max")
    st.sidebar.markdown("**Target filter (saturated)**")
    filter_tgt_min = st.sidebar.number_input("Tgt min", -10000, 10000, 1000, 10, key="tgt_min")
    filter_tgt_max = st.sidebar.number_input("Tgt max", -10000, 10000, 1950, 10, key="tgt_max")
    if "sat" in dicom_folder.name.lower():
        filter_min, filter_max = filter_tgt_min, filter_tgt_max
    else:
        filter_min, filter_max = filter_ref_min, filter_ref_max

if filter_ref_min >= filter_ref_max or filter_tgt_min >= filter_tgt_max:
    st.sidebar.error("Filter min must be less than filter max.")
    st.stop()


# ------------------------------------------------------------------
# Sidebar: display
# ------------------------------------------------------------------
st.sidebar.header("Display")

vm_min = st.sidebar.number_input("Color range — min (VMIN)",
    min_value=-10000, max_value=10000, value=int(DEFAULTS["vmin"]), step=10)
vm_max = st.sidebar.number_input("Color range — max (VMAX)",
    min_value=-10000, max_value=10000, value=int(DEFAULTS["vmax"]), step=10)
if vm_min >= vm_max:
    st.sidebar.error("VMIN must be less than VMAX.")
    st.stop()
vmin, vmax = int(vm_min), int(vm_max)

colormap_options = ["jet", "turbo", "viridis", "plasma", "inferno", "cividis", "gray"]
default_cmap = DEFAULTS.get("colormap", "jet")
cmap_index = colormap_options.index(default_cmap) if default_cmap in colormap_options else 0
colormap = st.sidebar.selectbox("Colormap", colormap_options, index=cmap_index)

st.sidebar.markdown("**Comparison preview only**")
preview_vmin = st.sidebar.number_input("Preview VMIN", -10000, 10000, 1400, 10, key="preview_vmin")
preview_vmax = st.sidebar.number_input("Preview VMAX", -10000, 10000, 1900, 10, key="preview_vmax")
if preview_vmin >= preview_vmax:
    st.sidebar.error("Preview VMIN must be less than Preview VMAX.")
    st.stop()


# ------------------------------------------------------------------
# General info
# ------------------------------------------------------------------
st.subheader("General information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rock type", info.rock_type if info else "—")
c2.metric("Sample ID", info.sample_id if info else sample_choice)
c3.metric("Dimensions (L × D, cm)",
          f"{info.length_cm} × {info.diameter_cm}" if info and
          info.length_cm and info.diameter_cm else "—")
c4.metric("Slices used", f"{len(kept_indices)} / {meta.n_slices}")


# ------------------------------------------------------------------
# Partner folder + saturated center
# ------------------------------------------------------------------

_sat_default = DEFAULTS.get("center_sat_px")
if "center_sat_row" not in st.session_state:
    st.session_state.center_sat_row = int(_sat_default[0] if _sat_default else st.session_state.center_row)
if "center_sat_col" not in st.session_state:
    st.session_state.center_sat_col = int(_sat_default[1] if _sat_default else st.session_state.center_col)

if partner_folder is not None:
    st.sidebar.header("Saturated geometry")
    st.sidebar.caption(f"Partner: {partner_folder.name}")

    if st.sidebar.button("Auto-detect saturated center"):
        try:
            pmeta = scan_folder(str(partner_folder))
            mid = kept_indices[len(kept_indices) // 2]
            mid = min(mid, pmeta.n_slices - 1)
            mid_img = _load_one(str(pmeta.files[mid]))
            cr, cc, dia = detect_center_and_diameter(
                mid_img, threshold=float(DEFAULTS.get("auto_center_threshold", 1300)))
            if cr and cc:
                st.session_state.center_sat_row = cr
                st.session_state.center_sat_col = cc
                st.sidebar.success(f"Sat center: row={cr}, col={cc}")
        except Exception as e:
            st.sidebar.error(f"Auto-detect (sat) failed: {e}")

    center_sat_row = st.sidebar.number_input(
        "Center row (sat)", 0, meta.rows - 1,
        value=int(st.session_state.center_sat_row), step=1, key="center_sat_row_input")
    center_sat_col = st.sidebar.number_input(
        "Center col (sat)", 0, meta.cols - 1,
        value=int(st.session_state.center_sat_col), step=1, key="center_sat_col_input")
    st.session_state.center_sat_row = int(center_sat_row)
    st.session_state.center_sat_col = int(center_sat_col)
    center_sat = (int(center_sat_row), int(center_sat_col))
else:
    center_sat = center


# ------------------------------------------------------------------
# Side-by-side slice preview
# ------------------------------------------------------------------
if partner_folder is not None and set2_folder is not None:
    theta_local = np.linspace(0, 2 * np.pi, 200)
    st.subheader("Slice preview — Dry vs Saturated")

    try:
        partner_meta = scan_folder(str(partner_folder))
    except Exception as e:
        partner_meta = None
        st.warning(f"Could not read partner folder {partner_folder}: {e}")

    if partner_meta is not None:
        # default preview index = middle of the kept range
        _default_idx = kept_indices[len(kept_indices) // 2] + 1
        _preview_idx = st.session_state.get("slice_idx_prev", _default_idx)
        _preview_idx = min(_preview_idx, len(kept_indices) + skip_first)
        if _preview_idx < (skip_first + 1):
            _preview_idx = skip_first + 1

        # clamp to available partner slices
        _preview_idx = min(_preview_idx, partner_meta.n_slices)

        # Set 1 uses the primary center; Set 2 uses the sat center if it's
        # the saturated folder, otherwise the same center.
        center_left  = center
        center_right = center_sat if "sat" in set2_folder.name.lower() else center

        colA, colB = st.columns(2)

        with colA:
            st.caption(f"**Set 1 — {set1_folder.name}** — slice {_preview_idx}")
            imgA = _load_one(str(meta.files[_preview_idx - 1]))
            figA = go.Figure(go.Heatmap(
                z=imgA, colorscale=colormap,
                zmin=preview_vmin, zmax=preview_vmax, showscale=False,
                hovertemplate="row=%{y}<br>col=%{x}<br>CT=%{z:.1f}<extra></extra>"))
            figA.add_trace(go.Scatter(
                x=center_left[1] + (region_diameter / 2) * np.cos(theta_local),
                y=center_left[0] + (region_diameter / 2) * np.sin(theta_local),
                mode='lines', line=dict(color='red', width=2),
                hoverinfo='skip', showlegend=False))
            figA.update_xaxes(range=[0, meta.cols], showticklabels=False)
            figA.update_yaxes(range=[meta.rows, 0], showticklabels=False,
                              scaleanchor='x', scaleratio=1)
            figA.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(figA, use_container_width=True)

        with colB:
            st.caption(f"**Set 2 — {set2_folder.name}** — slice {_preview_idx}")
            imgB = _load_one(str(partner_meta.files[_preview_idx - 1]))
            figB = go.Figure(go.Heatmap(
                z=imgB, colorscale=colormap,
                zmin=preview_vmin, zmax=preview_vmax,
                showscale=True, colorbar=dict(title="CT"),
                hovertemplate="row=%{y}<br>col=%{x}<br>CT=%{z:.1f}<extra></extra>"))
            figB.add_trace(go.Scatter(
                x=center_right[1] + (region_diameter / 2) * np.cos(theta_local),
                y=center_right[0] + (region_diameter / 2) * np.sin(theta_local),
                mode='lines', line=dict(color='red', width=2),
                hoverinfo='skip', showlegend=False))
            figB.update_xaxes(range=[0, meta.cols], showticklabels=False)
            figB.update_yaxes(range=[meta.rows, 0], showticklabels=False,
                              scaleanchor='x', scaleratio=1)
            figB.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(figB, use_container_width=True)

    st.divider()


# ------------------------------------------------------------------
# Slice viewer
# ------------------------------------------------------------------
st.subheader("Slice viewer")

# Slider bounded to the kept range
_slice_min = kept_indices[0] + 1
_slice_max = kept_indices[-1] + 1

s_col1, s_col2 = st.columns([3, 1])
with s_col1:
    slice_idx = st.slider("Slice index (kept range)",
                          _slice_min, _slice_max,
                          min(_slice_max, (_slice_min + _slice_max) // 2))
with s_col2:
    slice_idx = int(st.number_input("Jump to slice #",
                                    _slice_min, _slice_max,
                                    int(slice_idx), step=1))
st.session_state["slice_idx_prev"] = int(slice_idx)

slice_img = _load_one(str(meta.files[slice_idx - 1]))

# Pool the values from all selected lines in each enabled family
_h_vals, _h_rr, _h_cc, _h_per_line, _h_pts = pool_line_values(
    slice_img, 'H', h_selected, center, region_diameter,
    DEFAULTS["line_length_ratio"], DEFAULTS["num_selected_points"], n_per_side)

_v_vals, _v_rr, _v_cc, _v_per_line, _v_pts = pool_line_values(
    slice_img, 'V', v_selected, center, region_diameter,
    DEFAULTS["line_length_ratio"], DEFAULTS["num_selected_points"], n_per_side)

_d_vals, _d_rr, _d_cc, _d_per_line, _d_pts = pool_line_values(
    slice_img, 'D', d_selected, center, region_diameter,
    DEFAULTS["line_length_ratio"], DEFAULTS["num_selected_points"], n_per_side)

_e_vals, _e_rr, _e_cc, _e_per_line, _e_pts = pool_line_values(
    slice_img, 'E', e_selected, center, region_diameter,
    DEFAULTS["line_length_ratio"], DEFAULTS["num_selected_points"], n_per_side)

# Pack them into a structure the rest of the page can read
family_state = {
    'H': {'values': _h_vals, 'rr': _h_rr, 'cc': _h_cc,
          'per_line': _h_per_line, 'points': _h_pts, 'labels': h_selected},
    'V': {'values': _v_vals, 'rr': _v_rr, 'cc': _v_cc,
          'per_line': _v_per_line, 'points': _v_pts, 'labels': v_selected},
    'D': {'values': _d_vals, 'rr': _d_rr, 'cc': _d_cc,
          'per_line': _d_per_line, 'points': _d_pts, 'labels': d_selected},
    'E': {'values': _e_vals, 'rr': _e_rr, 'cc': _e_cc,
          'per_line': _e_per_line, 'points': _e_pts, 'labels': e_selected},
}

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=slice_img, colorscale=colormap, zmin=vmin, zmax=vmax,
    showscale=True, colorbar=dict(title="CT"),
    hovertemplate="row=%{y}<br>col=%{x}<br>CT=%{z:.1f}<extra></extra>"))

theta = np.linspace(0, 2 * np.pi, 200)
r = region_diameter / 2.0
fig.add_trace(go.Scatter(
    x=center[1] + r * np.cos(theta), y=center[0] + r * np.sin(theta),
    mode='lines', line=dict(color='red', width=2),
    name='ROI', hoverinfo='skip'))
fig.add_trace(go.Scatter(
    x=[center[1]], y=[center[0]], mode='markers',
    marker=dict(symbol='star', size=14, color='lime',
                line=dict(color='black', width=1)),
    name='center', hoverinfo='skip'))

line_colors = {'H': 'royalblue', 'V': 'crimson', 'D': 'limegreen', 'E': 'orange'}
for fam, st_fam in family_state.items():
    for lb, ld in st_fam['per_line'].items():
        if not ld.valid:
            continue
        fig.add_trace(go.Scatter(
            x=ld.cc, y=ld.rr, mode='lines',
            line=dict(color=line_colors[fam], width=2),
            name=f"{fam}: {lb}",
            hovertemplate=(f"{fam} {lb}<br>"
                           "row=%{y}<br>col=%{x}<extra></extra>"),
        ))
        if ld.selected_indices:
            sel_r = [ld.rr[i] for i in ld.selected_indices if 0 <= i < len(ld.rr)]
            sel_c = [ld.cc[i] for i in ld.selected_indices if 0 <= i < len(ld.cc)]
            fig.add_trace(go.Scatter(
                x=sel_c, y=sel_r, mode='markers',
                marker=dict(size=4, color='white',
                            line=dict(color=line_colors[fam], width=1)),
                name=f"{lb} pts", showlegend=False, hoverinfo='skip',
            ))

fig.update_xaxes(range=[0, meta.cols], title="col")
fig.update_yaxes(range=[meta.rows, 0], title="row",
                 scaleanchor="x", scaleratio=1)
fig.update_layout(height=750, margin=dict(l=0, r=0, t=20, b=0),
                  legend=dict(orientation='h', yanchor='bottom', y=1.02))
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Slice {slice_idx} / {meta.n_slices}  "
           f"(kept range {_slice_min}–{_slice_max}) — {meta.files[slice_idx-1].name}")


# ------------------------------------------------------------------
# Phase 4 — statistics table
# ------------------------------------------------------------------
st.subheader("Statistics — CT values")
st.caption(f"Filter [{filter_min}, {filter_max}]. "
           f"Slices used: {len(kept_indices)} of {meta.n_slices}.")

mask = circle_mask(slice_img.shape, center, region_diameter)
roi_values = apply_filter(slice_img[mask], filter_min, filter_max)
slice_stats = stats_dict(roi_values)

def _family_stats(fam_key: str):
    fam = family_state.get(fam_key, {})
    vals = fam.get('values', [])
    if not vals:
        return stats_dict([])
    return stats_dict(apply_filter(vals, filter_min, filter_max))

h_stats = _family_stats('H')
v_stats = _family_stats('V')
d_stats = _family_stats('D')
e_stats = _family_stats('E')

all_stats, kept_total, seen_total = _all_slices_stats(
    str(dicom_folder), tuple(kept_indices), center, region_diameter,
    float(filter_min), float(filter_max))


def _fmt(d):
    return {'Mean': d['mean'], 'Median': d['median'], 'SD': d['sd'],
            'Minimum': d['min'], 'Maximum': d['max']}


columns = {
    'All CT Slices':  _fmt(all_stats),
    'Selected Slice': _fmt(slice_stats),
}
if h_selected: columns[f'H Line ({len(h_selected)})'] = _fmt(h_stats)
if v_selected: columns[f'V Line ({len(v_selected)})'] = _fmt(v_stats)
if d_selected: columns[f'D Line ({len(d_selected)})'] = _fmt(d_stats)
if e_selected: columns[f'E Line ({len(e_selected)})'] = _fmt(e_stats)

table = pd.DataFrame(columns).round(2)
# Center + narrow the statistics table
st.markdown(
    """
    <style>
    .stats-table-wrapper {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stats-table {
        border-collapse: collapse;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stats-table th,
    .stats-table td {
        padding: 10px 26px;
        text-align: center;
        border-bottom: 1px solid #e6e6e6;
    }
     /* Column and row names — slightly larger */
    .stats-table th,
    .stats-table td:first-child {
        font-size: 18px;
    }

    /* Values — same or slightly different */
    .stats-table td:not(:first-child) {
        font-size: 18px;
    }
    .stats-table th {
        font-weight: 600;
        color: #444;
        background: #f7f7f7;
        border-bottom: 2px solid #ccc;
    }
    .stats-table td:first-child,
    .stats-table th:first-child {
        text-align: left;
        font-weight: 500;
        color: #333;
    }
    .stats-table {
        --table-font: 20px;
        font-size: var(--table-font);
    }
    .stats-table th,
    .stats-table td:first-child {
        font-size: var(--table-font);
    }
    .stats-table tr:last-child td {
        border-bottom: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Build the HTML table
def _fmt_num(v):
    if v is None:
        return "—"
    try:
        fv = float(v)
        if abs(fv - round(fv)) < 0.005:
            return f"{int(round(fv)):,}"
        return f"{fv:,.2f}"
    except Exception:
        return str(v)

_html_rows = ""
for metric in ('Mean', 'Median', 'SD', 'Minimum', 'Maximum'):
    _html_rows += "<tr>"
    _html_rows += f"<td>{metric}</td>"
    for col in table.columns:
        _html_rows += f"<td>{_fmt_num(table.loc[metric, col])}</td>"
    _html_rows += "</tr>"

_html_header = "<tr><th></th>" + "".join(
    f"<th>{c}</th>" for c in table.columns) + "</tr>"

_html = (
    '<div class="stats-table-wrapper">'
    '<table class="stats-table">'
    f'<thead>{_html_header}</thead>'
    f'<tbody>{_html_rows}</tbody>'
    '</table>'
    '</div>'
)
st.markdown(_html, unsafe_allow_html=True)

pct = 100.0 * kept_total / seen_total if seen_total else 0.0
st.caption(f"Pixels kept: {kept_total:,} of {seen_total:,} ({pct:.1f}%)")


# ------------------------------------------------------------------
# Phase 5 — Min/Mean/Max across slices
# ------------------------------------------------------------------
st.subheader("Min / Mean / Max trend across slices")
st.caption(f"Slices used: {len(kept_indices)} of {meta.n_slices}. "
           f"Filter [{filter_min}, {filter_max}].")

perslice = _per_slice_stats(str(dicom_folder), tuple(kept_indices),
                            center, region_diameter,
                            float(filter_min), float(filter_max))

pfig = go.Figure()
pfig.add_trace(go.Scatter(
    x=perslice['slices'], y=perslice['mean'], mode='lines+markers', name='Mean',
    line=dict(color='royalblue', width=2), marker=dict(size=4),
    hovertemplate="Slice %{x}<br>Mean=%{y:.2f}<extra></extra>"))
pfig.add_trace(go.Scatter(
    x=perslice['slices'], y=perslice['min'], mode='lines', name='Min',
    line=dict(color='crimson', width=1.5),
    hovertemplate="Slice %{x}<br>Min=%{y:.2f}<extra></extra>"))
pfig.add_trace(go.Scatter(
    x=perslice['slices'], y=perslice['max'], mode='lines', name='Max',
    line=dict(color='darkorange', width=1.5),
    hovertemplate="Slice %{x}<br>Max=%{y:.2f}<extra></extra>"))
pfig.add_trace(go.Scatter(
    x=perslice['slices'], y=perslice['std'], mode='lines',
    name='Std (right axis)',
    line=dict(color='dimgray', width=1.4, dash='dot'), yaxis='y2',
    hovertemplate="Slice %{x}<br>Std=%{y:.2f}<extra></extra>"))
pfig.add_vline(x=slice_idx, line=dict(color='green', width=2, dash='dash'),
               annotation_text=f"slice {slice_idx}",
               annotation_position="top right")
pfig.update_layout(
    height=520, margin=dict(l=20, r=20, t=60, b=20), font=dict(size=14),
    xaxis=dict(title="Slice number", title_font=dict(size=15), tickfont=dict(size=12)),
    yaxis=dict(title="CT value (Mean / Min / Max)",
               title_font=dict(size=15, color='royalblue'),
               tickfont=dict(size=12, color='royalblue')),
    yaxis2=dict(title="Std (right axis)",
                title_font=dict(size=15, color='dimgray'),
                tickfont=dict(size=12, color='dimgray'),
                overlaying='y', side='right', showgrid=False,
                range=[0, max(perslice['std']) * 1.2]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0,
                font=dict(size=13), bgcolor='rgba(255,255,255,0.7)',
                bordercolor='lightgray', borderwidth=1),
    hovermode='x unified')
st.plotly_chart(pfig, use_container_width=True)


# ------------------------------------------------------------------
# Phase 6 — Min/Mean/Max per family across slices (shared y-axis)
# ------------------------------------------------------------------
st.subheader("Min / Mean / Max per line family across slices")
st.caption(f"Slices used: {len(kept_indices)} of {meta.n_slices}. "
           f"Filter [{filter_min}, {filter_max}]. "
           "Only enabled families are shown. All family plots share the "
           "same y-axis range so they can be compared directly.")

family_titles = {
    'H': f"H ({len(h_selected)} lines)",
    'V': f"V ({len(v_selected)} lines)",
    'D': f"D ({len(d_selected)} lines)",
    'E': f"E ({len(e_selected)} lines)",
}
family_colors = {'H': 'royalblue', 'V': 'crimson',
                 'D': 'limegreen', 'E': 'darkorange'}

family_selections = {
    'H': h_selected, 'V': v_selected, 'D': d_selected, 'E': e_selected,
}

# ---- first pass: gather per-family stats so we can share the y-range ----
_fam_stats = {}
for fam in ('H', 'V', 'D', 'E'):
    labels_for_fam = family_selections[fam]
    if not labels_for_fam:
        continue
    _fam_stats[fam] = _per_slice_line_stats_multi(
        str(dicom_folder), tuple(kept_indices), center, region_diameter,
        float(filter_min), float(filter_max),
        float(DEFAULTS["line_length_ratio"]),
        int(DEFAULTS["num_selected_points"]),
        n_per_side,
        labels_for_fam,
    )

# ---- shared y-limits across all families ----
_all_min, _all_max, _all_std_max = [], [], 0.0
for ls in _fam_stats.values():
    for v in ls['min']:
        if np.isfinite(v): _all_min.append(v)
    for v in ls['max']:
        if np.isfinite(v): _all_max.append(v)
    for v in ls['std']:
        if np.isfinite(v): _all_std_max = max(_all_std_max, v)

if _all_min and _all_max:
    y_lo = min(_all_min)
    y_hi = max(_all_max)
    pad = 0.02 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    y_lo -= pad
    y_hi += pad
    std_hi = _all_std_max * 1.15 if _all_std_max > 0 else 1.0
else:
    y_lo, y_hi, std_hi = 0, 1, 1

# ---- second pass: draw all four plots with the same limits ----
for fam in ('H', 'V', 'D', 'E'):
    if fam not in _fam_stats:
        continue
    ls = _fam_stats[fam]
    if not any(np.isfinite(x) for x in ls['mean']):
        st.info(f"No valid data for {family_titles[fam]}.")
        continue

    lfig = go.Figure()
    lfig.add_trace(go.Scatter(
        x=ls['slices'], y=ls['mean'], mode='lines+markers', name='Mean',
        line=dict(color=family_colors[fam], width=2), marker=dict(size=4),
        hovertemplate="Slice %{x}<br>Mean=%{y:.2f}<extra></extra>"))
    lfig.add_trace(go.Scatter(
        x=ls['slices'], y=ls['min'], mode='lines', name='Min',
        line=dict(color='black', width=1.2, dash='dot'),
        hovertemplate="Slice %{x}<br>Min=%{y:.2f}<extra></extra>"))
    lfig.add_trace(go.Scatter(
        x=ls['slices'], y=ls['max'], mode='lines', name='Max',
        line=dict(color='red', width=1.2, dash='dot'),
        hovertemplate="Slice %{x}<br>Max=%{y:.2f}<extra></extra>"))
    lfig.add_trace(go.Scatter(
        x=ls['slices'], y=ls['std'], mode='lines', name='Std (right axis)',
        line=dict(color='dimgray', width=1.2, dash='dot'), yaxis='y2',
        hovertemplate="Slice %{x}<br>Std=%{y:.2f}<extra></extra>"))

    lfig.add_vline(x=slice_idx, line=dict(color='green', width=2, dash='dash'),
                   annotation_text=f"slice {slice_idx}",
                   annotation_position="top right")

    lfig.update_layout(
        height=460, margin=dict(l=20, r=20, t=70, b=20),
        title=dict(text=family_titles[fam], x=0.0, xanchor='left',
                   y=0.98, yanchor='top',
                   font=dict(size=18, color=family_colors[fam])),
        font=dict(size=14),
        xaxis=dict(title="Slice number", title_font=dict(size=15),
                   tickfont=dict(size=12)),
        yaxis=dict(title="CT value (Mean / Min / Max)",
                   title_font=dict(size=15, color=family_colors[fam]),
                   tickfont=dict(size=12),
                   range=[y_lo, y_hi]),          # <-- shared
        yaxis2=dict(title="Std (right axis)",
                    title_font=dict(size=15, color='dimgray'),
                    tickfont=dict(size=12, color='dimgray'),
                    overlaying='y', side='right', showgrid=False,
                    range=[0, std_hi]),          # <-- shared
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='right', x=1.0,
                    font=dict(size=13), bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='lightgray', borderwidth=1),
        hovermode='x unified')
    st.plotly_chart(lfig, use_container_width=True)

# ------------------------------------------------------------------
# Phase 6b — Per-slice deviation from the global (whole-ROI) Min/Mean/Max
# ------------------------------------------------------------------
st.subheader("Per-slice deviation from global Min / Mean / Max")
st.caption(
    "Three curves per dataset. For each slice i and each metric M (Min, Mean, Max): "
    "curve(i) = M_global − M_slice(i). "
    "Positive = this slice is below the global value; zero = equal to it. "
    "Computed over the whole ROI circle, kept slices only. "
    "The two figures share the same y-range so dry and saturated can be compared."
)

@st.cache_data(show_spinner="Computing per-slice deviations …")
def _per_slice_deviation(folder: str, indices: tuple[int, ...],
                         center: tuple[int, int], diameter: int,
                         lo: float, hi: float) -> dict:
    """Return per-slice min/mean/max plus the global values across the whole volume."""
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    rows, cols = m.rows, m.cols
    mk = circle_mask((rows, cols), center, diameter)

    slices, per_min, per_mean, per_max = [], [], [], []
    all_vals = []
    for i in indices:
        img = _load(str(m.files[i]))
        v = img[mk]
        v = v[np.isfinite(v)]
        v = v[(v >= lo) & (v <= hi)]
        slices.append(i + 1)
        if v.size:
            per_min.append(float(v.min()))
            per_mean.append(float(v.mean()))
            per_max.append(float(v.max()))
            all_vals.append(v)
        else:
            per_min.append(np.nan); per_mean.append(np.nan); per_max.append(np.nan)

    if all_vals:
        flat = np.concatenate(all_vals)
        g_min = float(flat.min())
        g_mean = float(flat.mean())
        g_max = float(flat.max())
    else:
        g_min = g_mean = g_max = np.nan

    return {
        'slices': slices,
        'per_min': per_min, 'per_mean': per_mean, 'per_max': per_max,
        'g_min': g_min, 'g_mean': g_mean, 'g_max': g_max,
    }


# Only run this section if there is a partner folder (dry + sat both available)
if partner_folder is not None:
    cur_is_sat = "sat" in dicom_folder.name.lower()
    dry_folder = partner_folder if cur_is_sat else dicom_folder
    sat_folder = dicom_folder if cur_is_sat else partner_folder

    dry_dev = _per_slice_deviation(
        str(dry_folder), tuple(kept_indices), center, region_diameter,
        float(filter_min), float(filter_max))
    sat_dev = _per_slice_deviation(
        str(sat_folder), tuple(kept_indices), center, region_diameter,
        float(filter_min), float(filter_max))

    # --- shared y-range across both figures ---
    _dev_vals = []
    for dev in (dry_dev, sat_dev):
        for k, g in (('per_max', dev['g_max']),
                     ('per_mean', dev['g_mean']),
                     ('per_min', dev['g_min'])):
            for v in dev[k]:
                if np.isfinite(v) and np.isfinite(g):
                    _dev_vals.append(g - v)
    if _dev_vals:
        _dlo = min(_dev_vals); _dhi = max(_dev_vals)
        _dpad = 0.05 * (_dhi - _dlo) if _dhi > _dlo else 1.0
        _dlo -= _dpad; _dhi += _dpad
    else:
        _dlo, _dhi = -1, 1

    def _dev_figure(dev: dict, title: str) -> go.Figure:
        fig = go.Figure()

        # per-slice deviation curves
        d_max  = [dev['g_max']  - v for v in dev['per_max']]
        d_mean = [dev['g_mean'] - v for v in dev['per_mean']]
        d_min  = [dev['g_min']  - v for v in dev['per_min']]

        fig.add_trace(go.Scatter(
            x=dev['slices'], y=d_max, mode='lines+markers', name='Max (global − slice)',
            line=dict(color='crimson', width=1.6), marker=dict(size=3),
            hovertemplate="Slice %{x}<br>Δ=%{y:.2f}<extra>Max</extra>"))
        fig.add_trace(go.Scatter(
            x=dev['slices'], y=d_mean, mode='lines+markers', name='Mean (global − slice)',
            line=dict(color='royalblue', width=1.6), marker=dict(size=3),
            yaxis='y2',
            hovertemplate="Slice %{x}<br>Δ=%{y:.2f}<extra>Mean</extra>"))
        fig.add_trace(go.Scatter(
            x=dev['slices'], y=d_min, mode='lines+markers', name='Min (global − slice)',
            line=dict(color='darkgreen', width=1.6), marker=dict(size=3),
            hovertemplate="Slice %{x}<br>Δ=%{y:.2f}<extra>Min</extra>"))

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

        fig.add_vline(x=slice_idx, line=dict(color='green', width=2, dash='dash'),
                      annotation_text=f"slice {slice_idx}",
                      annotation_position="top right")

        fig.update_layout(
            height=480, margin=dict(l=20, r=20, t=60, b=20),
            title=dict(text=title, x=0.0, xanchor='left', font=dict(size=16)),
            font=dict(size=13),
            xaxis=dict(title='Slice number'),
            yaxis=dict(title='Max / Min deviation (HU)',
                       range=[_dlo, _dhi]),
            yaxis2=dict(title='Mean deviation (HU)',
                        overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1.0,
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='lightgray', borderwidth=1),
            hovermode='x unified')
        return fig

    st.plotly_chart(
        _dev_figure(dry_dev, f"{dry_folder.name} — per-slice deviation from global"),
        use_container_width=True)
    st.plotly_chart(
        _dev_figure(sat_dev, f"{sat_folder.name} — per-slice deviation from global"),
        use_container_width=True)

    with st.expander("Global reference values"):
        st.write({
            dry_folder.name: {"Min": dry_dev['g_min'],
                              "Mean": dry_dev['g_mean'],
                              "Max": dry_dev['g_max']},
            sat_folder.name: {"Min": sat_dev['g_min'],
                              "Mean": sat_dev['g_mean'],
                              "Max": sat_dev['g_max']},
        })

# ------------------------------------------------------------------
# Phase 7 — Interactive 3D plot (family-aware, multi-line)
# ------------------------------------------------------------------
st.subheader("Interactive 3D — selected lines across slices")
st.caption(f"Slices used: {len(kept_indices)} of {meta.n_slices}. "
           f"Filter [{filter_min}, {filter_max}]. "
           "Only enabled families with at least one line appear below.")

family_order = ['H', 'V', 'D', 'E']
family_colors_3d = {'H': 'RoyalBlue', 'V': 'Crimson',
                    'D': 'LimeGreen', 'E': 'DarkOrange'}

# availability map (enabled AND has at least one line)
family_available = {
    'H': bool(h_selected),
    'V': bool(v_selected),
    'D': bool(d_selected),
    'E': bool(e_selected),
}

# --- checkboxes: disabled if the family is unavailable ---
sel_cols = st.columns(len(family_order))
show_family = {}
for i, fam in enumerate(family_order):
    with sel_cols[i]:
        avail = family_available[fam]
        n_lines = len({'H': h_selected, 'V': v_selected,
                       'D': d_selected, 'E': e_selected}[fam])
        label = f"{fam}  ({n_lines} line{'s' if n_lines != 1 else ''})" \
                if avail else f"{fam}  (disabled)"
        show_family[fam] = st.checkbox(
            label, value=avail, key=f"phase7_show_{fam}",
            disabled=not avail,
        )

# --- gather 3D data per family, only for available families ---
@st.cache_data(show_spinner="Gathering 3D data for the selected lines …")
def _per_slice_family_points(folder: str, indices: tuple[int, ...],
                             center: tuple[int, int], diameter: int,
                             lo: float, hi: float,
                             line_length_ratio: float,
                             num_selected_points: int,
                             n_per_side: int,
                             family: str,
                             labels: tuple[str, ...]) -> dict:
    """
    Per (slice, point) matrix for one family, averaged across all selected
    lines of that family. Missing values filled with NaN.
    """
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    P = int(num_selected_points)
    M = np.full((len(indices), P), np.nan)
    slice_numbers = [i + 1 for i in indices]

    for row, i in enumerate(indices):
        img = _load(str(m.files[i]))
        fam_data = build_family(img, family, n_per_side, n_per_side,
                                center, diameter, line_length_ratio, P)
        stacked = np.full((len(labels), P), np.nan)
        for k, lb in enumerate(labels):
            if lb in fam_data and fam_data[lb].valid:
                vals = np.asarray(fam_data[lb].values, dtype=float)
                sel_idx = fam_data[lb].selected_indices
                for j, idx in enumerate(sel_idx):
                    if 0 <= idx < len(vals) and j < P:
                        v = vals[idx]
                        if np.isfinite(v) and lo <= v <= hi:
                            stacked[k, j] = v
        # average across the selected lines (ignoring NaN)
        M[row, :] = np.nanmean(stacked, axis=0)

    return {'slices': slice_numbers, 'matrix': M, 'num_points': P,
            'labels': labels, 'family': family}

@st.cache_data(show_spinner="Gathering per-line 3D data for export …")
def _per_slice_per_line_points(folder: str, indices: tuple[int, ...],
                               center: tuple[int, int], diameter: int,
                               lo: float, hi: float,
                               line_length_ratio: float,
                               num_selected_points: int,
                               n_per_side: int,
                               family: str,
                               labels: tuple[str, ...]) -> dict:
    """
    Per-line matrices for one family. Unlike _per_slice_family_points,
    this keeps each line separate:
        matrices[label] = (n_slices, num_points)
    """
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    P = int(num_selected_points)
    out = {lb: np.full((len(indices), P), np.nan) for lb in labels}
    slice_numbers = [i + 1 for i in indices]

    for row, i in enumerate(indices):
        img = _load(str(m.files[i]))
        fam_data = build_family(img, family, n_per_side, n_per_side,
                                center, diameter, line_length_ratio, P)
        for lb in labels:
            if lb in fam_data and fam_data[lb].valid:
                vals = np.asarray(fam_data[lb].values, dtype=float)
                sel_idx = fam_data[lb].selected_indices
                for j, idx in enumerate(sel_idx):
                    if 0 <= idx < len(vals) and j < P:
                        v = vals[idx]
                        if np.isfinite(v) and lo <= v <= hi:
                            out[lb][row, j] = v

    return {'matrices': out, 'slices': slice_numbers,
            'labels': labels, 'num_points': P, 'family': family}

# fetch the matrices only for the visible families
z3d_fam = {}
for fam in family_order:
    if not family_available[fam]:
        continue
    labels_fam = tuple({'H': h_selected, 'V': v_selected,
                        'D': d_selected, 'E': e_selected}[fam])
    z3d_fam[fam] = _per_slice_family_points(
        str(dicom_folder), tuple(kept_indices), center, region_diameter,
        float(filter_min), float(filter_max),
        float(DEFAULTS["line_length_ratio"]),
        int(DEFAULTS["num_selected_points"]),
        int(n_per_side), fam, labels_fam,
    )

# --- build the 3D figure ---
fig3d = go.Figure()
point_labels = [f'P{i}' for i in range(int(DEFAULTS["num_selected_points"]))]

for fam in family_order:
    if not family_available[fam]:
        continue
    M = z3d_fam[fam]['matrix']
    n_lines = len(z3d_fam[fam]['labels'])
    name_str = f"{fam} (avg of {n_lines} line{'s' if n_lines != 1 else ''})"
    fig3d.add_trace(go.Surface(
        z=M,
        x=z3d_fam[fam]['slices'],
        y=list(range(M.shape[1])),
        colorscale=[[0, family_colors_3d[fam]], [1, family_colors_3d[fam]]],
        showscale=False,
        visible=show_family[fam],
        name=name_str,
        opacity=0.85,
        hovertemplate=(
            f"<b>{name_str}</b><br>"
            "Slice %{x}<br>Point P%{y}<br>CT=%{z:.1f}<extra></extra>"
        ),
    ))

# colourbar
fig3d.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None], mode='markers',
    marker=dict(size=0, colorscale='Jet',
                cmin=float(filter_min), cmax=float(filter_max),
                showscale=True, colorbar=dict(title="CT", x=1.02)),
    name='CT scale', hoverinfo='skip', showlegend=False,
))

fig3d.update_layout(
    height=800, margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(
        xaxis=dict(title='Slice number', title_font=dict(size=13),
                   tickfont=dict(size=11)),
        yaxis=dict(title='Selected point', title_font=dict(size=13),
                   tickfont=dict(size=11),
                   tickmode='array',
                   tickvals=list(range(len(point_labels))),
                   ticktext=point_labels),
        zaxis=dict(title='CT value', title_font=dict(size=13),
                   range=[float(filter_min), float(filter_max)],
                   tickfont=dict(size=11)),
        camera=dict(eye=dict(x=2, y=-1.2, z=0.8)),
        aspectmode='auto', bgcolor='white',
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0.0,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='lightgray', borderwidth=1),
    hovermode='closest',
)
st.plotly_chart(fig3d, use_container_width=True)


# ------------------------------------------------------------------
# Phase 8 — Dry vs Saturated comparison (family-aware, multi-line)
# ------------------------------------------------------------------
st.divider()
st.subheader("Phase 8 — Set 1 vs Set 2 comparison")

if partner_folder is None or set2_folder is None:
    st.info("No partner folder found next to the current condition.")
else:
    ref_folder = set1_folder
    tgt_folder = set2_folder
    ref_center = center
    tgt_center = center_sat if "sat" in set2_folder.name.lower() else center
    ref_filter = (filter_ref_min, filter_ref_max)
    tgt_filter = (filter_tgt_min, filter_tgt_max)

    st.caption(
        f"**Set 1 — {ref_folder.name}** center={ref_center} "
        f"filter=[{ref_filter[0]}, {ref_filter[1]}]  |  "
        f"**Set 2 — {tgt_folder.name}** center={tgt_center} "
        f"filter=[{tgt_filter[0]}, {tgt_filter[1]}]  |  "
        f"Slices used: {len(kept_indices)} of {meta.n_slices}."
    )

    # ----- 8a / 8b: whole-ROI comparison (unchanged) -----
    ref_roi = _per_slice_stats(str(ref_folder), tuple(kept_indices),
                               ref_center, region_diameter,
                               float(ref_filter[0]), float(ref_filter[1]))
    tgt_roi = _partner_roi_stats(str(tgt_folder), tuple(kept_indices),
                                 tgt_center, region_diameter,
                                 float(tgt_filter[0]), float(tgt_filter[1]))

    n = min(len(ref_roi['slices']), len(tgt_roi['slices']))
    slices = ref_roi['slices'][:n]
    ref_mean = np.array(ref_roi['mean'][:n])
    tgt_mean = np.array(tgt_roi['mean'][:n])
    delta_mean = tgt_mean - ref_mean

    summary_df = pd.DataFrame({
        'Slice': slices,
        'Set 1 Mean': np.round(ref_mean, 2),
        'Set 2 Mean': np.round(tgt_mean, 2),
        'Δ Mean': np.round(delta_mean, 2),
        'Set 1 SD': np.round(ref_roi['std'][:n], 2),
        'Set 2 SD': np.round(tgt_roi['std'][:n], 2),
        'Set 1 Min': np.round(ref_roi['min'][:n], 2),
        'Set 2 Min': np.round(tgt_roi['min'][:n], 2),
        'Set 1 Max': np.round(ref_roi['max'][:n], 2),
        'Set 2 Max': np.round(tgt_roi['max'][:n], 2),
    })
    st.markdown("**8a — Per-slice ROI statistics**")

    # Centered, smaller HTML table — same style as the Phase 4 table
    st.markdown(
        """
        <style>
        .cmp-table-wrapper {
            display: flex; justify-content: center;
            margin-top: 8px; margin-bottom: 12px;
            max-height: 300px; overflow-y: auto;
        }
        .cmp-table {
            border-collapse: collapse;
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .cmp-table th, .cmp-table td {
            padding: 4px 12px;
            text-align: center;
            border-bottom: 1px solid #e6e6e6;
            white-space: nowrap;
        }
        .cmp-table th {
            font-weight: 600;
            background: #f7f7f7;
            border-bottom: 2px solid #ccc;
            position: sticky; top: 0;
        }
        .cmp-table td:first-child {
            text-align: left;
            font-weight: 500;
            background: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _fmt_cmp(v):
        if v is None: return "—"
        try:
            fv = float(v)
            if not np.isfinite(fv): return "—"
            if abs(fv - round(fv)) < 0.005:
                return f"{int(round(fv)):,}"
            return f"{fv:,.2f}"
        except Exception:
            return str(v)

    _cmp_rows = ""
    for _, r in summary_df.iterrows():
        _cmp_rows += "<tr>"
        for i, c in enumerate(summary_df.columns):
            _cmp_rows += f"<td>{_fmt_cmp(r[c])}</td>"
        _cmp_rows += "</tr>"

    _cmp_header = "<tr>" + "".join(
        f"<th>{c}</th>" for c in summary_df.columns) + "</tr>"

    st.markdown(
        f'<div class="cmp-table-wrapper">'
        f'<table class="cmp-table">'
        f'<thead>{_cmp_header}</thead>'
        f'<tbody>{_cmp_rows}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )

    valid = np.isfinite(delta_mean)
    if valid.any():
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Mean ΔCT (whole ROI)",
                  f"{np.nanmean(delta_mean):+.2f}")
        cB.metric("SD of ΔCT", f"{np.nanstd(delta_mean):.2f}")
        cC.metric("Max |ΔCT| slice",
                  f"{np.nanmax(np.abs(delta_mean)):.2f}")
        cD.metric("Slices with |Δ| > 20 HU",
                  f"{int((np.abs(delta_mean) > 20).sum())} / {valid.sum()}")

    st.markdown(f"**8b — ΔCT across slices (whole ROI)  |  "
                f"Δ = Set 2 ({set2_folder.name}) − Set 1 ({set1_folder.name})**")
    figD = go.Figure()
    figD.add_trace(go.Scatter(
        x=slices, y=delta_mean, mode='lines+markers',
        name='Δ Mean (target − reference)',
        line=dict(color='seagreen', width=2), marker=dict(size=4),
        hovertemplate="Slice %{x}<br>ΔCT=%{y:+.2f}<extra></extra>"))
    figD.add_trace(go.Scatter(
        x=slices, y=ref_mean, mode='lines',
        name=f'{ref_folder.name} mean',
        line=dict(color='royalblue', width=1.3, dash='dot'),
        yaxis='y2',
        hovertemplate="Slice %{x}<br>Reference=%{y:.2f}<extra></extra>"))
    figD.add_trace(go.Scatter(
        x=slices, y=tgt_mean, mode='lines',
        name=f'{tgt_folder.name} mean',
        line=dict(color='crimson', width=1.3, dash='dot'),
        yaxis='y2',
        hovertemplate="Slice %{x}<br>Target=%{y:.2f}<extra></extra>"))
    figD.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
    figD.add_vline(x=slice_idx,
                   line=dict(color='green', width=2, dash='dash'),
                   annotation_text=f"slice {slice_idx}",
                   annotation_position="top right")
    figD.update_layout(
        height=480, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title='Slice number'),
        yaxis=dict(title='ΔCT (target − reference)',
                   title_font=dict(color='seagreen')),
        yaxis2=dict(title='Mean CT',
                    overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1.0),
        hovermode='x unified')
    st.plotly_chart(figD, use_container_width=True)

    # ----- 8c / 8d: family-level ΔCT (multi-line aware) -----
    st.markdown(f"**8c — ΔCT per family across slices  |  "
                f"Δ = Set 2 ({set2_folder.name}) − Set 1 ({set1_folder.name})  |  "
                f"avg of the selected lines, 21 points**")

    family_titles = {
        'H': f"H ({len(h_selected)} lines)",
        'V': f"V ({len(v_selected)} lines)",
        'D': f"D ({len(d_selected)} lines)",
        'E': f"E ({len(e_selected)} lines)",
    }

    family_selections = {
        'H': h_selected, 'V': v_selected, 'D': d_selected, 'E': e_selected,
    }

    # ---- first pass: gather all ΔCT curves ----
    _fam_delta = {}
    for fam in ('H', 'V', 'D', 'E'):
        labels_fam = family_selections[fam]
        if not labels_fam:
            continue
        ref_fam = _per_slice_family_points(
            str(ref_folder), tuple(kept_indices), ref_center, region_diameter,
            float(ref_filter[0]), float(ref_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )
        tgt_fam = _per_slice_family_points(
            str(tgt_folder), tuple(kept_indices), tgt_center, region_diameter,
            float(tgt_filter[0]), float(tgt_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )
        Mr = ref_fam['matrix']; Mt = tgt_fam['matrix']
        n_rows = min(Mr.shape[0], Mt.shape[0])
        Md = Mt[:n_rows] - Mr[:n_rows]
        _fam_delta[fam] = {
            'slices': list(ref_fam['slices'][:n_rows]),
            'mean_per_slice': np.nanmean(Md, axis=1),
        }

    # ---- shared y-limits across families ----
    _all_deltas = []
    for d in _fam_delta.values():
        for v in d['mean_per_slice']:
            if np.isfinite(v): _all_deltas.append(v)
    if _all_deltas:
        dl_lo = min(_all_deltas); dl_hi = max(_all_deltas)
        dpad = 0.05 * (dl_hi - dl_lo) if dl_hi > dl_lo else 1.0
        dl_lo -= dpad; dl_hi += dpad
    else:
        dl_lo, dl_hi = -1, 1

    # ---- second pass: draw ----
    for fam in ('H', 'V', 'D', 'E'):
        labels_fam = family_selections[fam]
        if not labels_fam:
            continue

        ref_fam = _per_slice_family_points(
            str(ref_folder), tuple(kept_indices), ref_center, region_diameter,
            float(ref_filter[0]), float(ref_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )
        tgt_fam = _per_slice_family_points(
            str(tgt_folder), tuple(kept_indices), tgt_center, region_diameter,
            float(tgt_filter[0]), float(tgt_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )
        Mr = ref_fam['matrix']; Mt = tgt_fam['matrix']
        n_rows = min(Mr.shape[0], Mt.shape[0])
        Md = Mt[:n_rows] - Mr[:n_rows]
        mean_per_slice = np.nanmean(Md, axis=1)
        slices_list = list(ref_fam['slices'][:n_rows])

        figL = go.Figure()
        figL.add_trace(go.Scatter(
            x=slices_list, y=mean_per_slice,
            mode='lines+markers', name=f'ΔCT along {family_titles[fam]}',
            line=dict(color='seagreen', width=2), marker=dict(size=4),
            hovertemplate="Slice %{x}<br>ΔCT=%{y:+.2f}<extra></extra>"))
        figL.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
        figL.update_layout(
            height=340, margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text=f"ΔCT along {family_titles[fam]}",
                       x=0.0, xanchor='left', font=dict(size=15)),
            xaxis=dict(title='Slice number'),
            yaxis=dict(title='ΔCT (target − reference)',
                       range=[dl_lo, dl_hi]),
            hovermode='x unified')
        st.plotly_chart(figL, use_container_width=True)

    st.markdown(f"**8d-2 — Residual ΔCT heatmaps per family  |  "
                f"Δ = Set 2 − Set 1, minus per-slice mean**")
    st.caption("For each slice, the mean ΔCT across the 21 points is "
               "subtracted. Red = above that slice's average; blue = below.")

    for fam in ('H', 'V', 'D', 'E'):
        labels_fam = family_selections[fam]
        if not labels_fam:
            continue

        ref_fam = _per_slice_family_points(
            str(ref_folder), tuple(kept_indices), ref_center, region_diameter,
            float(ref_filter[0]), float(ref_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )
        tgt_fam = _per_slice_family_points(
            str(tgt_folder), tuple(kept_indices), tgt_center, region_diameter,
            float(tgt_filter[0]), float(tgt_filter[1]),
            float(DEFAULTS["line_length_ratio"]),
            int(DEFAULTS["num_selected_points"]),
            int(n_per_side), fam, tuple(labels_fam),
        )

        Mr = ref_fam['matrix']; Mt = tgt_fam['matrix']
        n_rows = min(Mr.shape[0], Mt.shape[0])
        Md = Mt[:n_rows] - Mr[:n_rows]
        slice_mean = np.nanmean(Md, axis=1, keepdims=True)
        residual = Md - slice_mean

        finite = residual[np.isfinite(residual)]
        if finite.size == 0:
            continue
        lim = float(np.nanpercentile(np.abs(finite), 98)) or 1.0

        figR = go.Figure(go.Heatmap(
            z=residual,
            x=[f'P{i}' for i in range(residual.shape[1])],
            y=list(ref_fam['slices'][:n_rows]),
            colorscale='RdBu_r', zmin=-lim, zmax=lim,
            colorbar=dict(title='ΔCT − slice mean'),
            hovertemplate=("Slice %{y}<br>Point %{x}<br>"
                           "residual=%{z:+.2f} HU<extra></extra>")))
        figR.update_layout(
            height=440, margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text=f"Residual ΔCT — {family_titles[fam]}  "
                            f"(±{lim:.1f} HU)",
                       x=0.0, xanchor='left', font=dict(size=15)),
            xaxis=dict(title='Selected point'),
            yaxis=dict(title='Slice number'))
        st.plotly_chart(figR, use_container_width=True)

    # ----- 8e / 8f: whole-ROI histogram + summary (unchanged) -----
    st.markdown(f"**8e — Distribution of ΔCT (whole ROI volume)  |  "
                f"Δ = Set 2 ({set2_folder.name}) − Set 1 ({set1_folder.name})**")

    @st.cache_data(show_spinner="Computing ΔCT distribution …")
    def _delta_histogram(folderA: str, folderB: str,
                         indices: tuple[int, ...],
                         centerA: tuple[int, int],
                         centerB: tuple[int, int],
                         diameter: int,
                         loA: float, hiA: float,
                         loB: float, hiB: float) -> np.ndarray:
        volA = load_roi_volume(folderA, centerA, diameter,
                               max(indices) + 1)[list(indices)]
        volB = load_roi_volume(folderB, centerB, diameter,
                               max(indices) + 1)[list(indices)]
        Z, Y, X = volA.shape
        mkA = circle_mask((Y, X), centerA, diameter)
        mkB = circle_mask((Y, X), centerB, diameter)
        out = []
        for i in range(Z):
            a = volA[i][mkA]
            b = volB[i][mkB]
            d = b - a
            d = d[np.isfinite(d)]
            out.append(d)
        return np.concatenate(out) if out else np.array([])

    deltas = _delta_histogram(
        str(ref_folder), str(tgt_folder), tuple(kept_indices),
        ref_center, tgt_center, region_diameter,
        float(ref_filter[0]), float(ref_filter[1]),
        float(tgt_filter[0]), float(tgt_filter[1]))

    MAX_HIST_POINTS = 2_000_000
    if deltas.size > MAX_HIST_POINTS:
        rng = np.random.default_rng(42)
        deltas_plot = rng.choice(deltas, MAX_HIST_POINTS, replace=False)
    else:
        deltas_plot = deltas

    if deltas.size:
        CLIP_LO, CLIP_HI = -50, 400
        d_clip = deltas_plot[(deltas_plot >= CLIP_LO) &
                             (deltas_plot <= CLIP_HI)]
        n_total = deltas.size
        n_shown = d_clip.size

        figHist = go.Figure(go.Histogram(
            x=d_clip, nbinsx=180,
            marker=dict(color='teal'),
            hovertemplate="ΔCT=%{x}<br>count=%{y}<extra></extra>"))
        mean_full = float(np.mean(deltas))
        median_full = float(np.median(deltas))
        figHist.add_vline(
            x=mean_full, line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"mean = {mean_full:+.2f}",
            annotation_position="top right")
        figHist.add_vline(
            x=median_full,
            line=dict(color='orange', width=1.5, dash='dot'),
            annotation_text=f"median = {median_full:+.2f}",
            annotation_position="bottom right")
        figHist.update_layout(
            height=420, margin=dict(l=20, r=20, t=40, b=20),
            title=dict(
                text=(f"Distribution of ΔCT = Set 2 − Set 1, clipped to "
                      f"[{CLIP_LO}, {CLIP_HI}] HU; "
                      f"showing {n_shown:,} of {n_total:,} voxels "
                      f"({100*n_shown/n_total:.1f}%)"),
                x=0.0, xanchor='left', font=dict(size=14)),
            xaxis=dict(title='ΔCT (HU)', range=[CLIP_LO, CLIP_HI]),
            yaxis=dict(title='Number of voxels (downsampled)'))
        st.plotly_chart(figHist, use_container_width=True)

        st.markdown(f"**8f — Global summary  |  "
                    f"Δ = Set 2 ({set2_folder.name}) − Set 1 ({set1_folder.name})**")
        summary_metrics = pd.DataFrame({
            'Metric': [
                'Slices used', 'Voxels compared',
                'Mean ΔCT (HU)', 'Median ΔCT (HU)', 'SD of ΔCT (HU)',
                'Min ΔCT (HU)', 'Max ΔCT (HU)',
                'Voxels with |ΔCT| > 20 HU',
                'Voxels with |ΔCT| > 50 HU',
            ],
            'Value': [
                f"{len(kept_indices)} of {meta.n_slices}",
                f"{deltas.size:,}",
                f"{np.mean(deltas):+.2f}",
                f"{np.median(deltas):+.2f}",
                f"{np.std(deltas):.2f}",
                f"{np.min(deltas):+.2f}",
                f"{np.max(deltas):+.2f}",
                f"{int((np.abs(deltas) > 20).sum()):,}  "
                f"({100 * (np.abs(deltas) > 20).mean():.2f}%)",
                f"{int((np.abs(deltas) > 50).sum()):,}  "
                f"({100 * (np.abs(deltas) > 50).mean():.2f}%)",
            ]})
        st.dataframe(summary_metrics, use_container_width=True,
                     hide_index=True)

        # ------------------------------------------------------------------
    # Excel export — one column-group per selected line, per family
    # ------------------------------------------------------------------
    st.markdown("**Export — selected line points across slices**")
    st.caption(
        "One Excel file with up to 8 sheets: set1_H, set1_V, set1_D, set1_E, "
        "set2_H, set2_V, set2_D, set2_E. Each sheet has one column-group per "
        "selected line, e.g. H_C0_P0…P20, H_A1_P0…P20, …"
    )

    try:
        import openpyxl  # noqa: F401
        _HAS_XLSX = True
    except ImportError:
        _HAS_XLSX = False

    if st.button("Prepare export file"):
        if not _HAS_XLSX:
            st.error("Missing `openpyxl`. Run: pip install openpyxl")
        else:
            import io
            P = int(DEFAULTS["num_selected_points"])
            point_cols = [f'P{i}' for i in range(P)]

            # ---- gather per-line matrices for ref & tgt ----
            ref_plp = {}
            tgt_plp = {}
            for fam in ('H', 'V', 'D', 'E'):
                labels_fam = tuple(family_selections[fam])
                if not labels_fam:
                    continue
                ref_plp[fam] = _per_slice_per_line_points(
                    str(ref_folder), tuple(kept_indices), ref_center,
                    region_diameter,
                    float(ref_filter[0]), float(ref_filter[1]),
                    float(DEFAULTS["line_length_ratio"]),
                    int(DEFAULTS["num_selected_points"]),
                    int(n_per_side), fam, labels_fam,
                )
                tgt_plp[fam] = _per_slice_per_line_points(
                    str(tgt_folder), tuple(kept_indices), tgt_center,
                    region_diameter,
                    float(tgt_filter[0]), float(tgt_filter[1]),
                    float(DEFAULTS["line_length_ratio"]),
                    int(DEFAULTS["num_selected_points"]),
                    int(n_per_side), fam, labels_fam,
                )

            def _build_sheet_df(plp_entry):
                """One sheet: rows=slices, columns = Slice + line_P0..line_P20 groups."""
                slices_list = plp_entry['slices']
                labels_list = list(plp_entry['labels'])
                mats = plp_entry['matrices']
                n = min(len(slices_list), *(mats[lb].shape[0] for lb in labels_list))

                cols = {}
                for lb in labels_list:
                    M = mats[lb][:n]
                    for j in range(P):
                        cols[f'{lb}_P{j}'] = M[:, j]

                df = pd.DataFrame(cols)
                df.insert(0, 'Slice', slices_list[:n])
                return df

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as xw:
                # ----- Set 1 sheets -----
                for fam in ('H', 'V', 'D', 'E'):
                    if fam not in ref_plp:
                        continue
                    df = _build_sheet_df(ref_plp[fam])
                    df.to_excel(xw, sheet_name=f"set1_{fam}"[:31], index=False)

                # ----- Set 2 sheets -----
                for fam in ('H', 'V', 'D', 'E'):
                    if fam not in tgt_plp:
                        continue
                    df = _build_sheet_df(tgt_plp[fam])
                    df.to_excel(xw, sheet_name=f"set2_{fam}"[:31], index=False)

            buf.seek(0)
            fname = f"line_points__{ref_folder.name}__vs__{tgt_folder.name}.xlsx"
            st.download_button(
                label=f"⬇  Download {fname}",
                data=buf,
                file_name=fname,
                mime=("application/vnd.openxmlformats-officedocument."
                      "spreadsheetml.sheet"),
            )

st.caption("✅ Phase 8 complete.")