"""CT Core Sample Dashboard — Single Set Demo
Run:  streamlit run app.py
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import matplotlib.colors as mcolors

from ct_io import scan_folder, load_slice
from ct_metadata import parse_general_data, find_sample
from ct_lines import build_family, circle_mask, stats_dict

import plotly.io as pio
pio.templates.default = "plotly_white"

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
with open(HERE / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# --- NEW DEMO MODE LOGIC ---
# If the demo_data folder exists, we assume we are running the Cloud Demo
IS_DEMO = (HERE / "demo_data").exists()

if IS_DEMO:
    st.sidebar.success("✅ Running in CLOUD DEMO Mode")
    
    # 1. Point directly to the specific folders inside demo_data
    ROOT = HERE / "demo_data"
    GEN_DATA = HERE / "General_data.txt"
    
    # 2. Tell the app the "Sample" is the current demo folder itself
    ACTIVE_SAMPLE = "" 
    DEFAULTS = CFG["defaults"]
    DEFAULT_SET1 = "SW1_dry"
    
    # 3. FORCE the scan to look exactly at the data folder
    # This prevents it from looking for a sub-folder inside SW1_dry
    dicom_folder_override = ROOT / "SW1_dry"
else:
    # Local Mode (Your PC)
    ACTIVE_SAMPLE = CFG["active_sample"]
    DEFAULTS = CFG["defaults"]
    DEFAULT_SET1 = CFG.get("set1", "")
    
    _AUTO_ROOT = HERE.parent.parent
    _AUTO_GEN = _AUTO_ROOT / "General_data.txt"
    
    ROOT = Path(CFG["root"]) if CFG.get("root") else _AUTO_ROOT
    GEN_DATA = Path(CFG["general_data_file"]) if CFG.get("general_data_file") else _AUTO_GEN

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def apply_filter(arr, lo: float, hi: float):
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    return a[(a >= lo) & (a <= hi)]

def list_sample_folders(root: Path):
    if not root.exists(): return []
    return sorted([p for p in root.iterdir() if p.is_dir()])

def list_condition_folders(sample_folder: Path):
    if not sample_folder.exists(): return []
    
    # NEW: If the folder itself contains DICOMs, return it directly!
    has_dcm_here = any(f.is_file() and (f.name.startswith("CT.") or f.suffix.lower() == ".dcm") for f in sample_folder.iterdir())
    if has_dcm_here:
        return [sample_folder]
        
    keep = []
    for p in sample_folder.iterdir():
        if not p.is_dir() or p.name == "CT_Dashboard": continue
        try:
            has_dcm = any(f.is_file() and (f.name.startswith("CT.") or f.suffix.lower() == ".dcm") for f in p.iterdir())
            if has_dcm: keep.append(p)
        except Exception: pass
    return sorted(keep)

def build_family_labels(family: str, n_per_side: int) -> list[str]:
    labels = []
    for i in range(n_per_side, 0, -1): labels.append(f"{family}_A{i}")
    labels.append(f"{family}_C0")
    for i in range(1, n_per_side + 1): labels.append(f"{family}_B{i}")
    return labels

@st.cache_data(show_spinner="Computing per-slice statistics …")
def _per_slice_stats(folder: str, indices: tuple[int, ...], center: tuple[int, int], diameter: int, lo: float, hi: float) -> dict:
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
            mins.append(np.nan); means.append(np.nan); maxs.append(np.nan); stds.append(np.nan)
    return {'slices': slices, 'min': mins, 'mean': means, 'max': maxs, 'std': stds}

@st.cache_data(show_spinner="Computing per-slice stats for the selected lines …")
def _per_slice_line_stats(folder: str, indices: tuple[int, ...], center: tuple[int, int], diameter: int, lo: float, hi: float, line_length_ratio: float, num_selected_points: int, n_per_side: int, family: str, label: str) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    out = {'slices': [], 'min': [], 'mean': [], 'max': [], 'std': []}
    for i in indices:
        img = _load(str(m.files[i]))
        fam_data = build_family(img, family, n_per_side, n_per_side, center, diameter, line_length_ratio, num_selected_points)
        out['slices'].append(i + 1)
        if label in fam_data and fam_data[label].valid:
            v = np.asarray(fam_data[label].values, dtype=float)
            v = v[np.isfinite(v)]
            v = v[(v >= lo) & (v <= hi)]
            if v.size:
                out['min'].append(float(v.min())); out['mean'].append(float(v.mean()))
                out['max'].append(float(v.max())); out['std'].append(float(v.std(ddof=0)))
                continue
        out['min'].append(np.nan); out['mean'].append(np.nan); out['max'].append(np.nan); out['std'].append(np.nan)
    return out

@st.cache_data(show_spinner="Gathering 3D data for the selected lines …")
def _per_slice_line_points(folder: str, indices: tuple[int, ...], center: tuple[int, int], diameter: int, lo: float, hi: float, line_length_ratio: float, num_selected_points: int, n_per_side: int, family: str, label: str) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    P = int(num_selected_points)
    M = np.full((len(indices), P), np.nan)
    slice_numbers = [i + 1 for i in indices]
    for row, i in enumerate(indices):
        img = _load(str(m.files[i]))
        fam_data = build_family(img, family, n_per_side, n_per_side, center, diameter, line_length_ratio, num_selected_points)
        if label in fam_data and fam_data[label].valid:
            vals = np.asarray(fam_data[label].values, dtype=float)
            sel_idx = fam_data[label].selected_indices
            for j, idx in enumerate(sel_idx):
                if 0 <= idx < len(vals) and j < P:
                    v = vals[idx]
                    if np.isfinite(v) and lo <= v <= hi: M[row, j] = v
    return {'slices': slice_numbers, 'matrix': M, 'num_points': P, 'label': label, 'family': family}

# ------------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------------
st.set_page_config(page_title="CT Analysis Platform", layout="wide")
st.title("CT Rock Samples' Platform")
st.caption(f"Root: {ROOT}   |   Active sample: {ACTIVE_SAMPLE}")

sample_map = {}
try: sample_map = parse_general_data(GEN_DATA)
except Exception as e: st.warning(f"Could not read General_data.txt: {e}")
info = find_sample(sample_map, ACTIVE_SAMPLE) if sample_map else None

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Sample selection")
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
default_set1 = DEFAULT_SET1 if DEFAULT_SET1 in condition_names else condition_names[0]
set1_idx = condition_names.index(default_set1)
set1 = st.sidebar.selectbox("Set 1 (primary)", condition_names, index=set1_idx)
if IS_DEMO:
    dicom_folder = ROOT / set1
else:
    dicom_folder = sample_folder / set1

if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

try: meta = scan_folder(str(dicom_folder))
except Exception as e:
    st.error(f"Could not read DICOM folder {dicom_folder}: {e}")
    st.stop()

# ROI
st.sidebar.header("Region of interest")
if "center_row" not in st.session_state: st.session_state.center_row = int(DEFAULTS["center_px"][0])
if "center_col" not in st.session_state: st.session_state.center_col = int(DEFAULTS["center_px"][1])
if "region_diameter" not in st.session_state: st.session_state.region_diameter = int(DEFAULTS["region_diameter_px"])

region_diameter = st.sidebar.slider("Region diameter (px)", 50, min(meta.rows, meta.cols), int(st.session_state.region_diameter), 5)
st.session_state.region_diameter = region_diameter

center_row = st.sidebar.number_input("Center row (px)", 0, meta.rows - 1, int(st.session_state.center_row), 1)
center_col = st.sidebar.number_input("Center col (px)", 0, meta.cols - 1, int(st.session_state.center_col), 1)
st.session_state.center_row = int(center_row)
st.session_state.center_col = int(center_col)
center = (int(center_row), int(center_col))

# Line selection (Dropdown for 4 directions only)
st.sidebar.header("Line selection")
n_per_side = 2 # Fixed for demo
h_all = build_family_labels('H', n_per_side)
v_all = build_family_labels('V', n_per_side)
d_all = build_family_labels('D', n_per_side)
e_all = build_family_labels('E', n_per_side)

h_sel = st.sidebar.selectbox("H line (horizontal)", h_all, index=h_all.index('H_C0'))
v_sel = st.sidebar.selectbox("V line (vertical)", v_all, index=v_all.index('V_C0'))
d_sel = st.sidebar.selectbox("D line (diagonal \\)", d_all, index=d_all.index('D_C0'))
e_sel = st.sidebar.selectbox("E line (diagonal /)", e_all, index=e_all.index('E_C0'))

num_pts = st.sidebar.number_input("Points per line", 5, 201, int(DEFAULTS["num_selected_points"]), 2)
if num_pts % 2 == 0: num_pts += 1
DEFAULTS["num_selected_points"] = num_pts

# Statistics filter
st.sidebar.header("Statistics filter")
filter_min = st.sidebar.number_input("Filter min (inclusive)", -10000, 10000, 1000, 10)
filter_max = st.sidebar.number_input("Filter max (inclusive)", -10000, 10000, 1771, 10)
if filter_min >= filter_max:
    st.sidebar.error("Filter min must be less than filter max.")
    st.stop()

# Display
st.sidebar.header("Display")
vmin = st.sidebar.number_input("Color range — min (VMIN)", -10000, 10000, int(DEFAULTS["vmin"]), 10)
vmax = st.sidebar.number_input("Color range — max (VMAX)", -10000, 10000, int(DEFAULTS["vmax"]), 10)
if vmin >= vmax:
    st.sidebar.error("VMIN must be less than VMAX.")
    st.stop()

cmap = st.sidebar.selectbox("Colormap", ["gamma", "jet", "turbo", "viridis", "plasma", "inferno", "cividis", "gray"], index=0)

if cmap == "gamma":
    cmap_to_use = [
        [0.0, 'rgb(0, 0, 128)'],   # Dark Blue
        [0.25, 'rgb(0, 255, 255)'], # Cyan
        [0.5, 'rgb(255, 255, 0)'],  # Yellow
        [0.75, 'rgb(255, 165, 0)'], # Orange
        [1.0, 'rgb(200, 0, 0)']     # Red
    ]
else:
    cmap_to_use = cmap

# ------------------------------------------------------------------
# General Info
# ------------------------------------------------------------------
st.subheader("General information")
c1, c2, c3, c4 = st.columns(4)

# Safely get values with fallbacks for the demo
rock_type = info.rock_type if info else "Berea Sandstone"
sample_id = info.sample_id if info else "SW-1"
length_cm = info.length_cm if info else 30.0
diameter_cm = info.diameter_cm if info else 3.81
total_slices = 307 # Hardcoded total for the full sample

c1.metric("Rock type", rock_type)
c2.metric("Sample ID", sample_id)
c3.metric("Dimensions (L × D, cm)", f"{length_cm} × {diameter_cm}")
c4.metric("Slices used", f"{meta.n_slices} / {total_slices}")

# ------------------------------------------------------------------
# Slice Preview & Viewer
# ------------------------------------------------------------------
st.subheader("Slice preview")
slice_idx = st.slider("Slice index", 1, meta.n_slices, meta.n_slices // 2)

@st.cache_data(show_spinner=False)
def _load_one(path: str) -> np.ndarray: return load_slice(path)

slice_img = _load_one(str(meta.files[slice_idx - 1]))
theta = np.linspace(0, 2 * np.pi, 200)

fig = go.Figure()

# Heatmap with colorbar settings directly inside
fig.add_trace(go.Heatmap(
    z=slice_img, 
    colorscale=cmap_to_use, 
    zmin=vmin, 
    zmax=vmax, 
    showscale=True,
    colorbar=dict(
        title="CT",
        thickness=15, 
        len=0.8,      
        x=1.02        # Push slightly to the right
    )
))

fig.add_trace(go.Scatter(x=center[1] + (region_diameter / 2) * np.cos(theta), y=center[0] + (region_diameter / 2) * np.sin(theta), mode='lines', line=dict(color='red', width=2), name='ROI', hoverinfo='skip'))
fig.add_trace(go.Scatter(x=[center[1]], y=[center[0]], mode='markers', marker=dict(symbol='star', size=14, color='lime', line=dict(color='black', width=1)), name='center', hoverinfo='skip'))

# Draw the selected lines
for fam, label, color in [('H', h_sel, 'royalblue'), ('V', v_sel, 'crimson'), ('D', d_sel, 'limegreen'), ('E', e_sel, 'orange')]:
    fam_data = build_family(slice_img, fam, n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts)
    if label in fam_data and fam_data[label].valid:
        ld = fam_data[label]
        fig.add_trace(go.Scatter(x=ld.cc, y=ld.rr, mode='lines', line=dict(color=color, width=2), name=f"{fam}: {label}"))
        if ld.selected_indices:
            sel_r = [ld.rr[i] for i in ld.selected_indices]
            sel_c = [ld.cc[i] for i in ld.selected_indices]
            fig.add_trace(go.Scatter(x=sel_c, y=sel_r, mode='markers', marker=dict(size=4, color='white', line=dict(color=color, width=1)), showlegend=False))

# Layout settings (Legend to the left, full width)
fig.update_layout(
    height=700, 
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=-0.2, # Pushes legend to the left
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

fig.update_xaxes(range=[0, meta.cols], showticklabels=False)
fig.update_yaxes(range=[meta.rows, 0], showticklabels=False, scaleanchor='x', scaleratio=1)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 30-Slice Grid Preview (Bottom of App)
# ------------------------------------------------------------------
st.subheader("All Slices Preview")

# 1. Define custom Gamma colormap for Matplotlib
gamma_colors = [
    (0.0, 'darkblue'),   
    (0.25, 'cyan'),
    (0.5, 'yellow'),
    (0.75, 'orange'),
    (1.0, 'red')         
]
voxel_gamma = mcolors.LinearSegmentedColormap.from_list("voxel_gamma", gamma_colors)

# 2. Slider to select 30 slices (3 rows × 10 columns)
chunk_start = st.slider("Select 30-slice chunk to view", 0, max(0, meta.n_slices - 30), 0, step=30)

# 3. Render the image grid
for row_idx in range(3):
    cols = st.columns(10)
    for col_idx in range(10):
        slice_num = chunk_start + row_idx * 10 + col_idx
        with cols[col_idx]:
            if slice_num < meta.n_slices:
                img = _load_one(str(meta.files[slice_num]))
                
                # Normalize and apply the Gamma colormap
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                colored_img = voxel_gamma(norm(img))
                
                st.image(colored_img, use_container_width=True)
                st.caption(f"Slice {slice_num+1}")

# ------------------------------------------------------------------
# Statistics Table
# ------------------------------------------------------------------
st.subheader("Statistics — CT values")
st.caption(f"Filter [{filter_min}, {filter_max}].")

mask = circle_mask(slice_img.shape, center, region_diameter)
roi_values = apply_filter(slice_img[mask], filter_min, filter_max)
slice_stats = stats_dict(roi_values)

h_stats = stats_dict(apply_filter(build_family(slice_img, 'H', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts)[h_sel].values, filter_min, filter_max)) if h_sel in build_family(slice_img, 'H', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts) else stats_dict([])
v_stats = stats_dict(apply_filter(build_family(slice_img, 'V', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts)[v_sel].values, filter_min, filter_max)) if v_sel in build_family(slice_img, 'V', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts) else stats_dict([])
d_stats = stats_dict(apply_filter(build_family(slice_img, 'D', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts)[d_sel].values, filter_min, filter_max)) if d_sel in build_family(slice_img, 'D', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts) else stats_dict([])
e_stats = stats_dict(apply_filter(build_family(slice_img, 'E', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts)[e_sel].values, filter_min, filter_max)) if e_sel in build_family(slice_img, 'E', n_per_side, n_per_side, center, region_diameter, DEFAULTS["line_length_ratio"], num_pts) else stats_dict([])

def _fmt(d): return {'Mean': d['mean'], 'Median': d['median'], 'SD': d['sd'], 'Minimum': d['min'], 'Maximum': d['max']}

columns = {'Selected Slice': _fmt(slice_stats)}
if h_sel: columns[f'H Line'] = _fmt(h_stats)
if v_sel: columns[f'V Line'] = _fmt(v_stats)
if d_sel: columns[f'D Line'] = _fmt(d_stats)
if e_sel: columns[f'E Line'] = _fmt(e_stats)

table = pd.DataFrame(columns)
# Format to 0 decimals and center the text
styled_table = table.style.format("{:.0f}") \
    .set_properties(**{'text-align': 'center'}) \
    .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

# Center the table
t1, t2, t3 = st.columns([1, 2, 1])
with t2:
    st.dataframe(styled_table, use_container_width=True)

# ------------------------------------------------------------------
# Phase 5 — Min/Mean/Max across slices
# ------------------------------------------------------------------
st.subheader("Min / Mean / Max trend across slices")
perslice = _per_slice_stats(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max))

pfig = go.Figure()
pfig.add_trace(go.Scatter(x=perslice['slices'], y=perslice['mean'], mode='lines+markers', name='Mean', line=dict(color='royalblue', width=2)))
pfig.add_trace(go.Scatter(x=perslice['slices'], y=perslice['min'], mode='lines', name='Min', line=dict(color='crimson', width=1.5)))
pfig.add_trace(go.Scatter(x=perslice['slices'], y=perslice['max'], mode='lines', name='Max', line=dict(color='darkorange', width=1.5)))
pfig.add_vline(x=slice_idx, line=dict(color='green', width=2, dash='dash'))
pfig.update_layout(height=400, xaxis=dict(title="Slice number"), yaxis=dict(title="CT value"))
st.plotly_chart(pfig, use_container_width=True)

# ------------------------------------------------------------------
# Phase 6 — Min/Mean/Max per family
# ------------------------------------------------------------------
st.subheader("Min / Mean / Max per line family across slices")
for fam, label, color in [('H', h_sel, 'royalblue'), ('V', v_sel, 'crimson'), ('D', d_sel, 'limegreen'), ('E', e_sel, 'orange')]:
    if not label: continue
    ls = _per_slice_line_stats(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max), DEFAULTS["line_length_ratio"], num_pts, n_per_side, fam, label)
    lfig = go.Figure()
    lfig.add_trace(go.Scatter(x=ls['slices'], y=ls['mean'], mode='lines+markers', name='Mean', line=dict(color=color, width=2)))
    lfig.add_trace(go.Scatter(x=ls['slices'], y=ls['min'], mode='lines', name='Min', line=dict(color='black', width=1, dash='dot')))
    lfig.add_trace(go.Scatter(x=ls['slices'], y=ls['max'], mode='lines', name='Max', line=dict(color='red', width=1, dash='dot')))
    lfig.update_layout(height=300, title=dict(text=f"{fam} Line ({label})", font=dict(color=color)), xaxis=dict(title="Slice number"), yaxis=dict(title="CT value"))
    st.plotly_chart(lfig, use_container_width=True)

# ------------------------------------------------------------------
# Phase 6b — Per-slice deviation from global
# ------------------------------------------------------------------
st.subheader("Per-slice deviation from global Min / Mean / Max")
st.caption("Curve(i) = M_global − M_slice(i). Positive = this slice is below the global value.")

@st.cache_data(show_spinner="Computing per-slice deviations …")
def _per_slice_deviation(folder: str, indices: tuple[int, ...], center: tuple[int, int], diameter: int, lo: float, hi: float) -> dict:
    from ct_io import scan_folder as _scan, load_slice as _load
    m = _scan(folder)
    mk = circle_mask((m.rows, m.cols), center, diameter)
    slices, per_min, per_mean, per_max = [], [], [], []
    all_vals = []
    for i in indices:
        img = _load(str(m.files[i]))
        v = img[mk]; v = v[np.isfinite(v)]; v = v[(v >= lo) & (v <= hi)]
        slices.append(i + 1)
        if v.size:
            per_min.append(float(v.min())); per_mean.append(float(v.mean())); per_max.append(float(v.max()))
            all_vals.append(v)
        else:
            per_min.append(np.nan); per_mean.append(np.nan); per_max.append(np.nan)
    if all_vals:
        flat = np.concatenate(all_vals)
        g_min, g_mean, g_max = float(flat.min()), float(flat.mean()), float(flat.max())
    else: g_min = g_mean = g_max = np.nan
    return {'slices': slices, 'per_min': per_min, 'per_mean': per_mean, 'per_max': per_max, 'g_min': g_min, 'g_mean': g_mean, 'g_max': g_max}

dev = _per_slice_deviation(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max))
dfig = go.Figure()
dfig.add_trace(go.Scatter(x=dev['slices'], y=[dev['g_max'] - v for v in dev['per_max']], mode='lines', name='Max dev', line=dict(color='crimson')))
dfig.add_trace(go.Scatter(x=dev['slices'], y=[dev['g_mean'] - v for v in dev['per_mean']], mode='lines', name='Mean dev', line=dict(color='royalblue'), yaxis='y2'))
dfig.add_trace(go.Scatter(x=dev['slices'], y=[dev['g_min'] - v for v in dev['per_min']], mode='lines', name='Min dev', line=dict(color='darkgreen')))
dfig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
dfig.update_layout(height=400, xaxis=dict(title="Slice number"), yaxis=dict(title="Deviation (HU)"), yaxis2=dict(overlaying='y', side='right'))
st.plotly_chart(dfig, use_container_width=True)

# ------------------------------------------------------------------
# Phase 7 — Interactive 3D
# ------------------------------------------------------------------
st.subheader("Interactive 3D — selected lines across slices")
selected_fams = st.multiselect("Select directions for 3D viewer", ["H", "V", "D", "E"], default=["H", "V", "D", "E"])

fig3d = go.Figure()
for fam in selected_fams:
    label = {'H': h_sel, 'V': v_sel, 'D': d_sel, 'E': e_sel}[fam]
    color = {'H': 'RoyalBlue', 'V': 'Crimson', 'D': 'LimeGreen', 'E': 'DarkOrange'}[fam]
    pts = _per_slice_line_points(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max), DEFAULTS["line_length_ratio"], num_pts, n_per_side, fam, label)
    fig3d.add_trace(go.Surface(z=pts['matrix'], x=pts['slices'], y=list(range(pts['num_points'])), colorscale=[[0, color], [1, color]], showscale=False, name=f"{fam} ({label})", opacity=0.85))

fig3d.update_layout(
    height=700, 
    margin=dict(l=0, r=0, t=0, b=0), # Remove margins to use full width
    scene=dict(
        xaxis=dict(title='Slice'), 
        yaxis=dict(title='Point'), 
        zaxis=dict(title='CT'),
        aspectmode='cube' # This forces the axes to be proportional and fills the space better
    )
)
st.plotly_chart(fig3d, use_container_width=True,width = "stretch")

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
st.subheader("Export — CT values for selected lines")
if st.button("Prepare Excel Export"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        for fam in ['H', 'V', 'D', 'E']:
            label = {'H': h_sel, 'V': v_sel, 'D': d_sel, 'E': e_sel}[fam]
            if not label: continue
            
            pts = _per_slice_line_points(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max), DEFAULTS["line_length_ratio"], num_pts, n_per_side, fam, label)
            # Build DataFrame: Slice, Min, Mean, Max, P0...P20
            slice_stats = _per_slice_stats(str(dicom_folder), tuple(range(meta.n_slices)), center, region_diameter, float(filter_min), float(filter_max))
            
            df = pd.DataFrame({
                'Slice': pts['slices'],
                'Min_CT': slice_stats['min'],
                'Mean_CT': slice_stats['mean'],
                'Max_CT': slice_stats['max'],
            })
            for j in range(pts['num_points']):
                df[f'P{j}'] = pts['matrix'][:, j]
            
            df.to_excel(xw, sheet_name=f"{fam}_{label}"[:31], index=False)
            
    buf.seek(0)
    st.download_button(
        label="⬇ Download Excel Export",
        data=buf,
        file_name="CT_Selected_Lines.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

