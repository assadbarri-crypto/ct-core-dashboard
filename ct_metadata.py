"""
Parse the General_data.txt file that stores sample info.

Confirmed format (tab-separated, one key per line):

    Rock Type       Berea Sandstone
    Sample Index    SW-1
    Status          Dry
    Length (cm)     30
    Diameter (cm)   3.81

Only one sample per file, in the current layout. If in the future the file
grows to hold multiple samples, this parser still works because it just
returns a single SampleInfo keyed by the sample index.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class SampleInfo:
    sample_id: str = ""
    rock_type: str = ""
    status: str = ""
    length_cm: float | None = None
    diameter_cm: float | None = None
    n_slices_expected: int | None = None
    region_diameter_px: int | None = None
    center_px: tuple[int, int] | None = None
    n_h_lines: int | None = None
    n_v_lines: int | None = None
    n_d_lines: int | None = None
    n_e_lines: int | None = None
    line_spacing_px: float | None = None
    line_length_ratio: float | None = None
    n_selected_points: int | None = None
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def _to_float(s) -> float | None:
    try:
        return float(str(s).strip().replace(",", "."))
    except Exception:
        return None


def _to_int(s) -> int | None:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return None


def _normalise_key(k: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', k.strip().lower()).strip('_')


def _map_key(raw_key: str):
    """Map a raw file key (e.g. 'Rock Type', 'Length (cm)') to a SampleInfo field."""
    k = _normalise_key(raw_key)
    table = {
        'rock_type': 'rock_type',
        'rock': 'rock_type',
        'sample_index': 'sample_id',
        'sample_id': 'sample_id',
        'sample': 'sample_id',
        'status': 'status',
        'condition': 'status',
        'length_cm': 'length_cm',
        'length': 'length_cm',
        'diameter_cm': 'diameter_cm',
        'diameter': 'diameter_cm',
        'n_slices': 'n_slices_expected',
        'number_of_slices': 'n_slices_expected',
        'slices': 'n_slices_expected',
        'region_diameter_px': 'region_diameter_px',
        'region_diameter': 'region_diameter_px',
        'center_px': 'center_px',
        'n_h_lines': 'n_h_lines',
        'horizontal_lines': 'n_h_lines',
        'n_v_lines': 'n_v_lines',
        'vertical_lines': 'n_v_lines',
        'n_d_lines': 'n_d_lines',
        'diagonal_d_lines': 'n_d_lines',
        'n_e_lines': 'n_e_lines',
        'diagonal_e_lines': 'n_e_lines',
        'line_spacing_px': 'line_spacing_px',
        'spacing': 'line_spacing_px',
        'line_length_ratio': 'line_length_ratio',
        'n_selected_points': 'n_selected_points',
        'selected_points': 'n_selected_points',
        'notes': 'notes',
    }
    return table.get(k)


def parse_general_data(path: str | Path) -> Dict[str, SampleInfo]:
    """
    Returns {sample_id: SampleInfo}. Empty dict if file missing or unreadable.
    The current file layout is one sample, tab-separated key/value lines.
    """
    p = Path(path)
    if not p.exists():
        return {}

    text = p.read_text(encoding="utf-8", errors="ignore")

    # Try YAML first (in case the file is ever restructured)
    try:
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict) and loaded and any(
                isinstance(v, dict) for v in loaded.values()):
            return _from_yaml_dict(loaded)
    except Exception:
        pass

    # Default: tab / multi-space separated key<value> lines
    info = SampleInfo(raw={})
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # split on tabs first, then on 2+ spaces, then on single space after first token
        parts = re.split(r'\t+', raw_line)
        if len(parts) < 2:
            parts = re.split(r'\s{2,}', raw_line.strip())
        if len(parts) < 2:
            # try key: value or key = value
            m = re.match(r'^(.+?)\s*[:=]\s*(.+)$', raw_line)
            if not m:
                continue
            parts = [m.group(1), m.group(2)]

        key_raw, val_raw = parts[0].strip(), parts[1].strip()
        field_name = _map_key(key_raw)
        info.raw[key_raw] = val_raw

        if not field_name:
            continue

        if field_name in ('length_cm', 'diameter_cm', 'line_spacing_px',
                          'line_length_ratio'):
            setattr(info, field_name, _to_float(val_raw))
        elif field_name in ('n_slices_expected', 'region_diameter_px',
                            'n_h_lines', 'n_v_lines', 'n_d_lines', 'n_e_lines',
                            'n_selected_points'):
            setattr(info, field_name, _to_int(val_raw))
        elif field_name == 'center_px':
            nums = re.findall(r'-?\d+', val_raw)
            if len(nums) >= 2:
                info.center_px = (int(nums[0]), int(nums[1]))
        else:
            setattr(info, field_name, val_raw)

    out: Dict[str, SampleInfo] = {}
    if info.sample_id:
        out[info.sample_id] = info
    return out


def _from_yaml_dict(loaded: Dict[str, Any]) -> Dict[str, SampleInfo]:
    out: Dict[str, SampleInfo] = {}
    samples = loaded.get("samples") if "samples" in loaded else loaded
    for sid, data in samples.items():
        if not isinstance(data, dict):
            continue
        norm = {_normalise_key(k): v for k, v in data.items()}
        info = SampleInfo(sample_id=str(sid), raw=data)
        for k, v in norm.items():
            field_name = _map_key(k)
            if not field_name:
                continue
            if field_name in ('length_cm', 'diameter_cm', 'line_spacing_px',
                              'line_length_ratio'):
                setattr(info, field_name, _to_float(v))
            elif field_name in ('n_slices_expected', 'region_diameter_px',
                                'n_h_lines', 'n_v_lines', 'n_d_lines', 'n_e_lines',
                                'n_selected_points'):
                setattr(info, field_name, _to_int(v))
            else:
                setattr(info, field_name, v)
        out[str(sid)] = info
    return out


def find_sample(info: Dict[str, SampleInfo], sample_id: str) -> SampleInfo | None:
    """Case-insensitive, punctuation-insensitive lookup: 'SW-1' matches 'sw1', 'SW1'."""
    target = re.sub(r'[^a-z0-9]', '', sample_id.lower())
    for k, v in info.items():
        if re.sub(r'[^a-z0-9]', '', k.lower()) == target:
            return v
    return None