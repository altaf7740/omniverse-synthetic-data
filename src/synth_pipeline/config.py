"""
Loads tool-level settings from pyproject.toml's [tool.synth-pipeline] table
and exposes them as typed Python objects (Path, tuples) for every stage
script to import.

Only tool-level settings live here - per-run paths (input CAD folder, output
workspace) are passed explicitly via --input-dir/--output-dir on each stage
script's command line instead, since this is a general-purpose tool meant to
run against any folder of parts, not one fixed project.
"""

import tomllib
from pathlib import Path

# config.py -> synth_pipeline/ -> src/ -> repo root
_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
with open(_PYPROJECT_PATH, "rb") as f:
    _raw = tomllib.load(f)["tool"]["synth-pipeline"]

ISAAC_ROOT = Path(_raw["isaac"]["root"])
ISAAC_PYTHON = ISAAC_ROOT / "python.bat"

NUM_FRAMES = _raw["rendering"]["num_frames"]
RESOLUTION = tuple(_raw["rendering"]["resolution"])

_r = _raw["randomization"]
CAMERA_POSITION_RANGE = (tuple(_r["camera_position_min"]), tuple(_r["camera_position_max"]))
POINT_LIGHT_POSITION_RANGE = (tuple(_r["point_light_position_min"]), tuple(_r["point_light_position_max"]))
POINT_LIGHT_INTENSITY_RANGE = (_r["point_light_intensity_min"], _r["point_light_intensity_max"])
DOME_LIGHT_INTENSITY_RANGE = (_r["dome_light_intensity_min"], _r["dome_light_intensity_max"])
PART_POSITION_RANGE = (tuple(_r["part_position_min"]), tuple(_r["part_position_max"]))
PLANE_SCALE = _r["plane_scale"]
VAL_SPLIT = _r["val_split"]
