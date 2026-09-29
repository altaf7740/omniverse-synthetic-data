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


def _range(key: str) -> tuple:
    """[min, max] -> (min, max); nested [[x,y,z], [x,y,z]] -> ((x,y,z), (x,y,z))."""
    lo, hi = _r[key]
    return (tuple(lo), tuple(hi)) if isinstance(lo, list) else (lo, hi)


ISAAC_ROOT = Path(_raw["isaac"]["root"])
ISAAC_PYTHON = ISAAC_ROOT / "python.bat"

NUM_FRAMES = _raw["rendering"]["num_frames"]
RESOLUTION = tuple(_raw["rendering"]["resolution"])
SEED = _raw["rendering"]["seed"]
RT_SUBFRAMES = _raw["rendering"]["rt_subframes"]

GRAVITY = _raw["physics"]["gravity"]
SETTLE_SECONDS = _raw["physics"]["settle_seconds"]
PHYSICS_STEPS_PER_SECOND = _raw["physics"]["steps_per_second"]

VAL_SPLIT = _raw["dataset"]["val_split"]

_r = _raw["randomization"]
CAMERA_DISTANCE = _range("camera_distance")
CAMERA_ELEVATION_DEG = _range("camera_elevation_deg")
CAMERA_LOOK_AT_JITTER = _r["camera_look_at_jitter"]

PART_SPREAD = _r["part_spread"]
PART_VISIBLE_PROBABILITY = _r["part_visible_probability"]

DISTRACTORS_PER_SHAPE = _r["distractors_per_shape"]
DISTRACTOR_SIZE = _range("distractor_size")
DISTRACTOR_SPREAD = _r["distractor_spread"]
DISTRACTOR_VISIBLE_PROBABILITY = _r["distractor_visible_probability"]

GROUND_SIZE = _r["ground_size"]
GROUND_TEXTURE_TILE = _range("ground_texture_tile")

DOME_LIGHT_INTENSITY_RANGE = _range("dome_light_intensity")
POINT_LIGHT_POSITION_RANGE = _range("point_light_position")
POINT_LIGHT_INTENSITY_RANGE = _range("point_light_intensity")
