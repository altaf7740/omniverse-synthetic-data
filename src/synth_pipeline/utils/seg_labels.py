"""
Convert Replicator BasicWriter's instance_segmentation output into YOLO-seg
polygon labels.

Per frame, BasicWriter writes (verified against real output):
  instance_segmentation_NNNN.png
      RGBA image, one unique color per instance.
  instance_segmentation_semantics_mapping_NNNN.json
      {"(r, g, b, a)": {"class": "<name>"}, ...} - maps each instance color
      to its semantic class, plus "BACKGROUND"/"UNLABELLED" entries for
      pixels that aren't a labelled part (e.g. the ground plane).
Every instance has its own color even when two share a class, so each
still becomes its own polygon.

Runs in the uv venv (stage 3), not Isaac Sim's Python.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.converter import merge_multi_segment

_NON_INSTANCE_CLASSES = {"BACKGROUND", "UNLABELLED"}


def _parse_color_key(key: str) -> tuple:
    """'(33, 243, 3, 255)' -> (33, 243, 3, 255)"""
    return tuple(int(v) for v in key.strip("()").split(","))


def load_instance_class_map(semantics_mapping_path: Path) -> dict:
    """Load a frame's instance_segmentation_semantics_mapping_*.json.

    Returns:
        dict mapping RGBA color tuple -> class name, for labelled instances
        only (BACKGROUND/UNLABELLED dropped).
    """
    raw = json.loads(semantics_mapping_path.read_text())
    return {
        _parse_color_key(color): info["class"]
        for color, info in raw.items()
        if info.get("class") not in _NON_INSTANCE_CLASSES
    }


def instance_mask_to_polygon(mask: np.ndarray, min_area: float = 4.0):
    """Trace one binary instance mask into a single polygon.

    YOLO-seg has one polygon per instance and no holes. Tracing only outer
    contours would fill holes (a nut's bore) and emit each piece of an
    occluded part as a separate instance. Instead, every contour - outer
    pieces and holes - is joined into one polygon by zero-width cuts
    between nearest points. Rasterized with the even-odd rule (OpenCV's
    fillPoly, as Ultralytics does), holes stay empty.

    Args:
        mask: 2D boolean/uint8 array, non-zero where the instance is present.
        min_area: Discard contours smaller than this many pixels (noise).

    Returns:
        (N, 2) array of (x, y) pixel coordinates, or None if nothing is left.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c.reshape(-1, 2) for c in contours if len(c) >= 3 and cv2.contourArea(c) >= min_area]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(merge_multi_segment(parts), axis=0)


def frame_to_yolo_seg_lines(
    mask_rgba: np.ndarray,
    instance_class_map: dict,
    class_to_idx: dict,
    img_w: int,
    img_h: int,
) -> list:
    """Build YOLO-seg label lines ("class x1 y1 x2 y2 ...", normalized) for one frame.

    Args:
        mask_rgba: (H, W, 4) uint8 instance-segmentation image.
        instance_class_map: RGBA color tuple -> class name, from this same
            frame's instance_segmentation_semantics_mapping_*.json.
        class_to_idx: class name -> fixed YOLO class index.
        img_w, img_h: image dimensions, for normalizing coordinates.

    Returns:
        List of label lines, one per instance.
    """
    lines = []
    for color, class_name in instance_class_map.items():
        if class_name not in class_to_idx:
            continue
        mask = np.all(mask_rgba == np.array(color, dtype=mask_rgba.dtype), axis=-1)
        if not mask.any():
            continue
        poly = instance_mask_to_polygon(mask)
        if poly is None:
            continue
        norm = poly.astype(np.float64)
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        coords = " ".join(f"{v:.6f}" for v in norm.flatten())
        lines.append(f"{class_to_idx[class_name]} {coords}")
    return lines
