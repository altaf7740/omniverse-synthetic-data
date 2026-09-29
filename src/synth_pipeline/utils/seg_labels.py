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
"""

import json
from pathlib import Path

import cv2
import numpy as np

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


def instance_mask_to_polygons(mask: np.ndarray, min_area: float = 4.0) -> list:
    """Trace one binary instance mask into one or more polygons.

    A single instance can produce multiple disjoint contours (e.g. split by
    occlusion); all are kept as separate polygon entries.

    Args:
        mask: 2D boolean/uint8 array, non-zero where the instance is present.
        min_area: Discard contours smaller than this many pixels (noise).

    Returns:
        List of (N, 2) arrays of (x, y) pixel coordinates, each with at least
        3 points (a valid polygon).
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in contours if len(c) >= 3 and cv2.contourArea(c) >= min_area]


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
        List of label lines, one per polygon.
    """
    lines = []
    for color, class_name in instance_class_map.items():
        if class_name not in class_to_idx:
            continue
        mask = np.all(mask_rgba == np.array(color, dtype=mask_rgba.dtype), axis=-1)
        if not mask.any():
            continue
        for poly in instance_mask_to_polygons(mask):
            norm = poly.astype(np.float64)
            norm[:, 0] /= img_w
            norm[:, 1] /= img_h
            coords = " ".join(f"{v:.6f}" for v in norm.flatten())
            lines.append(f"{class_to_idx[class_name]} {coords}")
    return lines
