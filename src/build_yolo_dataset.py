"""
Stage 3: convert the Replicator segmentation output (stage 2) into an
Ultralytics YOLO-seg dataset. Classes are read from parts_manifest.json
(written in stage 1) rather than declared again here.

Run with a normal Python (not Isaac Sim's python.bat), from the repo root:
    uv sync
    uv run python src/build_yolo_dataset.py --output-dir <folder>
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from synth_pipeline import config
from synth_pipeline.utils.seg_labels import frame_to_yolo_seg_lines, load_instance_class_map
from synth_pipeline.utils.yolo_dataset import make_dataset_dirs, num_val, split_for_index, write_data_yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to earlier stages.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    manifest_path = args.output_dir / "parts_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    class_names = sorted(manifest.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    src = args.output_dir / "synthetic_dataset"
    dst = args.output_dir / "yolo_seg_dataset"
    # Start clean - images/labels left over from a previous, larger run would
    # otherwise stay in train/val alongside this run's.
    shutil.rmtree(dst, ignore_errors=True)
    make_dataset_dirs(dst)

    # BasicWriter writes flat files directly in src, one set per frame:
    # rgb_NNNN.png, instance_segmentation_NNNN.png and
    # instance_segmentation_semantics_mapping_NNNN.json. Each frame's mask and
    # mapping are looked up by its frame number, rather than zipping three
    # separately sorted file lists, which would silently misalign frames if
    # any single file were missing.
    rgb_files = sorted(src.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb_*.png frames in {src} - run generate_dataset.py first (stage 2).")

    total = len(rgb_files)
    seen_classes = set()

    for i, img_path in enumerate(rgb_files):
        frame = img_path.stem.removeprefix("rgb_")
        mask_path = src / f"instance_segmentation_{frame}.png"
        mapping_path = src / f"instance_segmentation_semantics_mapping_{frame}.json"
        missing = [p.name for p in (mask_path, mapping_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Frame {frame} is missing {missing} in {src}")

        split = split_for_index(i, total, config.VAL_SPLIT)

        img = Image.open(img_path)
        w, h = img.size
        shutil.copy(img_path, dst / "images" / split / img_path.name)

        mask_rgba = np.array(Image.open(mask_path).convert("RGBA"))
        instance_class_map = load_instance_class_map(mapping_path)
        seen_classes.update(instance_class_map.values())

        lines = frame_to_yolo_seg_lines(mask_rgba, instance_class_map, class_to_idx, w, h)

        label_txt = dst / "labels" / split / (img_path.stem + ".txt")
        label_txt.write_text("\n".join(lines))

    unknown = seen_classes - set(class_to_idx)
    if unknown:
        raise ValueError(f"Found class(es) {unknown} not in {manifest_path} - re-run stage 1/2 if parts changed.")

    write_data_yaml(dst, class_names)

    n_val = num_val(total, config.VAL_SPLIT)
    print(f"Done. {total - n_val} train / {n_val} val images written to {dst}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
