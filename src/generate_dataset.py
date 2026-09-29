"""
Stage 2: render the synthetic dataset with instance segmentation masks for
every part converted in stage 1 (see convert_assets.py / parts_manifest.json
under --output-dir). Clears <output-dir>/synthetic_dataset first - Replicator
numbers frames from 0 but never deletes old ones, so leftovers from a
previous, longer run would otherwise silently mix into this one.

Run with Isaac Sim's bundled Python, from the repo root:
    <isaac_root>\\python.bat src\\generate_dataset.py --output-dir <folder> [--num-frames N]
"""

import argparse
import json
import shutil
import traceback
from pathlib import Path

from synth_pipeline import config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to convert_assets.py.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=config.NUM_FRAMES,
        help=f"Frames to render (default from pyproject.toml: {config.NUM_FRAMES}). Use a small value to smoke-test.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    manifest_path = args.output_dir / "parts_manifest.json"
    if not manifest_path.exists():
        parser.error(f"{manifest_path} not found - run convert_assets.py first (stage 1).")
    args.manifest = json.loads(manifest_path.read_text())
    if not args.manifest:
        parser.error(f"{manifest_path} is empty - no parts were converted in stage 1.")
    return args


# Parse/validate args before booting Isaac Sim (~15-20s startup) - so --help
# and a missing manifest fail instantly instead of paying that cost first.
args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep


def render(manifest: dict, dataset_dir: Path, num_frames: int) -> None:
    with rep.new_layer():
        # Non-degenerate initial pose (eye != look_at) - a camera created with
        # position == look_at crashes Replicator before any frame renders.
        camera = rep.create.camera(position=(0, 0, 200), look_at=(0, 0, 0))
        render_product = rep.create.render_product(camera, config.RESOLUTION)

        plane = rep.create.plane(scale=config.PLANE_SCALE, position=(0, 0, 0), rotation=(0, 0, 0))

        # Light intensities are created as floats (not ints) - the initial
        # value's type fixes the USD attribute's schema type, and the
        # randomizer below writes float samples.
        dome_light = rep.create.light(light_type="Dome", intensity=1000.0)
        point_light = rep.create.light(light_type="Sphere", intensity=5000.0)

        # Each part keeps its own class label; the group is only a logical
        # collection of prim paths (it doesn't reparent anything).
        parts = [
            rep.create.from_usd(usd_path, semantics=[("class", class_name)])
            for class_name, usd_path in manifest.items()
        ]
        parts_group = rep.create.group(parts)

        def randomize_camera():
            with camera:
                rep.modify.pose(
                    position=rep.distribution.uniform(*config.CAMERA_POSITION_RANGE),
                    look_at=(0, 0, 0),
                )
            return camera.node

        # All parts are randomized through ONE group rather than one
        # randomizer per part. Replicator's graph builder
        # (_get_last_exec_attrs) re-walks everything downstream of the
        # trigger each time a randomizer is attached, with no visited-set, so
        # chaining one randomizer per part scales exponentially - 2 parts
        # worked, 4 hung indefinitely. A group keeps the graph the same size
        # no matter how many classes the input folder has. Distributions
        # inside `with parts_group:` are sampled independently per part.
        def randomize_parts():
            with parts_group:
                rep.modify.pose(
                    position=rep.distribution.uniform(*config.PART_POSITION_RANGE),
                    rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                )
                rep.randomizer.materials(
                    materials=rep.create.material_omnipbr(
                        diffuse=rep.distribution.uniform((0.3, 0.3, 0.3), (0.9, 0.9, 0.9)),
                        metallic=rep.distribution.uniform(0.6, 1.0),
                        roughness=rep.distribution.uniform(0.1, 0.6),
                        count=len(parts),
                    )
                )
            return parts_group.node

        def randomize_lights():
            with dome_light:
                # Target "inputs:intensity"/"inputs:color", not bare "intensity"/
                # "color" - lights also carry stale legacy attributes of those
                # bare names typed int with no value; writing float samples
                # there raises a fatal USD type-mismatch assertion at render time.
                rep.modify.attribute(
                    "inputs:intensity",
                    rep.distribution.uniform(*config.DOME_LIGHT_INTENSITY_RANGE),
                    attribute_type="float",
                )
                rep.modify.attribute(
                    "inputs:color",
                    rep.distribution.uniform((0.7, 0.7, 0.7), (1.0, 1.0, 1.0)),
                    attribute_type="color3f",
                )
            with point_light:
                rep.modify.pose(position=rep.distribution.uniform(*config.POINT_LIGHT_POSITION_RANGE))
                rep.modify.attribute(
                    "inputs:intensity",
                    rep.distribution.uniform(*config.POINT_LIGHT_INTENSITY_RANGE),
                    attribute_type="float",
                )
            return dome_light.node

        def randomize_background():
            with plane:
                rep.randomizer.materials(
                    materials=rep.create.material_omnipbr(
                        diffuse=rep.distribution.uniform((0.05, 0.05, 0.05), (0.95, 0.95, 0.95)),
                        roughness=rep.distribution.uniform(0.2, 1.0),
                        count=1,
                    )
                )
            return plane.node

        rep.randomizer.register(randomize_camera)
        rep.randomizer.register(randomize_parts)
        rep.randomizer.register(randomize_lights)
        rep.randomizer.register(randomize_background)

        with rep.trigger.on_frame(max_execs=num_frames):
            rep.randomizer.randomize_camera()
            rep.randomizer.randomize_parts()
            rep.randomizer.randomize_lights()
            rep.randomizer.randomize_background()

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(dataset_dir),
            rgb=True,
            instance_segmentation=True,
            bounding_box_2d_tight=True,  # cheap to keep; useful as a sanity check
        )
        writer.attach([render_product])

        # run() only submits the Start command and returns immediately - in a
        # standalone script that leaves nothing captured before
        # simulation_app.close() runs. run_until_complete() blocks until the
        # writer has actually finished.
        rep.orchestrator.run_until_complete()


def main() -> None:
    dataset_dir = args.output_dir / "synthetic_dataset"
    shutil.rmtree(dataset_dir, ignore_errors=True)
    render(args.manifest, dataset_dir, args.num_frames)
    print(f"\nRendered {args.num_frames} frames for {sorted(args.manifest)} to {dataset_dir}")


# An unhandled exception here would otherwise still exit 0: SimulationApp's
# atexit handler calls close() with the default exit_code=0, and Kit's fast
# shutdown terminates the process with that. Passing exit_code explicitly is
# NVIDIA's documented way to keep a failure visible to callers (main.py, make).
exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    simulation_app.close(exit_code=exit_code)
