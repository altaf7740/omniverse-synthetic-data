"""
Stage 1: for each class subfolder in --input-dir, resolve its STEP file
(extracting a zip first if that's what's there) and convert it to USD.
Writes <output-dir>/usd/<class_name>.usd and <output-dir>/parts_manifest.json
for stage 2 to read.

Fails (nonzero exit, no manifest written) if ANY class can't be converted -
silently dropping a class would produce a dataset/model missing that part.

Run with Isaac Sim's bundled Python, from the repo root:
    <isaac_root>\\python.bat src\\convert_assets.py --input-dir <folder> --output-dir <folder>
"""

import argparse
import asyncio
import json
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with one subfolder per part class.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where the USD files/manifest are written.")
    args = parser.parse_args()
    # Resolve to absolute paths immediately - relative paths are relative to
    # wherever the caller ran this command from, which may not match this
    # process's own working directory (e.g. when launched via main.py).
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir not found: {args.input_dir}")
    return args


# Parse/validate args before booting Isaac Sim (~15-20s startup) - so --help
# and bad arguments fail instantly instead of paying that cost first.
args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.kit.converter.cad", True)
manager.set_extension_enabled_immediate("omni.kit.converter.hoops_core", True)

from synth_pipeline.utils.discover import discover_classes, resolve_step_file
from synth_pipeline.utils.step_to_usd import convert_step_to_usd


async def convert_all(input_dir: Path, output_dir: Path) -> dict:
    usd_dir = output_dir / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    failures = {}
    for class_name in discover_classes(input_dir):
        try:
            step_path = resolve_step_file(input_dir / class_name)
            print(f"[{class_name}] STEP file: {step_path}")
            usd_path = await convert_step_to_usd(str(step_path), str(usd_dir / f"{class_name}.usd"))
            print(f"[{class_name}] converted -> {usd_path}")
            manifest[class_name] = usd_path
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[{class_name}] FAILED: {e}")
            failures[class_name] = str(e)

    # Try every class before failing, so one run reports every problem at once.
    if failures:
        raise RuntimeError(f"{len(failures)} class(es) failed to convert: {sorted(failures)}")
    return manifest


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = asyncio.get_event_loop().run_until_complete(convert_all(args.input_dir, args.output_dir))

    manifest_path = args.output_dir / "parts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest for {len(manifest)} classes to {manifest_path}")


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
