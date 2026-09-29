"""
Pipeline entry point. Point it at a folder of CAD parts and it runs the
whole thing: STEP -> USD -> synthetic renders (with instance-segmentation
masks) -> YOLO-seg dataset -> trained model.

Input folder contract: one subfolder per class, each containing either a
.step/.stp file directly, or a single .zip that contains one
(see utils/discover.py). Nothing about the parts needs to be declared
anywhere else - classes are discovered from the folder names.

Stages run under different Python interpreters (Isaac Sim bundles its own,
separate from the system one - see README.md for why):
  1. convert_assets.py      - Isaac Sim's python.bat
  2. generate_dataset.py    - Isaac Sim's python.bat
  3. build_yolo_dataset.py  - system Python
  4. train.py               - system Python

Usage (from the repo root):
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace --stage convert
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace --from generate
"""

import argparse
import subprocess
import sys
from pathlib import Path

from synth_pipeline import config

HERE = Path(__file__).resolve().parent

# (name, script, interpreter, needs_input_dir)
STAGES = [
    ("convert", HERE / "convert_assets.py", "isaac", True),
    ("generate", HERE / "generate_dataset.py", "isaac", False),
    ("dataset", HERE / "build_yolo_dataset.py", "system", False),
    ("train", HERE / "train.py", "system", False),
]
STAGE_NAMES = [s[0] for s in STAGES]


def run_stage(name: str, script: Path, interpreter: str, needs_input: bool, input_dir: Path, output_dir: Path) -> None:
    py = str(config.ISAAC_PYTHON) if interpreter == "isaac" else sys.executable
    cmd = [py, str(script), "--output-dir", str(output_dir)]
    if needs_input:
        cmd += ["--input-dir", str(input_dir)]

    print(f"\n=== stage '{name}': {' '.join(cmd)} ===\n")
    # No cwd= override here - input_dir/output_dir are already resolved to
    # absolute paths in main() below, so it doesn't matter what directory
    # each stage subprocess actually runs in.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Stage '{name}' failed (exit {result.returncode}) - stopping pipeline.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with one subfolder per part class.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where generated USDs/dataset/runs go.")
    parser.add_argument("--stage", choices=STAGE_NAMES, help="Run only this stage.")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, help="Run this stage and all after it.")
    args = parser.parse_args()

    # Resolve to absolute paths immediately - user-typed relative paths are
    # relative to wherever they ran this command from, not to any directory
    # a downstream subprocess happens to run in.
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.stage:
        stages_to_run = [s for s in STAGES if s[0] == args.stage]
    elif args.from_stage:
        start = STAGE_NAMES.index(args.from_stage)
        stages_to_run = STAGES[start:]
    else:
        stages_to_run = STAGES

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, script, interpreter, needs_input in stages_to_run:
        run_stage(name, script, interpreter, needs_input, args.input_dir, args.output_dir)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
