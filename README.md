# Synthetic Data Pipeline

Generate an instance-segmentation training dataset — and train a model on
it — from CAD parts alone. No real photos required. Point it at a folder of
STEP files; it converts each to USD, renders thousands of domain-randomized
frames with per-pixel instance masks, builds a YOLO-seg dataset, and trains
a YOLO26-seg model.

## How it works

```
<input folder>/<class_name>/*.step
        │
        ▼
 1. src/convert_assets.py     STEP → USD (Isaac Sim's Hoops CAD converter)
        │
        ▼
 2. src/generate_dataset.py   USD → 2000 rendered frames + instance masks
        │                     (Replicator + PhysX, domain randomization)
        ▼
 3. src/build_yolo_dataset.py masks → YOLO-seg polygon labels, train/val split
        │
        ▼
 4. src/train.py              YOLO26n-seg training (Ultralytics)
```

### What each rendered frame varies

| | |
|---|---|
| **Placement** | Parts and clutter are dropped onto the surface and settled with physics, so they rest in natural poses (on their side, flat, leaning on each other) |
| **Surface** | 36 built-in procedural textures (wood, brushed metal, concrete, fabric, tiles, cardboard, speckle, rubber mat, plain), random tiling, rotation and tint, plus your own photos via `--textures-dir` |
| **Backdrop** | Textured environment (studio gradients, tinted rooms and skies), rotated per frame; visible in low-angle shots, and reflected by metal parts |
| **Part finishes** | Steel/zinc, black oxide, brass/yellow zinc, white and black nylon, plus a share of random colors |
| **Clutter** | Unlabelled cubes, spheres, cylinders, cones and tori with random finishes, including metallic greys |
| **Composition** | Each part is independently present or absent, so frames range from single parts to all of them |
| **Camera** | Orbit around the parts: distance, elevation (15–85°), azimuth and off-center aim |
| **Lights** | Environment intensity; point light position, intensity and warm/neutral tint |

Every range is in `pyproject.toml`'s `[tool.synth-pipeline.*]` tables, and a
fixed `seed` makes runs reproducible.

Parts are loaded with their pivot moved to their geometric center. CAD
exports often put geometry far from the file's origin (the example nut is
~78 mm off it), which would otherwise make parts land out of frame or rotate
into the ground.

If any stage fails, the pipeline stops there with a nonzero exit code, so
`make` and CI can see the failure. Stage 1 also fails if even one class
can't be converted, rather than quietly leaving that class out of the
dataset.

## Requirements

- Windows, NVIDIA GPU (RTX recommended — Isaac Sim needs RTX for rendering)
- [Isaac Sim](https://developer.nvidia.com/isaac/sim) 6.0.x installed locally
- Python 3.11+ for the system-side stages
- [uv](https://docs.astral.sh/uv/) — `pip install uv`, or `winget install astral-sh.uv`

## Setup

```
uv sync
```

(or `make setup`)

This creates `.venv/` and installs everything from `pyproject.toml` -
including a **CUDA build of torch**, not the default CPU-only PyPI wheel.
That routing is declared in `pyproject.toml`'s `[tool.uv.sources]` /
`[[tool.uv.index]]` (torch/torchvision are pulled from PyTorch's `cu128`
index instead of PyPI) - `uv sync` alone handles it correctly, no separate
manual install step needed.

Then point `pyproject.toml`'s `[tool.synth-pipeline.isaac].root` at your own
Isaac Sim install path if it isn't at the default
(`C:\isaac\isaac-sim-standalone-6.0.1-windows-x86_64`).

## Usage

Prepare an input folder — one subfolder per class, each holding a `.step`
file (or a `.zip` that contains one):

```
my_parts/
├── bolt/
│   └── bolt.step
└── nut/
    └── nut.zip        # gets extracted in place automatically
```

Run everything:

```
uv run python src/main.py --input-dir my_parts --output-dir workspace
```

Or run one stage at a time (useful while iterating):

```
uv run python src/main.py --input-dir my_parts --output-dir workspace --stage convert
uv run python src/main.py --input-dir my_parts --output-dir workspace --stage generate
uv run python src/main.py --input-dir my_parts --output-dir workspace --from dataset
```

`main.py` itself must be launched via `uv run python` (not a bare `python`)
- stages 3-4 shell out using whatever interpreter ran `main.py`
(`sys.executable`), so that's what puts them inside the `uv`-managed venv
where opencv/ultralytics/torch are actually installed. Stages 1-2 always use
Isaac's own `python.bat` regardless, so this only matters for `main.py`
itself.

`--input-dir`/`--output-dir` can be given as relative paths (resolved
against wherever you ran the command from) or absolute - every script
resolves them to absolute paths immediately after parsing, so it doesn't
matter that `main.py` and the stage scripts it launches live one directory
down in `src/`.

`workspace/` (or whatever `--output-dir` you choose) ends up with:

```
workspace/
├── usd/<class>.usd
├── parts_manifest.json      # class -> USD path, written by stage 1
├── textures/                # built-in textures generated by stage 2
├── synthetic_dataset/       # raw Replicator output (rgb, masks, mappings)
├── yolo_seg_dataset/        # images/, labels/, data.yaml
├── preview/                 # review sheets (make preview)
└── runs/seg/                # Ultralytics training run, weights/best.pt
```

Pretrained weights are downloaded once into `~/.cache/synth-pipeline/`,
outside the repo.

### Using your own surface photos

Photos of real workbenches, floors, tables or mats narrow the gap to real
images more than anything procedural. Put them in a folder and pass it in;
they're mixed into both the ground and backdrop textures:

```
make generate TEXTURES_DIR=my_photos
```

### Reviewing a dataset

After stage 3, render review sheets with the labels drawn on exactly as
training will read them:

```
make preview                # 5 images per class
make preview PER_CLASS=10
```

This writes `preview/<class>.png` (one sheet per class),
`preview/<class>/` (full-size images) and `preview/all.png`. The sheet takes
the first N frames containing each class, so one frame can appear in more
than one class's row.

Stages 2 and 3 clear their own output folder before writing, so re-running
them never mixes in frames left over from an earlier run.

A ready-to-use example input folder (4 fastener parts) is included at
`examples/fasteners/` — `uv run python src/main.py --input-dir
examples/fasteners --output-dir workspace` runs end to end against it.

### Smoke-testing a new part set

Before committing to a full 2000-frame render (about 40 minutes at ~1.2 s a
frame), check a new input folder with a handful of frames and look at them:

```
make convert INPUT_DIR=my_parts
make generate INPUT_DIR=my_parts NUM_FRAMES=24
make dataset preview
uv run python src/train.py --output-dir workspace --epochs 1 --batch 2
```

`train.py` also takes `--imgsz` and `--name` (the run folder under
`runs/`, default `seg`).

### Makefile

Same commands, shorter. Every target goes through `src/main.py`, so the
Isaac Sim path is only ever read from `pyproject.toml`:

```
make help
make setup          # uv sync
make convert
make generate
make dataset
make train
make all            # every stage, in order
make preview        # review sheets
make clean          # delete OUTPUT_DIR
```

Variables: `INPUT_DIR`, `OUTPUT_DIR`, `NUM_FRAMES`, `TEXTURES_DIR`,
`PER_CLASS`, e.g. `make all INPUT_DIR=my_parts OUTPUT_DIR=out`.

Windows doesn't ship `make`. Install it with
`winget install ezwinports.make`, then **open a new terminal** (terminals
that were already open won't see it on PATH). The Makefile avoids shell
built-ins, so it runs the same from PowerShell/cmd and from Git Bash.

## Why two Python environments

Isaac Sim bundles its **own** standalone Python interpreter (`python.bat`,
wrapping `kit\python\kit.exe`) — a separate distribution from your system
Python, with its own site-packages. That's the only place `omni.*`,
`isaacsim`, and Replicator exist; they aren't pip-installable. So:

- **Stages 1-2** (`convert_assets.py`, `generate_dataset.py`) need
  `omni.*`/Replicator → must run via Isaac's `python.bat`. `uv` is irrelevant
  here; Isaac Sim's interpreter is a separate distribution uv doesn't manage.
- **Stages 3-4** (`build_yolo_dataset.py`, `train.py`) need
  `numpy`/`opencv`/`ultralytics`/`torch` → run inside the `uv`-managed venv.

`main.py` handles the dispatch for you (it shells out to the right
interpreter per stage) as long as you launch it with
`uv run python src/main.py` - see the note in Usage above.

## Repo layout

```
synthetic-data-pipeline/
├── README.md
├── Makefile
├── pyproject.toml       dependencies, uv config, pipeline settings
├── uv.lock
├── .gitignore
├── examples/fasteners/  sample input data - not code, stays out of src/
└── src/
    ├── main.py                  entry points (one per stage + orchestrator)
    ├── convert_assets.py
    ├── generate_dataset.py
    ├── build_yolo_dataset.py
    ├── train.py
    ├── preview_dataset.py       review sheets (not a pipeline stage)
    └── synth_pipeline/          shared library code
        ├── config.py
        └── utils/
            ├── discover.py
            ├── step_to_usd.py
            ├── textures.py
            ├── seg_labels.py
            └── yolo_dataset.py
```

| Path | What |
|---|---|
| `pyproject.toml` | Dependencies + `uv` config (`[tool.uv]`) + pipeline settings (`[tool.synth-pipeline]`: Isaac Sim path, render count/resolution, domain-randomization ranges) |
| `src/main.py` | Orchestrator — dispatches each stage to the right interpreter |
| `src/convert_assets.py` | Stage 1 |
| `src/generate_dataset.py` | Stage 2 |
| `src/build_yolo_dataset.py` | Stage 3 |
| `src/train.py` | Stage 4 |
| `src/preview_dataset.py` | Review sheets: N images per class with labels drawn on |
| `src/synth_pipeline/config.py` | Loads `pyproject.toml`'s `[tool.synth-pipeline]` table into typed Python objects (`Path`, tuples) |
| `src/synth_pipeline/utils/discover.py` | Finds each class's STEP file in the input folder (extension-based, not hardcoded filenames — source CAD downloads name these inconsistently) |
| `src/synth_pipeline/utils/step_to_usd.py` | The actual STEP→USD conversion call |
| `src/synth_pipeline/utils/textures.py` | Built-in tileable surface and environment textures, and finding user photos |
| `src/synth_pipeline/utils/seg_labels.py` | Instance mask → YOLO-seg polygon conversion |
| `src/synth_pipeline/utils/yolo_dataset.py` | Dataset folder layout / `data.yaml` helpers |
| `examples/fasteners/` | 4 sample parts, ready to run against |

The shared code lives in a uniquely named `synth_pipeline` package, not in
top-level `utils`/`config` modules. That's deliberate: once Isaac Sim loads
its bundled OpenCV, `cv2`'s own package directory is added to `sys.path`,
and its internal `utils/` and `config.py` then shadow any top-level
modules with those names.

## Known limitations

- **Sim-to-real gap is unmeasured.** Training only validates against
  synthetic data from the same distribution. Collect real photos of your
  parts and run `model.predict()` against them before trusting deployment
  accuracy; widen `pyproject.toml`'s randomization ranges if it's weak.
- **Ranges are tuned for small parts (~5-40 mm).** Camera distance, clutter
  size, spawn spread and gravity (`[tool.synth-pipeline.physics]`, in stage
  units per s²) all assume parts of that size in mm. For a very different
  part set, adjust them in `pyproject.toml` and check with `make preview`.
- **Procedural textures are a stand-in for real ones.** They vary the
  surface a lot but don't look like real materials up close; photos passed
  via `--textures-dir` are the better source when you have them.
- **Rendering is ~1.2 s per frame** (mostly physics settling), so a
  2000-frame dataset takes about 40 minutes.
- **Noisy but harmless log warnings.** Stage 2 logs Replicator
  `Illegal cycle connection ... WriterSyncGate` warnings on every frame, and
  occasional `rtx.scenedb ... exceed the recommended extents limit`
  warnings. Neither affects the output; errors still stop the run.
- **Weight downloads from GitHub can be flaky.** Ultralytics retries on its
  own; once downloaded, weights are reused from `~/.cache/synth-pipeline/`.
