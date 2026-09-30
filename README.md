# Synthetic Data Pipeline

Generate an instance-segmentation or object-detection training dataset —
and train a model on it — from CAD parts alone. No real photos required.
Point it at a folder of STEP files; it converts each to USD, renders
thousands of domain-randomized frames with per-pixel instance masks, builds
a YOLO dataset with outline or box labels, and trains a YOLO26 model. Use it
from the command line or from a drag-and-drop web page.

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
 3. src/build_yolo_dataset.py masks → YOLO labels (outlines or boxes), train/val split
        │
        ▼
 4. src/train.py              YOLO26n-seg or YOLO26n training (Ultralytics)
```

### What each rendered frame varies

| | |
|---|---|
| **Placement** | Parts and clutter are dropped onto the surface and settled with physics, so they rest in natural poses (on their side, flat, leaning on each other) |
| **Composition** | 0–3 copies of each part per frame, dropped close together (piles, overlaps) or far apart (scattered); 3% of frames have no parts at all |
| **Surface** | 48 built-in procedural textures, mostly realistic (wood, brushed metal, concrete, fabric, tiles, cardboard, speckled laminate, rubber mat, anti-static mat, paper, plain) plus a small share of loud colors; random tiling, rotation and brightness; plus your own photos via `--textures-dir` |
| **Backdrop** | Textured environment (studio gradients, mildly tinted rooms and skies), rotated per frame; visible in low-angle shots, and reflected by metal parts |
| **Part finishes** | Steel/zinc, black oxide, brass/yellow zinc, white and black nylon, plus a share of random colors |
| **Clutter** | Unlabelled cubes, spheres, cylinders and cones with random finishes, including metallic greys |
| **Camera** | Frames one random part at a random apparent size (50–400 px), so small parts are seen up close as often as large ones; random lens (18–60 mm), elevation (20–88°), azimuth and off-center aim |
| **Lights** | Environment intensity; point light position, intensity and warm/neutral tint; a sun-like light with crisp shadows in half the frames |
| **Camera effects** | Stage 3 adds blur, motion blur, sensor noise and JPEG compression to a share of the training images (validation images stay clean) |

Every range is in `pyproject.toml`'s `[tool.synth-pipeline.*]` tables, and a
fixed `seed` makes runs reproducible.

### Works for any part set without retuning

Nothing is tuned to the example fasteners. Stage 2 measures your parts and
scales the scene to them:

- **Size:** every length setting (spacing, clutter size, ground, texture
  scale, light placement) is a multiple of the parts' median bounding
  radius, so 3 mm components and 500 mm brackets get the same kind of
  scene. The camera frames each part by its own size.
- **Units:** gravity comes from the unit the parts were converted in, so mm,
  inch and meter CAD all fall and settle correctly.
- **Class count:** spacing grows with the number of parts dropped, so 2
  classes or 20 give similar crowding. Classes come from the input folders.

Stage 2 prints the scale it measured, e.g.
`Scene scale: median part radius 8.5 units (8.5 mm), 1 unit = 0.001 m`.

Parts are loaded with their pivot moved to their geometric center. CAD
exports often put geometry far from the file's origin (the example nut is
~78 mm off it), which would otherwise make parts land out of frame or rotate
into the ground. Instanced sub-shapes from the CAD converter are also
de-instanced on load; rendered as shared prototypes they picked up corrupt
transforms, and whole frames lost their labels.

Parts that are "absent" from a frame are parked far out of view rather than
hidden. Toggling visibility made the renderer leave a part unlabelled in
the masks for a few frames after it reappeared.

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

### Segmentation or detection

Labels are outlines for instance segmentation by default. For object
detection, build box labels instead:

```
uv run python src/main.py --input-dir my_parts --output-dir workspace --task detect
make all TASK=detect
make dataset TASK=detect        # re-label frames you already rendered
```

Rendering is the same for both, so you can re-run just the dataset stage
to switch. Boxes are tight around each part's visible pixels (the same
region its outline covers). The task is recorded in `data.yaml`, and
`train.py` and `make preview` follow it: training starts from
`yolo26n-seg.pt` for outlines and `yolo26n.pt` for boxes (override with
`--model`).

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
├── yolo_dataset/            # images/, labels/, data.yaml
├── preview/                 # review sheets (make preview)
└── runs/segment/            # Ultralytics training run (runs/detect/ for boxes), weights/best.pt
```

Pretrained weights are downloaded once into `~/.cache/synth-pipeline/`,
outside the repo.

`data.yaml` uses paths relative to itself, so `yolo_dataset/` can be moved
or copied to another machine and still train.

### Web UI

For a drag-and-drop version that anyone can use without the command line:

```
make ui                     # or: uv run python src/server.py --port 8000
```

Then open http://127.0.0.1:8000. Drop one STEP file per part and type each
one's **class name**: what the model will call that part, e.g. `hex_nut_M5`
(letters, digits, `-` and `_`; no spaces, since class names also become
file and USD names). It's pre-filled from the file name and selected when
you drop the file, so you can just type over it. Pick how many images to
render and the label type (**Outlines** for segmentation or **Boxes** for
detection), and start. The page shows each stage's progress with a time
estimate, then a labelled preview and a **Download dataset (.zip)** button.
The zip holds the `yolo_dataset/` folder.

Jobs run one at a time in the order they're started, each in its own folder
under `jobs/`, through the same stages as the command line (so Isaac Sim
must be set up as in Setup). The job list is kept in memory, so restarting
the server clears it (finished zips stay in `jobs/`). The server has no
login: it listens on localhost by default, and should only be opened to a
trusted network (`--host 0.0.0.0`).

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

Before committing to a full 2000-frame render (about 30 minutes), check a new input folder with a handful of frames and look at them:

```
make convert INPUT_DIR=my_parts
make generate INPUT_DIR=my_parts NUM_FRAMES=24
make dataset preview
uv run python src/train.py --output-dir workspace --epochs 1 --batch 2
```

`train.py` also takes `--imgsz`, `--device`, `--name` (the run folder under
`runs/`, default the task name) and `--model` (default `yolo26n-seg.pt` or
`yolo26n.pt` to match the dataset; `yolo26s` or larger is more accurate,
especially on small parts, but slower).

Stage 3 ends with a per-class health check: how many instances, their
median size in pixels, and the share smaller than 20×20 px. A class that is
rare or mostly tiny will train poorly; fix it in the settings before
spending time on training.

Each instance's outline label is one polygon: pieces split by occlusion are
joined, and holes (a nut's bore) stay open.

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
make ui             # web UI on http://127.0.0.1:8000
make clean          # delete OUTPUT_DIR
```

Variables: `INPUT_DIR`, `OUTPUT_DIR`, `NUM_FRAMES`, `TEXTURES_DIR`, `TASK`
(`segment` or `detect`), `PER_CLASS`, `PORT`, e.g.
`make all INPUT_DIR=my_parts OUTPUT_DIR=out TASK=detect`.

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
    ├── server.py                web UI backend (make ui)
    ├── web/index.html           web UI page
    └── synth_pipeline/          shared library code
        ├── config.py
        └── utils/
            ├── discover.py
            ├── step_to_usd.py
            ├── textures.py
            ├── labels.py
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
| `src/server.py`, `src/web/index.html` | Web UI: upload STEP files, run the pipeline, download the dataset |
| `src/synth_pipeline/config.py` | Loads `pyproject.toml`'s `[tool.synth-pipeline]` table into typed Python objects (`Path`, tuples) |
| `src/synth_pipeline/utils/discover.py` | Finds each class's STEP file in the input folder (extension-based, not hardcoded filenames — source CAD downloads name these inconsistently) |
| `src/synth_pipeline/utils/step_to_usd.py` | The actual STEP→USD conversion call |
| `src/synth_pipeline/utils/textures.py` | Built-in tileable surface and environment textures, and finding user photos |
| `src/synth_pipeline/utils/labels.py` | Instance mask → YOLO outline polygons or boxes |
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
- **Part finishes assume mechanical parts** (metals, black oxide, plastics).
  For other kinds of objects, edit `PART_FINISHES` in `generate_dataset.py`.
- **Very mixed sizes in one set** (say 3 mm and 500 mm parts together) work,
  but clutter and spacing follow the median part, so the extremes look less
  natural. Check with `make preview`.
- **More classes render slower**: each class adds up to 3 physics bodies.
- **Procedural textures are a stand-in for real ones.** They vary the
  surface a lot but don't look like real materials up close; photos passed
  via `--textures-dir` are the better source when you have them.
- **Rendering is ~0.9 s per frame** (mostly physics settling), so a
  2000-frame dataset takes about 30 minutes.
- **Stage 2's console shows errors only**, plus a progress bar with ETA.
  Warnings (e.g. Replicator's harmless per-frame `Illegal cycle connection
  ... WriterSyncGate`) still go to Kit's log file, whose path is printed at
  startup.
- **Weight downloads from GitHub can be flaky.** Ultralytics retries on its
  own; once downloaded, weights are reused from `~/.cache/synth-pipeline/`.
