"""
Stage 2: render the synthetic dataset with instance segmentation masks for
every part converted in stage 1 (see convert_assets.py / parts_manifest.json
under --output-dir).

Each frame:
  1. randomizes appearance - ground texture/tiling/tint, environment backdrop,
     lights, part and clutter materials, which parts and clutter are present;
  2. drops the parts and clutter onto the ground and lets physics settle
     them, so they rest in natural poses (on their side, flat, leaning on
     each other) instead of being placed at arbitrary, half-buried angles;
  3. moves the camera to a random point on an orbit around them and captures.
Ranges live in pyproject.toml's [tool.synth-pipeline.*] tables.

Clears <output-dir>/synthetic_dataset first - Replicator numbers frames from
0 but never deletes old ones, so leftovers from a previous, longer run would
otherwise silently mix into this one.

Run with Isaac Sim's bundled Python, from the repo root:
    <isaac_root>\\python.bat src\\generate_dataset.py --output-dir <folder>
        [--num-frames N] [--textures-dir <folder of photos>] [--seed S]
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

from synth_pipeline import config
from synth_pipeline.utils.textures import find_images, generate_procedural_textures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to convert_assets.py.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=config.NUM_FRAMES,
        help=f"Frames to render (default from pyproject.toml: {config.NUM_FRAMES}). Use a small value to smoke-test.",
    )
    parser.add_argument(
        "--textures-dir",
        type=Path,
        help="Optional folder of photos (workbenches, floors, tables...) mixed into the ground and backdrop "
        "textures alongside the built-in procedural set.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help=f"Random seed (default: {config.SEED}).")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    manifest_path = args.output_dir / "parts_manifest.json"
    if not manifest_path.exists():
        parser.error(f"{manifest_path} not found - run convert_assets.py first (stage 1).")
    args.manifest = json.loads(manifest_path.read_text())
    if not args.manifest:
        parser.error(f"{manifest_path} is empty - no parts were converted in stage 1.")

    args.user_textures = []
    if args.textures_dir:
        args.textures_dir = args.textures_dir.resolve()
        if not args.textures_dir.is_dir():
            parser.error(f"--textures-dir not found: {args.textures_dir}")
        args.user_textures = find_images(args.textures_dir)
        if not args.user_textures:
            parser.error(f"--textures-dir has no images: {args.textures_dir}")
    return args


# Parse/validate args before booting Isaac Sim (~15-20s startup) - so --help
# and bad arguments fail instantly instead of paying that cost first.
args = parse_args()

from isaacsim import SimulationApp

# Console shows errors only, so the progress bar stays readable: Replicator
# logs a harmless "Illegal cycle connection ... WriterSyncGate" warning every
# frame. Warnings still go to Kit's log file. (Must be a launch argument -
# changing the setting after boot has no effect.)
simulation_app = SimulationApp({"headless": True, "extra_args": ["--/log/outputStreamLevel=Error"]})

import numpy as np
import omni.physx
import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from tqdm import tqdm  # bundled with Isaac Sim's Python

# The imperative rep.functional API is used throughout, not the trigger-graph
# API: it builds no OmniGraph, so it avoids a Replicator graph-builder bug
# that scales exponentially with the number of randomized objects, and it
# lets physics settle the scene between randomizing and capturing.
F = rep.functional

# Material families as (count, diffuse min/max RGB, metallic range, roughness
# range). The count sets each family's share of the pool. Channels are
# sampled independently, so every family keeps its color range narrow; one
# wide range produces saturated pastels, not finishes a fastener has.
PART_FINISHES = [
    (10, ((0.60, 0.60, 0.62), (0.75, 0.75, 0.78)), (0.85, 1.0), (0.15, 0.5)),  # steel / zinc plated
    (6, ((0.02, 0.02, 0.02), (0.10, 0.10, 0.10)), (0.4, 0.9), (0.3, 0.7)),  # black oxide
    (5, ((0.70, 0.50, 0.15), (0.90, 0.75, 0.40)), (0.8, 1.0), (0.2, 0.5)),  # brass / yellow zinc
    (3, ((0.80, 0.80, 0.78), (0.95, 0.95, 0.92)), (0.0, 0.1), (0.3, 0.7)),  # white nylon / plastic
    (3, ((0.02, 0.02, 0.02), (0.08, 0.08, 0.08)), (0.0, 0.1), (0.3, 0.8)),  # black nylon / plastic
    (4, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 1.0), (0.1, 0.9)),  # anything - keeps some wildness
]
# Clutter includes metallic greys too, so "shiny grey" alone never means "part".
DISTRACTOR_FINISHES = [
    (20, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 0.3), (0.2, 1.0)),
    (8, ((0.45, 0.45, 0.45), (0.8, 0.8, 0.8)), (0.8, 1.0), (0.15, 0.6)),
]
DISTRACTOR_SHAPES = (F.create.cube, F.create.sphere, F.create.cylinder, F.create.cone, F.create.torus)

SPAWN_GAP = 1.0  # clearance between spawned bodies, and above the ground
PARK_X = 1e4  # hidden bodies wait here on the (much larger) ground collider, far out of view


def _world_range(prim):
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()


class Scene:
    def __init__(self, manifest: dict, ground_textures: list, env_textures: list, rng):
        self.ground_textures, self.env_textures = ground_textures, env_textures
        self.physx = omni.physx.get_physx_interface()

        F.physics.create_physics_scene("/PhysicsScene", gravityMagnitude=config.GRAVITY)

        # Visual ground plane (default plane is 1x1). Collisions use a thick,
        # invisible slab instead: a zero-thickness plane lets thin, fast parts
        # tunnel straight through it. The slab is far larger than the visible
        # ground so hidden bodies can be parked on it out of view.
        self.ground = F.create.plane(scale=config.GROUND_SIZE, name="Ground")
        slab_size = 4 * PARK_X
        slab = F.create.cube(position=(0, 0, -25), scale=(slab_size, slab_size, 50), name="GroundCollider")
        F.modify.visibility(slab, False)
        F.physics.apply_collider(slab)
        self.ground_material = F.create.material(mdl="OmniPBR.mdl", bind_prims=[self.ground], name="GroundMaterial")

        self.dome = F.create.dome_light(texture=env_textures[0], intensity=1000.0, name="Environment")
        self.point_light = F.create.sphere_light(name="PointLight")
        self.camera = F.create.camera(position=(0, -150, 100), look_at=(0, 0, 0), name="Camera")

        self.parts = [self._load_centered_part(i, name, path) for i, (name, path) in enumerate(manifest.items())]
        self.distractors = [
            shape(name=f"Distractor_{shape.__name__}_{i}")
            for shape in DISTRACTOR_SHAPES
            for i in range(config.DISTRACTORS_PER_SHAPE)
        ]
        # A sane depenetration limit: the default (1e5 units/s) fires any body
        # that starts slightly overlapping another clean through the floor.
        for body in self.parts + self.distractors:
            F.physics.apply_rigid_body(
                body, with_collider=True, angularDamping=2.0, linearDamping=0.5, maxDepenetrationVelocity=200.0
            )
        # Radius of each part around its (centered) pivot: a bound that holds in any orientation.
        self.part_radii = [_world_range(p).GetSize().GetLength() / 2 for p in self.parts]

        self.part_materials = self._material_pool(PART_FINISHES, rng, "PartMaterial")
        self.distractor_materials = self._material_pool(DISTRACTOR_FINISHES, rng, "DistractorMaterial")

    @staticmethod
    def _load_centered_part(index: int, class_name: str, usd_path: str):
        """Load a part so its pivot sits at its geometric center.

        CAD exports often put geometry far from the file's origin (the example
        nut is ~78 mm off it, the hex screw's origin is at one end). Posing or
        rotating such a part about that origin swings it out of frame or into
        the ground. So: root (posed, carries the class label) -> pivot
        (offset by -center) -> the referenced file, left untouched.
        """
        root = F.create.xform(semantics={"class": class_name}, name=f"Part_{index}_{class_name}")
        pivot = F.create.xform(parent=root, name="Pivot")
        geometry = F.create.reference(usd_path, parent=pivot, name="Geometry")
        # The CAD converter instances repeated sub-shapes. Rendered from shared
        # prototypes, those picked up corrupt (inf) transforms once the part
        # was moved, and whole frames lost their labels - so de-instance them.
        instanced = [p for p in Usd.PrimRange(geometry) if p.IsInstance()]
        while instanced:
            for prim in instanced:
                prim.SetInstanceable(False)
            instanced = [p for p in Usd.PrimRange(geometry) if p.IsInstance()]
        center = _world_range(geometry).GetMidpoint()
        F.modify.pose(pivot, position_value=tuple(-v for v in center), write_to_usd=True)
        return root

    @staticmethod
    def _material_pool(finishes, rng, name: str) -> list:
        pool = []
        for count, diffuse, metallic, roughness in finishes:
            for _ in range(count):
                pool.append(
                    F.create.material(
                        mdl="OmniPBR.mdl",
                        name=f"{name}_{len(pool)}",
                        diffuse_color_constant=Gf.Vec3f(*rng.uniform(*diffuse)),
                        metallic_constant=float(rng.uniform(*metallic)),
                        reflection_roughness_constant=float(rng.uniform(*roughness)),
                    )
                )
        return pool

    # ---- per frame ---------------------------------------------------------

    def randomize_appearance(self, rng) -> tuple:
        """Everything except body placement. Returns (visible parts, visible distractors)."""
        tile = rng.uniform(*config.GROUND_TEXTURE_TILE)
        mat = self.ground_material
        # Plain string: Replicator's material-attribute helper builds the AssetPath itself.
        F.modify.attribute(mat, "diffuse_texture", str(rng.choice(self.ground_textures)), Sdf.ValueTypeNames.Asset)
        F.modify.attribute(mat, "texture_scale", Gf.Vec2f(config.GROUND_SIZE / tile, config.GROUND_SIZE / tile), Sdf.ValueTypeNames.Float2)
        F.modify.attribute(mat, "texture_rotate", float(rng.uniform(0, 360)), Sdf.ValueTypeNames.Float)
        F.modify.attribute(mat, "diffuse_tint", Gf.Vec3f(*rng.uniform(0.75, 1.0, 3)), Sdf.ValueTypeNames.Color3f)
        F.modify.attribute(mat, "reflection_roughness_constant", float(rng.uniform(0.3, 1.0)), Sdf.ValueTypeNames.Float)

        F.modify.attribute(self.dome, "inputs:texture:file", Sdf.AssetPath(str(rng.choice(self.env_textures))))
        F.modify.attribute(self.dome, "inputs:intensity", float(rng.uniform(*config.DOME_LIGHT_INTENSITY_RANGE)))
        F.modify.pose(self.dome, rotation_value=(0.0, 0.0, float(rng.uniform(0, 360))))

        lo, hi = config.POINT_LIGHT_POSITION_RANGE
        F.modify.pose(self.point_light, position_value=tuple(float(v) for v in rng.uniform(lo, hi)))
        F.modify.attribute(self.point_light, "inputs:intensity", float(rng.uniform(*config.POINT_LIGHT_INTENSITY_RANGE)))
        F.modify.attribute(self.point_light, "inputs:color", Gf.Vec3f(*rng.uniform((0.8, 0.75, 0.65), (1.0, 1.0, 1.0))))

        strength = UsdShade.Tokens.strongerThanDescendants  # beat the materials baked into the CAD files
        for bodies, pool in ((self.parts, self.part_materials), (self.distractors, self.distractor_materials)):
            chosen = [pool[i] for i in rng.integers(0, len(pool), len(bodies))]
            F.modify.material(bodies, chosen, strength=[strength] * len(bodies))

        # "Absent" bodies are parked far out of view by drop_and_settle, not
        # hidden: toggling visibility made the renderer drop a part's class
        # label for a few frames after it was shown again, so it came out
        # unlabelled in the masks.
        part_on = rng.random(len(self.parts)) < config.PART_VISIBLE_PROBABILITY
        dist_on = rng.random(len(self.distractors)) < config.DISTRACTOR_VISIBLE_PROBABILITY
        return part_on, dist_on

    def drop_and_settle(self, rng, part_on, dist_on) -> int:
        """Drop visible bodies onto the ground, let physics settle them, apply the result.

        Returns how many visible parts fell through the ground (should be 0).
        """
        sizes = rng.uniform(*config.DISTRACTOR_SIZE, len(self.distractors))
        F.modify.pose(self.distractors, scale_value=[(float(s),) * 3 for s in sizes], write_to_usd=True)
        # A unit primitive fits in a sphere of radius sqrt(3)/2 around its center.
        dist_radii = list(sizes * 0.87)

        F.physics.reset()
        bodies = self.parts + self.distractors
        radii = self.part_radii + dist_radii
        visible = list(part_on) + list(dist_on)
        spreads = [config.PART_SPREAD] * len(self.parts) + [config.DISTRACTOR_SPREAD] * len(self.distractors)

        positions, placed, stack = [], [], 0.0
        for i, (r, on, spread) in enumerate(zip(radii, visible, spreads)):
            if not on:
                positions.append((PARK_X + 100.0 * i, 0.0, r + SPAWN_GAP))
                continue
            for _ in range(100):  # non-overlapping spot; bodies that start interpenetrating get launched
                xy = rng.uniform(-spread, spread, 2)
                if all(np.hypot(*(xy - q)) > r + rq + SPAWN_GAP for q, rq in placed):
                    z = r + SPAWN_GAP + rng.uniform(0, r)
                    break
            else:  # too crowded: drop it from above the others instead
                stack += 2 * r + SPAWN_GAP
                z = r + SPAWN_GAP + stack
            placed.append((xy, r))
            positions.append((float(xy[0]), float(xy[1]), float(z)))
        rotations = [tuple(float(a) for a in rng.uniform(0, 360, 3)) for _ in bodies]
        F.modify.pose(bodies, position_value=positions, rotation_value=rotations, write_to_usd=True)  # PhysX parses USD

        F.physics.simulate(time=config.SETTLE_SECONDS, step_dt=1.0 / config.PHYSICS_STEPS_PER_SECOND)

        # Read the settled poses from PhysX and apply them ourselves: PhysX's
        # own write-back to USD/Fabric only happens on some frames, which left
        # bodies rendered floating at their drop height. Parked bodies too, so
        # none is left rendering at last frame's spot.
        final_pos, final_rot = [], []
        for body in bodies:
            t = self.physx.get_rigidbody_transformation(str(body.GetPath()))
            x, y, z, w = t["rotation"]  # PhysX quaternions are imaginary-first
            final_pos.append(tuple(float(v) for v in t["position"]))
            final_rot.append(Gf.Rotation(Gf.Quatd(w, x, y, z)))
        for to_usd in (True, False):  # USD, then Fabric (what the renderer reads)
            F.modify.pose(bodies, position_value=final_pos, rotation_value=final_rot, write_to_usd=to_usd)

        return sum(1 for p, r, on in zip(final_pos, self.part_radii, part_on) if on and p[2] < -r)

    def place_camera(self, rng) -> None:
        dist = rng.uniform(*config.CAMERA_DISTANCE)
        elev = np.radians(rng.uniform(*config.CAMERA_ELEVATION_DEG))
        azim = rng.uniform(0.0, 2 * np.pi)
        j = config.CAMERA_LOOK_AT_JITTER
        target = np.array([rng.uniform(-j, j), rng.uniform(-j, j), 0.0])
        offset = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        F.modify.pose(
            self.camera,
            position_value=tuple(float(v) for v in target + offset * dist),
            look_at_value=tuple(float(v) for v in target),
        )


def render(manifest: dict, dataset_dir: Path, ground_textures: list, env_textures: list, num_frames: int, seed: int):
    rng = np.random.default_rng(seed)
    rep.orchestrator.set_capture_on_play(False)
    scene = Scene(manifest, ground_textures, env_textures, rng)

    render_product = rep.create.render_product(str(scene.camera.GetPath()), config.RESOLUTION)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        output_dir=str(dataset_dir),
        rgb=True,
        instance_segmentation=True,
        bounding_box_2d_tight=True,  # cheap to keep; useful as a sanity check
    )
    writer.attach(render_product)

    fell_through = 0
    with tqdm(total=num_frames, desc="Rendering", unit="frame", file=sys.stdout, dynamic_ncols=True) as bar:
        for _ in range(num_frames):
            part_on, dist_on = scene.randomize_appearance(rng)
            fell_through += scene.drop_and_settle(rng, part_on, dist_on)
            scene.place_camera(rng)
            # delta_time=0: capture without advancing the timeline.
            rep.orchestrator.step(rt_subframes=config.RT_SUBFRAMES, delta_time=0.0)
            bar.update()
            if fell_through:
                bar.set_postfix(fell_through=fell_through)

    rep.orchestrator.wait_until_complete()
    writer.detach()
    if fell_through:
        print(f"WARNING: {fell_through} part placement(s) fell through the ground and are missing from their frames.")


def main() -> None:
    dataset_dir = args.output_dir / "synthetic_dataset"
    shutil.rmtree(dataset_dir, ignore_errors=True)

    builtin_ground, builtin_env = generate_procedural_textures(args.output_dir / "textures", args.seed)
    ground_textures = builtin_ground + args.user_textures
    env_textures = builtin_env + args.user_textures
    print(
        f"Textures: {len(builtin_ground)} built-in ground, {len(builtin_env)} built-in environment, "
        f"{len(args.user_textures)} from --textures-dir"
    )

    render(args.manifest, dataset_dir, ground_textures, env_textures, args.num_frames, args.seed)
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
