"""
Stage 2: render the synthetic dataset with instance segmentation masks for
every part converted in stage 1 (see convert_assets.py / parts_manifest.json
under --output-dir).

Each frame:
  1. randomizes appearance - ground texture/tiling/tint, environment backdrop,
     lights, part and clutter materials, how many of each part (0-N) and
     which clutter are present;
  2. drops the parts and clutter onto the ground and lets physics settle
     them, so they rest in natural poses (on their side, flat, leaning on
     each other, in piles) instead of arbitrary, half-buried angles;
  3. frames one random part at a random apparent size, lens and viewing
     angle, and captures.
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
import re
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
DISTRACTOR_SHAPES = (F.create.cube, F.create.sphere, F.create.cylinder, F.create.cone)

def _world_range(prim):
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()


def _meters_per_unit(manifest: dict) -> float:
    """The parts' length unit. Parts are used at their authored size (no
    rescaling), so the scene works in that unit - which sets gravity."""
    units = {name: UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(path)) for name, path in manifest.items()}
    if len({round(u, 9) for u in units.values()}) > 1:
        raise ValueError(f"Parts use different length units (meters per unit): {units}. Re-export them in one unit.")
    return next(iter(units.values()))


class Scene:
    def __init__(self, manifest: dict, ground_textures: list, env_textures: list, rng):
        self.ground_textures, self.env_textures = ground_textures, env_textures
        self.physx = omni.physx.get_physx_interface()

        # Enough copies of every class for the most crowded frame; each frame
        # uses a random number of them and parks the rest.
        copies = config.INSTANCES_PER_CLASS[1]
        self.parts, self.part_class = [], []
        for name, path in manifest.items():
            for _ in range(copies):
                self.parts.append(self._load_centered_part(len(self.parts), name, path))
                self.part_class.append(name)
        self.classes = list(manifest)
        # Radius of each part around its (centered) pivot: a bound that holds in any orientation.
        self.part_radii = [_world_range(p).GetSize().GetLength() / 2 for p in self.parts]

        # Scene scale R: every length setting is a multiple of it, so the same
        # settings fit 5 mm screws and 500 mm brackets, in any CAD unit.
        self.R = R = float(np.median(self.part_radii))
        mpu = _meters_per_unit(manifest)
        print(f"Scene scale: median part radius {R:.3g} units ({R * mpu * 1000:.3g} mm), 1 unit = {mpu:g} m")
        F.physics.create_physics_scene("/PhysicsScene", gravityMagnitude=9.81 / mpu)
        self.spawn_gap = 0.12 * R  # clearance between spawned bodies, and above the ground
        self.park_x = 1000 * R  # absent bodies wait here on the (much larger) ground collider, far out of view

        # Visual ground plane (default plane is 1x1). Collisions use a thick,
        # invisible slab instead: a zero-thickness plane lets thin, fast parts
        # tunnel straight through it. The slab is far larger than the visible
        # ground so absent bodies can be parked on it out of view.
        self.ground = F.create.plane(scale=config.GROUND_SIZE * R, name="Ground")
        slab_size = 4 * self.park_x
        slab = F.create.cube(position=(0, 0, -3 * R), scale=(slab_size, slab_size, 6 * R), name="GroundCollider")
        F.modify.visibility(slab, False)
        F.physics.apply_collider(slab)
        self.ground_material = F.create.material(mdl="OmniPBR.mdl", bind_prims=[self.ground], name="GroundMaterial")

        self.dome = F.create.dome_light(texture=env_textures[0], intensity=1000.0, name="Environment")
        # Light radius scales with the scene, so the same intensity gives the same exposure at any size.
        self.point_light = F.create.sphere_light(name="PointLight")
        F.modify.attribute(self.point_light, "inputs:radius", float(config.POINT_LIGHT_RADIUS * R))
        self.sun = F.create.distant_light(name="Sun")
        self.camera = F.create.camera(
            position=(0, -20 * R, 12 * R), look_at=(0, 0, 0), clipping_range=(0.01 * R, 1e4 * R), name="Camera"
        )

        self.distractors = [
            shape(name=f"Distractor_{shape.__name__}_{i}")
            for shape in DISTRACTOR_SHAPES
            for i in range(config.DISTRACTORS_PER_SHAPE)
        ]
        # A sane depenetration limit: the default (1e5 units/s) fires any body
        # that starts slightly overlapping another clean through the floor.
        for body in self.parts + self.distractors:
            F.physics.apply_rigid_body(
                body, with_collider=True, angularDamping=2.0, linearDamping=0.5, maxDepenetrationVelocity=25.0 * R
            )

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
        # The label keeps the class name as-is; the prim name can't hold "-" or spaces.
        prim_name = re.sub(r"[^A-Za-z0-9_]", "_", class_name)
        root = F.create.xform(semantics={"class": class_name}, name=f"Part_{index}_{prim_name}")
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
        repeats = config.GROUND_SIZE / rng.uniform(*config.GROUND_TEXTURE_TILE)  # both in units of R
        mat = self.ground_material
        # Plain string: Replicator's material-attribute helper builds the AssetPath itself.
        F.modify.attribute(mat, "diffuse_texture", str(rng.choice(self.ground_textures)), Sdf.ValueTypeNames.Asset)
        F.modify.attribute(mat, "texture_scale", Gf.Vec2f(repeats, repeats), Sdf.ValueTypeNames.Float2)
        F.modify.attribute(mat, "texture_rotate", float(rng.uniform(0, 360)), Sdf.ValueTypeNames.Float)
        # Brightness plus a slight warm/cool cast - not a per-channel tint, which turns grey surfaces pink or green.
        tint = rng.uniform(0.7, 1.0) * (1.0 + rng.uniform(-0.05, 0.05, 3))
        F.modify.attribute(mat, "diffuse_tint", Gf.Vec3f(*tint), Sdf.ValueTypeNames.Color3f)
        F.modify.attribute(mat, "reflection_roughness_constant", float(rng.uniform(0.3, 1.0)), Sdf.ValueTypeNames.Float)

        F.modify.attribute(self.dome, "inputs:texture:file", Sdf.AssetPath(str(rng.choice(self.env_textures))))
        F.modify.attribute(self.dome, "inputs:intensity", float(rng.uniform(*config.DOME_LIGHT_INTENSITY_RANGE)))
        F.modify.pose(self.dome, rotation_value=(0.0, 0.0, float(rng.uniform(0, 360))))

        lo, hi = config.POINT_LIGHT_POSITION_RANGE
        F.modify.pose(self.point_light, position_value=tuple(float(v) * self.R for v in rng.uniform(lo, hi)))
        F.modify.attribute(self.point_light, "inputs:intensity", float(rng.uniform(*config.POINT_LIGHT_INTENSITY_RANGE)))
        F.modify.attribute(self.point_light, "inputs:color", Gf.Vec3f(*rng.uniform((0.8, 0.75, 0.65), (1.0, 1.0, 1.0))))

        sun_on = rng.random() < config.DISTANT_LIGHT_PROBABILITY
        F.modify.attribute(self.sun, "inputs:intensity", float(rng.uniform(*config.DISTANT_LIGHT_INTENSITY_RANGE)) if sun_on else 0.0)
        tilt = 90.0 - rng.uniform(*config.DISTANT_LIGHT_ELEVATION_DEG)  # the light shines along its -Z
        F.modify.pose(self.sun, rotation_value=(float(tilt), 0.0, float(rng.uniform(0, 360))))

        strength = UsdShade.Tokens.strongerThanDescendants  # beat the materials baked into the CAD files
        for bodies, pool in ((self.parts, self.part_materials), (self.distractors, self.distractor_materials)):
            chosen = [pool[i] for i in rng.integers(0, len(pool), len(bodies))]
            F.modify.material(bodies, chosen, strength=[strength] * len(bodies))

        # "Absent" bodies are parked far out of view by drop_and_settle, not
        # hidden: toggling visibility made the renderer drop a part's class
        # label for a few frames after it was shown again, so it came out
        # unlabelled in the masks.
        counts = {name: int(rng.integers(config.INSTANCES_PER_CLASS[0], config.INSTANCES_PER_CLASS[1] + 1)) for name in self.classes}
        if rng.random() < config.EMPTY_FRAME_PROBABILITY:
            counts = dict.fromkeys(counts, 0)
        part_on = np.zeros(len(self.parts), bool)
        for i, name in enumerate(self.part_class):
            if counts[name] > 0:
                part_on[i], counts[name] = True, counts[name] - 1
        dist_on = rng.random(len(self.distractors)) < config.DISTRACTOR_VISIBLE_PROBABILITY
        return part_on, dist_on

    def drop_and_settle(self, rng, part_on, dist_on) -> int:
        """Drop visible bodies onto the ground, let physics settle them, apply the result.

        Returns how many visible parts fell through the ground (should be 0).
        """
        sizes = rng.uniform(*config.DISTRACTOR_SIZE, len(self.distractors)) * self.R
        F.modify.pose(self.distractors, scale_value=[(float(s),) * 3 for s in sizes], write_to_usd=True)
        # A unit primitive fits in a sphere of radius sqrt(3)/2 around its center.
        dist_radii = list(sizes * 0.87)

        F.physics.reset()
        bodies = self.parts + self.distractors
        radii = self.part_radii + dist_radii
        visible = list(part_on) + list(dist_on)
        # Spread grows with the number of parts dropped, so crowding (and how
        # often parts pile up) doesn't depend on how many classes there are.
        part_spread = rng.uniform(*config.PART_SPREAD) * self.R * np.sqrt(max(int(np.sum(part_on)), 1))
        spreads = [part_spread] * len(self.parts) + [config.DISTRACTOR_SPREAD * self.R] * len(self.distractors)
        gap = self.spawn_gap

        positions, placed, stack = [], [], 0.0
        for i, (r, on, spread) in enumerate(zip(radii, visible, spreads)):
            if not on:
                positions.append((self.park_x + 20 * self.R * i, 0.0, r + gap))
                continue
            for _ in range(100):  # non-overlapping spot; bodies that start interpenetrating get launched
                xy = rng.uniform(-spread, spread, 2)
                if all(np.hypot(*(xy - q)) > r + rq + gap for q, rq in placed):
                    z = r + gap + rng.uniform(0, r)
                    break
            else:  # too crowded: drop it from above the others instead
                stack += 2 * r + gap
                z = r + gap + stack
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

        # What the camera can frame: visible parts, else visible clutter, with their bounding radii.
        on_parts = [(p, r) for p, r, on in zip(final_pos, self.part_radii, part_on) if on]
        on_clutter = [(p, r) for p, r, on in zip(final_pos[len(self.parts) :], dist_radii, dist_on) if on]
        self.framing_candidates = on_parts or on_clutter or [((0.0, 0.0, 0.0), self.R)]
        return sum(1 for p, r, on in zip(final_pos, self.part_radii, part_on) if on and p[2] < -r)

    def place_camera(self, rng) -> None:
        """Frame one random visible part at a random apparent size.

        Picking the distance from a target size in pixels, instead of a fixed
        distance range, keeps small parts (a 5 mm nut) from being only a few
        pixels in most frames while large parts fill the view.
        """
        focal = float(rng.uniform(*config.CAMERA_FOCAL_LENGTH))
        aperture = self.camera.GetAttribute("horizontalAperture").Get()
        focal_px = config.RESOLUTION[0] * focal / aperture
        center, radius = self.framing_candidates[rng.integers(len(self.framing_candidates))]
        size_px = rng.uniform(*config.CAMERA_TARGET_PIXELS)
        dist = max(focal_px * 2 * radius / size_px, 3 * radius)

        # Shift the aim point so the framed part lands off-center, but stays in frame.
        half_view = dist * aperture / (2 * focal)
        shift = rng.uniform(-1, 1, 2) * config.CAMERA_FRAME_OFFSET * half_view
        target = np.array(center) + np.array([shift[0], shift[1], 0.0])

        elev = np.radians(rng.uniform(*config.CAMERA_ELEVATION_DEG))
        azim = rng.uniform(0.0, 2 * np.pi)
        offset = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        F.modify.attribute(self.camera, "focalLength", focal)
        F.modify.attribute(self.camera, "focusDistance", float(dist))
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
