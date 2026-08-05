#!/usr/bin/env python3
"""
Generate a Rerun figure of the Fetch mobile manipulator: its visual mesh (solid), with the
three characteristic radii mvtable-bench records for it (mobile_max_radius, robot_max_radius,
true_max_query_radius) drawn as translucent spheres at their real locations on the robot, i.e.
r_mobile, r_max, and r_query.

Fetch was chosen over the panda for this figure because its query/mobile size ratio (0.42/0.15
= 2.8x) is the smallest of the four robots mvtable-bench tracks, so the r_query sphere doesn't
swallow r_mobile as badly as it does on the panda (3.7x) or ur5 (4.4x, where r_max and r_mobile
are also exactly equal in size, so ur5 was a non-starter regardless).

Robot/URDF data is read from a local checkout of the `vamp` project
(https://github.com/KavrakiLab/vamp), which is where mvtable-bench's own hardcoded per-robot
radius constants (crates/mvtable-bench/src/lib.rs) were traced from. Nothing is written back to
that checkout; this script only reads resources/fetch/fetch.urdf and
resources/fetch/fetch_spherized.urdf from it (the latter only to look up where r_max's and
r_mobile's spheres actually sit; r_query has no physical sphere of its own since the BVH it
bounds is implicit inside FK/CC, not a separate structure, so it's drawn centered on r_mobile's
sphere, which it dwarfs).

The full set of collision spheres for the whole robot (translucent, unhighlighted) is instead
computed by `carom` itself — Clayton's unpublished forward-kinematics/collision-checking
library, the actual "carom's generated fkcc" mvtable-bench's docs refer to — via the
`fetch_fk_spheres` helper crate next to this script, so that this layer is ground truth rather
than a second hand-rolled reimplementation of FK. As a sanity check, that crate's r_max sphere
(the base_link sphere position it reports) lands exactly on this script's own URDF-derived
r_max_center, to float precision.

Requires a local clone of `vamp` (path overridable via the VAMP_DIR env var, defaults to
~/projects/vamp), a local clone of `rumple` (github.com/claytonwramsey/rumple, hardcoded in
fetch_fk_spheres/Cargo.toml as ~/projects/rumple since Cargo path deps can't take env vars) for
the `carom` crate, a nightly Rust toolchain (`carom` uses `#![feature(portable_simd)]`), plus
`pip install rerun-sdk yourdfpy trimesh numpy`.

Running this script spawns a Rerun Viewer window directly (via `rr.init(..., spawn=True)`);
it doesn't write an .rrd file.
"""

import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yourdfpy

VAMP_DIR = os.environ.get("VAMP_DIR", os.path.expanduser("~/projects/vamp"))
VAMP_FETCH_DIR = os.path.join(VAMP_DIR, "resources", "fetch")
URDF_PATH = os.path.join(VAMP_FETCH_DIR, "fetch.urdf")
SPHERIZED_URDF_PATH = os.path.join(VAMP_FETCH_DIR, "fetch_spherized.urdf")
FK_SPHERES_CRATE_DIR = os.path.join(os.path.dirname(__file__), "fetch_fk_spheres")

# mvtable-bench's traced-by-hand radii for the fetch (crates/mvtable-bench/src/lib.rs).
R_MOBILE = 0.15  # largest sphere on a link that actually moves during planning
R_MAX = 0.24  # largest sphere on any link (the fixed base, base_link)
R_QUERY = 0.42  # true max collision-query radius carom's generated fkcc ever issues

# The "arm_with_torso" goal pose from resources/fetch/problems/bookshelf_small_fetch/
# request0001.yaml, one of mvtable's own planning problems for this robot.
READY_CFG = {
    "torso_lift_joint": 0.05580749394926036,
    "shoulder_pan_joint": 0.2319594187719277,
    "shoulder_lift_joint": -0.7632272745271215,
    "upperarm_roll_joint": 0.7273950815863892,
    "elbow_flex_joint": 1.421938462868271,
    "forearm_roll_joint": 2.57193125373631,
    "wrist_flex_joint": 0.4549567580598895,
    "wrist_roll_joint": -3.141592599877235,
}

MESH_COLOR = (0.82, 0.83, 0.86, 1.0)
FK_SPHERE_COLOR = (90, 150, 255, 55)  # light blue, translucent — the whole-robot BVH
R_MAX_COLOR = (214, 39, 40, 130)  # red
R_MOBILE_COLOR = (44, 160, 44, 130)  # green — distinct from r_max/r_query even when blended
R_QUERY_COLOR = (148, 103, 189, 90)  # purple


def load_urdf_with_meshes(urdf_path: str) -> yourdfpy.URDF:
    base_dir = os.path.dirname(urdf_path)

    def filename_handler(fname: str) -> str:
        # This URDF's mesh filenames look like "package://meshes/base_link.dae", but "meshes"
        # is a path segment, not a ROS package name, so yourdfpy's stock package-aware handlers
        # strip the wrong thing. Just resolve relative to the URDF.
        if fname.startswith("package://"):
            fname = fname[len("package://") :]
        return os.path.join(base_dir, fname)

    return yourdfpy.URDF.load(
        urdf_path,
        load_meshes=True,
        build_scene_graph=True,
        filename_handler=filename_handler,
    )


def sphere_local_xyz(spherized_urdf_path: str, link_name: str, radius: float) -> np.ndarray:
    """Local-frame xyz of the first sphere of the given radius on the given link."""
    root = ET.parse(spherized_urdf_path).getroot()
    link = next(l for l in root.findall("link") if l.get("name") == link_name)
    for collision in link.findall("collision"):
        sphere = collision.find("geometry/sphere")
        if sphere is not None and float(sphere.get("radius")) == radius:
            origin = collision.find("origin")
            return np.array([float(v) for v in origin.get("xyz").split()])
    raise ValueError(f"no radius-{radius} sphere found on {link_name}")


def transform_point(transform: np.ndarray, local_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.append(local_xyz, 1.0)
    return (transform @ homogeneous)[:3]


def fk_spheres() -> list[dict[str, float]]:
    """The whole robot's ground-truth FK'd collision spheres, from `carom` itself (see
    fetch_fk_spheres/src/main.rs), as a list of {"x", "y", "z", "r"} dicts."""
    result = subprocess.run(
        ["cargo", "run", "--release", "--quiet"],
        cwd=FK_SPHERES_CRATE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    urdf = load_urdf_with_meshes(URDF_PATH)
    urdf.update_cfg(READY_CFG)

    rr.init("fetch_bvh_figure", spawn=True)
    rr.send_blueprint(
        rrb.Spatial3DView(
            origin="/",
            background=rrb.Background(kind=rrb.BackgroundKind.SolidColor, color=(255, 255, 255)),
        )
    )
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # The robot's visual mesh, solid/opaque.
    for i, mesh in enumerate(urdf.scene.dump(concatenate=False)):
        rr.log(
            f"mesh/part_{i:02d}",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                albedo_factor=MESH_COLOR,
            ),
        )

    # The whole robot's collision spheres, ground-truth FK'd by carom itself, translucent and
    # unhighlighted underneath r_max/r_mobile/r_query.
    spheres = fk_spheres()
    rr.log(
        "fk_spheres",
        rr.Ellipsoids3D(
            centers=[(s["x"], s["y"], s["z"]) for s in spheres],
            radii=[s["r"] for s in spheres],
            colors=FK_SPHERE_COLOR,
            fill_mode="solid",
        ),
    )

    # r_max sits on the (fixed) base link.
    r_max_center = transform_point(
        urdf.get_transform("base_link"),
        sphere_local_xyz(SPHERIZED_URDF_PATH, "base_link", R_MAX),
    )
    # r_mobile sits on torso_lift_link: it only translates vertically, but that still counts
    # as "actually moves during planning".
    r_mobile_local = sphere_local_xyz(SPHERIZED_URDF_PATH, "torso_lift_link", R_MOBILE)
    r_mobile_center = transform_point(urdf.get_transform("torso_lift_link"), r_mobile_local)
    # r_query has no physical sphere (the BVH it bounds is implicit inside FK/CC); drawn
    # centered on r_mobile's sphere, which it dwarfs.
    r_query_center = r_mobile_center

    rr.log(
        "highlighted/r_max",
        rr.Ellipsoids3D(
            centers=[r_max_center],
            radii=[R_MAX],
            colors=[R_MAX_COLOR],
            fill_mode="solid",
        ),
    )
    rr.log(
        "highlighted/r_mobile",
        rr.Ellipsoids3D(
            centers=[r_mobile_center],
            radii=[R_MOBILE],
            colors=[R_MOBILE_COLOR],
            fill_mode="solid",
        ),
    )
    rr.log(
        "highlighted/r_query",
        rr.Ellipsoids3D(
            centers=[r_query_center],
            radii=[R_QUERY],
            colors=[R_QUERY_COLOR],
            fill_mode="solid",
        ),
    )

    print("logged to Rerun Viewer")


if __name__ == "__main__":
    main()
