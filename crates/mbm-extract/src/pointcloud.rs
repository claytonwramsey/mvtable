//! Surface-sampling of MotionBenchMaker scene geometry into point clouds.

use std::f32::consts::{FRAC_PI_2, TAU};

use carom::env::{Capsule, Cuboid, World3d, ZCuboid};
use nalgebra::{Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, Vector3};
use rand::{Rng, RngExt};

/// Surface-sample every primitive in `world` (Z-aligned cuboids, arbitrarily-rotated cuboids, and
/// capsules) into a point cloud, at `density` points per unit area (or, for capsules, points per
/// unit tube area).
///
/// `world` is expected to contain no balls, AABBs, or existing point clouds; callers should check
/// for these before calling.
#[must_use]
pub fn sample_scene(world: &World3d<f32>, density: f32, rng: &mut impl Rng) -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    for z_cuboid in &world.z_cuboids {
        sample_z_cuboid(z_cuboid, density, &mut points, rng);
    }
    for cuboid in &world.cuboids {
        sample_cuboid(cuboid, density, &mut points, rng);
    }
    for capsule in &world.capsules {
        sample_capsule(capsule, density, &mut points, rng);
    }
    points
}

fn sample_capsule(
    capsule: &Capsule<3, f32>,
    density: f32,
    points: &mut Vec<[f32; 3]>,
    rng: &mut impl Rng,
) {
    let cap_start = Translation3::from(capsule.c1);
    let cap_vec = Translation3::from(*capsule.vector());
    let length = cap_vec.vector.norm();
    let tube_area = capsule.radius * TAU * length;
    let tf = cap_start
        * UnitQuaternion::rotation_between(&Vector3::new(0.0, 0.0, 1.0), &cap_vec.vector)
            .unwrap_or_else(UnitQuaternion::identity);

    for _ in 0..(tube_area * density) as usize {
        let theta = rng.random_range(0.0..TAU);
        let h = rng.random_range(0.0..=length);
        let tf_point = tf
            * UnitQuaternion::from_euler_angles(0.0, 0.0, theta)
            * Point3::new(capsule.radius, 0.0, h);
        points.push(tf_point.coords.data.0[0]);
    }
}

fn sample_z_cuboid(
    z_cuboid: &ZCuboid<f32>,
    density: f32,
    points: &mut Vec<[f32; 3]>,
    rng: &mut impl Rng,
) {
    let [wx, wy, wz] = z_cuboid.half_widths;

    let &[[r00, r01], [r10, r11]] = z_cuboid.axes();
    let r_mat = Matrix3::new(r00, r01, 0.0, r10, r11, 0.0, 0.0, 0.0, 1.0).transpose();
    let center_iso = Translation3::from(z_cuboid.position) * UnitQuaternion::from_matrix(&r_mat);

    sample_rect_pair(&center_iso, [wx, wy], wz, density, points, rng);
    sample_rect_pair(
        &(center_iso * UnitQuaternion::from_euler_angles(0.0, FRAC_PI_2, 0.0)),
        [wz, wy],
        wx,
        density,
        points,
        rng,
    );
    sample_rect_pair(
        &(center_iso * UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0)),
        [wx, wz],
        wy,
        density,
        points,
        rng,
    );
}

/// Like [`sample_z_cuboid`], but for a cuboid rotated arbitrarily in 3D (not just about the Z
/// axis). `cuboid.axes()` gives the world-frame unit vectors of the cuboid's local axes, which
/// are exactly the columns of the local-to-world rotation matrix.
fn sample_cuboid(
    cuboid: &Cuboid<3, f32>,
    density: f32,
    points: &mut Vec<[f32; 3]>,
    rng: &mut impl Rng,
) {
    let [wx, wy, wz] = cuboid.half_widths;
    let &[[a1x, a1y, a1z], [a2x, a2y, a2z], [a3x, a3y, a3z]] = cuboid.axes();
    #[rustfmt::skip]
    let r_mat = Matrix3::new(
        a1x, a2x, a3x,
        a1y, a2y, a3y,
        a1z, a2z, a3z,
    );
    let center_iso = Translation3::from(cuboid.position) * UnitQuaternion::from_matrix(&r_mat);

    sample_rect_pair(&center_iso, [wx, wy], wz, density, points, rng);
    sample_rect_pair(
        &(center_iso * UnitQuaternion::from_euler_angles(0.0, FRAC_PI_2, 0.0)),
        [wz, wy],
        wx,
        density,
        points,
        rng,
    );
    sample_rect_pair(
        &(center_iso * UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0)),
        [wx, wz],
        wy,
        density,
        points,
        rng,
    );
}

fn sample_rect_pair(
    center: &Isometry3<f32>,
    half_widths: [f32; 2],
    offset: f32,
    density: f32,
    points: &mut Vec<[f32; 3]>,
    rng: &mut impl Rng,
) {
    let a = half_widths[0] * half_widths[1] * 4.0;
    for _ in 0..(a * density) as usize {
        let top_transform = Translation3::new(0.0, 0.0, offset);
        let top_iso = center * top_transform;
        points.push(sample_face_1(&top_iso, half_widths, rng));
        let bottom_iso = center * top_transform.inverse();
        points.push(sample_face_1(&bottom_iso, half_widths, rng));
    }
}

fn sample_face_1(center: &Isometry3<f32>, half_widths: [f32; 2], rng: &mut impl Rng) -> [f32; 3] {
    let x = rng.random_range(-half_widths[0]..=half_widths[0]);
    let y = rng.random_range(-half_widths[1]..=half_widths[1]);
    (center * Point3::new(x, y, 0.0)).coords.data.0[0]
}
