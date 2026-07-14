//! End-to-end motion planning benchmarks.
#![feature(portable_simd)]

use std::{
    simd::{Simd, SimdElement},
    time::Instant,
};

use carom_core::{BlockValidate, Collide3, Robot};
use mbm::{Problem, SolveStatus};
use mvtable_bench::{SimdStructure, Structure};
use rumple::{
    geo::{AdaptiveSettings, adaptive_ddrrtc},
    metric::SquaredEuclidean,
    nn::KiddoMap,
    sample::{HaltonState, Rectangle},
    space::Vector,
    time::{Alarm, LimitNodes, LimitSamples, Solved},
    valid::Validate,
};

pub use mbm_extract::sample_scene;

/// A [`Collide3`] world backed by any point-cloud collision structure implementing
/// [`mvtable_bench::Structure`], so the same planner code can be driven against
/// `mvtable::Mvt`, `mvtable::MutableMvt`, `capt::Capt`, or `kiddo::ImmutableKdTree`
/// interchangeably.
pub struct PointCloudWorld<S>(pub S);

/// A SIMD-accelerated point cloud collision checker, akin to [`PointCloudWorld`].
pub struct SimdPointCloudWorld<S>(pub S);

impl<S: Structure<3>> Collide3<f32> for PointCloudWorld<S> {
    fn collides_ball(&self, x: f32, y: f32, z: f32, r: f32) -> bool {
        self.0.collides(&[x, y, z], r)
    }

    fn collides_balls<const L: usize>(
        &self,
        xs: Simd<f32, L>,
        ys: Simd<f32, L>,
        zs: Simd<f32, L>,
        rs: Simd<f32, L>,
    ) -> bool
    where
        f32: SimdElement,
    {
        let xs = xs.to_array();
        let ys = ys.to_array();
        let zs = zs.to_array();
        let rs = rs.to_array();
        (0..L).any(|l| self.0.collides(&[xs[l], ys[l], zs[l]], rs[l]))
    }
}

impl<S: SimdStructure<3>> Collide3<f32> for SimdPointCloudWorld<S> {
    fn collides_ball(&self, x: f32, y: f32, z: f32, r: f32) -> bool {
        self.0.collides(&[x, y, z], r)
    }

    fn collides_balls<const L: usize>(
        &self,
        xs: Simd<f32, L>,
        ys: Simd<f32, L>,
        zs: Simd<f32, L>,
        rs: Simd<f32, L>,
    ) -> bool
    where
        f32: SimdElement,
    {
        self.0.collides_simd(&[xs, ys, zs], rs)
    }
}

/// Solve a MBM problem and report its status.
pub fn solve_with_backend<R, W, const N: usize>(
    robot: R,
    problem: &Problem<N>,
    world: W,
    max_solve_time: std::time::Duration,
) -> Result<SolveStatus<N>, Box<dyn std::error::Error>>
where
    R: Robot<N, f32> + BlockValidate<N, f32, W> + Clone,
{
    let sampler = Rectangle::from(robot.bounds().map(Vector));
    let valid = carom::Rake::<_, _, _, 8> {
        robot,
        world,
        step_size: 1.0 / 24.0,
    };

    let mut rng = HaltonState::new();

    let mut samples = LimitSamples::new(1_000_000);
    let mut nodes = LimitNodes::new(1_000_000);
    let mut alarm = Alarm::from_now(max_solve_time);

    let start = Vector(problem.start_cfg);
    let goal = Vector(problem.end_cfg);

    if !sampler.contains(&start) {
        return Err("start out of bounds".into());
    }
    if !sampler.contains(&goal) {
        return Err("goal out of bounds".into());
    }
    if !valid.is_valid_configuration(&start) {
        return Err("invalid start".into());
    }
    if !valid.is_valid_configuration(&goal) {
        return Err("invalid goal".into());
    }

    let tic = Instant::now();
    let trajectory =
        adaptive_ddrrtc::<_, KiddoMap<_, N, SquaredEuclidean>, _, _, _, SquaredEuclidean, _, _>(
            Vector(problem.start_cfg),
            Vector(problem.end_cfg),
            &valid,
            &sampler,
            &AdaptiveSettings {
                range: 1.0,
                r_nom: 25.0,
                alpha: 1e-4,
            },
            &mut (Solved::new() | &mut samples | &mut nodes | &mut alarm),
            &mut rng,
        );
    let toc = Instant::now();

    Ok(SolveStatus {
        time: toc - tic,
        samples: samples.n_sampled(),
        nodes: nodes.n_nodes(),
        trajectory,
    })
}

/// Whether [`sample_scene`] can convert every primitive in `world` into a point cloud.
#[must_use]
pub fn is_sampleable(world: &carom::env::World3d<f32>) -> bool {
    world.aabbs.is_empty() && world.balls.is_empty() && world.point_clouds.is_empty()
}
