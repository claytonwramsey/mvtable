//! Support code for extracting realistic point-cloud collision-checking workloads from the
//! MotionBenchMaker dataset, for later replay in `mbm_bench`.
#![feature(portable_simd)]

use core::cell::RefCell;
use std::{
    simd::{Simd, SimdElement},
    time::Instant,
};

use carom::env::World3d;
use carom_core::{BlockValidate, Collide3, Robot};
use mbm::{Problem, SolveStatus};
use rumple::{
    geo::{AdaptiveSettings, adaptive_ddrrtc},
    metric::SquaredEuclidean,
    nn::KiddoMap,
    sample::{HaltonState, Rectangle},
    space::Vector,
    time::{LimitNodes, LimitSamples, Solved},
    valid::Validate,
};

mod pointcloud;

pub use pointcloud::sample_scene;

/// A single collision-checking query and its ground-truth result.
#[derive(Clone, Copy, Debug)]
pub struct RecordedQuery {
    /// Query sphere center, x coordinate.
    pub x: f32,
    /// Query sphere center, y coordinate.
    pub y: f32,
    /// Query sphere center, z coordinate.
    pub z: f32,
    /// Query sphere radius.
    pub r: f32,
    /// Whether this query collided with the world it was checked against.
    pub collided: bool,
}

/// A [`Collide3`] implementation that wraps a point-cloud-only [`World3d`] and records every
/// scalar-equivalent collision query it is asked, in the order they occur.
///
/// SIMD-batched queries (via [`Collide3::collides_balls`]) are unpacked and checked one lane at a
/// time against the wrapped world.
pub struct RecordingWorld {
    world: World3d<f32>,
    log: RefCell<Vec<RecordedQuery>>,
}

impl RecordingWorld {
    #[must_use]
    pub fn new(world: World3d<f32>) -> Self {
        Self {
            world,
            log: RefCell::new(Vec::new()),
        }
    }

    /// Remove and return every query recorded so far.
    pub fn take_queries(&mut self) -> Vec<RecordedQuery> {
        core::mem::take(self.log.get_mut())
    }
}

impl Collide3<f32> for RecordingWorld {
    fn collides_ball(&self, x: f32, y: f32, z: f32, r: f32) -> bool {
        let collided = self.world.collides_ball(x, y, z, r);
        self.log.borrow_mut().push(RecordedQuery {
            x,
            y,
            z,
            r,
            collided,
        });
        collided
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
        let any = self.world.collides_balls(xs, ys, zs, rs);

        // separately record whether any query collided
        let xs = xs.to_array();
        let ys = ys.to_array();
        let zs = zs.to_array();
        let rs = rs.to_array();
        let mut log = self.log.borrow_mut();
        for l in 0..L {
            let collided = self.world.collides_ball(xs[l], ys[l], zs[l], rs[l]);
            log.push(RecordedQuery {
                x: xs[l],
                y: ys[l],
                z: zs[l],
                r: rs[l],
                collided,
            });
        }
        any
    }
}

/// The solution status and queries from solving a planning problem.
pub struct RecordingResult<const N: usize> {
    /// The planner's solve status.
    pub status: SolveStatus<N>,
    /// Every collision query issued while solving, in order.
    pub queries: Vec<RecordedQuery>,
}

/// Run the same adaptive RRT-Connect planner that `mbm::solve` uses, but against a
/// [`RecordingWorld`] instead of a bare `World3d`, so that every collision query issued during
/// planning is captured.
///
/// This mirrors `mbm::solve` closely.
pub fn solve_recording<R, const N: usize>(
    robot: R,
    problem: &Problem<N>,
    world: RecordingWorld,
) -> Result<RecordingResult<N>, Box<dyn std::error::Error>>
where
    R: Robot<N, f32> + BlockValidate<N, f32, RecordingWorld> + Clone,
{
    let sampler = Rectangle::from(robot.bounds().map(Vector));
    let mut valid = carom::Rake::<_, _, _, 8> {
        robot,
        world,
        step_size: 1.0 / 24.0,
    };

    let mut rng = HaltonState::new();

    let mut samples = LimitSamples::new(1_000_000);
    let mut nodes = LimitNodes::new(1_000_000);

    let start = Vector(problem.start_cfg);
    let goal = Vector(problem.end_cfg);

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
            &mut (Solved::new() | &mut samples | &mut nodes),
            &mut rng,
        );
    let toc = Instant::now();

    Ok(RecordingResult {
        status: SolveStatus {
            time: toc - tic,
            samples: samples.n_sampled(),
            nodes: nodes.n_nodes(),
            trajectory,
        },
        queries: valid.world.take_queries(),
    })
}
