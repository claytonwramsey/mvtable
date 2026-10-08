//! Construction timing that never builds the same input twice in a row.
//!
//! Construction time for a point cloud depends on whether the CPU has recently seen that same
//! cloud.
//! Rebuilding one cloud over and over trains the branch predictor on that cloud's data and
//! understates the cost of building a fresh one.
//! [`round_robin_medians`] instead cycles through every input once per round.

use std::time::{Duration, Instant};

/// Number of rounds [`round_robin_medians`] runs.
pub const ROUNDS: usize = 5;

/// A first build slower than this is timed only once.
/// At this scale, run-to-run variation is negligible next to the build itself.
pub const REPEAT_LIMIT: Duration = Duration::from_millis(10);

/// Time a call to `build`, excluding the time to drop its result.
pub fn time_build<T>(build: impl FnOnce() -> T) -> (Duration, T) {
    let tic = Instant::now();
    let built = build();
    (tic.elapsed(), built)
}

/// Time `build` on items `0..n_items` for [`ROUNDS`] rounds, and return each item's median time.
///
/// Each round times every item once, in index order.
/// `build` returns the time taken to build item `i`, or `None` if item `i` cannot be built.
/// An item that cannot be built, or whose first build takes longer than [`REPEAT_LIMIT`], is not
/// built again.
pub fn round_robin_medians(
    n_items: usize,
    mut build: impl FnMut(usize) -> Option<Duration>,
) -> Vec<Option<Duration>> {
    let mut times: Vec<Vec<Duration>> = vec![Vec::with_capacity(ROUNDS); n_items];
    let mut repeat = vec![true; n_items];
    for _ in 0..ROUNDS {
        for (i, (item_times, repeat)) in times.iter_mut().zip(&mut repeat).enumerate() {
            if !*repeat {
                continue;
            }
            match build(i) {
                Some(t) => {
                    *repeat = t <= REPEAT_LIMIT;
                    item_times.push(t);
                }
                None => *repeat = false,
            }
        }
    }
    times
        .into_iter()
        .map(|mut item_times| {
            item_times.sort_unstable();
            item_times.get(item_times.len() / 2).copied()
        })
        .collect()
}
