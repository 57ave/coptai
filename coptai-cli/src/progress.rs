//! # progress
//!
//! Thin wrapper around `indicatif` providing reusable progress bars for coptai-cli.
//! Agnostic from core — implements `coptai_core::ProgressReporter` so it can
//! be passed directly into any loader function.

use std::cell::Cell;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// ---------------------------------------------------------------------------
// Bar constructors (reusable by any command)
// ---------------------------------------------------------------------------

pub fn multi() -> MultiProgress {
    MultiProgress::new()
}

pub fn shard_bar(mp: &MultiProgress, total: u64) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:.bold} [{bar:35.cyan/blue}] {pos}/{len}  {wide_msg}",
        )
        .expect("valid template")
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.set_prefix("shards ");
    pb
}

pub fn tensor_bar(mp: &MultiProgress, total: u64) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:.bold} [{bar:35.green/black}] {pos}/{len}  {wide_msg:.dim}",
        )
        .expect("valid template")
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.set_prefix("tensors");
    pb
}

// ---------------------------------------------------------------------------
// CliProgress — implements ProgressReporter for the loader
// ---------------------------------------------------------------------------

/// Two live bars: one for shards, one for tensors within the current shard.
/// The tensor bar is reset at the start of each shard with the correct total.
pub struct CliProgress {
    #[allow(dead_code)]
    pub mp: MultiProgress,
    pub shard_bar: ProgressBar,
    pub tensor_bar: ProgressBar,
    /// Tracks whether we've set the tensor bar length yet (reset each shard).
    initialised: Cell<bool>,
}

// CliProgress is only used from main (single thread) so this is safe.
unsafe impl Sync for CliProgress {}

impl CliProgress {
    pub fn new(shard_count: u64) -> Self {
        let mp = multi();
        let shard_bar = shard_bar(&mp, shard_count);
        // Tensor bar starts with length 1 — reset to real count in on_shard_start.
        let tensor_bar = tensor_bar(&mp, 1);
        Self {
            mp,
            shard_bar,
            tensor_bar,
            initialised: Cell::new(false),
        }
    }

    pub fn finish(&self) {
        self.tensor_bar.finish_and_clear();
        self.shard_bar.finish_with_message("done ✓");
    }
}

impl coptai_core::ProgressReporter for CliProgress {
    fn on_shard_start(&self, _idx: usize, name: &str, tensor_count: usize) {
        // Reset the tensor bar with the real count for this shard.
        self.tensor_bar.reset();
        self.tensor_bar.set_length(tensor_count as u64);
        self.tensor_bar.set_message(name.to_string());
        self.initialised.set(true);
    }

    fn on_tensor(&self, tensor_name: &str) {
        self.tensor_bar.inc(1);
        self.tensor_bar.set_message(tensor_name.to_string());
    }

    fn on_shard_done(&self, _idx: usize, quantized: usize, kept: usize) {
        self.shard_bar.inc(1);
        self.shard_bar
            .set_message(format!("q={quantized} kept={kept}"));
    }
}

