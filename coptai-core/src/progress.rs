//! # progress
//!
//! Tiny trait that lets consumers of the loader receive progress notifications
//! without pulling in any UI dependency into coptai-core.
//!
//! The CLI implements `ProgressReporter` using `indicatif`.
//! Library / headless callers use `NoopProgress` (the default).

/// Implemented by anything that wants to receive loader progress events.
pub trait ProgressReporter: Send + Sync {
    /// Called once after a shard header is parsed, before any tensor is loaded.
    /// `tensor_count` is the total number of tensors in this shard — use it
    /// to initialise a progress bar with the correct total.
    fn on_shard_start(&self, idx: usize, name: &str, tensor_count: usize);

    /// Called after each tensor inside a shard has been processed.
    fn on_tensor(&self, tensor_name: &str);

    /// Called after all tensors in a shard have been processed.
    fn on_shard_done(&self, idx: usize, quantized: usize, kept: usize);
}

/// Zero-cost no-op implementation — used by default in all loader functions.
pub struct NoopProgress;

impl ProgressReporter for NoopProgress {
    fn on_shard_start(&self, _idx: usize, _name: &str, _tensor_count: usize) {}
    fn on_tensor(&self, _tensor_name: &str) {}
    fn on_shard_done(&self, _idx: usize, _quantized: usize, _kept: usize) {}
}
