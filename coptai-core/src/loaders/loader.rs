use std::{collections::HashMap, fs::File, path::Path};

use candle_core::{Device, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::instrument;

use crate::{
    error::CoptaiError,
    progress::{NoopProgress, ProgressReporter},
};
use super::helpers::decode_tensor;

pub type ModelWeights = HashMap<String, Tensor>;

/// Load model weights from one or more `.safetensors` shards.
///
/// Each file is memory-mapped so the OS can page weight data in on demand rather than reading the entire file upfront.
/// opti-check-need need to migrate that to a trait to be agnotisc from candle
#[instrument(skip(paths, device))]
pub fn load_from_safetensors(
    paths: &[&Path],
    device: &Device,
) -> Result<ModelWeights, CoptaiError> {
    let mut weights: ModelWeights = HashMap::new();
    let reporter = &NoopProgress;
    for (idx, path) in paths.iter().enumerate() {
        let shard_name = path.file_name().unwrap_or_default().to_string_lossy();
        tracing::info!(path = %path.display(), "Memory-mapping safetensors shard");

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? }; // SAFETY: file handle kept alive
        let tensors = SafeTensors::deserialize(&mmap)?;

        reporter.on_shard_start(idx, &shard_name, tensors.len());

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let tensor = decode_tensor(view.data(), view.dtype(), &shape, device)?;
            reporter.on_tensor(&name);
            weights.insert(name.to_string(), tensor);
        }

        reporter.on_shard_done(idx, 0, weights.len());
        tracing::info!(path = %path.display(), tensors_loaded = weights.len(), "Shard loaded");
    }

    Ok(weights)
}
