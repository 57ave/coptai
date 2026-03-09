use std::path::Path;

use candle_core::Device;

use crate::{
    error::CoptaiError,
    loader::{load_from_safetensors, ModelWeights},
    OptimizerConfig,
};

/// An unoptimised model loaded from disk.
pub struct CoptaiModel {
    pub weights: ModelWeights,
    pub source_paths: Vec<std::path::PathBuf>,
}

impl CoptaiModel {
    /// Convenience constructor — load all shards from a directory.
    pub fn from_dir(dir: &Path, device: &Device) -> Result<Self, CoptaiError> {
        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .collect();

        shard_paths.sort(); // deterministic shard ordering

        if shard_paths.is_empty() {
            return Err(CoptaiError::Unsupported(format!(
                "No .safetensors files found in {}",
                dir.display()
            )));
        }

        let refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
        let weights = load_from_safetensors(&refs, device)?;

        Ok(Self {
            weights,
            source_paths: shard_paths,
        })
    }
}

/// A model that has passed through the optimisation pipeline.
pub struct OptimizedModel {
    pub inner: CoptaiModel,
    pub device: Device,
    pub config: OptimizerConfig,
}

impl OptimizedModel {
    /// Return a reference to the underlying weight map.
    pub fn weights(&self) -> &ModelWeights {
        &self.inner.weights
    }

    /// Report the number of parameters (tensors) in the model.
    pub fn num_tensors(&self) -> usize {
        self.inner.weights.len()
    }
}
