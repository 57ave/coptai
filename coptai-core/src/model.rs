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
    /// Convenience constructor — load all shards from a directory eagerly into RAM.
    /// Use this when you need F32 weights directly (inference, info, bench).
    /// For quantization passes, use `from_dir_lazy` to avoid a double load.
    pub fn from_dir(dir: &Path, device: &Device) -> Result<Self, CoptaiError> {
        let shard_paths = Self::collect_shards(dir)?;
        let refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
        let weights = load_from_safetensors(&refs, device)?;
        Ok(Self { weights, source_paths: shard_paths })
    }

    /// Discover shard paths without loading any weights into RAM.
    /// `weights` will be empty — the optimizer will stream-load during its pass.
    pub fn from_dir_lazy(dir: &Path) -> Result<Self, CoptaiError> {
        let shard_paths = Self::collect_shards(dir)?;
        tracing::info!(shards = shard_paths.len(), "Shards discovered (lazy — no data loaded yet)");
        Ok(Self {
            weights: std::collections::HashMap::new(),
            source_paths: shard_paths,
        })
    }

    fn collect_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>, CoptaiError> {
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

        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(CoptaiError::Unsupported(format!(
                "No .safetensors files found in {}",
                dir.display()
            )));
        }

        Ok(shard_paths)
    }
}

/// A model that has passed through the optimisation pipeline.
pub struct OptimizedModel {
    pub inner: CoptaiModel,
    pub device: Device,
    pub config: OptimizerConfig,
    /// INT8-quantized weights produced by the quantization pass.
    pub quantized_weights: std::collections::HashMap<String, crate::quantization::int8::Int8QuantizedTensor>,
}

impl OptimizedModel {
    /// Return a reference to the underlying F32 weight map (non-quantized tensors).
    pub fn weights(&self) -> &ModelWeights {
        &self.inner.weights
    }

    /// Number of tensors in the original model (F32 + quantized combined).
    pub fn num_tensors(&self) -> usize {
        self.inner.weights.len() + self.quantized_weights.len()
    }

    /// Number of weight matrices that were INT8-quantized.
    pub fn num_quantized(&self) -> usize {
        self.quantized_weights.len()
    }
}
