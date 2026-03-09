pub mod error;
pub mod loader;
pub mod model;

pub use error::CoptaiError;
pub use model::{CoptaiModel, OptimizedModel};

pub use candle_core;

use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Top-level configuration driving all optimisation passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub quantization: Option<QuantizationConfig>,
    pub pruning: Option<PruningConfig>,
    pub graph_fusion: bool,
    pub target_device: DeviceConfig,
}

//Default config
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            quantization: None,
            pruning: None,
            graph_fusion: true,
            target_device: DeviceConfig::Cpu,
        }
    }
}

/// Serializable device selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl TryFrom<&DeviceConfig> for Device {
    type Error = CoptaiError;

    fn try_from(cfg: &DeviceConfig) -> Result<Self, Self::Error> {
        match cfg {
            DeviceConfig::Cpu => Ok(Device::Cpu),
            DeviceConfig::Cuda(ordinal) => {
                Device::new_cuda(*ordinal).map_err(CoptaiError::Candle)
            }
            DeviceConfig::Metal(ordinal) => {
                Device::new_metal(*ordinal).map_err(CoptaiError::Candle)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationTarget {
    Int8,
    Int4Awq,
    Int4Gptq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub target: QuantizationTarget,
    /// Number of calibration samples used for activation-aware methods.
    pub calibration_samples: usize,
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Fraction of weights to prune (0.0 – 1.0).
    pub sparsity: f32,
    pub structured: bool,
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Apply all enabled optimisation passes described in `config` to `model`.
/// Only public entrypoint of coptai

pub fn optimize(
    model: CoptaiModel,
    config: OptimizerConfig,
) -> Result<OptimizedModel, CoptaiError> {
    tracing::info!(?config, "Starting optimisation pipeline");

    let device = Device::try_from(&config.target_device)?;

    // --- Graph fusion (no-op stub until pass is implemented) ---
    if config.graph_fusion {
        tracing::debug!("Graph fusion pass: enabled (stub)");
    }

    // --- Quantization pass ---
    if let Some(ref _quant) = config.quantization {
        tracing::debug!("Quantization pass: enabled (stub)");
    }

    // --- Pruning pass ---
    if let Some(ref _prune) = config.pruning {
        tracing::debug!("Pruning pass: enabled (stub)");
    }

    Ok(OptimizedModel {
        inner: model,
        device,
        config,
    })
}
