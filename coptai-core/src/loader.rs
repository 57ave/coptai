use std::{collections::HashMap, fs::File, path::Path};

use candle_core::{Device, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::instrument;

use crate::{
    error::CoptaiError,
    progress::{NoopProgress, ProgressReporter},
    quantization::int8::{int8_quantize, Int8QuantizedTensor},
};

/// Raw weight storage — a name-keyed map of candle `Tensor`s.
pub type ModelWeights = HashMap<String, Tensor>;

/// Load model weights from one or more `.safetensors` shards.
///
/// Each file is memory-mapped so the OS can page weight data in on demand
/// rather than reading the entire file upfront.
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

// ---------------------------------------------------------------------------
// Streaming INT8 loader
// ---------------------------------------------------------------------------

pub struct QuantizedShards {
    pub quantized: HashMap<String, Int8QuantizedTensor>,
    pub other: ModelWeights,
}

/// Stream shards and INT8-quantize 2-D weight matrices immediately.
/// Default no-op progress — use `load_and_quantize_int8_with_progress` to attach a reporter.
#[instrument(skip(paths, device))]
pub fn load_and_quantize_int8(
    paths: &[&Path],
    device: &Device,
) -> Result<QuantizedShards, CoptaiError> {
    load_and_quantize_int8_with_progress(paths, device, &NoopProgress)
}

/// Same as `load_and_quantize_int8` but drives any `ProgressReporter` you pass in.
/// The CLI passes its `CliProgress` here; library users pass `&NoopProgress`.
pub fn load_and_quantize_int8_with_progress(
    paths: &[&Path],
    device: &Device,
    reporter: &dyn ProgressReporter,
) -> Result<QuantizedShards, CoptaiError> {
    let mut quantized: HashMap<String, Int8QuantizedTensor> = HashMap::new();
    let mut other: ModelWeights = HashMap::new();

    for (idx, path) in paths.iter().enumerate() {
        let shard_name = path.file_name().unwrap_or_default().to_string_lossy();
        tracing::info!(path = %path.display(), "Streaming INT8: memory-mapping shard");

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? }; // SAFETY: file handle kept alive
        let tensors = SafeTensors::deserialize(&mmap)?;

        reporter.on_shard_start(idx, &shard_name, tensors.len());

        let mut shard_q = 0usize;
        let mut shard_other = 0usize;

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let tensor = decode_tensor(view.data(), view.dtype(), &shape, device)?;

            if tensor.rank() == 2 {
                let qt = int8_quantize(&tensor)?;
                quantized.insert(name.to_string(), qt);
                shard_q += 1;
            } else {
                other.insert(name.to_string(), tensor);
                shard_other += 1;
            }
            reporter.on_tensor(&name);
        }

        reporter.on_shard_done(idx, shard_q, shard_other);
        tracing::info!(
            path = %path.display(),
            quantized = shard_q,
            kept_f32 = shard_other,
            "Shard done (INT8 streaming)"
        );
    }

    Ok(QuantizedShards { quantized, other })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytemuck_cast_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn decode_tensor(
    data: &[u8],
    dtype: safetensors::Dtype,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, CoptaiError> {
    match dtype {
        safetensors::Dtype::F32 => {
            Tensor::from_vec(bytemuck_cast_f32(data), shape.to_vec(), device)
                .map_err(CoptaiError::Candle)
        }
        safetensors::Dtype::BF16 => {
            Tensor::from_raw_buffer(data, candle_core::DType::BF16, shape, device)
                .map_err(CoptaiError::Candle)
        }
        safetensors::Dtype::F16 => {
            Tensor::from_raw_buffer(data, candle_core::DType::F16, shape, device)
                .map_err(CoptaiError::Candle)
        }
        safetensors::Dtype::I32 => {
            Tensor::from_raw_buffer(data, candle_core::DType::I64, shape, device)
                .map_err(CoptaiError::Candle)
        }
        other => Err(CoptaiError::Unsupported(format!(
            "dtype {other:?} not yet handled by the loader"
        ))),
    }
}
