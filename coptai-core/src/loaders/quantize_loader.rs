use std::{collections::HashMap, fs::File, path::Path};

use candle_core::{Device, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::instrument;

use crate::{
    error::CoptaiError,
    progress::{NoopProgress, ProgressReporter},
    quantization::int8::{int8_quantize_raw, Int8QuantizedTensor},
};
use super::helpers::decode_tensor;

pub type ModelWeights = HashMap<String, Tensor>;


pub struct QuantizedShards {
    pub quantized: HashMap<String, Int8QuantizedTensor>,
    pub other: ModelWeights,
}

#[instrument(skip(paths, device))]
pub fn load_and_quantize_int8(
    paths: &[&Path],
    device: &Device,
) -> Result<QuantizedShards, CoptaiError> {
    let mut quantized: HashMap<String, Int8QuantizedTensor> = HashMap::new();
    let mut other: ModelWeights = HashMap::new();
    let reporter = &NoopProgress;

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
            let dtype = view.dtype();

            if shape.len() == 2
                && matches!(
                    dtype,
                    safetensors::Dtype::F32
                        | safetensors::Dtype::BF16
                        | safetensors::Dtype::F16
                )
            {
                // Zero-copy path: quantize directly from mmap bytes.
                let qt = int8_quantize_raw(view.data(), dtype, shape[0], shape[1])?;
                quantized.insert(name.to_string(), qt);
                shard_q += 1;
            } else {
                let tensor = decode_tensor(view.data(), dtype, &shape, device)?;
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