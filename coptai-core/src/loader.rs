use std::{collections::HashMap, fs::File, path::Path};

use candle_core::{Device, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::instrument;

use crate::error::CoptaiError;

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

    for path in paths {
        tracing::info!(path = %path.display(), "Memory-mapping safetensors shard");

        let file = File::open(path)?;
        // SAFETY: We hold the file handle open for the lifetime of `mmap`.
        let mmap = unsafe { Mmap::map(&file)? };

        let tensors = SafeTensors::deserialize(&mmap)?;

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            // Convert the raw bytes into a candle Tensor on the target device.
            let tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let floats: Vec<f32> = bytemuck_cast_f32(data);
                    Tensor::from_vec(floats, shape, device).map_err(CoptaiError::Candle)?
                }
                safetensors::Dtype::BF16 => {
                    Tensor::from_raw_buffer(data, candle_core::DType::BF16, &shape, device)
                        .map_err(CoptaiError::Candle)?
                }
                safetensors::Dtype::F16 => {
                    Tensor::from_raw_buffer(data, candle_core::DType::F16, &shape, device)
                        .map_err(CoptaiError::Candle)?
                }
                other => {
                    return Err(CoptaiError::Unsupported(format!(
                        "dtype {other:?} not yet handled by the loader"
                    )))
                }
            };

            weights.insert(name.to_string(), tensor);
        }

        tracing::info!(
            path = %path.display(),
            tensors_loaded = weights.len(),
            "Shard loaded"
        );
    }

    Ok(weights)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reinterpret a `&[u8]` byte slice as `Vec<f32>` via a plain copy.

fn bytemuck_cast_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
