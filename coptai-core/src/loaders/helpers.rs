use candle_core::{Device,Tensor};

use crate::CoptaiError;



pub(super) fn bytemuck_cast_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub(super) fn decode_tensor(
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
            Tensor::from_raw_buffer(data, candle_core::DType::I32, shape, device)
                .map_err(CoptaiError::Candle)
        }
        other => Err(CoptaiError::Unsupported(format!(
            "dtype {other:?} not yet handled by the loader"
        ))),
    }
}
