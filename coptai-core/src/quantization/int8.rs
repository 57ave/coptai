//! # quantization::int8


use candle_core::{DType, Tensor};

use crate::error::CoptaiError;

/// Output of the INT8 quantization pass for a single weight tensor.
#[derive(Debug)]
pub struct Int8QuantizedTensor {
    pub quantized: Tensor,
    // scales to recover quantized value
    pub scales: Tensor,
}


pub fn int8_quantize(w: &Tensor) -> Result<Int8QuantizedTensor, CoptaiError> {
    // Work in F32 — convert BF16/F16 inputs once upfront.
    let w_f32 = w.to_dtype(DType::F32).map_err(CoptaiError::Candle)?;

    // Absolute max per output channel → shape [out_ch, 1]
    let abs_max = w_f32
        .abs()
        .map_err(CoptaiError::Candle)?
        .max_keepdim(1)
        .map_err(CoptaiError::Candle)?;

    // scale = abs_max / 127
    let scale = (abs_max / 127.0_f64)
        .map_err(CoptaiError::Candle)?
        .clamp(1e-8_f64, f64::MAX)
        .map_err(CoptaiError::Candle)?; // shape [out_ch, 1]

    // W_q = round(W / scale), clamped to [-127, 127]
    let w_q = w_f32
        .broadcast_div(&scale)
        .map_err(CoptaiError::Candle)?
        .round()
        .map_err(CoptaiError::Candle)?
        .clamp(-127_f64, 127_f64)
        .map_err(CoptaiError::Candle)?;

    // Squeeze scale back to 1-D  [out_ch]
    let scales_1d = scale.squeeze(1).map_err(CoptaiError::Candle)?;

    tracing::debug!(
        shape = ?w.shape(),
        "INT8 quantization complete"
    );

    Ok(Int8QuantizedTensor {
        quantized: w_q,
        scales: scales_1d,
    })
}

/// Dequantize: reconstruct an approximate F32 weight from INT8 + scales.
pub fn int8_dequantize(qt: &Int8QuantizedTensor) -> Result<Tensor, CoptaiError> {
    // Broadcast scale back to [out_ch, 1] so it multiplies row-wise.
    let scale_2d = qt.scales.unsqueeze(1).map_err(CoptaiError::Candle)?;
    qt.quantized
        .broadcast_mul(&scale_2d)
        .map_err(CoptaiError::Candle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn int8_roundtrip_error_bound() -> Result<(), CoptaiError> {
        let device = Device::Cpu;
        // Small matrix: 4 output channels, 8 input channels
        let data: Vec<f32> = (0..32).map(|x| x as f32 - 16.0).collect();
        let w = Tensor::from_vec(data, (4, 8), &device).map_err(CoptaiError::Candle)?;

        let qt = int8_quantize(&w)?;
        let w_approx = int8_dequantize(&qt)?;

        // Max absolute error must be ≤ 1 LSB = max(|w|)/127
        let abs_max = 15.0_f32; // max |x| in [-16..15]
        let lsb = abs_max / 127.0;

        let diff = (w.to_dtype(DType::F32).map_err(CoptaiError::Candle)?
            - w_approx)
            .map_err(CoptaiError::Candle)?
            .abs()
            .map_err(CoptaiError::Candle)?
            .max(0)
            .map_err(CoptaiError::Candle)?;

        let max_err: f32 = diff
            .to_vec1::<f32>()
            .map_err(CoptaiError::Candle)?
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            max_err <= lsb + 1e-5,
            "INT8 round-trip error {max_err} exceeds 1 LSB {lsb}"
        );
        Ok(())
    }
}
