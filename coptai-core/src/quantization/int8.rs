//! # quantization::int8

use rayon::prelude::*;
use safetensors::Dtype as SfDtype;

use crate::quantization::QuantizedTensorSerde;

use crate::error::CoptaiError;

/// Output of the INT8 quantization pass for a single weight tensor.
#[derive(Debug, Clone)]
pub struct Int8QuantizedTensor {
    pub quantized: Vec<i8>,
    pub scales: Vec<f32>,
    pub shape: [usize; 2],
}

// ---------------------------------------------------------------------------
// Zero-copy raw-byte entry point (called from the loader)
// ---------------------------------------------------------------------------

pub fn int8_quantize_raw(
    data: &[u8],
    dtype: SfDtype,
    rows: usize,
    cols: usize,
) -> Result<Int8QuantizedTensor, CoptaiError> {
    // --- decode one row at a time and quantize in parallel ----------------
    let n = rows * cols;
    if data.len() < n * dtype_bytes(dtype) {
        return Err(CoptaiError::Unsupported(format!(
            "int8_quantize_raw: buffer too short ({} bytes for {rows}×{cols} {dtype:?})",
            data.len()
        )));
    }

    // Allocate outputs once.
    let mut quantized = vec![0i8; n];
    let mut scales = vec![0f32; rows];

    // Process each row in parallel.
    quantized
        .par_chunks_mut(cols)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(row, (q_row, scale_out))| {
            let byte_start = row * cols * dtype_bytes(dtype);
            let byte_end = byte_start + cols * dtype_bytes(dtype);
            let row_bytes = &data[byte_start..byte_end];

            // Find abs-max for this row.
            let abs_max = row_abs_max(row_bytes, dtype, cols);
            let scale = (abs_max / 127.0).max(1e-8);
            *scale_out = scale;

            // Quantize each element.
            let inv_scale = 1.0 / scale;
            for (col, q) in q_row.iter_mut().enumerate() {
                let v = read_f32(row_bytes, dtype, col);
                *q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        });

    Ok(Int8QuantizedTensor {
        quantized,
        scales,
        shape: [rows, cols],
    })
}

// ---------------------------------------------------------------------------
// Candle-based entry point (kept for tests / inference use)
// ---------------------------------------------------------------------------

/// Quantize a candle `Tensor` (2-D).  Converts to F32 first, then delegates
/// to the raw path via `to_vec2`.
pub fn int8_quantize_tensor(
    w: &candle_core::Tensor,
) -> Result<Int8QuantizedTensor, CoptaiError> {
    use candle_core::DType;
    let [rows, cols] = match w.dims() {
        [r, c] => [*r, *c],
        _ => {
            return Err(CoptaiError::Unsupported(
                "int8_quantize_tensor: expected 2-D tensor".into(),
            ))
        }
    };
    // to_vec2 does a single contiguous copy into a flat Vec — much cheaper
    // than multiple candle op chains.
    let flat: Vec<f32> = w
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(CoptaiError::Candle)?;

    let mut quantized = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];

    quantized
        .par_chunks_mut(cols)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(row, (q_row, scale_out))| {
            let row_f = &flat[row * cols..(row + 1) * cols];
            let abs_max: f32 = row_f.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let scale = (abs_max / 127.0).max(1e-8_f32);
            *scale_out = scale;
            let inv = 1.0 / scale;
            for (col, q) in q_row.iter_mut().enumerate() {
                *q = (row_f[col] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        });

    Ok(Int8QuantizedTensor {
        quantized,
        scales,
        shape: [rows, cols],
    })
}

/// Dequantize back to a candle F32 tensor (for tests / inference).
pub fn int8_dequantize(qt: &Int8QuantizedTensor) -> Result<candle_core::Tensor, CoptaiError> {
    use candle_core::Device;
    let [rows, cols] = qt.shape;
    let mut out = vec![0f32; rows * cols];
    for (row, (&scale, q_row)) in qt
        .scales
        .iter()
        .zip(qt.quantized.chunks(cols))
        .enumerate()
    {
        for (col, &q) in q_row.iter().enumerate() {
            out[row * cols + col] = q as f32 * scale;
        }
    }
    candle_core::Tensor::from_vec(out, (rows, cols), &Device::Cpu)
        .map_err(CoptaiError::Candle)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn dtype_bytes(dtype: SfDtype) -> usize {
    match dtype {
        SfDtype::F32 => 4,
        SfDtype::BF16 | SfDtype::F16 => 2,
        _ => 4, // fallback — caller validates
    }
}

#[inline(always)]
fn read_f32(row_bytes: &[u8], dtype: SfDtype, col: usize) -> f32 {
    match dtype {
        SfDtype::F32 => {
            let b = &row_bytes[col * 4..col * 4 + 4];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }
        SfDtype::BF16 => {
            let b = &row_bytes[col * 2..col * 2 + 2];
            let bits = u16::from_le_bytes([b[0], b[1]]);
            // BF16 → F32: upper 16 bits of the F32 bit pattern
            f32::from_bits((bits as u32) << 16)
        }
        SfDtype::F16 => {
            let b = &row_bytes[col * 2..col * 2 + 2];
            half::f16::from_le_bytes([b[0], b[1]]).to_f32()
        }
        _ => {
            // Unreachable for well-formed weight files
            0.0
        }
    }
}

#[inline(always)]
fn row_abs_max(row_bytes: &[u8], dtype: SfDtype, cols: usize) -> f32 {
    let mut max = 0.0_f32;
    for col in 0..cols {
        let v = read_f32(row_bytes, dtype, col).abs();
        if v > max {
            max = v;
        }
    }
    max
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl QuantizedTensorSerde for Int8QuantizedTensor {
    /// Yields two entries per weight tensor:
    ///   `{base_name}.qint8`  — I8,  shape [rows, cols]
    ///   `{base_name}.scales` — F32, shape [rows]
    fn to_serde_entries(
        &self,
        base_name: &str,
    ) -> Vec<(String, SfDtype, Vec<usize>, Vec<u8>)> {
        let [rows, cols] = self.shape;

        // i8 → u8 reinterpret (same bit pattern, 1-byte elements)
        let qbytes: Vec<u8> = self.quantized.iter().map(|&b| b as u8).collect();

        let scale_bytes: Vec<u8> = self
            .scales
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        vec![
            (format!("{base_name}.qint8"),  SfDtype::I8,  vec![rows, cols], qbytes),
            (format!("{base_name}.scales"), SfDtype::F32, vec![rows],        scale_bytes),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_error_bound() -> Result<(), CoptaiError> {
        // Small matrix: 4 output channels, 8 input channels
        let data: Vec<f32> = (0..32).map(|x| x as f32 - 16.0).collect();

        // Encode as F32 bytes (little-endian)
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();

        let qt = int8_quantize_raw(&bytes, SfDtype::F32, 4, 8)?;
        let w_approx = int8_dequantize(&qt)?;

        // Max absolute error must be ≤ 1 LSB = max(|w|)/127
        let abs_max = 15.0_f32; // max |x| in {-16..15}
        let lsb = abs_max / 127.0;

        let approx_flat: Vec<f32> = w_approx
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(CoptaiError::Candle)?;

        let max_err = data
            .iter()
            .zip(approx_flat.iter())
            .map(|(orig, approx)| (orig - approx).abs())
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            max_err <= lsb + 1e-5,
            "INT8 round-trip error {max_err} exceeds 1 LSB {lsb}"
        );
        Ok(())
    }

    #[test]
    fn bf16_roundtrip() -> Result<(), CoptaiError> {
        // Encode 4×4 matrix as BF16
        let data: Vec<f32> = (0..16).map(|x| x as f32 * 0.5).collect();
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                let bf16_bits = (bits >> 16) as u16;
                bf16_bits.to_le_bytes()
            })
            .collect();

        let qt = int8_quantize_raw(&bytes, SfDtype::BF16, 4, 4)?;
        assert_eq!(qt.quantized.len(), 16);
        assert_eq!(qt.scales.len(), 4);
        Ok(())
    }
}
