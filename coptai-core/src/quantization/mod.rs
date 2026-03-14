
//! Quantization passes for coptai-core.

// pub mod int4;
pub mod int8;

pub use int8::{int8_dequantize, int8_quantize};
