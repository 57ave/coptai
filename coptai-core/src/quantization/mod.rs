
//! Quantization passes for coptai-core.

// pub mod int4;
pub mod int8;

pub use int8::int8_quantize_raw;
pub use int8::int8_quantize_tensor;
pub use int8::int8_dequantize;
pub use int8::Int8QuantizedTensor;

pub trait QuantizedTensorSerde {
    /// Returns `(tensor_name, safetensors_dtype, shape, raw_le_bytes)` for
    /// every safetensors entry needed to fully encode this tensor.
    fn to_serde_entries(&self, base_name: &str) -> Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>;
}      